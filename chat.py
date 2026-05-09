#!/usr/bin/env python3
"""
Cliente de consola interactivo para hablar con Gemma 4 servido por llama.cpp.

Características:
- Conecta al servidor local en http://127.0.0.1:8080
- Streaming de tokens en tiempo real (los ves aparecer según se generan)
- Mantiene historial completo de la conversación
- Colores en terminal
- Comandos especiales: /salir, /reset, /stats, /system, /ayuda

Uso:
    python3 chat.py

Requisitos:
    - llama-server corriendo en localhost:8080
    - Python 3.8+
    - librería requests (instalada por defecto en Ubuntu)
"""

import json
import re
import sys
import time
import requests

# ──── Configuración ────────────────────────────────────────────────────────
SERVER_URL = "http://127.0.0.1:8080"
MODEL_NAME = "gemma-4-E4B"
DEFAULT_SYSTEM = (
    "Te llamas Mabel, asistente de apoyo emocional para estudiantes "
    "universitarios colombianos de la UMB. Escucha activa, valida emociones, "
    "haz preguntas exploratorias antes de dar consejos. No eres psicóloga "
    "profesional. Responde en español, breve (máx 3-4 frases), conversacional, "
    "sin Markdown ni listas. Si hay crisis (suicidio, autolesión), mantén la "
    "calma, valida, y deriva a Línea 123, Línea 106 o Bienestar UMB."
)

# ──── Colores ANSI ─────────────────────────────────────────────────────────
C_RESET   = "\033[0m"
C_USER    = "\033[1;36m"   # cian negrita
C_MODEL   = "\033[1;32m"   # verde negrita
C_SYSTEM  = "\033[1;33m"   # amarillo negrita
C_DIM     = "\033[2m"      # tenue
C_ERROR   = "\033[1;31m"   # rojo negrita


def detect_model_name():
    """Consulta /v1/models y devuelve un nombre legible del modelo activo."""
    try:
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        if r.status_code != 200:
            return "Gemma 4"
        data = r.json()
        items = data.get("data") or []
        if not items:
            return "Gemma 4"
        raw = items[0].get("id", "") or ""
        return format_model_name(raw)
    except Exception:
        return "Gemma 4"


def format_model_name(raw):
    """
    Convierte nombres crudos del GGUF a etiquetas legibles.

    Ejemplos:
      gemma-4-26B-A4B-it-UD-Q4_K_M.gguf → Gemma 4 26B MoE (UD-Q4_K_M)
      gemma-4-E4B-it-Q4_K_M.gguf        → Gemma 4 E4B (Q4_K_M)
      gemma-4-31B-it-Q5_K_M.gguf        → Gemma 4 31B (Q5_K_M)
    """
    name = raw.rsplit("/", 1)[-1]  # si viene con ruta, quedarse con el basename
    name = name.replace(".gguf", "")

    if "26B-A4B" in name or "26B_A4B" in name:
        base = "Gemma 4 26B MoE"
    elif "E4B" in name:
        base = "Gemma 4 E4B"
    elif "E2B" in name:
        base = "Gemma 4 E2B"
    elif "31B" in name:
        base = "Gemma 4 31B"
    else:
        base = "Gemma 4"

    quant_match = re.search(r"((?:UD-|IQ|MXFP)?[IQ]\d+(?:_[KMSXL0-9]+)*)", name)
    if quant_match:
        quant = quant_match.group(1)
        return f"{base} ({quant})"
    return base


def print_header(model_label):
    print(f"{C_SYSTEM}{'═' * 70}{C_RESET}")
    title = f"  Mabel — Asistente de apoyo emocional ({model_label})"
    print(f"{C_SYSTEM}{title}{C_RESET}")
    print(f"{C_SYSTEM}{'═' * 70}{C_RESET}")
    print(f"{C_DIM}  Servidor: {SERVER_URL}{C_RESET}")
    print(f"{C_DIM}  Comandos: /salir  /reset  /stats  /system  /ayuda{C_RESET}")
    print(f"{C_SYSTEM}{'═' * 70}{C_RESET}\n")


def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def stream_response(messages):
    """Envía mensajes y hace streaming de la respuesta. Devuelve (texto, stats)."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    try:
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=600,
        )
    except requests.exceptions.ConnectionError:
        print(f"\n{C_ERROR}✗ No se puede conectar al servidor en {SERVER_URL}{C_RESET}")
        print(f"{C_DIM}  ¿Está corriendo llama-server? Verifica con: ps aux | grep llama-server{C_RESET}\n")
        return None, None

    if response.status_code != 200:
        print(f"\n{C_ERROR}✗ Servidor devolvió HTTP {response.status_code}{C_RESET}")
        print(response.text[:500])
        return None, None

    response.encoding = "utf-8"

    full_text = ""
    first_token_time = None
    start_time = time.time()
    token_count = 0
    state = None  # None | "reasoning" | "content"

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = data.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        reasoning_chunk = delta.get("reasoning_content")
        content_chunk = delta.get("content")

        if reasoning_chunk:
            if state != "reasoning":
                print(f"{C_DIM}💭 [pensando...]{C_RESET}")
                print(f"{C_DIM}", end="", flush=True)
                state = "reasoning"
            if first_token_time is None:
                first_token_time = time.time()
            print(reasoning_chunk, end="", flush=True)

        if content_chunk:
            if state != "content":
                if state == "reasoning":
                    print(f"{C_RESET}\n")  # cerrar dim y separar del reasoning
                print(f"{C_MODEL}Mabel:{C_RESET} ", end="", flush=True)
                state = "content"
            if first_token_time is None:
                first_token_time = time.time()
            print(content_chunk, end="", flush=True)
            full_text += content_chunk
            token_count += 1  # aproximado (delta puede traer varios tokens)

    if state == "reasoning":
        # El modelo se cortó antes de salir del reasoning — cerrar dim
        print(C_RESET)
    print("\n")
    elapsed = time.time() - start_time
    ttft = (first_token_time - start_time) if first_token_time else None

    return full_text, {
        "elapsed": elapsed,
        "ttft": ttft,
        "deltas": token_count,
    }


def main():
    if not check_server():
        print_header("sin conexión")
        print(f"{C_ERROR}✗ El servidor en {SERVER_URL} no responde.{C_RESET}")
        print(f"{C_DIM}  Lánzalo primero con:")
        print(f"    llama-server -m modelos/gemma-4-E4B-it-Q4_K_M.gguf \\")
        print(f"                 -c 4096 -t 6 --host 127.0.0.1 --port 8080 --mlock{C_RESET}\n")
        sys.exit(1)

    model_label = detect_model_name()
    print_header(model_label)

    print(f"{C_DIM}✓ Servidor activo. Escribe tu mensaje y pulsa Enter.{C_RESET}\n")

    system_prompt = DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system_prompt}]
    stats_total = {"turns": 0, "total_time": 0.0}

    while True:
        try:
            user_input = input(f"{C_USER}Tú:{C_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_DIM}Adiós.{C_RESET}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/salir", "/exit", "/quit"):
                print(f"{C_DIM}Adiós.{C_RESET}")
                break
            elif cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                stats_total = {"turns": 0, "total_time": 0.0}
                print(f"{C_SYSTEM}✓ Conversación reiniciada.{C_RESET}\n")
                continue
            elif cmd == "/stats":
                print(f"{C_SYSTEM}  Turnos: {stats_total['turns']}{C_RESET}")
                print(f"{C_SYSTEM}  Tiempo total de generación: {stats_total['total_time']:.1f}s{C_RESET}")
                print(f"{C_SYSTEM}  Mensajes en historial: {len(messages)}{C_RESET}\n")
                continue
            elif cmd == "/system":
                print(f"{C_SYSTEM}System prompt actual:{C_RESET}")
                print(f"{C_DIM}{system_prompt}{C_RESET}\n")
                continue
            elif cmd in ("/ayuda", "/help"):
                print(f"{C_SYSTEM}Comandos disponibles:{C_RESET}")
                print(f"  {C_DIM}/salir     — salir del chat{C_RESET}")
                print(f"  {C_DIM}/reset     — limpiar el historial de la conversación{C_RESET}")
                print(f"  {C_DIM}/stats     — ver estadísticas de la sesión{C_RESET}")
                print(f"  {C_DIM}/system    — ver el system prompt actual{C_RESET}")
                print(f"  {C_DIM}/ayuda     — mostrar esta ayuda{C_RESET}\n")
                continue
            else:
                print(f"{C_ERROR}Comando desconocido: {user_input}{C_RESET}")
                print(f"{C_DIM}Usa /ayuda para ver los comandos disponibles.{C_RESET}\n")
                continue

        messages.append({"role": "user", "content": user_input})
        reply, stats = stream_response(messages)

        if reply is None:
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        stats_total["turns"] += 1
        stats_total["total_time"] += stats["elapsed"]

        ttft_str = f"{stats['ttft']:.1f}s" if stats["ttft"] else "N/A"
        print(f"{C_DIM}   [{stats['elapsed']:.1f}s total · primer token en {ttft_str}]{C_RESET}\n")


if __name__ == "__main__":
    main()
