#!/usr/bin/env python3
# Self-generated evaluation text for the KV-cache KL divergence of chat models: the model (llama-server with an f16
# KV cache) writes ~450-word answers on 40 topics; the answers, without chat markup, are text from its own
# distribution (raw wikitext is far out of distribution for chat models such as gpt-oss and Gemma 4).
# usage: kld-gen-text.py <server port> <output file> [chat_template_kwargs as JSON]
import json, sys, urllib.request
port, out = int(sys.argv[1]), sys.argv[2]
kwargs = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
topics = [
 "how a refrigerator works", "the history of the printing press", "why the sky is blue", "how vaccines train the immune system",
 "the causes of the French Revolution", "how a CPU executes instructions", "the water cycle", "how bees communicate",
 "the basics of compound interest", "how airplanes stay in the air", "the life cycle of a star", "how the internet routes packets",
 "the rules and strategy of chess openings", "how photosynthesis works", "the Roman road network", "how earthquakes happen",
 "how to bake sourdough bread", "the theory of evolution by natural selection", "how electric cars store energy", "the origins of jazz",
 "how a bill becomes law in a parliament", "the structure of DNA", "how noise-cancelling headphones work", "the Silk Road trade routes",
 "how glaciers shape landscapes", "the basics of supply and demand", "how GPS determines position", "the history of the Olympic Games",
 "how the human heart pumps blood", "why we have seasons", "how a nuclear reactor produces power", "the art of the Renaissance",
 "how coffee is grown and roasted", "the basics of machine learning", "how volcanoes form", "the history of writing systems",
 "how tides are caused by the moon", "how a camera lens focuses light", "the migration of birds", "the invention of the telephone",
]
with open(out, "w") as f:
    for i, t in enumerate(topics):
        body = {"messages": [{"role": "user", "content": f"Write a clear, detailed explanation of {t} in about 450 words, in flowing paragraphs without headings or lists."}],
                "max_tokens": 900, "temperature": 0.8, "seed": 1000 + i, "chat_template_kwargs": kwargs}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        msg = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]
        f.write((msg.get("content") or "").strip() + "\n\n")
        f.flush()
        print(i, t, len(msg.get("content") or ""), flush=True)
