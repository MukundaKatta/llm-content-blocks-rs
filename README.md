# llm-content-blocks

[![Crates.io](https://img.shields.io/crates/v/llm-content-blocks.svg)](https://crates.io/crates/llm-content-blocks)
[![Documentation](https://docs.rs/llm-content-blocks/badge.svg)](https://docs.rs/llm-content-blocks)
[![License](https://img.shields.io/crates/l/llm-content-blocks.svg)](https://crates.io/crates/llm-content-blocks)

**Typed fluent builder for Anthropic Messages-API content blocks.** No SDK requirement, narrow deps (`serde`, `serde_json`, `base64`).

```rust
use llm_content_blocks::{Blocks, CacheControl};

let content = Blocks::new()
    .text("Look at this image:")
    .image_b64(&png_bytes, "image/png").unwrap()
    .text_with_cache("What's in it?", CacheControl::Ephemeral)
    .build();

// wrap as a full message
let user_msg = Blocks::new().text("Hi").build_message("user");
// Message { role: "user", content: vec![Text { text: "Hi", ... }] }

// tool result (one-shot, not chained)
use serde_json::json;
let tr = Blocks::tool_result_block("toolu_1", json!("the answer"), false);
```

## Why

The Anthropic Messages content-block shapes are simple but fiddly when you assemble them programmatically: tool responses, multi-image inputs, cached system prompts, retry-with-correction loops. `Blocks` is a fluent builder that emits the exact JSON shape the API expects.

No SDK dependency. The crate doesn't talk to the API. It just makes the right shape, so you can use it from any HTTP client (`reqwest`, `ureq`, `hyper`) or pass the output through `serde_json` to the official SDK.

Supported blocks: `text`, `image` (base64 or URL), `tool_use`, `tool_result`, `document` (PDF, plain text).

## Install

```toml
[dependencies]
llm-content-blocks = "0.1"
```

## API

```rust
use llm_content_blocks::{Blocks, CacheControl};
use serde_json::json;

// chained appends, return &mut self
let mut b = Blocks::new();
b.text("hello");
b.text_with_cache("cached body", CacheControl::Ephemeral);
b.image_b64(&bytes, "image/png").unwrap();
b.image_url("https://example.com/x.png");
b.tool_use("toolu_1", "search", json!({"q": "anthropic"}));
b.tool_result("toolu_1", json!("the answer"), false);
b.document_b64(&pdf_bytes, "application/pdf").unwrap();
b.document_pdf_b64(&pdf_bytes);

// terminal
let blocks = b.build();                              // Vec<ContentBlock>
let msg = Blocks::new().text("hi").build_message("user"); // Message { role, content }

// one-shot helpers
let tr = Blocks::tool_result_block("toolu_1", json!("answer"), false);
```

`CacheControl::Ephemeral` is the only valid value today (Anthropic's prompt cache). The cache_control field is omitted from JSON when unset. The crate rejects unknown image and document media types up front with a typed [`BlockError`], so you get a build-site error instead of a 400 from the API.

## Serialization

`ContentBlock` derives `serde::Serialize` with `#[serde(tag = "type", rename_all = "snake_case")]`. The produced JSON matches the Anthropic dict shape exactly. Pass `&content` (or the `Message` envelope) to `serde_json::to_value` / `serde_json::to_string` and ship it.

## Companion libraries

- [`claude-cost`](https://crates.io/crates/claude-cost) - cache-aware cost calculator for Anthropic + Bedrock model IDs.
- [`agentprompt-rs`](https://crates.io/crates/agentprompt-rs) - Jinja-style prompt template render; pair with this builder for final message construction.
- [`claude-stream-rs`](https://crates.io/crates/claude-stream-rs) - incremental SSE parser for Anthropic streamed responses.

## License

MIT OR Apache-2.0
