//! End-to-end shape checks for the Anthropic content-block builder.
//!
//! Each test asserts the serialized JSON matches the dict shape the
//! Python reference library emits.

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use llm_content_blocks::{
    BlockError, Blocks, CacheControl, ContentBlock, DocumentSource, ImageSource,
    VALID_DOCUMENT_MEDIA_TYPES, VALID_IMAGE_MEDIA_TYPES,
};
use serde_json::json;

// ---- text --------------------------------------------------------------

#[test]
fn text_basic() {
    let out = Blocks::new().text("hello").build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(json, json!([{"type": "text", "text": "hello"}]));
}

#[test]
fn text_cache_control() {
    let out = Blocks::new()
        .text_with_cache("x", CacheControl::Ephemeral)
        .build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(json[0]["cache_control"], json!({"type": "ephemeral"}));
}

#[test]
fn text_no_cache_control_by_default() {
    let out = Blocks::new().text("x").build();
    let json = serde_json::to_value(&out).unwrap();
    assert!(json[0].get("cache_control").is_none());
}

// ---- image_b64 ---------------------------------------------------------

#[test]
fn image_b64_with_bytes_roundtrip() {
    let out = Blocks::new()
        .image_b64(b"\x89PNG", "image/png")
        .unwrap()
        .build();
    let ContentBlock::Image { source, .. } = &out[0] else {
        panic!("expected Image variant");
    };
    let ImageSource::Base64 { media_type, data } = source else {
        panic!("expected Base64 image source");
    };
    assert_eq!(media_type, "image/png");
    let decoded = BASE64_STANDARD.decode(data).unwrap();
    assert_eq!(decoded, b"\x89PNG");
}

#[test]
fn image_b64_rejects_unknown_media_type() {
    let err = Blocks::new().image_b64(b"x", "image/tiff").unwrap_err();
    assert!(matches!(&err, BlockError::UnsupportedImageMediaType(t) if t == "image/tiff"));
    // Display formatting also works (Display + std::error::Error).
    let msg = format!("{err}");
    assert!(msg.contains("image/tiff"));
}

#[test]
fn image_b64_cache_control() {
    let out = Blocks::new()
        .image_b64_with_cache(b"x", "image/png", CacheControl::Ephemeral)
        .unwrap()
        .build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(json[0]["cache_control"], json!({"type": "ephemeral"}));
}

#[test]
fn valid_image_media_types_table() {
    for t in ["image/png", "image/jpeg", "image/webp", "image/gif"] {
        assert!(VALID_IMAGE_MEDIA_TYPES.contains(&t), "missing: {t}");
    }
}

// ---- image_url ---------------------------------------------------------

#[test]
fn image_url_block_shape() {
    let out = Blocks::new().image_url("https://example.com/x.png").build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(
        json,
        json!([{
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/x.png"},
        }])
    );
}

// ---- tool_use ----------------------------------------------------------

#[test]
fn tool_use_shape_and_input_preserved() {
    let out = Blocks::new()
        .tool_use("toolu_1", "search", json!({"q": "anthropic"}))
        .build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(
        json,
        json!([{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "search",
            "input": {"q": "anthropic"},
        }])
    );
}

// ---- tool_result -------------------------------------------------------

#[test]
fn tool_result_block_via_builder() {
    let out = Blocks::new()
        .tool_result("u1", json!("answer"), false)
        .build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(
        json,
        json!([{
            "type": "tool_result",
            "tool_use_id": "u1",
            "content": "answer",
        }])
    );
}

#[test]
fn tool_result_block_is_error_flag_serializes() {
    let out = Blocks::new().tool_result("u1", json!("boom"), true).build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(json[0]["is_error"], json!(true));
}

#[test]
fn tool_result_one_shot_static() {
    let block = Blocks::tool_result_block("u1", json!("answer"), false);
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(
        json,
        json!({
            "type": "tool_result",
            "tool_use_id": "u1",
            "content": "answer",
        })
    );
}

#[test]
fn tool_result_one_shot_static_is_error() {
    let block = Blocks::tool_result_block("u1", json!("boom"), true);
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["is_error"], json!(true));
}

#[test]
fn tool_result_omits_is_error_when_false() {
    let block = Blocks::tool_result_block("u1", json!("ok"), false);
    let json = serde_json::to_value(&block).unwrap();
    assert!(json.get("is_error").is_none());
}

// ---- document ---------------------------------------------------------

#[test]
fn document_b64_default_pdf() {
    let out = Blocks::new().document_pdf_b64(b"%PDF-").build();
    let json = serde_json::to_value(&out).unwrap();
    assert_eq!(json[0]["type"], "document");
    assert_eq!(json[0]["source"]["media_type"], "application/pdf");
    assert_eq!(json[0]["source"]["type"], "base64");
}

#[test]
fn document_b64_rejects_unknown_media_type() {
    let err = Blocks::new()
        .document_b64(b"x", "application/xml")
        .unwrap_err();
    assert!(matches!(&err, BlockError::UnsupportedDocumentMediaType(t) if t == "application/xml"));
}

#[test]
fn document_b64_text_plain_accepted() {
    let out = Blocks::new()
        .document_b64(b"hello", "text/plain")
        .unwrap()
        .build();
    let ContentBlock::Document { source, .. } = &out[0] else {
        panic!("expected Document variant");
    };
    let DocumentSource::Base64 { media_type, data } = source;
    assert_eq!(media_type, "text/plain");
    assert_eq!(BASE64_STANDARD.decode(data).unwrap(), b"hello");
}

#[test]
fn valid_document_media_types_table() {
    assert!(VALID_DOCUMENT_MEDIA_TYPES.contains(&"application/pdf"));
    assert!(VALID_DOCUMENT_MEDIA_TYPES.contains(&"text/plain"));
}

// ---- composition / extend --------------------------------------------

#[test]
fn chain_multiple_blocks() {
    let out = Blocks::new()
        .text("Look:")
        .image_b64(b"\x89PNG", "image/png")
        .unwrap()
        .text("done")
        .build();
    let types: Vec<&str> = out
        .iter()
        .map(|b| match b {
            ContentBlock::Text { .. } => "text",
            ContentBlock::Image { .. } => "image",
            ContentBlock::ToolUse { .. } => "tool_use",
            ContentBlock::ToolResult { .. } => "tool_result",
            ContentBlock::Document { .. } => "document",
        })
        .collect();
    assert_eq!(types, vec!["text", "image", "text"]);
}

#[test]
fn extend_from_existing_blocks() {
    let existing = vec![ContentBlock::Text {
        text: "from-elsewhere".to_string(),
        cache_control: None,
    }];
    let out = Blocks::new().text("first").extend(existing).build();
    assert_eq!(out.len(), 2);
    let ContentBlock::Text { text, .. } = &out[1] else {
        panic!("expected Text variant");
    };
    assert_eq!(text, "from-elsewhere");
}

// ---- build / build_message -------------------------------------------

#[test]
fn build_consumes_self() {
    let blocks = Blocks::new().text("x").build();
    assert_eq!(blocks.len(), 1);
    // builder is gone; this is a compile-time guarantee.
    // (uncommenting `blocks_builder.build()` again would not compile.)
}

#[test]
fn build_message_user_shape() {
    let msg = Blocks::new().text("Hi").build_message("user");
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        json,
        json!({
            "role": "user",
            "content": [{"type": "text", "text": "Hi"}],
        })
    );
}

#[test]
fn build_message_assistant_shape() {
    let msg = Blocks::new()
        .text("OK")
        .tool_use("u", "search", json!({"q": "x"}))
        .build_message("assistant");
    assert_eq!(msg.role, "assistant");
    assert_eq!(msg.content.len(), 2);
}

#[test]
fn len_and_is_empty_reflect_blocks() {
    let mut builder = Blocks::new();
    assert_eq!(builder.len(), 0);
    assert!(builder.is_empty());
    builder.text("a").text("b");
    assert_eq!(builder.len(), 2);
    assert!(!builder.is_empty());
}

// ---- JSON shape regression vs Anthropic dict shape --------------------

#[test]
fn json_output_matches_anthropic_text_shape() {
    // The Python reference: [{"type": "text", "text": "hi"}]
    let blocks = Blocks::new().text("hi").build();
    let json = serde_json::to_string(&blocks).unwrap();
    assert_eq!(json, r#"[{"type":"text","text":"hi"}]"#);
}

#[test]
fn json_output_matches_anthropic_image_b64_shape() {
    // The Python reference shape for image_b64:
    // {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    let blocks = Blocks::new()
        .image_b64(b"abc", "image/png")
        .unwrap()
        .build();
    let v = serde_json::to_value(&blocks).unwrap();
    assert_eq!(v[0]["type"], "image");
    assert_eq!(v[0]["source"]["type"], "base64");
    assert_eq!(v[0]["source"]["media_type"], "image/png");
    assert_eq!(v[0]["source"]["data"], BASE64_STANDARD.encode(b"abc"));
}

#[test]
fn json_output_matches_anthropic_tool_use_shape() {
    let blocks = Blocks::new()
        .tool_use("toolu_1", "search", json!({"q": "x"}))
        .build();
    let v = serde_json::to_value(&blocks).unwrap();
    assert_eq!(
        v,
        json!([{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "search",
            "input": {"q": "x"},
        }])
    );
}

#[test]
fn json_output_matches_anthropic_document_b64_shape() {
    // The Anthropic document base64 source shape:
    // {"type": "document", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    let blocks = Blocks::new()
        .document_b64(b"%PDF-", "application/pdf")
        .unwrap()
        .build();
    let v = serde_json::to_value(&blocks).unwrap();
    assert_eq!(v[0]["type"], "document");
    assert_eq!(v[0]["source"]["type"], "base64");
    assert_eq!(v[0]["source"]["media_type"], "application/pdf");
    assert_eq!(v[0]["source"]["data"], BASE64_STANDARD.encode(b"%PDF-"));
}

// ---- builder lifecycle -------------------------------------------------

#[test]
fn default_matches_new_empty_builder() {
    let mut builder = Blocks::default();
    assert!(builder.is_empty());
    assert_eq!(builder.len(), 0);
    builder.text("x");
    assert_eq!(builder.len(), 1);
}

#[test]
fn build_drains_builder_so_second_call_is_empty() {
    // Documented contract: build() takes &mut self and leaves the builder
    // empty, so a second build() yields no blocks.
    let mut builder = Blocks::new();
    builder.text("first");
    let first = builder.build();
    assert_eq!(first.len(), 1);
    let second = builder.build();
    assert!(second.is_empty());
    assert!(builder.is_empty());
}
