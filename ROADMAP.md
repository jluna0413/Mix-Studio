# Mix Studio roadmap

This file tracks deferred work that is useful but not ready to prioritize over generation reliability and the core creation experience.

## Intel Arc / XPU fork track (this repository)

This fork ([jluna0413/Mix-Studio](https://github.com/jluna0413/Mix-Studio), branch `intel-arc-xpu`) is being developed to make Mix Studio usable on Intel Arc / XPU-backed ComfyUI setups where practical.

**Current focus (validated in this tree: `node --check` clean, `node --test` 1584/1584 pass):**
- H3 text/frame/reference graph compatibility fixes for XPU-oriented stacks (capability probing, loader/`weight_dtype` handling, attention backend selection, CLIP safety).
- Prevent H3 DiT shape-mismatch crashes from loading a 4B lightweight CLIP without its required projection/adapter path.
- Splat/orbit workflow correctness fixes and stranded-job/device-lost recovery (`UR_RESULT_ERROR_DEVICE_LOST`).

**Known gaps / bugs to track:** slow H3/3D performance on Arc, driver/device-loss instability, ComfyUI-side patch requirements for quantized H3 CLIP/vision loading, conditional lightweight/distilled encoder support, limited XPU attention acceleration parity, and experimental splat quality. See [docs/intel-arc-compatibility.md](docs/intel-arc-compatibility.md).

**Planning alignment with upstream:** monitor upstream `BlackMixture/Mix-Studio` releases/PRs/issues (no published v1.4 roadmap/milestone in-repo as of 2026-11-16; recent/open signals include #75 H3 SageAttention inspection, #80 OpenAI-compatible local provider, #83 Krea/Klein vs Qwen steps/CFG separation, #85/#84 smart-planner shot continuity) for conflicts and upstreamable fixes.

## Deferred integrations

### Gemini Spark remote control

**Status:** Deferred while Gemini Spark's custom MCP support remains in beta.

What we established during the first integration pass:

- Mix Studio exposes a narrow, secret-path MCP endpoint through Tailscale Funnel without publishing the web app itself.
- The endpoint successfully handles Streamable HTTP initialization, tool discovery, and tool calls.
- Browser-origin requests from `https://gemini.google.com` receive the required CORS response.
- Gemini Spark still rejects the custom URL before establishing an MCP session, despite those protocol and browser checks succeeding.

Re-evaluate this integration after Gemini Spark's validator requirements or beta documentation change. At that point:

1. Test the endpoint on default HTTPS port 443 to determine whether Spark rejects nonstandard port 8443.
2. Confirm whether Spark requires OAuth, dynamic client registration, or another authentication handshake instead of a secret URL.
3. Re-run initialization, discovery, invocation, and browser-origin checks against Spark's current requirements.
4. Preserve the current security boundary: expose only the MCP endpoint and its limited tool set, not the Mix Studio web interface.
