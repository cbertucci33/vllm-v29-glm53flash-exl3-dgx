# GLM chat-template provenance

`chat_template_glm53_official-690b705.jinja` is based on the template in
`zai-org/GLM-5.3-Flash` revision
`690b705278a3a58e538fcb37c2ca8b5f9511213c`.

- Official SHA-256: `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`
- Packaged SHA-256: `7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb`
- Local behavior: request-level thinking control, explicit reasoning effort,
  ordered parallel tool results, multimodal markers, and fail-safe handling
  of ambiguous tool-result IDs.

The image stores the template at `/opt/glm53/chat_template.jinja`. The launch
contract must select it explicitly together with the `glm47` tool parser and
`glm45` reasoning parser.
