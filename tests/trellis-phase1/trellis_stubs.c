// Placeholder encoder/decoder + code function. Phase-1 Task #90+91+92
// will replace these with real implementations.

#include "trellis_phase1.h"
#include <string.h>

float trellis_code(trellis_code_fn fn, uint32_t state, int state_bits) {
    (void)fn; (void)state; (void)state_bits;
    return 0.0f;
}

float trellis_encode_block(const trellis_config * cfg, const float * x, trellis_block * out) {
    (void)cfg; (void)x;
    memset(out, 0, sizeof(*out));
    return 0.0f;
}

void trellis_decode_block(const trellis_config * cfg, const trellis_block * in, float * y) {
    (void)in;
    memset(y, 0, (size_t)cfg->block_size * sizeof(float));
}
