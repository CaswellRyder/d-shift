/* Exact RGB24 table lookup. No HSV approximation, global state or actuation. */
#include <stddef.h>
#include <stdint.h>

uint32_t dtr_color_lookup_abi(void) { return 1; }

void dtr_color_lookup(const uint8_t *rgb, size_t pixels, const uint8_t *table,
                      uint8_t *red, uint8_t *blue) {
    for (size_t i = 0; i < pixels; ++i) {
        uint32_t index = ((uint32_t)rgb[3*i] << 16) |
                         ((uint32_t)rgb[3*i+1] << 8) | rgb[3*i+2];
        uint8_t flags = table[index];
        red[i] = (flags & 1) ? 255 : 0;
        blue[i] = (flags & 2) ? 255 : 0;
    }
}
