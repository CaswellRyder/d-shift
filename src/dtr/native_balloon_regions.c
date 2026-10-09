/* Research-only exact MSER reduction. No model, camera or actuator access. */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

uint32_t dtr_balloon_regions_abi(void) { return 1; }

/* Keep horizontal extrema of each row: every removed point is on the segment
 * joining its row extrema, so the convex hull is unchanged. Plane/points are
 * fixed-contract contiguous buffers; all coordinates are checked before use.
 * Return -1 for invalid input, 0 for an original-search rejection, otherwise
 * the number of output points (at most 480). Caller owns 960 int32 slots. */
int32_t dtr_balloon_region(const uint8_t *plane, const int32_t *points,
                          size_t count, int32_t *out) {
    if (!plane || !points || !out || count == 0 || count > 76800) return -1;
    int32_t lo[240], hi[240];
    for (int y = 0; y < 240; ++y) { lo[y] = INT_MAX; hi[y] = -1; }
    int32_t xmin = 320, xmax = -1, ymin = 240, ymax = -1;
    uint32_t sum = 0; /* At most 76800 * 255, no uint32 overflow. */
    for (size_t i = 0; i < count; ++i) {
        int32_t x = points[2*i], y = points[2*i+1];
        if (x < 0 || x >= 320 || y < 0 || y >= 240) return -1;
        sum += plane[y*320+x];
        if (x < lo[y]) lo[y] = x;
        if (x > hi[y]) hi[y] = x;
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
    }
    int32_t w = xmax-xmin+1, h = ymax-ymin+1;
    /* Exactly equivalent to mean < 15 and the original bounding checks. */
    if (sum < 15*count || w < 4 || h < 4 || w > 3*h || h > 3*w ||
        w*h > 57600) return 0;
    int32_t n = 0;
    for (int y = ymin; y <= ymax; ++y) {
        if (hi[y] < 0) continue;
        out[2*n] = lo[y]; out[2*n+1] = y; ++n;
        if (hi[y] != lo[y]) {
            out[2*n] = hi[y]; out[2*n+1] = y; ++n;
        }
    }
    return n;
}
