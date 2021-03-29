#include "rtm.hpp"
const int BITSET_WIDTH = 8;
using namespace SSE_SHIFT;
#define _CYCLIC_SHIFT1_ 57    //_MM_SHUFFLE(0,3,2,1); //rotating right
#define _CYCLIC_SHIFT2_ 147   //_MM_SHUFFLE(2,1,0,3); //rotating left
#define _CYCLIC_SHIFT3_ 78    //_MM_SHUFFLE(1,0,3,2);
#define _MASK_SHIFT_ 27       //_MM_SHUFFLE(0,1,2,3);
// #define STACK_THRESHOLD 16368 // 4 * 8k: 2 ** 16 bytes
namespace RTM_AVX2 {

void intersect(uint8_t *bitmap_a, uint8_t *bitmap_b, uint8_t *out, int vector_size)
{
    for (int i = 0; i < vector_size; i++)
    {
        out[i] = bitmap_a[i] & bitmap_b[i];
    }
};

int get_vector_size(int deg)
{
    return (deg % BITSET_WIDTH) ? (deg / BITSET_WIDTH + 1) : (deg / BITSET_WIDTH);
}
const __m256i add8 = _mm256_set1_epi32(8);
const __m256i add64 = _mm256_set1_epi32(64);
int count_bitmap(uint8_t *bitmap, int vector_size)
{
    int offset = vector_size & 7, res = 0;
    unsigned char *end = bitmap + vector_size - offset;
    while (bitmap < end)
    {
        res += __builtin_popcountll(*(unsigned long long *)bitmap);
        bitmap += 8;
    }
    if (offset)
    {
        res += __builtin_popcountll((*(unsigned long long *)bitmap) << (64 - offset * 8));
    }
    return res;
}
int count_bitmap(uint8_t *bitmap, int start, int end)
{
    if (start == end)
        return 0;
    int vector_end = get_vector_size(end);
    int vector_start = start / BITSET_WIDTH;
    auto bitset = bitmap[vector_start] & (0xff >> (start % BITSET_WIDTH));
    int vector_size = vector_end - ++vector_start;
    if (vector_size > 0)
    {
        return lengthTable[bitset] + count_bitmap(bitmap + vector_start, vector_size);
    }
    else
    {
        return lengthTable[bitset];
    }
}

int expandToIndex(uint8_t *bitmap, int *out, int vector_size)
{
    int *initout = out;
    __m256i baseVec = _mm256_set1_epi32(0);
    uint8_t mask;
    int i = 0;
    while (i < vector_size)
    {
        if (*(uint64_t *)(bitmap + i) == 0)
        {
            baseVec = _mm256_add_epi32(baseVec, add64);
            i += 8;
        }
        else
        {
            mask = bitmap[i];
            __m256i vecA = _mm256_load_si256((const __m256i *)vecDecodeTable[mask]);
            vecA = _mm256_add_epi32(baseVec, vecA);
            _mm256_storeu_si256((__m256i *)out, vecA);
            out += lengthTable[mask];
            baseVec = _mm256_add_epi32(baseVec, add8);
            i++;
        }
    }
    return out - initout;
};

int expandToIndex(uint8_t *bitmap, int *out, int start, int end)
{
    if (start == end)
        return 0;
    int *initout = out;
    uint8_t filter = 0xff;
    uint8_t mask;
    int start_vec = start / BITSET_WIDTH;
    int end_vec = get_vector_size(end);
    __m256i baseVec = _mm256_set1_epi32(start_vec * 8);
    mask = bitmap[start_vec];
    mask = mask & (filter >> (start % BITSET_WIDTH));
    __m256i vecA = _mm256_load_si256((const __m256i *)vecDecodeTable[mask]);
    vecA = _mm256_add_epi32(baseVec, vecA);
    _mm256_storeu_si256((__m256i *)out, vecA);
    out += lengthTable[mask];
    baseVec = _mm256_add_epi32(baseVec, add8);
    int i = start_vec + 1;
    while (i < end_vec)
    {
        if (*(uint64_t *)(bitmap + i) == 0)
        {
            baseVec = _mm256_add_epi32(baseVec, add64);
            i += 8;
        }
        else
        {
            mask = bitmap[i];
            __m256i vecA = _mm256_load_si256((const __m256i *)vecDecodeTable[mask]);
            vecA = _mm256_add_epi32(baseVec, vecA);
            _mm256_storeu_si256((__m256i *)out, vecA);
            out += lengthTable[mask];
            baseVec = _mm256_add_epi32(baseVec, add8);
            i++;
        }
    }
    return out - initout;
};

int expandToID(uint8_t *bitmap, int *out, int *id_list, int start, int end)
{
    int p = expandToIndex(bitmap, out, start, end);
    for (int i = 0; i < p; i++)
    {
        out[i] = id_list[out[i]];
    }
    return p;
};

int expandToID(uint8_t *bitmap, int *out, int vector_size, int *id_list)
{
    int p = expandToIndex(bitmap, out, vector_size);
    for (int i = 0; i < p; i++)
    {
        out[i] = id_list[out[i]];
    }
    return p;
};
int mark_intersect(int *set_a, int size_a, int *set_b, int size_b, uint8_t *bitmap)
{
    int i = 0, j = 0, k = 0, cnt = 0;
    int qs_a = size_a - (size_a & 7);
    int qs_b = size_b - (size_b & 7);
    const __m256i reverse_mask = _mm256_loadu_si256((__m256i *)reverese);

    while (i < qs_a && j < qs_b)
    {
        __m256i v_a = _mm256_lddqu_si256((__m256i *)(set_a + i));
        __m256i v_b = _mm256_lddqu_si256((__m256i *)(set_b + j));
        k = i / 8;
        int a_max = set_a[i + 7];
        int b_max = set_b[j + 7];
        if (a_max == b_max)
        {
            i += 8;
            j += 8;
            _mm_prefetch((const char *)set_a + i, _MM_HINT_NTA);
            _mm_prefetch((const char *)set_b + j, _MM_HINT_NTA);
        }
        else if (a_max > b_max)
        {
            j += 8;
            _mm_prefetch((const char *)set_b + j, _MM_HINT_NTA);
        }
        else
        {
            i += 8;
            _mm_prefetch((const char *)set_a + i, _MM_HINT_NTA);
        }

        __m256i cmp_mask0 = _mm256_cmpeq_epi32(v_a, v_b);
        __m256 rot1 = _mm256_permute_ps((__m256)v_b, _CYCLIC_SHIFT1_);
        __m256i cmp_mask1 = _mm256_cmpeq_epi32(v_a, (__m256i)rot1);
        __m256 rot2 = _mm256_permute_ps((__m256)v_b, _CYCLIC_SHIFT2_);
        __m256i cmp_mask2 = _mm256_cmpeq_epi32(v_a, (__m256i)rot2);
        __m256 rot3 = _mm256_permute_ps((__m256)v_b, _CYCLIC_SHIFT3_);
        __m256i cmp_mask3 = _mm256_cmpeq_epi32(v_a, (__m256i)rot3);

        __m256 rot4 = _mm256_permute2f128_ps((__m256)v_b, (__m256)v_b, 1);
        __m256i cmp_mask4 = _mm256_cmpeq_epi32(v_a, (__m256i)rot4);
        __m256 rot5 = _mm256_permute_ps(rot4, _CYCLIC_SHIFT1_);
        __m256i cmp_mask5 = _mm256_cmpeq_epi32(v_a, (__m256i)rot5);
        __m256 rot6 = _mm256_permute_ps(rot4, _CYCLIC_SHIFT2_);
        __m256i cmp_mask6 = _mm256_cmpeq_epi32(v_a, (__m256i)rot6);
        __m256 rot7 = _mm256_permute_ps(rot4, _CYCLIC_SHIFT3_);
        __m256i cmp_mask7 = _mm256_cmpeq_epi32(v_a, (__m256i)rot7);

        __m256i cmp_mask = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_or_si256(cmp_mask0, cmp_mask1),
                _mm256_or_si256(cmp_mask2, cmp_mask3)),
            _mm256_or_si256(
                _mm256_or_si256(cmp_mask4, cmp_mask5),
                _mm256_or_si256(cmp_mask6, cmp_mask7)));
        __m256 shifted_mask = _mm256_permutevar8x32_ps((__m256)cmp_mask, reverse_mask);
        uint8_t mask = _mm256_movemask_ps(shifted_mask);
        bitmap[k] |= mask;
        cnt += lengthTable[mask];
    }

    while (i < size_a && j < size_b)
    {
        if (set_a[i] == set_b[j])
        {
            bitmap[i / 8] |= 1 << (7 - i % 8);
            i++;
            j++;
            cnt++;
        }
        else if (set_a[i] < set_b[j])
        {
            i++;
        }
        else
        {
            j++;
        }
    }
    return cnt;
};

bool all_zero(Bitmap *bitmap, int vector_size)
{
    for (int i = 0; i < vector_size; i++)
    {
        if (bitmap[i])
            return false;
    }
    return true;
}
}