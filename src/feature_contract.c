#include <math.h>
#include "feature_contract.h"

static int finite_nonneg(float x)
{
    return isfinite(x) && x >= 0.0f;
}

static int validate_iot(const struct feature_vector *fv)
{
    if (fv->n != 8) return 0;
    float min_len = fv->v[0], max_len = fv->v[1], mean_len = fv->v[2];
    float iat_min = fv->v[3], iat_max = fv->v[4], iat_mean = fv->v[5];
    if (!finite_nonneg(min_len) || !finite_nonneg(max_len) || !finite_nonneg(mean_len)) return 0;
    if (min_len > mean_len || mean_len > max_len) return 0;
    if (!finite_nonneg(iat_min) || !finite_nonneg(iat_max) || !finite_nonneg(iat_mean)) return 0;
    if (iat_min > iat_mean || iat_mean > iat_max) return 0;
    if (!finite_nonneg(fv->v[6]) || !finite_nonneg(fv->v[7])) return 0;
    return 1;
}

static int validate_doh(const struct feature_vector *fv, uint16_t window)
{
    if (fv->n != 16) return 0;
    float n_client = fv->v[1];
    float n_server = fv->v[3];
    float byte_frac_client = fv->v[2];
    float pkt_frac_client = fv->v[4];
    float size_min = fv->v[7], size_mean = fv->v[8], size_max = fv->v[12];
    float dir_switches = fv->v[10];

    if (!finite_nonneg(n_client) || !finite_nonneg(n_server)) return 0;
    if ((uint32_t)(n_client + n_server + 0.5f) != window) return 0;
    if (byte_frac_client < 0.0f || byte_frac_client > 1.0f) return 0;
    if (pkt_frac_client < 0.0f || pkt_frac_client > 1.0f) return 0;
    if (size_min > size_mean || size_mean > size_max) return 0;
    if (dir_switches < 0.0f || dir_switches > (float)(window - 1)) return 0;
    if (dir_switches > 2.0f * fminf(n_client, n_server)) return 0;
    return 1;
}

static int validate_ics(const struct feature_vector *fv)
{
    if (fv->n != 16) return 0;
    float minv = fv->v[0], maxv = fv->v[1], mean = fv->v[2];
    float iat_min = fv->v[3], iat_max = fv->v[4], iat_mean = fv->v[5];
    float range = fv->v[6];
    if (!isfinite(minv) || !isfinite(maxv) || !isfinite(mean)) return 0;
    if (minv > mean || mean > maxv) return 0;
    if (!finite_nonneg(iat_min) || !finite_nonneg(iat_max) || !finite_nonneg(iat_mean)) return 0;
    if (iat_min > iat_mean || iat_mean > iat_max) return 0;
    if (range < 0.0f) return 0;
    if (fabsf(range - (maxv - minv)) > 1e-3f * fmaxf(1.0f, fabsf(maxv - minv))) return 0;
    if (!(fv->v[14] == 0.0f || fv->v[14] == 1.0f)) return 0;
    return 1;
}

int feature_contract_validate(enum app_profile_type profile,
                              const struct feature_vector *fv,
                              uint16_t window)
{
#if APP_ENABLE_CONTRACTS
    switch (profile) {
    case APP_PROFILE_IOT: return validate_iot(fv);
    case APP_PROFILE_DOH: return validate_doh(fv, window);
    case APP_PROFILE_ICS: return validate_ics(fv);
    default: return 0;
    }
#else
    (void)profile; (void)fv; (void)window; return 1;
#endif
}
