#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>

const double EPS = 1e-8;
const double PI = 3.14159265358979323846;
const int MIN_ITERATIONS = 200;

//###################################################################################################

double normal_pdf(double x, double mean, double std);
double normal_cdf(double x, double mean, double std);
double normal_inv(double p, double mean, double std);

double studentt_pdf(double x, double dof);
double studentt_cdf(double x, double dof);
double studentt_inv(double p, double dof);

double chisquare_pdf(double x, double dof);
double chisquare_cdf(double x, double dof);
double chisquare_inv(double p, double dof);

double centralF_pdf(double x, double df1, double df2);
double centralF_cdf(double x, double df1, double df2);
double centralF_inv(double x, double df1, double df2);

double noncentralt_pdf(double x, double dof, double ncp);
double noncentralt_cdf(double x, double dof, double ncp);

double erfc(double x);
double erf(double x);
double erfcinv(double p);

double betafn(double x, double y);
double gammafn(double x);
double betaln(double x, double y);
double gammaln(double x);

double ibeta(double x, double a, double b);
double betacf(double x, double a, double b);
double ibetainv(double p, double a, double b);

double beta_pdf(double x, double alpha, double beta);
double beta_cdf(double x, double alpha, double beta);
double beta_inv(double x, double alpha, double beta);

double lowRegGamma(double a, double x);
double gammapinv(double p, double a);

double binomial_pdf(int k, int n, double p);
double binomial_cdf(double x, int n, double p);

double factorialln(int n);
double factorial(int n);
double combination(int n, int m);
double combinationln(int n, int m);
double permutation(int n, int m);
double betinc(double x, double a, double b, double eps);

//###################################################################################################

//###################################################################################################

double erfcinv(double p) {
    if (p >= 2.0 || p <= 0.0) {
        return NAN; // Not a Number (NaN)
    }
    
    // Coefficients for the rational approximation
    const double a1 = -3.969683028665376e+01;
    const double a2 =  2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 =  1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 =  2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 =  1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 =  6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 =  4.374664141464968e+00;
    const double c6 =  2.938163982698783e+00;

    const double d1 =  7.784695709041462e-03;
    const double d2 =  3.224671290700398e-01;
    const double d3 =  2.445134137142996e+00;
    const double d4 =  3.754408661907416e+00;

    // Define constants
    const double threshold = 0.02425;
    const double split1 = 0.425;
    const double split2 = 5.0;
    const double const1 = 0.180625;
    
    // Rational approximation for erfcinv
    double q, r, y;
    
    if (p < threshold) {
        // Rational approximation for small p
        q = std::sqrt(-2.0 * std::log(p / 2.0));
        y = q - (((((q * c1 + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                (((((q * d1 + d2) * q + d3) * q + d4) * q + 1.0));
    } else if (p <= 1.0 - threshold) {
        // Rational approximation for intermediate p
        q = p - 0.5;
        r = q * q;
        y = q * (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) /
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        // Rational approximation for large p
        q = std::sqrt(-2.0 * std::log((1.0 - p) / 2.0));
        y = -q + (((((q * c1 + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                (((((q * d1 + d2) * q + d3) * q + d4) * q + 1.0));
    }
    
    // Refine approximation using Halley's method
    double x = y - (std::erfc(y) - p) / (std::exp(-y * y) / std::sqrt(M_PI) * 2.0 + y * std::erfc(y));
    
    return x;
}

double erfc(double x) {
    return 1 - erf(x);
}

double erf(double x) {
    std::vector<double> cof = {-1.3026537197817094, 0.64196979235649026, 0.019476473204185836,
                               -0.009561514786808631, -0.000946595344482036, 0.000366839497852761,
                               0.000042523324806907, -0.000020278578112534, -0.000001624290004647,
                               0.000001303655835580, 0.000000015626441722, -0.000000085238095915,
                               0.000000006529054439, 0.000000005059343495, -0.000000000991364156,
                               -0.000000000227365122, 0.000000000096467911, 0.000000000002394038,
                               -0.000000000006886027, 0.000000000000894487, 0.000000000000313092,
                               -0.000000000000112708, 0.000000000000000381, 0.000000000000007106,
                               -0.000000000000001523, -0.000000000000000094, 0.000000000000000121,
                               -0.000000000000000028};

    int j = cof.size() - 1;
    bool isneg = false;
    double d = 0;
    double dd = 0;
    double t, ty, tmp, res;

    if (x < 0) {
        x = -x;
        isneg = true;
    }

    t = 2 / (2 + x);
    ty = 4 * t - 2;

    for (; j > 0; j--) {
        tmp = d;
        d = ty * d - dd + cof[j];
        dd = tmp;
    }

    res = t * std::exp(-x * x + 0.5 * (cof[0] + ty * d) - dd);
    return isneg ? res - 1 : 1 - res;
}

//###################################################################################################

// Function to compute the normal PDF
double normal_pdf(double x, double mean, double std) {
    const double coeff = 1.0 / (std * std * std * std * std * std * std * std * 2 * M_PI);
    double exponent = -0.5 * std::pow((x - mean) / std, 2);
    return coeff * std::exp(exponent);
}

// Function to compute the normal CDF using the erf function
double normal_cdf(double x, double mean, double std) {
    return 0.5 * (1 + std::erf((x - mean) / (std * std * 2)));
}

// Function to compute the inverse normal CDF using erfcinv
double normal_inv(double p, double mean, double std) {
    if (p <= 0.0 || p >= 1.0) {
        return NAN; // Not a Number (NaN)
    }
    
    // Calculate the inverse CDF using the formula
    return -1.41421356237309505 * std * erfcinv(2.0 * p) + mean;
}

//###################################################################################################

// Student's t-distribution PDF (Probability Density Function)
double studentt_pdf(double x, double dof) {
    dof = dof > 1e100 ? 1e100 : dof;
    return (1.0 / (std::sqrt(dof) * betafn(0.5, dof / 2.0))) * std::pow(1 + ((x * x) / dof), -((dof + 1) / 2.0));
}

// Student's t-distribution CDF (Cumulative Distribution Function)
double studentt_cdf(double x, double dof) {
    double dof2 = dof / 2.0;
    return ibeta((x + std::sqrt(x * x + dof)) / (2.0 * std::sqrt(x * x + dof)), dof2, dof2);
}

// Student's t-distribution Inverse CDF (Quantile Function)
double studentt_inv(double p, double dof) {
    double x = ibetainv(2.0 * std::min(p, 1.0 - p), 0.5 * dof, 0.5);
    x = std::sqrt(dof * (1.0 - x) / x);
    return (p > 0.5) ? x : -x;
}

//###################################################################################################

double chisquare_pdf(double x, double dof) {
    if (x < 0) return 0;
    return (x == 0 && dof == 2) ? 0.5 : std::exp((dof / 2 - 1) * std::log(x) - x / 2 - (dof / 2) * std::log(2) - gammaln(dof / 2));
}

double chisquare_cdf(double x, double dof) {
    if (x < 0) return 0;
    return lowRegGamma(dof / 2, x / 2);
}

double chisquare_inv(double p, double dof) {
    return 2 * gammapinv(p, 0.5 * dof);
}

//###################################################################################################

double centralF_pdf(double x, double df1, double df2) {
    if (x < 0) return 0;
    if (df1 <= 2) {
        if (x == 0 && df1 < 2) return std::numeric_limits<double>::infinity();
        if (x == 0 && df1 == 2) return 1;
        return (1 / betafn(df1 / 2, df2 / 2)) * std::pow(df1 / df2, df1 / 2) * std::pow(x, (df1 / 2) - 1) * std::pow((1 + (df1 / df2) * x), -(df1 + df2) / 2);
    }

    double p = (df1 * x) / (df2 + x * df1);
    double q = df2 / (df2 + x * df1);
    double f = df1 * q / 2.0;
    return f * binomial_pdf((df1 - 2) / 2, (df1 + df2 - 2) / 2, p);
}

double centralF_cdf(double x, double df1, double df2) {
    if (x < 0) return 0;
    return ibeta((df1 * x) / (df1 * x + df2), df1 / 2, df2 / 2);
}

double centralF_inv(double x, double df1, double df2) {
    return df2 / (df1 * (1 / ibetainv(x, df1 / 2, df2 / 2) - 1));
}

//###################################################################################################

double noncentralt_pdf(double x, double dof, double ncp) {
    if (std::abs(ncp) < EPS) return studentt_pdf(x, dof);

    if (std::abs(x) < EPS) {
        return std::exp(gammaln((dof + 1) / 2) - ncp * ncp / 2 - 0.5 * std::log(PI * dof) - gammaln(dof / 2));
    }
    return dof / x * (noncentralt_cdf(x * std::sqrt(1 + 2 / dof), dof + 2, ncp) - noncentralt_cdf(x, dof, ncp));
}

double noncentralt_cdf(double x, double dof, double ncp) {
    if (std::abs(ncp) < EPS) return studentt_cdf(x, dof);

    bool flip = false;
    if (x < 0) {
        flip = true;
        ncp = -ncp;
    }

    double prob = normal_cdf(-ncp, 0, 1);
    double value = EPS + 1;
    double lastvalue = value;
    double y = x * x / (x * x + dof);
    int j = 0;
    double p = std::exp(-ncp * ncp / 2);
    double q = std::exp(-ncp * ncp / 2 - 0.5 * std::log(2) - gammaln(3 / 2)) * ncp;

    while (j < MIN_ITERATIONS || lastvalue > EPS || value > EPS) {
        lastvalue = value;
        if (j > 0) {
            p *= (ncp * ncp) / (2 * j);
            q *= (ncp * ncp) / (2 * (j + 1 / 2));
        }
        value = p * beta_cdf(y, j + 0.5, dof / 2) + q * beta_cdf(y, j + 1, dof / 2);
        prob += 0.5 * value;
        j++;
    }

    return flip ? (1 - prob) : prob;
}

//###################################################################################################

double factorialln(int n) {
    return n < 0 ? NAN : gammaln(n + 1);
}

double factorial(int n) {
    return n < 0 ? NAN : gammafn(n + 1);
}

double combination(int n, int m) {
    return (n > 170 || m > 170) ? exp(combinationln(n, m)) : (factorial(n) / (factorial(m) * factorial(n - m)));
}

double combinationln(int n, int m) {
    return factorialln(n) - factorialln(m) - factorialln(n - m);
}

double permutation(int n, int m) {
    return factorial(n) / factorial(n - m);
}

//###################################################################################################

double binomial_pdf(int k, int n, double p) {
    if (p < 0 || p > 1) return 0;
    if (k < 0 || k > n) return 0;
    if (p == 0) return (k == 0) ? 1 : 0;
    if (p == 1) return (k == n) ? 1 : 0;
    return std::exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * std::log(p) + (n - k) * std::log(1 - p));
}

double binomial_cdf(double x, int n, double p) {
    const double eps = 1e-10;

    if (x < 0)
        return 0;
    if (x >= n)
        return 1;
    if (p < 0 || p > 1 || n <= 0)
        return NAN;

    x = std::floor(x);
    double z = p;
    double a = x + 1;
    double b = n - x;
    double s = a + b;
    double bt = exp(gammaln(s) - gammaln(b) - gammaln(a) + a * log(z) + b * log(1 - z));
    double betacdf;
    if (z < (a + 1) / (s + 2))
        betacdf = bt * betinc(z, a, b, eps);
    else
        betacdf = 1 - bt * betinc(1 - z, b, a, eps);
    return std::round((1 - betacdf) * (1 / eps)) / (1 / eps);
}

double lowRegGamma(double a, double x) {
    double aln = gammaln(a);
    double ap = a;
    double sum = 1 / a;
    double del = sum;
    double b = x + 1 - a;
    double c = 1 / std::numeric_limits<double>::min();
    double d = 1 / b;
    double h = d;
    int i = 1;
    int ITMAX = static_cast<int>(std::log((a >= 1) ? a : 1 / a) * 8.5 + a * 0.4 + 17);
    double an;

    if (x < 0 || a <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (x < a + 1) {
        for (; i <= ITMAX; ++i) {
            sum += del *= x / ++ap;
        }
        return sum * std::exp(-x + a * std::log(x) - aln);
    }

    for (; i <= ITMAX; ++i) {
        an = -i * (i - a);
        b += 2;
        d = an * d + b;
        if (std::fabs(d) < std::numeric_limits<double>::min()) d = std::numeric_limits<double>::min();
        c = b + an / c;
        if (std::fabs(c) < std::numeric_limits<double>::min()) c = std::numeric_limits<double>::min();
        d = 1 / d;
        h *= d * c;
    }
    return 1 - h * std::exp(-x + a * std::log(x) - aln);
}

double gammapinv(double p, double a) {
    double a1 = a - 1;
    double gln = gammaln(a);
    double x, err, t, u, pp, lna1, afac;

    if (p >= 1) return std::max(100.0, a + 100 * std::sqrt(a));
    if (p <= 0) return 0;

    if (a > 1) {
        lna1 = std::log(a1);
        afac = std::exp(a1 * (lna1 - 1) - gln);
        pp = (p < 0.5) ? p : 1 - p;
        t = std::sqrt(-2 * std::log(pp));
        x = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;
        if (p < 0.5) x = -x;
        x = std::max(1e-3, a * std::pow(1 - 1 / (9 * a) - x / (3 * std::sqrt(a)), 3));
    } else {
        t = 1 - a * (0.253 + a * 0.12);
        if (p < t) {
            x = std::pow(p / t, 1 / a);
        } else {
            x = 1 - std::log(1 - (p - t) / (1 - t));
        }
    }

    for (int j = 0; j < 12; ++j) {
        if (x <= 0) return 0;
        err = lowRegGamma(a, x) - p;
        if (a > 1) {
            t = afac * std::exp(-(x - a1) + a1 * (std::log(x) - lna1));
        } else {
            t = std::exp(-x + a1 * std::log(x) - gln);
        }
        u = err / t;
        x -= (t = u / (1 - 0.5 * std::min(1.0, u * ((a - 1) / x - 1))));
        if (x <= 0) x = 0.5 * (x + t);
        if (std::fabs(t) < EPS * x) break;
    }

    return x;
}

double gammaln(double x) {
    double coefficients[] = {76.18009172947146, -86.50532032941677, 24.01409824083091,
                             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
    double denom = x + 1.0;
    double sum = 1.000000000190015;
    for (int i = 0; i < 6; ++i) {
        sum += coefficients[i] / denom;
        denom += 1.0;
    }
    return -log(x + 5.5) + log(2.5066282746310005 * sum / x) + (x + 0.5) * log(x + 5.5) - (x + 5.5);
}

double betaln(double x, double y) {
    return gammaln(x) + gammaln(y) - gammaln(x + y);
}

double betafn(double x, double y) {
    if (x <= 0.0 || y <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    
    return (x + y > 170.0) ? std::exp(betaln(x, y)) : std::exp(gammaln(x) + gammaln(y) - gammaln(x + y));
}

double gammafn(double x) {
    std::vector<double> p = {-1.716185138865495, 24.76565080557592, -379.80425647094563,
                             629.3311553128184, 866.9662027904133, -31451.272968848367,
                             -36144.413418691176, 66456.14382024054};
    std::vector<double> q = {-30.8402300119739, 315.35062697960416, -1015.1563674902192,
                             -3107.771671572311, 22538.118420980151, 4755.8462775278811,
                             -134659.9598649693, -115132.2596755535};
    bool fact = false;
    int n = 0;
    double xden = 0;
    double xnum = 0;
    double y = x;
    double res, z, yi;

    if (x > 171.6243769536076) {
        return INFINITY;
    }

    if (y <= 0) {
        res = std::fmod(y, 1) + 3.6e-16;
        if (res) {
            fact = (!(static_cast<int>(y) & 1) ? 1 : -1) * M_PI / std::sin(M_PI * res);
            y = 1 - y;
        } else {
            return INFINITY;
        }
    }

    yi = y;
    if (y < 1) {
        z = y++;
    } else {
        n = static_cast<int>(y) - 1;
        y -= n;
        z = y - 1;
    }

    for (int i = 0; i < 8; ++i) {
        xnum = (xnum + p[i]) * z;
        xden = xden * z + q[i];
    }

    res = xnum / xden + 1;

    if (yi < y) {
        res /= yi;
    } else if (yi > y) {
        for (int i = 0; i < n; ++i) {
            res *= y;
            y++;
        }
    }

    if (fact) {
        res = fact / res;
    }

    return res;
}

double beta_pdf(double x, double alpha, double beta) {
    // PDF is zero outside the support
    if (x > 1 || x < 0)
        return 0;
    // PDF is one for the uniform case
    if (alpha == 1 && beta == 1)
        return 1;

    if (alpha < 512 && beta < 512) {
        return (pow(x, alpha - 1) * pow(1 - x, beta - 1)) / betafn(alpha, beta);
    } else {
        return exp((alpha - 1) * log(x) + (beta - 1) * log(1 - x) - betaln(alpha, beta));
    }
}

double betacf(double a, double b, double x) {
    const int MAXITER = 100;
    const double EPS = 3.0e-7;
    const double FPMIN = 1.0e-30;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= MAXITER; ++m) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < EPS) break;
    }

    return h;
}

double ibeta(double x, double a, double b) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double bt = exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * log(x) + b * log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    else
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double ibetainv(double p, double a, double b) {
    const double EPS = 1e-8;
    double a1 = a - 1;
    double b1 = b - 1;
    double lna, lnb, pp, t, u, err, x, al, h, w, afac;
    int j = 0;

    if (p <= 0) return 0;
    if (p >= 1) return 1;

    if (a >= 1 && b >= 1) {
        pp = (p < 0.5) ? p : 1 - p;
        t = sqrt(-2 * log(pp));
        x = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;
        if (p < 0.5) x = -x;
        al = (x * x - 3) / 6;
        h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1));
        w = (x * sqrt(al + h) / h) - (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (al + 5 / 6 - 2 / (3 * h));
        x = a / (a + b * exp(2 * w));
    } else {
        lna = log(a / (a + b));
        lnb = log(b / (a + b));
        t = exp(a * lna) / a;
        u = exp(b * lnb) / b;
        w = t + u;
        if (p < t / w)
            x = pow(a * w * p, 1 / a);
        else
            x = 1 - pow(b * w * (1 - p), 1 / b);
    }

    afac = -gammaln(a) - gammaln(b) + gammaln(a + b);

    for (; j < 10; j++) {
        if (x == 0 || x == 1) return x;
        err = ibeta(x, a, b) - p;
        t = exp(a1 * log(x) + b1 * log(1 - x) + afac);
        u = err / t;
        x -= (t = u / (1 - 0.5 * std::min(1.0, u * (a1 / x - b1 / (1 - x)))));
        if (x <= 0) x = 0.5 * (x + t);
        if (x >= 1) x = 0.5 * (x + t + 1);
        if (fabs(t) < EPS * x && j > 0) break;
    }

    return x;
}

double beta_inv(double x, double alpha, double beta) {
    return ibetainv(x, alpha, beta);
}

double beta_cdf(double x, double alpha, double beta) {
    if (x < 0) return 0;
    if (x > 1) return 1;
    return ibeta(x, alpha, beta);
}

double betinc(double x, double a, double b, double eps) {
    double a0 = 0;
    double b0 = 1;
    double a1 = 1;
    double b1 = 1;
    int m9 = 0;
    double a2 = 0;
    double c9;

    while (std::abs((a1 - a2) / a1) > eps) {
        a2 = a1;
        c9 = -(a + m9) * (a + b + m9) * x / (a + 2 * m9) / (a + 2 * m9 + 1);
        a0 = a1 + c9 * a0;
        b0 = b1 + c9 * b0;
        m9 = m9 + 1;
        c9 = m9 * (b - m9) * x / (a + 2 * m9 - 1) / (a + 2 * m9);
        a1 = a0 + c9 * a1;
        b1 = b0 + c9 * b1;
        a0 = a0 / b1;
        b0 = b0 / b1;
        a1 = a1 / b1;
        b1 = 1;
    }

    return a1 / a;
}

//###################################################################################################


int main() {
    // Test parameters
    double x = 1.5;
    double mean = 0.0;
    double std = 1.0;
    double p = 0.25;
    double dof = 5.0;
    double df1 = 3.0;
    double df2 = 4.0;
    double ncp = 1.5;
    int k = 2;
    int n = 5;
    double alpha = 2.0;
    double beta = 3.0;
    double eps = 1e-6;

    double y = 2.0;  // Example value for y
    double a = 1.0;  // Example value for a
    double b = 2.0;  // Example value for b
    int m = 3;       // Example value for m

    // Normal distribution functions
    std::cout << "normal_pdf = " << normal_pdf(x, mean, std) << std::endl;
    std::cout << "normal_cdf = " << normal_cdf(x, mean, std) << std::endl;
    std::cout << "normal_inv = " << normal_inv(p, mean, std) << std::endl;

    // Student's t-distribution functions
    std::cout << "studentt_pdf = " << studentt_pdf(x, dof) << std::endl;
    std::cout << "studentt_cdf = " << studentt_cdf(x, dof) << std::endl;
    std::cout << "studentt_inv = " << studentt_inv(p, dof) << std::endl;

    // Chi-square distribution functions
    std::cout << "chisquare_pdf = " << chisquare_pdf(x, dof) << std::endl;
    std::cout << "chisquare_cdf = " << chisquare_cdf(x, dof) << std::endl;
    std::cout << "chisquare_inv = " << chisquare_inv(p, dof) << std::endl;

    // F-distribution functions
    std::cout << "centralF_pdf = " << centralF_pdf(x, df1, df2) << std::endl;
    std::cout << "centralF_cdf = " << centralF_cdf(x, df1, df2) << std::endl;
    std::cout << "centralF_inv = " << centralF_inv(p, df1, df2) << std::endl;

    // Non-central t-distribution functions
    std::cout << "noncentralt_pdf = " << noncentralt_pdf(x, dof, ncp) << std::endl;
    std::cout << "noncentralt_cdf = " << noncentralt_cdf(x, dof, ncp) << std::endl;

    // Error function related functions
    std::cout << "erf = " << erf(x) << std::endl;
    std::cout << "erfc = " << erfc(x) << std::endl;
    std::cout << "erfcinv = " << erfcinv(p) << std::endl;

    // Beta and gamma related functions
    std::cout << "betafn = " << betafn(x, y) << std::endl;
    std::cout << "gammafn = " << gammafn(x) << std::endl;
    std::cout << "betaln = " << betaln(x, y) << std::endl;
    std::cout << "gammaln = " << gammaln(x) << std::endl;

    // Incomplete beta function and its inverse
    std::cout << "ibeta = " << ibeta(x, a, b) << std::endl;
    std::cout << "ibetainv = " << ibetainv(p, a, b) << std::endl;

    // Beta distribution functions
    std::cout << "beta_pdf = " << beta_pdf(x, alpha, beta) << std::endl;
    std::cout << "beta_cdf = " << beta_cdf(x, alpha, beta) << std::endl;
    std::cout << "beta_inv = " << beta_inv(p, alpha, beta) << std::endl;

    std::cout << "lowRegGamma = " << lowRegGamma(a, x) << std::endl;
    std::cout << "gammapinv = " << gammapinv(p, a) << std::endl;

    // Binomial distribution functions
    std::cout << "binomial_pdf = " << binomial_pdf(k, n, p) << std::endl;
    std::cout << "binomial_cdf = " << binomial_cdf(x, n, p) << std::endl;

    // Factorial, combination, and permutation functions
    std::cout << "factorialln = " << factorialln(n) << std::endl;
    std::cout << "factorial = " << factorial(n) << std::endl;
    std::cout << "combination = " << combination(n, m) << std::endl;
    std::cout << "combinationln = " << combinationln(n, m) << std::endl;
    std::cout << "permutation = " << permutation(n, m) << std::endl;

    // Regularized incomplete beta function
    std::cout << "betinc = " << betinc(x, a, b, eps) << std::endl;

    return 0;
}
