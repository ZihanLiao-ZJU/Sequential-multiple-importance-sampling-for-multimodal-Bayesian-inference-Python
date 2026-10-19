function [xf, fval] = fminbnd(func, x1, x2, varargin)
% Bounded Brent minimisation with MATLAB's default options (TolX = 1e-4,
% MaxFunEvals = 500), written step by step like scipy.optimize's bounded
% method, which the Python port uses. Octave's own fminbnd has other
% defaults, so it is shadowed when the reference streams are recorded.
xatol = 1e-4; maxfun = 500;
sqrt_eps = sqrt(2.2e-16);
golden_mean = 0.5 * (3.0 - sqrt(5.0));
a = x1; b = x2;
fulc = a + golden_mean * (b - a);
nfc = fulc; xf = fulc;
rat = 0; e = 0;
x = xf;
fx = func(x);
num = 1;
fu = inf;
ffulc = fx; fnfc = fx;
xm = 0.5 * (a + b);
tol1 = sqrt_eps * abs(xf) + xatol / 3.0;
tol2 = 2.0 * tol1;
while abs(xf - xm) > (tol2 - 0.5 * (b - a))
    golden = 1;
    if abs(e) > tol1
        golden = 0;
        r = (xf - nfc) * (fx - ffulc);
        q = (xf - fulc) * (fx - fnfc);
        p = (xf - fulc) * q - (xf - nfc) * r;
        q = 2.0 * (q - r);
        if q > 0.0, p = -p; end
        q = abs(q);
        r = e;
        e = rat;
        if (abs(p) < abs(0.5*q*r)) && (p > q*(a - xf)) && (p < q*(b - xf))
            rat = (p + 0.0) / q;
            x = xf + rat;
            if ((x - a) < tol2) || ((b - x) < tol2)
                si = sign(xm - xf) + ((xm - xf) == 0);
                rat = tol1 * si;
            end
        else
            golden = 1;
        end
    end
    if golden
        if xf >= xm, e = a - xf; else, e = b - xf; end
        rat = golden_mean * e;
    end
    si = sign(rat) + (rat == 0);
    x = xf + si * max(abs(rat), tol1);
    fu = func(x);
    num = num + 1;
    if fu <= fx
        if x >= xf, a = xf; else, b = xf; end
        fulc = nfc; ffulc = fnfc;
        nfc = xf; fnfc = fx;
        xf = x; fx = fu;
    else
        if x < xf, a = x; else, b = x; end
        if (fu <= fnfc) || (nfc == xf)
            fulc = nfc; ffulc = fnfc;
            nfc = x; fnfc = fu;
        elseif (fu <= ffulc) || (fulc == xf) || (fulc == nfc)
            fulc = x; ffulc = fu;
        end
    end
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0;
    tol2 = 2.0 * tol1;
    if num >= maxfun, break; end
end
fval = fx;
end
