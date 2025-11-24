(function () {
    'use strict';

    // Copia array
    function cloneArray(a) { return a.slice(); }
    // Verifica se é número finito
    function isFiniteNumber(x) { return typeof x === 'number' && isFinite(x); }
    // Norma euclidiana (tamanho do vetor)
    function norm2(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }

    // Função que retorna demanda para um dia específico
    // Se dia é decimal, faz interpolação linear simples
    function getDemandAtDay(vendas, dia) {
        const dia_int = Math.floor(dia);
        const frac = dia - dia_int;
        if (dia_int < 0 || dia_int >= vendas.length) return 0;
        if (dia_int === vendas.length - 1) return vendas[dia_int];
        const v0 = vendas[dia_int];
        const v1 = vendas[dia_int + 1];
        return v0 * (1 - frac) + v1 * frac;
    }

    function partialDerivative(f, t, s, varName = 't', h = 1e-4) {
        if (varName === 't') {
            const fp = f(t + h, s);
            const fm = f(t - h, s);
            return (fp - fm) / (2 * h);
        } else { // 's'
            const fp = f(t, s + h);
            const fm = f(t, s - h);
            return (fp - fm) / (2 * h);
        }
    }

    // Calcula o vetor gradiente [∂f/∂t, ∂f/∂s]
    function gradient(f, t, s, h = 1e-4) {
        const dfdt = partialDerivative(f, t, s, 't', h);
        const dfds = partialDerivative(f, t, s, 's', h);
        return [dfdt, dfds];
    }

    // Hessiana 2x2 aproximada por diferenças centrais
    function hessian(f, t, s, h = 1e-3) {
        const f_tt = (f(t + h, s) - 2 * f(t, s) + f(t - h, s)) / (h * h);
        const f_ss = (f(t, s + h) - 2 * f(t, s) + f(t, s - h)) / (h * h);
        const f_ts = (f(t + h, s + h) - f(t + h, s - h) - f(t - h, s + h) + f(t - h, s - h)) / (4 * h * h);
        return [[f_tt, f_ts], [f_ts, f_ss]];
    }

    // Inverte matriz 2x2 (retorna null se singular)
    function invert2x2(m) {
        const [[a, b], [c, d]] = m;
        const det = a * d - b * c;
        if (!isFiniteNumber(det) || Math.abs(det) < 1e-12) return null;
        const inv = [[d / det, -b / det], [-c / det, a / det]];
        return inv;
    }

    // Multiplica matriz 2x2 por vetor 2x1
    function mat2x2MulVec(m, v) {
        return [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
    }

    // Método de Newton para resolver ∇f = 0 em 2 variáveis
    function newtonSolveGradZero(f, init, opts = {}) {
        const maxIter = opts.maxIter || 50;
        const tol = opts.tol || 1e-6;
        const h = opts.h || 1e-3;
        let x = [init[0], init[1]];
        for (let k = 0; k < maxIter; k++) {
            const g = gradient(f, x[0], x[1], h);
            const gnorm = norm2(g);
            if (gnorm < tol) return { converged: true, x: cloneArray(x), iter: k, grad: g };
            const H = hessian(f, x[0], x[1], h);
            const Hinv = invert2x2(H);
            if (!Hinv) return { converged: false, reason: 'singular_hessian', x: cloneArray(x), grad: g };
            const delta = mat2x2MulVec(Hinv, g).map(v => -v);
            x[0] += delta[0];
            x[1] += delta[1];
            if (norm2(delta) < tol) return { converged: true, x: cloneArray(x), iter: k + 1, grad: g };
        }
        return { converged: false, reason: 'max_iter', x: cloneArray(x) };
    }

    // Classifica ponto crítico usando a Hessiana (2x2)
    function classifyCriticalPoint(H) {
        const a = H[0][0], b = H[0][1], c = H[1][0], d = H[1][1];
        const det = a * d - b * c;
        const trace = a + d;
        if (!isFiniteNumber(det)) return 'indeterminate';
        if (det > 0) {
            if (a > 0) return 'local_min';
            if (a < 0) return 'local_max';
            return 'indeterminate';
        } else if (det < 0) {
            return 'saddle';
        } else {
            return 'degenerate';
        }
    }

    // Busca pontos críticos em uma grade e refina com Newton
    function findCriticalPointsGrid(f, tRange, sRange, nt = 50, ns = 50, opts = {}) {
        const [t0, t1] = tRange, [s0, s1] = sRange;
        const candidates = [];
        for (let i = 0; i < nt; i++) {
            const t = t0 + (t1 - t0) * (i / (nt - 1));
            for (let j = 0; j < ns; j++) {
                const s = s0 + (s1 - s0) * (j / (ns - 1));
                const g = gradient(f, t, s, opts.h || 1e-3);
                if (norm2(g) < (opts.eps || 1e-2)) {
                    // refina
                    const sol = newtonSolveGradZero(f, [t, s], { maxIter: opts.maxIter || 20, tol: opts.tol || 1e-6, h: opts.h || 1e-3 });
                    if (sol.converged) {
                        const key = `${sol.x[0].toFixed(6)}|${sol.x[1].toFixed(6)}`;
                        candidates.push({ t: sol.x[0], s: sol.x[1], grad: sol.grad, iter: sol.iter, key });
                    }
                }
            }
        }
        // único por chave
        const uniq = {};
        const out = [];
        for (const c of candidates) {
            if (!uniq[c.key]) { uniq[c.key] = true; out.push(c); }
        }
        // anexa Hessiana e classificação
        for (const c of out) {
            const H = hessian(f, c.t, c.s, opts.h || 1e-3);
            c.H = H;
            c.type = classifyCriticalPoint(H);
        }
        return out;
    }

    function sampleGrid(f, tRange, sRange, nt = 50, ns = 50, opts = {}) {
        const [t0, t1] = tRange, [s0, s1] = sRange;
        const Ts = new Array(nt);
        const Ss = new Array(ns);
        const Z = Array.from({ length: nt }, () => new Array(ns).fill(0));
        const Gt = Array.from({ length: nt }, () => new Array(ns).fill(0));
        const Gs = Array.from({ length: nt }, () => new Array(ns).fill(0));
        for (let i = 0; i < nt; i++) Ts[i] = t0 + (t1 - t0) * (i / (nt - 1));
        for (let j = 0; j < ns; j++) Ss[j] = s0 + (s1 - s0) * (j / (ns - 1));
        for (let i = 0; i < nt; i++) {
            for (let j = 0; j < ns; j++) {
                const t = Ts[i], s = Ss[j];
                Z[i][j] = f(t, s);
                const gradv = gradient(f, t, s, opts.h || 1e-3);
                Gt[i][j] = gradv[0];
                Gs[i][j] = gradv[1];
            }
        }
        return { Ts, Ss, Z, Gt, Gs };
    }

    // Simulação simples de estoque no tempo (diferença discreta)
    // Parâmetros: S0, t0, tEnd, dt, demandFunc, reposições
    function simulateStock(params) {
        const S0 = params.S0 || 100;
        const t0 = params.t0 || 0;
        const tEnd = params.tEnd || 30;
        const dt = params.dt || 1;
        const demandFunc = params.demandFunc || ((t, S) => 0);
        const replenishments = Array.isArray(params.replenishments) ? params.replenishments.slice() : [];
        const periodic = params.periodic || null;
        const steps = Math.max(1, Math.ceil((tEnd - t0) / dt));
        const out = [];
        let S = S0;
        for (let k = 0; k <= steps; k++) {
            const t = t0 + k * dt;
            // aplica reposição periódica
            let repQty = 0;
            if (periodic && periodic.interval && periodic.qty) {
                if (t >= (periodic.start || t0) && Math.abs((t - (periodic.start || t0)) % periodic.interval) < 1e-6) {
                    repQty += periodic.qty;
                }
            }
            // aplica reposições programadas em tempos exatos (com tolerância)
            for (const r of replenishments) {
                if (Math.abs(r.time - t) < dt * 0.5) repQty += r.qty;
            }
            // demanda neste passo (assume taxa de demanda por unidade de tempo)
            const d = demandFunc(t, S);
            // atualiza (Euler explícito)
            const Snext = S - d * dt + repQty;
            out.push({ t, S, demand: d, replenishment: repQty });
            S = Snext;
            // detecção antecipada de ruptura
            if (S <= 0) {
                // inclui ponto de ruptura ligeiramente ajustado
                out.push({ t: t + dt, S: S, demand: d, replenishment: 0, rupture: true });
                break;
            }
        }
        return out;
    }

    // Parser CSV simples: retorna array de objetos (converte números)
    function parseCSV(text, sep = ',') {
        const lines = text.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
        if (lines.length === 0) return [];
        const hdr = lines[0].split(sep).map(s => s.trim());
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(sep).map(s => s.trim());
            const obj = {};
            for (let j = 0; j < hdr.length; j++) {
                const key = hdr[j] || `c${j}`;
                const val = cols[j] !== undefined ? cols[j] : '';
                const n = Number(val);
                obj[key] = (!isNaN(n) && val !== '') ? n : val;
            }
            rows.push(obj);
        }
        return rows;
    }

    // Constrói função f(t, s) = demanda(t) * fator(s)
    function makeDemandFunction(vendas, fatorEstoque) {
        const g = (typeof fatorEstoque === 'function') ? fatorEstoque : (s => 1);
        return function (t, s) {
            const demand_t = getDemandAtDay(vendas, t);
            return demand_t * g(s);
        };
    }

    const MathEngine = {
        getDemandAtDay,
        partialDerivative,
        gradient,
        hessian,
        newtonSolveGradZero,
        classifyCriticalPoint,
        findCriticalPointsGrid,
        sampleGrid,
        simulateStock,
        parseCSV,
        makeDemandFunction,
        util: { norm2, cloneArray, isFiniteNumber }
    };

    // Expõe para o escopo global
    if (typeof window !== 'undefined') window.MathEngine = MathEngine;
    if (typeof module !== 'undefined' && module.exports) module.exports = MathEngine;

})();