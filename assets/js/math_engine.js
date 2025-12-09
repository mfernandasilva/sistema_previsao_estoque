(function () {
    'use strict';

    /**
     * Copia um array criando um novo array com os mesmos elementos
     * @param {Array} a - Array a ser copiado
     * @returns {Array} Novo array com cópia superficial dos elementos
     */
    function cloneArray(a) { return a.slice(); }
    
    /**
     * Verifica se um valor é um número finito válido
     * @param {*} x - Valor a ser verificado
     * @returns {boolean} true se x é um número finito, false caso contrário
     */
    function isFiniteNumber(x) { return typeof x === 'number' && isFinite(x); }
    
    /**
     * Calcula a norma euclidiana (tamanho) de um vetor
     * @param {Array<number>} v - Vetor (array de números)
     * @returns {number} Norma euclidiana: √(v[0]² + v[1]² + ... + v[n]²)
     */
    function norm2(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }

    /**
     * Obtém o valor de demanda para um dia específico (pode ser decimal)
     * Usa interpolação linear simples entre dias inteiros
     * 
     * @param {Array<number>} vendas - Array de vendas diárias [v₀, v₁, v₂, ...]
     * @param {number} dia - Dia a consultar (pode ser decimal, ex: 1.5 entre dias 1 e 2)
     * @returns {number} Valor de demanda interpolado linearmente, ou 0 se fora do intervalo
     * 
     * @example
     * const vendas = [10, 7, 5, 8, 12];
     * getDemandAtDay(vendas, 1)    // retorna 7
     * getDemandAtDay(vendas, 1.5)  // retorna 6 (média entre 7 e 5)
     */
    function getDemandAtDay(vendas, dia) {
        const dia_int = Math.floor(dia);
        const frac = dia - dia_int;
        if (dia_int < 0 || dia_int >= vendas.length) return 0;
        if (dia_int === vendas.length - 1) return vendas[dia_int];
        const v0 = vendas[dia_int];
        const v1 = vendas[dia_int + 1];
        return v0 * (1 - frac) + v1 * frac;
    }

    /**
     * Calcula a derivada parcial de uma função em um ponto usando diferenças centrais
     * 
     * Aproximação: ∂f/∂x ≈ (f(x+h) - f(x-h)) / (2h)
     * 
     * @param {Function} f - Função f(t, s) a derivar
     * @param {number} t - Coordenada t (tempo/dia)
     * @param {number} s - Coordenada s (estoque)
     * @param {string} [varName='t'] - Variável em relação à qual derivar: 't' ou 's'
     * @param {number} [h=1e-4] - Passo para aproximação numérica
     * @returns {number} Valor da derivada parcial: ∂f/∂t ou ∂f/∂s
     * 
     * @example
     * const f = (t, s) => t*s;
     * partialDerivative(f, 2, 3, 't')  // ∂f/∂t ≈ 3
     * partialDerivative(f, 2, 3, 's')  // ∂f/∂s ≈ 2
     */
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

    /**
     * Calcula o vetor gradiente de uma função em um ponto
     * 
     * Gradiente: ∇f = [∂f/∂t, ∂f/∂s] - direção de maior crescimento
     * 
     * @param {Function} f - Função f(t, s)
     * @param {number} t - Coordenada t (tempo/dia)
     * @param {number} s - Coordenada s (estoque)
     * @param {number} [h=1e-4] - Passo para aproximação numérica
     * @returns {Array<number>} Vetor gradiente [∂f/∂t, ∂f/∂s]
     * 
     * @example
     * const f = (t, s) => t*s;
     * const grad = gradient(f, 2, 3);  // [3, 2]
     */
    function gradient(f, t, s, h = 1e-4) {
        const dfdt = partialDerivative(f, t, s, 't', h);
        const dfds = partialDerivative(f, t, s, 's', h);
        return [dfdt, dfds];
    }

    /**
     * Calcula a matriz Hessiana (matriz de segundas derivadas parciais) 2×2
     * 
     * Hessiana: H = [[∂²f/∂t², ∂²f/∂t∂s], [∂²f/∂s∂t, ∂²f/∂s²]]
     * Caracteriza a curvatura da função (aceleração)
     * 
     * @param {Function} f - Função f(t, s)
     * @param {number} t - Coordenada t (tempo/dia)
     * @param {number} s - Coordenada s (estoque)
     * @param {number} [h=1e-3] - Passo para aproximação numérica
     * @returns {Array<Array<number>>} Matriz Hessiana 2×2: [[f_tt, f_ts], [f_ts, f_ss]]
     * 
     * @example
     * const f = (t, s) => t*t + s*s;
     * const H = hessian(f, 1, 2);  // [[2, 0], [0, 2]]
     */
    function hessian(f, t, s, h = 1e-3) {
        const f_tt = (f(t + h, s) - 2 * f(t, s) + f(t - h, s)) / (h * h);
        const f_ss = (f(t, s + h) - 2 * f(t, s) + f(t, s - h)) / (h * h);
        const f_ts = (f(t + h, s + h) - f(t + h, s - h) - f(t - h, s + h) + f(t - h, s - h)) / (4 * h * h);
        return [[f_tt, f_ts], [f_ts, f_ss]];
    }

    /**
     * Inverte uma matriz 2×2
     * 
     * Para M = [[a, b], [c, d]], a inversa é (1/det) * [[d, -b], [-c, a]]
     * 
     * @param {Array<Array<number>>} m - Matriz 2×2: [[a, b], [c, d]]
     * @returns {Array<Array<number>>|null} Matriz inversa 2×2, ou null se singular (det ≈ 0)
     * 
     * @example
     * const m = [[1, 0], [0, 1]];
     * invert2x2(m);  // [[1, 0], [0, 1]]
     */
    function invert2x2(m) {
        const [[a, b], [c, d]] = m;
        const det = a * d - b * c;
        if (!isFiniteNumber(det) || Math.abs(det) < 1e-12) return null;
        const inv = [[d / det, -b / det], [-c / det, a / det]];
        return inv;
    }

    /**
     * Multiplica uma matriz 2×2 por um vetor 2×1
     * 
     * Resultado: [m[0][0]*v[0] + m[0][1]*v[1], m[1][0]*v[0] + m[1][1]*v[1]]
     * 
     * @param {Array<Array<number>>} m - Matriz 2×2
     * @param {Array<number>} v - Vetor 2×1
     * @returns {Array<number>} Vetor resultado 2×1
     * 
     * @example
     * const m = [[1, 2], [3, 4]];
     * const v = [5, 6];
     * mat2x2MulVec(m, v);  // [17, 39]
     */
    function mat2x2MulVec(m, v) {
        return [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
    }

    /**
     * Resolve o sistema ∇f = 0 (encontra pontos críticos) usando Método de Newton
     * em 2 variáveis
     * 
     * Iteração: x_{k+1} = x_k - H⁻¹ · ∇f(x_k)
     * onde H é a Hessiana
     * 
     * @param {Function} f - Função f(t, s)
     * @param {Array<number>} init - Ponto inicial [t₀, s₀]
     * @param {Object} [opts={}] - Opções
     * @param {number} [opts.maxIter=50] - Número máximo de iterações
     * @param {number} [opts.tol=1e-6] - Tolerância de convergência
     * @param {number} [opts.h=1e-3] - Passo para aproximação de derivadas
     * @returns {Object} Objeto com {converged, x, iter, grad, reason}
     * 
     * @example
     * const f = (t, s) => (t-1)**2 + (s-2)**2;
     * const result = newtonSolveGradZero(f, [0, 0]);
     * // result.converged === true, result.x ≈ [1, 2]
     */
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

    /**
     * Classifica um ponto crítico (máximo local, mínimo local ou ponto de sela)
     * usando o teste da Hessiana
     * 
     * - Se det(H) > 0 e H[0][0] > 0: mínimo local
     * - Se det(H) > 0 e H[0][0] < 0: máximo local
     * - Se det(H) < 0: ponto de sela
     * - Se det(H) = 0: teste inconclusivo
     * 
     * @param {Array<Array<number>>} H - Matriz Hessiana 2×2
     * @returns {string} Tipo do ponto: 'local_min', 'local_max', 'saddle', 'degenerate', 'indeterminate'
     * 
     * @example
     * const H = [[2, 0], [0, 2]];
     * classifyCriticalPoint(H);  // 'local_min'
     */
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

    /**
     * Busca pontos críticos em uma região retangular usando varredura em grade
     * seguida de refinamento com Método de Newton
     * 
     * Estratégia:
     * 1. Cria grade regular em [tRange] × [sRange]
     * 2. Identifica candidatos onde ||∇f|| < eps
     * 3. Refina cada candidato com Newton
     * 4. Remove duplicatas
     * 5. Classifica cada ponto usando Hessiana
     * 
     * @param {Function} f - Função f(t, s)
     * @param {Array<number>} tRange - Intervalo [t_min, t_max]
     * @param {Array<number>} sRange - Intervalo [s_min, s_max]
     * @param {number} [nt=50] - Número de pontos na grade para t
     * @param {number} [ns=50] - Número de pontos na grade para s
     * @param {Object} [opts={}] - Opções
     * @param {number} [opts.eps=1e-2] - Tolerância para identificar candidatos
     * @param {number} [opts.maxIter=20] - Máx iterações do Newton
     * @param {number} [opts.tol=1e-6] - Tolerância do Newton
     * @param {number} [opts.h=1e-3] - Passo para derivadas
     * @returns {Array<Object>} Array de pontos críticos com {t, s, H, type, grad, iter, key}
     * 
     * @example
     * const f = (t, s) => t*t - s*s;
     * const points = findCriticalPointsGrid(f, [0, 1], [0, 1], 20, 20);
     */
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

    /**
     * Amostrer uma função em uma grade regular e calcula gradientes
     * 
     * Útil para visualizar a superfície e sua sensibilidade
     * 
     * @param {Function} f - Função f(t, s)
     * @param {Array<number>} tRange - Intervalo [t_min, t_max]
     * @param {Array<number>} sRange - Intervalo [s_min, s_max]
     * @param {number} [nt=50] - Número de pontos para t
     * @param {number} [ns=50] - Número de pontos para s
     * @param {Object} [opts={}] - Opções
     * @param {number} [opts.h=1e-3] - Passo para gradientes
     * @returns {Object} Objeto com {Ts, Ss, Z, Gt, Gs}
     *   - Ts: Array de valores de t
     *   - Ss: Array de valores de s
     *   - Z: Matriz nt×ns com valores de f
     *   - Gt: Matriz nt×ns com ∂f/∂t
     *   - Gs: Matriz nt×ns com ∂f/∂s
     * 
     * @example
     * const f = (t, s) => t*s;
     * const grid = sampleGrid(f, [0, 1], [0, 1], 10, 10);
     */
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

    /**
     * Simula a evolução do estoque no tempo usando Método de Euler explícito
     * 
     * Discretização: S(t+dt) = S(t) - d(t,S)*dt + reposição(t)
     * onde d(t,S) é a taxa de demanda
     * 
     * @param {Object} params - Parâmetros da simulação
     * @param {number} [params.S0=100] - Estoque inicial
     * @param {number} [params.t0=0] - Tempo inicial
     * @param {number} [params.tEnd=30] - Tempo final
     * @param {number} [params.dt=1] - Passo de tempo
     * @param {Function} [params.demandFunc=(t,S)=>0] - Função demanda(t, S)
     * @param {Array<{time, qty}>} [params.replenishments] - Reposições programadas
     * @param {Object} [params.periodic] - Reposição periódica
     * @param {number} [params.periodic.interval] - Intervalo entre reposições
     * @param {number} [params.periodic.qty] - Quantidade por reposição
     * @param {number} [params.periodic.start=t0] - Tempo de início
     * @returns {Array<Object>} Array de passos com {t, S, demand, replenishment, rupture?}
     * 
     * @example
     * const f = (t, S) => 5 + 0.1*S;
     * const sim = simulateStock({S0: 100, tEnd: 10, dt: 0.5, demandFunc: f, periodic: {interval: 3, qty: 50}});
     */
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

    /**
     * Parser CSV simples: retorna array de objetos (converte números)
     * 
     * Processa uma string CSV linha por linha, criando array de objetos
     * onde cada chave é definida pelo cabeçalho (primeira linha)
     * Tenta converter valores para número automaticamente
     * 
     * @param {string} text - Conteúdo CSV completo
     * @param {string} [sep=','] - Delimitador entre colunas (padrão vírgula)
     * @returns {Array<Object>} Array onde cada item representa uma linha com chaves do cabeçalho
     * 
     * @example
     * const csv = "dia,vendas\\n1,100\\n2,120\\n3,95";
     * const data = parseCSV(csv);  // [{dia: 1, vendas: 100}, {dia: 2, vendas: 120}, ...]
     */
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

    /**
     * Fábrica que constrói uma função f(t, s) = demanda(t) × fator(s)
     * 
     * Combina uma função de demanda temporal (interpolada de dados discretos)
     * com um fator de modulação dependente de estoque
     * 
     * @param {Array<number>} vendas - Dados de vendas discretas por dia
     * @param {Function|undefined} [fatorEstoque] - Função fator(s) ou undefined para fator=1
     * @returns {Function} Função f(t,s) que calcula demanda interpolada × fator(s)
     * 
     * @example
     * const f = makeDemandFunction([100, 120, 95], (s) => 1 + 0.01*s);
     * const demand = f(1.5, 50);  // Demanda em t=1.5 com s=50
     */
    function makeDemandFunction(vendas, fatorEstoque) {
        const g = (typeof fatorEstoque === 'function') ? fatorEstoque : (s => 1);
        return function (t, s) {
            const demand_t = getDemandAtDay(vendas, t);
            return demand_t * g(s);
        };
    }

    /**
     * Prevê a demanda para os próximos N dias com base em padrão temporal
     * 
     * Usa função de demanda com variação sazonal e crescimento linear
     * 
     * @param {Object} demandParams - Parâmetros de demanda
     * @param {number} demandParams.media - Demanda média diária (kg)
     * @param {number} demandParams.variacao - Variação típica ±% em torno da média
     * @param {boolean} [demandParams.crescimento=false] - Se true, aplica crescimento linear +5% ao longo do período
     * @param {number} [dias=30] - Número de dias para prever
     * @returns {Array<Object>} Array com {dia, demanda, estoque_necessario}
     * 
     * @example
     * const pred = forecastDemand({media: 35, variacao: 13, crescimento: true}, 30);
     */
    function forecastDemand(demandParams, dias = 30) {
        const { media, variacao, crescimento } = demandParams;
        const forecast = [];
        for (let d = 1; d <= dias; d++) {
            // variação sazonal (senoidal)
            const phase = (d / dias) * 2 * Math.PI;
            const sazonal = media * (variacao / 100) * Math.sin(phase);
            // crescimento linear opcional
            const crescFator = crescimento ? 1 + 0.05 * (d / dias) : 1;
            const demanda = (media + sazonal) * crescFator;
            forecast.push({ dia: d, demanda: Math.max(0, demanda), estoque_necessario: demanda * dias / d });
        }
        return forecast;
    }

    /**
     * Simula degradação de produto com taxa de perda diária
     * 
     * @param {Object} perecParams - Parâmetros de perecibilidade
     * @param {number} perecParams.taxa_perda_pct - Taxa de perda diária (%)
     * @param {number} perecParams.prazo_validade - Prazo de validade em dias
     * @param {number} quantidade_inicial - Quantidade inicial (kg)
     * @param {number} [dias=30] - Número de dias a simular
     * @returns {Array<Object>} Array com {dia, quantidade, perda_acumulada, disponivel_pct}
     * 
     * @example
     * const perec = simulatePerecibility({taxa_perda_pct: 1.5, prazo_validade: 15}, 100, 30);
     */
    function simulatePerecibility(perecParams, quantidade_inicial, dias = 30) {
        const { taxa_perda_pct, prazo_validade } = perecParams;
        const taxaDiaria = taxa_perda_pct / 100;
        const timeline = [];
        let qtd = quantidade_inicial;
        for (let d = 1; d <= dias; d++) {
            // perda exponencial decrescente
            const perda = qtd * taxaDiaria;
            qtd = Math.max(0, qtd - perda);
            // após prazo de validade, todo produto é descartado
            const disponivel = d <= prazo_validade ? qtd : 0;
            timeline.push({
                dia: d,
                quantidade: qtd,
                perda_acumulada: quantidade_inicial - qtd,
                disponivel_pct: (disponivel / quantidade_inicial) * 100
            });
        }
        return timeline;
    }

    /**
     * Otimiza quantidade de reposição considerando demanda, custo e perecibilidade
     * 
     * Usa Hessiana de custo = (replenishment_qty)² - (leadtime_days * demand) * replenishment_qty
     * para encontrar ponto ótimo
     * 
     * @param {Object} params - Parâmetros de otimização
     * @param {number} params.demanda_media - Demanda média diária
     * @param {number} params.custo_estoque - Custo de manutenção por kg/dia
     * @param {number} params.custo_reposicao - Custo fixo por ordem
     * @param {number} params.lead_time - Dias para entrega
     * @param {number} params.taxa_perda - Taxa de perda diária (%)
     * @returns {Object} {quantidade_otima, ciclo_reposicao_dias, custo_total_diario}
     * 
     * @example
     * const opt = optimizeReplenishment({demanda_media: 35, custo_estoque: 2, custo_reposicao: 50, lead_time: 3, taxa_perda: 1.5});
     */
    function optimizeReplenishment(params) {
        const { demanda_media, custo_estoque, custo_reposicao, lead_time, taxa_perda } = params;
        // Fórmula EOQ modificada com perecibilidade: Q* = sqrt(2*D*S / (H*(1+taxa_perda)))
        const taxa = taxa_perda / 100;
        const fatorPerda = 1 + taxa;
        const Q = Math.sqrt((2 * demanda_media * custo_reposicao) / (custo_estoque * fatorPerda));
        const ciclo = Q / demanda_media;
        const custo_diario = (custo_reposicao / ciclo) + (Q / 2) * custo_estoque;
        return {
            quantidade_otima: Math.ceil(Q),
            ciclo_reposicao_dias: Math.ceil(ciclo),
            custo_total_diario: custo_diario,
            estoque_medio: Q / 2
        };
    }

    /**
     * Simula estoque completo considerando demanda, perecibilidade e reposição
     * 
     * @param {Object} config - Configuração completa
     * @param {number} config.estoque_inicial - Estoque inicial (kg)
     * @param {number} config.demanda_media - Demanda média (kg/dia)
     * @param {number} config.demanda_variacao - Variação % da demanda
     * @param {number} config.taxa_perda - Taxa de perda diária (%)
     * @param {number} config.prazo_validade - Prazo de validade (dias)
     * @param {number} config.qtd_reposicao - Quantidade por reposição (kg)
     * @param {number} config.lead_time - Dias para chegar reposição
     * @param {number} config.estoque_minimo - Estoque mínimo desejado (kg)
     * @param {number} [config.dias=30] - Dias a simular
     * @returns {Array<Object>} Timeline com {dia, demanda, estoque, perda, ruptura, reposicao_em_transito}
     * 
     * @example
     * const sim = completeStockSimulation({...config, dias: 30});
     */
    function completeStockSimulation(config) {
        const {
            estoque_inicial, demanda_media, demanda_variacao, taxa_perda, prazo_validade,
            qtd_reposicao, lead_time, estoque_minimo, dias = 30
        } = config;
        
        const timeline = [];
        let estoque = estoque_inicial;
        let em_transito = 0;
        let dias_para_chegada = 0;
        let acumulado_perda = 0;
        
        for (let d = 1; d <= dias; d++) {
            // demanda do dia (sazonal)
            const phase = (d / dias) * 2 * Math.PI;
            const sazonal = demanda_media * (demanda_variacao / 100) * Math.sin(phase);
            const demanda = Math.max(0, demanda_media + sazonal);
            
            // perda por perecibilidade
            const perda = estoque * (taxa_perda / 100);
            
            // verificar se reposição chega hoje
            if (dias_para_chegada > 0) {
                dias_para_chegada--;
                if (dias_para_chegada === 0) {
                    estoque += em_transito;
                    em_transito = 0;
                }
            }
            
            // deduz demanda e perda
            estoque = Math.max(0, estoque - demanda - perda);
            acumulado_perda += perda;
            
            // decision: fazer reposição?
            let reposicao_disparada = false;
            if (estoque <= estoque_minimo && dias_para_chegada === 0 && em_transito === 0) {
                em_transito = qtd_reposicao;
                dias_para_chegada = lead_time;
                reposicao_disparada = true;
            }
            
            timeline.push({
                dia: d,
                demanda: parseFloat(demanda.toFixed(2)),
                estoque: parseFloat(estoque.toFixed(2)),
                perda: parseFloat(perda.toFixed(2)),
                perda_acumulada: parseFloat(acumulado_perda.toFixed(2)),
                ruptura: estoque <= 0,
                reposicao_disparada,
                em_transito,
                dias_para_chegada
            });
        }
        return timeline;
    }

    /**
     * Objeto público do MathEngine com todos os métodos matemáticos e utilitários
     * 
     * Implementa conceitos de Cálculo 2: derivadas parciais, gradientes, Hessianas,
     * pontos críticos, e aplicações a problemas de previsão de estoque
     * 
     * @namespace MathEngine
     * @property {Function} getDemandAtDay - Interpola demanda em dia fracionário
     * @property {Function} partialDerivative - Calcula ∂f/∂(t|s) via diferenças centrais
     * @property {Function} gradient - Retorna vetor gradiente ∇f = [∂f/∂t, ∂f/∂s]
     * @property {Function} hessian - Retorna matriz Hessiana 2×2 das derivadas segundas
     * @property {Function} newtonSolveGradZero - Resolve ∇f = 0 usando Método de Newton
     * @property {Function} classifyCriticalPoint - Classifica pontos críticos (min, max, sela, etc)
     * @property {Function} findCriticalPointsGrid - Busca pontos críticos em região retangular
     * @property {Function} sampleGrid - Amostrou função e gradientes em grade regular
     * @property {Function} simulateStock - Simula evolução de estoque via Método de Euler
     * @property {Function} parseCSV - Parser CSV que retorna array de objetos
     * @property {Function} makeDemandFunction - Cria f(t,s) = demanda(t) × fator(s)
     * @property {Function} forecastDemand - Prevê demanda para próximos N dias
     * @property {Function} simulatePerecibility - Simula degradação com taxa de perda
     * @property {Function} optimizeReplenishment - Calcula quantidade ótima de reposição (EOQ)
     * @property {Function} completeStockSimulation - Simula estoque completo (demanda+perda+reposição)
     * @property {Object} util - Utilitários internos {norm2, cloneArray, isFiniteNumber}
     */
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
        forecastDemand,
        simulatePerecibility,
        optimizeReplenishment,
        completeStockSimulation,
        util: { norm2, cloneArray, isFiniteNumber }
    };

    // Expõe para o escopo global
    if (typeof window !== 'undefined') window.MathEngine = MathEngine;
    if (typeof module !== 'undefined' && module.exports) module.exports = MathEngine;

})();