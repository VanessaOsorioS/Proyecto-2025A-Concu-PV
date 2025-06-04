import time
import os
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Union, Tuple, List, Dict, Set, Any
from copy import deepcopy

from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_NEG,
    INFTY_POS,
    LAST_IDX,
    EFECTO,
    ACTUAL,
)

def _process_batch(args):
    """Worker function for parallel batch processing"""
    deltas_batch, omegas, subsistema, sia_dists_marginales = args
    results = []
    for delta in deltas_batch:
        emd_union, emd_delta, vector_delta_marginal = funcion_submodular(delta, omegas, subsistema, sia_dists_marginales)
        results.append((delta, emd_union, emd_delta, vector_delta_marginal))
    return results

class QNodes(SIA):
    """
    Optimized QNodes class with deterministic parallelization.
    """
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.memoria_omega = {}
        self.memoria_particiones = {}
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(QNODES_STRAREGY_TAG)
        
        # Optimize worker count based on CPU cores, but cap at 8 for deterministic behavior
        self.num_workers = min(4, mp.cpu_count())
        self.logger.info(f"Initialized with {self.num_workers} workers")

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        mp.freeze_support()

        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        futuro = [(EFECTO, e) for e in self.sia_subsistema.indices_ncubos]
        presente = [(ACTUAL, a) for a in self.sia_subsistema.dims_ncubos]
        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )
        vertices = presente + futuro
        self.vertices = set(vertices)

        vertices_totales = sorted(vertices)
        start_time = time.time()

        # Dynamic batch size based on total workload and number of workers
        total_vertices = len(vertices_totales)
        batch_size = max(2, total_vertices // (self.num_workers * 2))
        
        omegas_ciclo = [vertices_totales[0]]
        deltas_restantes = sorted(vertices_totales[1:])
        resultados = {}

        with mp.Pool(processes=self.num_workers) as pool:
            while deltas_restantes:
                # Create batches for parallel processing
                batches = [
                    deltas_restantes[i:i + batch_size]
                    for i in range(0, len(deltas_restantes), batch_size)
                ]

                # Process batches in parallel
                args = [
                    (batch, omegas_ciclo, self.sia_subsistema, self.sia_dists_marginales)
                    for batch in batches
                ]
                
                all_results = []
                for batch_results in pool.map(_process_batch, args):
                    all_results.extend(batch_results)

                # Deterministic selection
                all_results.sort(key=lambda x: (round(x[1] - x[2], 10), str(x[0])))
                
                if not all_results:
                    break

                # Select best result deterministically
                best_delta, best_emd_union, best_emd_delta, best_dist = all_results[0]
                
                # Store result and update working sets
                resultados[best_delta] = (best_emd_delta, best_dist)
                omegas_ciclo.append(best_delta)
                deltas_restantes.remove(best_delta)

        # Final solution selection
        if resultados:
            best_key = min(resultados.keys(), key=lambda k: (resultados[k][0], str(k)))
            perdida, dist_marginal = resultados[best_key]
            key = (best_key,) if isinstance(best_key[0], int) else best_key
            fmt = fmt_biparte_q(list(key), self.nodes_complement(list(key)))
        else:
            self.logger.warning("No partitions found, using default values")
            key = []
            perdida = INFTY_NEG
            dist_marginal = np.zeros(1)
            fmt = {"left": [], "right": []}

        tiempo_total = time.time() - start_time
        self.logger.info(f"Total execution time: {tiempo_total:.4f} seconds")

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal,
            tiempo_total=tiempo_total,
            particion=fmt,
        )

    def nodes_complement(self, nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(list(set(self.vertices) - set(nodes)))

def funcion_submodular(
    deltas: Union[tuple, List[tuple]],
    omegas: List[Union[tuple, List[tuple]]],
    subsistema,  # tipo: System
    sia_dists_marginales: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    temporal = [[], []]

    if isinstance(deltas, tuple):
        d_tiempo, d_indice = deltas
        temporal[d_tiempo].append(d_indice)
    else:
        for delta in sorted(deltas):
            d_tiempo, d_indice = delta
            temporal[d_tiempo].append(d_indice)

    particion_delta = subsistema.bipartir(
        np.array(sorted(temporal[EFECTO]), dtype=np.int8),
        np.array(sorted(temporal[ACTUAL]), dtype=np.int8),
    )
    vector_delta_marginal = particion_delta.distribucion_marginal()
    emd_delta = emd_efecto(vector_delta_marginal, sia_dists_marginales)

    for omega in sorted(omegas, key=str):
        if isinstance(omega, list):
            for omg in sorted(omega):
                o_tiempo, o_indice = omg
                temporal[o_tiempo].append(o_indice)
        else:
            o_tiempo, o_indice = omega
            temporal[o_tiempo].append(o_indice)

    particion_union = subsistema.bipartir(
        np.array(sorted(temporal[EFECTO]), dtype=np.int8),
        np.array(sorted(temporal[ACTUAL]), dtype=np.int8),
    )
    vector_union_marginal = particion_union.distribucion_marginal()
    emd_union = emd_efecto(vector_union_marginal, sia_dists_marginales)

    return emd_union, emd_delta, vector_delta_marginal


