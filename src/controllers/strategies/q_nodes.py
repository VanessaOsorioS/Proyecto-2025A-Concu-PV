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

def _process_submodular(args):
    """Worker function for parallel processing"""
    qnodes, delta, omegas, sia_dists_marginales = args
    emd_union, emd_delta, vector_delta_marginal = qnodes.funcion_submodular(delta, omegas, sia_dists_marginales)
    return delta, emd_union, emd_delta, vector_delta_marginal

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

        self.num_workers = 4  # Fixed for deterministic behavior
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

        manager = mp.Manager()
        shared_particiones = manager.dict()

        vertices_totales = sorted(vertices)
        start_time = time.time()

        chunk_size = 2
        omegas_ciclo = [vertices_totales[0]]
        deltas_restantes = sorted(vertices_totales[1:])

        with mp.Pool(processes=self.num_workers) as pool:
            while len(deltas_restantes) > 0:
                chunks = [deltas_restantes[i:i + chunk_size] for i in range(0, len(deltas_restantes), chunk_size)]

                all_results = []
                for chunk in chunks:
                    args = [
                        (self, delta, omegas_ciclo, deepcopy(self.sia_dists_marginales))
                        for delta in chunk
                    ]
                    chunk_results = pool.map(_process_submodular, args)
                    all_results.extend(chunk_results)

                # Deterministic selection: sort by cost then delta
                all_results.sort(key=lambda x: (round(x[1] - x[2], 10), str(x[0])))

                min_cost = float('inf')
                best_delta = None
                best_emd_delta = None
                best_dist = None

                for delta, emd_union, emd_delta, dist in all_results:
                    cost = emd_union - emd_delta
                    if cost < min_cost or (cost == min_cost and str(delta) < str(best_delta)):
                        min_cost = cost
                        best_delta = delta
                        best_emd_delta = emd_delta
                        best_dist = dist

                if best_delta is not None:
                    shared_particiones[tuple(best_delta)] = (best_emd_delta, best_dist)
                    omegas_ciclo.append(best_delta)
                    deltas_restantes = [d for d in deltas_restantes if d != best_delta]

        if shared_particiones:
            raw_key = min(shared_particiones.keys(), key=lambda k: (shared_particiones[k][0], str(k)))
            key = (raw_key,) if isinstance(raw_key[0], int) else raw_key
            perdida, dist_marginal = shared_particiones[raw_key]
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

    def funcion_submodular(
        self,
        deltas: Union[tuple, List[tuple]],
        omegas: List[Union[tuple, List[tuple]]],
        sia_dists_marginales: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        emd_delta = INFTY_NEG
        temporal = [[], []]

        if isinstance(deltas, tuple):
            d_tiempo, d_indice = deltas
            temporal[d_tiempo].append(d_indice)
        else:
            for delta in sorted(deltas):
                d_tiempo, d_indice = delta
                temporal[d_tiempo].append(d_indice)

        copia_delta = self.sia_subsistema
        particion_delta = copia_delta.bipartir(
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

        copia_union = self.sia_subsistema
        particion_union = copia_union.bipartir(
            np.array(sorted(temporal[EFECTO]), dtype=np.int8),
            np.array(sorted(temporal[ACTUAL]), dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, sia_dists_marginales)

        return emd_union, emd_delta, vector_delta_marginal

    def nodes_complement(self, nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(list(set(self.vertices) - set(nodes)))