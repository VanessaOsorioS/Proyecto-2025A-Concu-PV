import time
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from concurrent.futures import ProcessPoolExecutor

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


class QNodes(SIA):
    """
    Clase QNodes para el análisis de redes mediante el algoritmo Q.

    Esta clase implementa un gestor principal para el análisis de redes que utiliza
    el algoritmo Q para encontrar la partición óptima que minimiza la
    pérdida de información en el sistema. Hereda de la clase base SIA (Sistema de
    Información Activo) y proporciona funcionalidades para analizar la estructura
    y dinámica de la red.

    Args:
    ----
        config (Loader):
            Instancia de la clase Loader que contiene la configuración del sistema
            y los parámetros necesarios para el análisis.

    Attributes:
    ----------
        m (int):
            Número de elementos en el conjunto de purview (vista).

        n (int):
            Número de elementos en el conjunto de mecanismos.

        tiempos (tuple[np.ndarray, np.ndarray]):
            Tupla de dos arrays que representan los tiempos para los estados
            actual y efecto del sistema.

        etiquetas (list[tuple]):
            Lista de tuplas conteniendo las etiquetas para los nodos,
            con versiones en minúsculas y mayúsculas del abecedario.

        vertices (set[tuple]):
            Conjunto de vértices que representan los nodos de la red,
            donde cada vértice es una tupla (tiempo, índice).

        memoria (dict):
            Diccionario para almacenar resultados intermedios y finales
            del análisis (memoización).

        logger:
            Instancia del logger configurada para el análisis Q.

    Methods:
    -------
        run(condicion, purview, mechanism):
            Ejecuta el análisis principal de la red con las condiciones,
            purview y mecanismo especificados.

        algorithm(vertices):
            Implementa el algoritmo Q para encontrar la partición
            óptima del sistema.

        funcion_submodular(deltas, omegas):
            Calcula la función submodular para evaluar particiones candidatas.

        view_solution(mip):
            Visualiza la solución encontrada en términos de las particiones
            y sus valores asociados.

        nodes_complement(nodes):
            Obtiene el complemento de un conjunto de nodos respecto a todos
            los vértices del sistema.

    Notes:
    -----
    - La clase implementa una versión secuencial del algoritmo Q para encontrar la partición que minimiza la pérdida de información.
    - Utiliza memoización para evitar recálculos innecesarios durante el proceso.
    - El análisis se realiza considerando dos tiempos: actual (presente) y
      efecto (futuro).
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        # self.memoria_delta = dict() # Se puede integrar en memoria_omega si la clave es hashable (tuple)
        self.memoria_omega = dict() # Almacena {tupla_nodos: (emd, dist_marginal)}
        self.memoria_particiones = dict()

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        self.logger = SafeLogger(QNODES_STRAREGY_TAG)

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # Optimizando la construcción de presente y futuro
        # Usamos listas de tuplas directamente y luego una única conversión a set
        futuro = [(EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos]
        presente = [(ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos]

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size

        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        # Solo convertimos a set una vez para `self.vertices`
        self.vertices = set(presente + futuro)
        vertices_list = list(presente + futuro) # Usamos una lista para el algoritmo

        mip = self.algorithm(vertices_list) # Pasamos la lista directamente
        
        # Asegurarse de que mip sea una tupla hashable para buscar en el diccionario
        if not isinstance(mip, tuple):
             mip = tuple(mip)

        fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
        perdida_mip, dist_marginal_mip = self.memoria_particiones[mip]

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=perdida_mip,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal_mip,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def algorithm(self, vertices: list[tuple[int, int]]):
        """
        Implementa el algoritmo Q para encontrar la partición óptima de un sistema que minimiza la pérdida de información, basándose en principios de submodularidad dentro de la teoría de lainformación.

        El algoritmo opera sobre un conjunto de vértices que representan nodos en diferentes tiempos del sistema (presente y futuro). La idea fundamental es construir incrementalmente grupos de nodos que, cuando se particionan, producen la menor pérdida posible de información en el sistema.

        Proceso Principal:
        -----------------
        El algoritmo comienza estableciendo dos conjuntos fundamentales: omega (W) y delta.
        Omega siempre inicia con el primer vértice del sistema, mientras que delta contiene todos los vértices restantes. Esta decisión no es arbitraria - al comenzar con un
        solo elemento en omega, podemos construir grupos de manera incremental evaluando cómo cada adición afecta la pérdida de información.

        La ejecución se desarrolla en fases, ciclos e iteraciones, donde cada fase representa un nivel diferente y conlleva a la formación de una partición candidata, cada ciclo representa un incremento de elementos al conjunto W y cada iteración determina al final cuál es el mejor elemento/cambio/delta para añadir en W.
        Fase >> Ciclo >> Iteración.

        1. Formación Incremental de Grupos:
        El algoritmo mantiene un conjunto omega que crece gradualmente en cada j-iteración. En cada paso, evalúa todos los deltas restantes para encontrar cuál, al unirse con omega produce la menor pérdida de información. Este proceso utiliza la función submodular para calcular la diferencia entre la EMD (Earth Mover's Distance) de la combinación y la EMD individual del delta evaluado.

        2. Evaluación de deltas:
        Para cada delta candidato el algoritmo:
        - Calcula su EMD individual si no está en memoria.
        - Calcula la EMD de su combinación con el conjunto omega actual
        - Determina la diferencia entre estas EMDs (el "costo" de la combinación)
        El delta que produce el menor costo se selecciona y se añade a omega.

        3. Formación de Nuevos Grupos:
        Al final de cada fase cuando omega crezca lo suficiente, el algoritmo:
        - Toma los últimos elementos de omega y delta (par candidato).
        - Los combina en un nuevo grupo
        - Actualiza la lista de vértices para la siguiente fase
        Este proceso de agrupamiento permite que el algoritmo construya particiones
        cada vez más complejas y reutilice estos "pares candidatos" para particiones en conjunto.

        Optimización y Memoria:
        ----------------------
        El algoritmo utiliza dos estructuras de memoria clave:
        - individual_memory: Almacena las EMDs y distribuciones de nodos individuales, evitando recálculos muy costosos.
        - partition_memory: Guarda las EMDs y distribuciones de las particiones completas, permitiendo comparar diferentes combinaciones de grupos teniendo en cuenta que su valor real está asociado al valor individual de su formación delta.

        La memoización es relevante puesto muchos cálculos de EMD son computacionalmente costos y se repiten durante la ejecución del algoritmo.

        Resultado:
        ---------------
        Al terminar todas las fases, el algoritmo selecciona la partición que produjo la menor EMD global, representando la división del sistema que mejor preserva su información causal.

        Args:
            vertices (list[tuple[int, int]]): Lista de vértices donde cada uno es una
                tupla (tiempo, índice). tiempo=0 para presente (t_0), tiempo=1 para futuro (t_1).

        Returns:
            tuple[float, tuple[tuple[int, int], ...]]: El valor de pérdida en la primera posición, asociado con la partición óptima encontrada, identificada por la clave en partition_memory que produce la menor EMD.
        """
        # Aseguramos que la lista inicial de vértices sea mutable para las operaciones pop/append
        vertices_fase = list(vertices)

        # Inicializamos omegas y deltas como listas
        # omegas_ciclo y deltas_ciclo serán modificadas, por lo que deben ser listas
        omegas_ciclo = [vertices_fase[0]]
        deltas_ciclo = list(vertices_fase[1:]) # Aseguramos que sea una nueva lista mutable

        total = len(vertices_fase) - 2
        for i in range(len(vertices_fase) - 2):
            self.logger.debug(f"total: {total-i}")
            # Re-inicializamos omegas_ciclo y deltas_ciclo para cada 'fase' (bucle 'i')
            # Esto es crucial si `vertices_fase` se modifica en el bucle interior
            # Si `vertices_fase` NO se modifica en el bucle 'j', estas líneas pueden ir fuera del bucle 'i'
            # o su comportamiento cambiará. Asumo que es el comportamiento deseado re-inicializar
            # para cada fase 'i'.
            if i > 0: # Para la primera iteración, ya están inicializados
                omegas_ciclo = [vertices_fase[0]]
                deltas_ciclo = list(vertices_fase[1:])

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None # Inicializar para asegurar que siempre tenga un valor

            # Este bucle `j` es donde se construyen los grupos
            for j in range(len(deltas_ciclo)): # Cambiado de -1 a sin -1, ya que pop() reducirá el tamaño
                emd_local = INFTY_POS # Inicializar en cada iteración 'j'
                indice_mip: int = -1 # Inicializar para evitar UnboundLocalError

                # Bucle 'k' para evaluar cada delta candidato
                for k in range(len(deltas_ciclo)):
                    current_delta = deltas_ciclo[k]
                    
                    # Llamada a la función submodular.
                    # Aquí la MEMOIZACIÓN se vuelve crucial (ver comentarios en funcion_submodular)
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        current_delta, omegas_ciclo
                    )
                    emd_iteracion = emd_union - emd_delta

                    if emd_iteracion < emd_local:
                        emd_local = emd_iteracion
                        indice_mip = k
                        # Almacenamos el delta que dio el mejor resultado en esta iteración 'j'
                        # para usarlo para la partición candidata
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                # Asegurarse de que `indice_mip` se haya asignado antes de usarlo
                if indice_mip != -1:
                    # Mover el mejor delta de `deltas_ciclo` a `omegas_ciclo`
                    # `pop` devuelve el elemento eliminado
                    best_delta_to_add = deltas_ciclo.pop(indice_mip)
                    omegas_ciclo.append(best_delta_to_add)
                else:
                    # Esto podría indicar un problema si no se encontró un delta óptimo
                    # Manejar el error o lógica de salida según sea necesario
                    self.logger.error("No se encontró un delta óptimo en la iteración.")
                    break # Salir del bucle 'j' si no se encuentra un delta

            # Almacenar la mejor partición para esta "fase" (bucle `i`)
            # La clave del diccionario debe ser hashable (una tupla)
            # Asegúrate de que `deltas_ciclo` (la partición restante) sea una tupla
            # si va a ser la clave del diccionario.
            final_delta_group = tuple(deltas_ciclo) # Convertir a tupla para hashable
            self.memoria_particiones[final_delta_group] = emd_particion_candidata, dist_particion_candidata

            # Preparar para la siguiente fase: combinar los últimos elementos
            # Esta lógica puede ser compleja si omegas_ciclo contiene listas de tuplas
            # Asegúrate de que `par_candidato` se convierta en una tupla de tuplas
            # si los elementos dentro de omegas_ciclo y deltas_ciclo pueden ser grupos.

            # Simplificación para el par_candidato:
            # Si `omegas_ciclo` y `deltas_ciclo` solo contienen tuplas individuales,
            # la lógica puede ser más sencilla. Si pueden contener listas de tuplas (grupos),
            # la "aplanación" del par candidato es necesaria.
            
            # Asumiendo que `omegas_ciclo` al final de la fase 'j' tiene el grupo principal
            # y `deltas_ciclo` tiene el grupo restante.
            # Convertimos todo a una sola lista de tuplas para la nueva `vertices_fase`
            
            # Construcción del `par_candidato` y `vertices_fase` para la siguiente iteración del bucle `i`.
            # Esta parte puede necesitar un ajuste fino dependiendo de la lógica de agrupamiento deseada.
            # Por ahora, simplemente tomamos `omegas_ciclo` como la nueva `vertices_fase`
            # para la próxima iteración del bucle 'i'.
            vertices_fase = list(omegas_ciclo) # Reinicia la `vertices_fase` con los `omegas_ciclo` del final de esta fase.


        # Retornar la clave (la tupla de nodos) de la partición con la menor pérdida
        # Asegurarse de que las claves en `memoria_particiones` sean tuplas de tuplas
        # para que la comparación sea correcta.
        return min(
            self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0]
        )

    def funcion_submodular(
        self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]
    ):
        """
        Evalúa el impacto de combinar el conjunto de nodos individual delta y su agrupación con el conjunto omega, calculando la diferencia entre EMD (Earth Mover's Distance) de las configuraciones, en conclusión los nodos delta evaluados individualmente y su combinación con el conjunto omega.

        El proceso se realiza en dos fases principales:

        1. Evaluación Individual:
            - Crea una copia del estado temporal del subsistema.
            - Activa los nodos delta en su tiempo correspondiente (presente/futuro).
            - Si el delta ya fue evaluado antes, recupera su EMD y distribución marginal de memoria
            - Si no, ha de:
              * Identificar dimensiones activas en presente y futuro.
              * Realiza bipartición del subsistema con esas dimensiones.
              * Calcular la distribución marginal y EMD respecto al subsistema.
              * Guarda resultados en memoria para seguro un uso futuro.

        2. Evaluación Combinada:
            - Sobre la misma copia temporal, activa también los nodos omega.
            - Calcula dimensiones activas totales (delta + omega).
            - Realiza bipartición del subsistema completo.
            - Obtiene EMD de la combinación.

        Args:
            deltas: Un nodo individual (tupla) o grupo de nodos (lista de tuplas)
                    donde cada tupla está identificada por su (tiempo, índice), sea el tiempo t_0 identificado como 0, t_1 como 1 y, el índice hace referencia a las variables/dimensiones habilitadas para operaciones de substracción/marginalización sobre el subsistema, tal que genere la partición.
            omegas: Lista de nodos ya agrupados, puede contener tuplas individuales
                    o listas de tuplas para grupos formados por los pares candidatos o más uniones entre sí (grupos candidatos).

        Returns:
            tuple: (
                EMD de la combinación omega y delta,
                EMD del delta individual,
                Distribución marginal del delta individual
            )
            Esto lo hice así para hacer almacenamiento externo de la emd individual y su distribución marginal en las particiones candidatas.
        """
        # Clave hashable para memoización de 'deltas'
        delta_key = tuple(deltas) if isinstance(deltas, list) else deltas

        emd_delta = INFTY_NEG
        vector_delta_marginal = None

        # --- Memoización para el cálculo individual de delta ---
        if delta_key in self.memoria_omega: # O usar un diccionario específico para 'deltas_individuales'
            emd_delta, vector_delta_marginal = self.memoria_omega[delta_key]
        else:
            temporal_delta_dims = [[], []]
            if isinstance(deltas, tuple):
                d_tiempo, d_indice = deltas
                temporal_delta_dims[d_tiempo].append(d_indice)
            else: # Es una lista de tuplas
                for delta in deltas:
                    d_tiempo, d_indice = delta
                    temporal_delta_dims[d_tiempo].append(d_indice)

            copia_delta = self.sia_subsistema # Asumo que es una copia o que la bipartición no modifica el original
            
            # Asegurarse de que los arrays sean np.ndarray para `bipartir`
            dims_alcance_delta = np.array(temporal_delta_dims[EFECTO], dtype=np.int8)
            dims_mecanismo_delta = np.array(temporal_delta_dims[ACTUAL], dtype=np.int8)

            particion_delta = copia_delta.bipartir(
                dims_alcance_delta,
                dims_mecanismo_delta,
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            
            # Guardar en memoria
            self.memoria_omega[delta_key] = (emd_delta, vector_delta_marginal)


        # --- Cálculo de la Unión ---
        temporal_union_dims = [[], []]

        # Añadir las dimensiones de deltas a la unión
        if isinstance(deltas, tuple):
            temporal_union_dims[deltas[0]].append(deltas[1])
        else:
            for delta in deltas:
                temporal_union_dims[delta[0]].append(delta[1])

        # Añadir las dimensiones de omegas a la unión
        for omega_group in omegas:
            if isinstance(omega_group, list): # Si es un grupo de nodos (lista de tuplas)
                for omg in omega_group:
                    temporal_union_dims[omg[0]].append(omg[1])
            else: # Si es un nodo individual (tupla)
                temporal_union_dims[omega_group[0]].append(omega_group[1])

        # Convertir a tuplas inmutables y ordenadas para una clave de diccionario fiable
        # Esto es crucial para la memoización de la unión
        # Se pueden usar sets para garantizar unicidad y luego convertir a tupla para hashable
        dims_alcance_union_sorted = tuple(sorted(set(temporal_union_dims[EFECTO])))
        dims_mecanismo_union_sorted = tuple(sorted(set(temporal_union_dims[ACTUAL])))
        
        # Clave para la memoización de la unión
        union_key = (dims_alcance_union_sorted, dims_mecanismo_union_sorted)

        emd_union = INFTY_NEG

        # --- Memoización para el cálculo de la unión ---
        if union_key in self.memoria_omega: # Reutilizamos memoria_omega para la unión también
            emd_union, _ = self.memoria_omega[union_key] # Solo necesitamos la EMD aquí
        else:
            copia_union = self.sia_subsistema
            
            # Asegurarse de que los arrays sean np.ndarray para `bipartir`
            particion_union = copia_union.bipartir(
                np.array(dims_alcance_union_sorted, dtype=np.int8),
                np.array(dims_mecanismo_union_sorted, dtype=np.int8),
            )
            vector_union_marginal = particion_union.distribucion_marginal()
            emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)
            
            # Guardar en memoria
            self.memoria_omega[union_key] = (emd_union, vector_union_marginal)


        return emd_union, emd_delta, vector_delta_marginal

    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))