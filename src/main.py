from src.controllers.manager import Manager

from src.controllers.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                    # 20 bits
    estado_inicial = "100000000000000"
    condiciones =    "111111111111111"
    alcance =        "111111111111111"
    mecanismo =      "111111111111111"

    gestor_sistema = Manager(estado_inicial)

    # ✅ Verifica que existe TPM de 20 nodos, o créala si no
    if not gestor_sistema.tpm_filename.exists():
        print(f"Archivo TPM de 20 nodos no encontrado. Generando uno nuevo...")
        gestor_sistema.generar_red(dimensiones=20, datos_discretos=True)

    # 🧠 Ejecutar estrategia
    analizador_qn = QNodes(gestor_sistema)
    sia_uno = analizador_qn.aplicar_estrategia(condiciones, alcance, mecanismo)

    print(sia_uno)