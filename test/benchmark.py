import time
import os
import gc
from typing import Dict, Any, Optional

# Importación opcional de pyarrow
try:
    import pyarrow  # type: ignore
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

class PerformanceBenchmark:
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time: float = 0.0
        self.test_name: Optional[str] = None
        
    def start_measurement(self, test_name: str) -> None:
        self.test_name = test_name
        self.start_time = time.time()
        gc.collect()
        print(f"🎯 Iniciando benchmark: {test_name}")

    def end_measurement(self, rows_processed=0, files_processed=0):
        """Termina medición y guarda resultados"""
        if self.test_name is None:
            print("❌ Error: No se ha iniciado ninguna medición")
            return
            
        end_time = time.time()
        duration = end_time - self.start_time
        
        self.results[self.test_name] = {
            'duration': duration,
            'rows_processed': rows_processed,
            'files_processed': files_processed,
            'rows_per_second': rows_processed / duration if duration > 0 else 0,
            'files_per_second': files_processed / duration if duration > 0 else 0
        }
        
        print(f"✅ {self.test_name} completado en {duration:.2f}s")
        print(f"   📊 {rows_processed:,} filas | {files_processed} archivos")
        print(f"   ⚡ Velocidad: {self.results[self.test_name]['rows_per_second']:,.0f} filas/seg")
        
    def compare_results(self):
        """Compara resultados entre diferentes métodos"""
        if len(self.results) < 2:
            print("⚠️ Necesitas al menos 2 resultados para comparar")
            return
            
        print("\n" + "="*80)
        print("📊 COMPARACIÓN DE RENDIMIENTO")
        print("="*80)
        
        methods = list(self.results.keys())
        old_result = self.results[methods[0]]
        new_result = self.results[methods[1]]
        
        speed_improvement = new_result['rows_per_second'] / old_result['rows_per_second']
        time_improvement = old_result['duration'] / new_result['duration']
        
        print(f"🚀 MEJORAS OBTENIDAS:")
        print(f"   ⚡ Velocidad: {speed_improvement:.1f}x más rápido")
        print(f"   ⏱️ Tiempo: {time_improvement:.1f}x menor duración")

def benchmark_system_info():
    """Muestra información del sistema"""
    cpu_count = os.cpu_count()
    
    print("🖥️ INFORMACIÓN DEL SISTEMA")
    print("="*50)
    print(f"CPUs: {cpu_count}")
    
    # Verificar dependencias
    optimizations = []
    if PYARROW_AVAILABLE:
        optimizations.append("✅ PyArrow (Parquet rápido)")
    else:
        optimizations.append("❌ PyArrow (no disponible)")
    
    print(f"\n🔧 Optimizaciones:")
    for opt in optimizations:
        print(f"   {opt}")
    print()

def run_quick_benchmark():
    """Ejecuta un benchmark rápido simulado"""
    print("🏃‍♂️ Ejecutando benchmark rápido...")
    
    benchmark = PerformanceBenchmark()
    
    # Simular procesamiento tradicional
    benchmark.start_measurement("Threading Original")
    time.sleep(2)  # Simular procesamiento lento
    benchmark.end_measurement(rows_processed=100000, files_processed=10)
    
    # Simular procesamiento optimizado
    benchmark.start_measurement("Multiprocessing + Parquet")
    time.sleep(0.3)  # Simular procesamiento rápido
    benchmark.end_measurement(rows_processed=100000, files_processed=10)
    
    benchmark.compare_results()

def main():
    """Función principal del benchmark"""
    print("🚀 BENCHMARK DE RENDIMIENTO - LANDSIDE PROCESSOR")
    print("=" * 80)
    
    benchmark_system_info()
    run_quick_benchmark()
    
    print("\n📋 PARA BENCHMARK REAL:")
    print("   1. Instala dependencias: python install_dependencies.py")
    print("   2. Ejecuta optimizado: python classes/LandSide.py")

if __name__ == "__main__":
    main()
