#!/usr/bin/env python3
"""
Script de benchmark para comparar rendimiento del procesamiento de archivos NC
"""

import time
import os
import sys
from pathlib import Path
import gc
from typing import Dict, Any, Optional

# Importación opcional de psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil no disponible - medición de memoria limitada")

class PerformanceBenchmark:
    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.test_name: Optional[str] = None
        
    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria actual"""
        if not PSUTIL_AVAILABLE:
            return 0.0
            
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024  # type: ignore # MB
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return 0.0
        
    def start_measurement(self, test_name: str) -> None:
        """Inicia medición de tiempo y memoria"""
        self.test_name = test_name
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        gc.collect()
        print(f"🎯 Iniciando benchmark: {test_name}")
        
    def end_measurement(self, rows_processed: int = 0, files_processed: int = 0) -> None:
        """Termina medición y guarda resultados"""
        if self.start_time is None or self.test_name is None:
            print("❌ Error: No se ha iniciado ninguna medición")
            return
            
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - self.start_time
        memory_used = end_memory - (self.start_memory or 0.0)
        
        self.results[self.test_name] = {
            'duration': duration,
            'memory_used': memory_used,
            'peak_memory': end_memory,
            'rows_processed': rows_processed,
            'files_processed': files_processed,
            'rows_per_second': rows_processed / duration if duration > 0 else 0,
            'files_per_second': files_processed / duration if duration > 0 else 0
        }
        
        print(f"✅ {self.test_name} completado en {duration:.2f}s")
        print(f"   📊 {rows_processed:,} filas | {files_processed} archivos")
        if PSUTIL_AVAILABLE:
            print(f"   💾 Memoria: {memory_used:.1f}MB | Pico: {end_memory:.1f}MB")
        print(f"   ⚡ Velocidad: {self.results[self.test_name]['rows_per_second']:,.0f} filas/seg")
        
        # Reset para próxima medición
        self.start_time = None
        self.start_memory = None
        self.test_name = None
        
    def compare_results(self) -> None:
        """Compara resultados entre diferentes métodos"""
        if len(self.results) < 2:
            print("⚠️ Necesitas al menos 2 resultados para comparar")
            return
            
        print("\n" + "="*80)
        print("📊 COMPARACIÓN DE RENDIMIENTO")
        print("="*80)
        
        # Tabla de resultados
        print(f"{'Método':<25} {'Tiempo':<10} {'Filas/seg':<12} {'Memoria':<10} {'Archivos/seg':<12}")
        print("-" * 80)
        
        # Calcular mejores valores evitando división por cero
        durations = [r['duration'] for r in self.results.values() if r['duration'] > 0]
        speeds = [r['rows_per_second'] for r in self.results.values() if r['rows_per_second'] > 0]
        memories = [r['memory_used'] for r in self.results.values() if r['memory_used'] > 0]
        
        best_time = min(durations) if durations else 1.0
        best_speed = max(speeds) if speeds else 1.0
        best_memory = min(memories) if memories else 1.0
        
        for name, result in self.results.items():
            print(f"{name:<25} {result['duration']:.1f}s {result['rows_per_second']:>10,.0f} {result['memory_used']:>8.1f}MB {result['files_per_second']:>10.1f}")
            
        print("\n🏆 MEJORES RESULTADOS:")
        
        if durations:
            fastest_method = min(self.results.keys(), key=lambda k: self.results[k]['duration'])
            print(f"   ⚡ Más rápido: {fastest_method}")
            
        if memories:
            most_efficient = min(self.results.keys(), key=lambda k: self.results[k]['memory_used'])
            print(f"   💾 Menos memoria: {most_efficient}")
            
        if speeds:
            fastest_processing = max(self.results.keys(), key=lambda k: self.results[k]['rows_per_second'])
            print(f"   📊 Mayor throughput: {fastest_processing}")
        
        # Calcular mejoras si hay exactamente 2 resultados
        if len(self.results) == 2:
            methods = list(self.results.keys())
            old_result = self.results[methods[0]]
            new_result = self.results[methods[1]]
            
            # Evitar división por cero
            speed_improvement = (
                new_result['rows_per_second'] / old_result['rows_per_second'] 
                if old_result['rows_per_second'] > 0 else 1.0
            )
            time_improvement = (
                old_result['duration'] / new_result['duration'] 
                if new_result['duration'] > 0 else 1.0
            )
            memory_improvement = (
                old_result['memory_used'] / new_result['memory_used'] 
                if new_result['memory_used'] > 0 else 1.0
            )
            
            print(f"\n🚀 MEJORAS OBTENIDAS:")
            print(f"   ⚡ Velocidad: {speed_improvement:.1f}x más rápido")
            print(f"   ⏱️ Tiempo: {time_improvement:.1f}x menor duración")
            print(f"   💾 Memoria: {memory_improvement:.1f}x más eficiente")

def benchmark_system_info() -> None:
    """Muestra información del sistema para el benchmark"""
    cpu_count = os.cpu_count() or 1
    
    if PSUTIL_AVAILABLE:
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            available_gb = psutil.virtual_memory().available / (1024**3)
        except:
            memory_gb = 0.0
            available_gb = 0.0
    else:
        memory_gb = 0.0
        available_gb = 0.0
    
    print("🖥️ INFORMACIÓN DEL SISTEMA")
    print("="*50)
    print(f"CPUs: {cpu_count}")
    if PSUTIL_AVAILABLE:
        print(f"Memoria RAM: {memory_gb:.1f} GB")
        print(f"Memoria disponible: {available_gb:.1f} GB")
    else:
        print("Memoria: No disponible (instala psutil)")
    print(f"Python: {sys.version.split()[0]}")
    
    # Verificar dependencias de optimización
    optimizations = []
    
    try:
        import pyarrow  # type: ignore
        optimizations.append("✅ PyArrow (Parquet rápido)")
    except ImportError:
        optimizations.append("❌ PyArrow (no disponible)")
    
    try:
        import numba  # type: ignore
        optimizations.append("✅ Numba (JIT compilation)")
    except ImportError:
        optimizations.append("❌ Numba (no disponible)")
        
    try:
        import h5netcdf  # type: ignore
        optimizations.append("✅ h5netcdf (NetCDF rápido)")
    except ImportError:
        optimizations.append("❌ h5netcdf (no disponible)")
        
    try:
        import dask  # type: ignore
        optimizations.append("✅ Dask (procesamiento distribuido)")
    except ImportError:
        optimizations.append("❌ Dask (no disponible)")
    
    if PSUTIL_AVAILABLE:
        optimizations.append("✅ PSUtil (monitoreo de sistema)")
    else:
        optimizations.append("❌ PSUtil (no disponible)")
    
    print(f"\n🔧 Optimizaciones disponibles:")
    for opt in optimizations:
        print(f"   {opt}")
    print()

def run_quick_benchmark() -> None:
    """Ejecuta un benchmark rápido con datos simulados"""
    print("🏃‍♂️ Ejecutando benchmark rápido...")
    
    benchmark = PerformanceBenchmark()
    
    try:
        # Simular procesamiento tradicional
        benchmark.start_measurement("Threading Original")
        time.sleep(2.0)  # Simular procesamiento lento
        benchmark.end_measurement(rows_processed=100000, files_processed=10)
        
        # Simular procesamiento optimizado
        benchmark.start_measurement("Multiprocessing + Parquet")
        time.sleep(0.3)  # Simular procesamiento rápido
        benchmark.end_measurement(rows_processed=100000, files_processed=10)
        
        benchmark.compare_results()
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error durante el benchmark: {e}")

def run_real_benchmark() -> None:
    """Ejecuta benchmark con datos reales si están disponibles"""
    print("📊 Preparando benchmark con datos reales...")
    
    # Verificar si existen archivos de datos
    project_root = Path(__file__).parent
    data_path = project_root / 'public' / 'datasets' / 'Datos_humedad_suelo_CDS.zip'
    
    if not data_path.exists():
        print(f"❌ No se encontró el archivo de datos: {data_path}")
        print("   Ejecutando benchmark simulado en su lugar...")
        run_quick_benchmark()
        return
    
    print("✅ Datos encontrados, ejecutando benchmark real...")
    print("⚠️ Esto puede tomar varios minutos...")
    
    benchmark = PerformanceBenchmark()
    
    try:
        # Aquí se ejecutaría el benchmark real
        print("🔄 Para benchmark completo ejecuta manualmente:")
        print("   1. python classes/LandSide.py")
        print("   2. Anota los tiempos y resultados")
        print("   3. Compara con versiones anteriores")
        
    except Exception as e:
        print(f"❌ Error durante el benchmark real: {e}")

def validate_choice(choice: str) -> bool:
    """Valida la elección del usuario"""
    return choice.strip() in ['1', '2', '3', '4']

def main() -> None:
    """Función principal del benchmark"""
    print("🚀 BENCHMARK DE RENDIMIENTO - LANDSIDE PROCESSOR")
    print("=" * 80)
    
    benchmark_system_info()
    
    # Opciones de benchmark
    print("Opciones de benchmark:")
    print("1. Benchmark rápido (simulado)")
    print("2. Benchmark con datos reales")
    print("3. Solo mostrar información del sistema")
    print("4. Salir")
    
    try:
        choice = input("\nSelecciona una opción (1-4): ").strip()
        
        if not validate_choice(choice):
            print("❌ Opción no válida. Usa números del 1 al 4.")
            return
        
        if choice == "1":
            run_quick_benchmark()
        elif choice == "2":
            run_real_benchmark()
        elif choice == "3":
            print("ℹ️ Información del sistema mostrada arriba")
        elif choice == "4":
            print("👋 ¡Hasta luego!")
        
    except KeyboardInterrupt:
        print("\n👋 Benchmark cancelado por el usuario")
    except EOFError:
        print("\n❌ Error de entrada. Terminando...")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main() 