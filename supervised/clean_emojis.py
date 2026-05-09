"""Script para limpiar emojis de archivos"""
import re
from pathlib import Path

archivos = ['src/data_loader.py', 'src/Evaluation.py', 'src/utils.py', 'src/Models.py']
patron_emoji = r'[\U0001F300-\U0001F9FF]|[✅⚠️📊🔄🏆📈📁💾🎯👇📂]'

for archivo in archivos:
    path = Path(archivo)
    if not path.exists():
        print(f'[SKIP] {archivo}: no existe')
        continue
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Reemplazar emojis simples con texto
        replacements = {
            '✅': '[OK]',
            '⚠️': '[ADVERTENCIA]',
            '📊': '[DATOS]',
            '🔄': '[PROCESANDO]',
            '🏆': '[GANADOR]',
            '📈': '[RANKING]',
            '📁': '[ARCHIVOS]',
            '💾': '[GUARDANDO]',
            '🎯': '[OBJETIVO]',
            '👇': '',
            '📂': '[CARPETA]',
        }
        
        contenido_limpio = contenido
        for emoji, reemplazo in replacements.items():
            contenido_limpio = contenido_limpio.replace(emoji, reemplazo)
        
        # Limpiar emojis Unicode complejos
        contenido_limpio = re.sub(patron_emoji, '', contenido_limpio)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contenido_limpio)
        
        print(f'[OK] {archivo}: limpiado')
    except Exception as e:
        print(f'[ERROR] {archivo}: {e}')

print('\n[COMPLETO] Limpieza de emojis terminada')
