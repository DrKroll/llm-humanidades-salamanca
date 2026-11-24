"""Cuadernos del día 1: introducción a Python para humanistas.

Tres cuadernos (P1–P3); cada uno junta dos módulos antiguos como «Parte A» y «Parte B»:
    P1 = Colab + cadenas · P2 = listas/diccionarios + bucles/funciones · P3 = ficheros + análisis
"""

from nbtools import SETUP_CODE_DATA_ONLY, code, md, part, write_notebook

DIA1 = "notebooks/dia1_python"


def solucion(text: str):
    """Celda de markdown con la solución plegada."""
    return md(f"<details><summary>💡 Ver una solución</summary>\n\n```python\n{text.strip()}\n```\n</details>")


def main():
    # ------------------------------------------------------------ P1
    P1 = [
        md("""
# P1 · Primeros pasos en Colab

**Objetivo:** aprender a usar un cuaderno (*notebook*): celdas, ejecutar código, `print`, comentarios y errores.

Un cuaderno mezcla dos tipos de celdas:
- **Celdas de texto** (como esta): explicaciones.
- **Celdas de código**: instrucciones para el ordenador. Se ejecutan con ▶ o con **Mayús + Intro**.

> 💾 **Guarda tu propia copia:** menú *Archivo → Guardar una copia en Drive*. Así tus cambios no se pierden.
"""),
        md("## 1. Tu primera instrucción\nEjecuta la celda. `print` muestra algo en pantalla."),
        code('print("¡Hola, Salamanca!")'),
        md("## 2. Comentarios\nTodo lo que va detrás de `#` es un **comentario**: el ordenador lo ignora. Sirve para explicar."),
        code("""
# Esto es un comentario
print("Esto sí se ejecuta")   # y esto también es un comentario
"""),
        md("## 3. Python como calculadora"),
        code("""
print(2 + 3)
print(1605 - 1547)     # ¿qué edad tenía Cervantes al publicar la primera parte del Quijote?
print(10 / 4)
print(2 ** 10)         # potencia
"""),
        md("## 4. Variables: poner nombre a las cosas\nUna **variable** es una etiqueta para un valor. El `=` significa «guarda esto aquí»."),
        code("""
autor = "Miguel de Cervantes"
nacimiento = 1547
muerte = 1616
print(autor, "vivió", muerte - nacimiento, "años")
"""),
        md("""
## 5. Los errores son normales 🙂
Ejecuta la celda: verás un error en rojo. Léelo desde **abajo**: la última línea dice qué ha pasado.
Aquí usamos una variable que no existe (`autora` en lugar de `autor`).
"""),
        code("""
try:
    print(autora)
except NameError as error:
    print("Error:", error)   # (sin el try/except, Colab mostraría el error en rojo)
"""),
        md("""
Errores frecuentes:
- `NameError`: nombre mal escrito o celda anterior sin ejecutar.
- `SyntaxError`: falta una comilla, un paréntesis o dos puntos.
- `IndentationError`: los espacios al principio de la línea no cuadran.

## 🖐️ Tu turno
1. Crea una variable `mi_nombre` con tu nombre e imprime «Me llamo ...».
2. Calcula cuántos años han pasado desde 1605 hasta hoy.
"""),
        code("""
# Escribe aquí tu código

"""),
        solucion("""
mi_nombre = "Teresa"
print("Me llamo", mi_nombre)
print(2026 - 1605)
"""),
    ]

    # ------------------------------------------------------------ P2
    P2 = [
        md("""
# P2 · Cadenas de texto (*strings*)

**Objetivo:** manipular texto con Python: longitud, trozos, mayúsculas, dividir, reemplazar...

Un texto entre comillas es una **cadena** (`str`). Para las humanidades, ¡es el tipo de dato más importante!
"""),
        code("""
frase = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme"
print(frase)
print(type(frase))
print(len(frase), "caracteres")
"""),
        md("## 1. Unir cadenas"),
        code("""
nombre = "Sancho"
apellido = "Panza"
print(nombre + " " + apellido)
print(nombre * 3)
"""),
        md("## 2. Trozos (*slicing*)\nCada carácter tiene una posición (**índice**). ¡Python empieza a contar en 0!"),
        code("""
palabra = "Quijote"
print(palabra[0])      # primera letra
print(palabra[-1])     # última letra
print(palabra[0:4])    # de la posición 0 a la 3
print(frase[:22])      # desde el principio hasta la 21
"""),
        md("## 3. Métodos útiles"),
        code("""
print(frase.upper())
print(frase.lower())
print(frase.replace("Mancha", "Castilla"))
print(frase.count("de"))
print(frase.startswith("En"))
print("   texto con espacios   ".strip())
"""),
        md("## 4. Dividir y unir: `split` y `join`"),
        code("""
palabras = frase.split()          # divide por espacios
print(palabras)
print(len(palabras), "palabras")
print(" | ".join(palabras))
"""),
        md("## 5. f-strings: insertar valores en un texto"),
        code("""
obra = "Lazarillo de Tormes"
anio = 1554
print(f"El {obra} se publicó en {anio}, hace {2026 - anio} años.")
"""),
        md("""
## 6. Acentos, eñes y Unicode
Python 3 trabaja con **Unicode**: á, ñ, ü, ¿, ¡ no son un problema. Pero cuidado:
una «á» puede escribirse como un carácter o como «a» + acento combinado.
"""),
        code("""
import unicodedata
a1 = "á"
a2 = "a\\u0301"          # «a» + acento combinado
print(a1, a2, a1 == a2)
print(unicodedata.normalize("NFC", a1) == unicodedata.normalize("NFC", a2))
print("¿Qué es España?".lower())
"""),
        md("""
## 🖐️ Tu turno
Con la cita `cita = "Yo sé quién soy"`:
1. ¿Cuántos caracteres tiene?
2. Imprímela en mayúsculas.
3. Sustituye «soy» por «eres».
4. ¿Cuántas palabras tiene?
"""),
        code("""
cita = "Yo sé quién soy"
# Escribe aquí tu código

"""),
        solucion("""
print(len(cita))
print(cita.upper())
print(cita.replace("soy", "eres"))
print(len(cita.split()))
"""),
    ]

    # ------------------------------------------------------------ P3
    P3 = [
        md("""
# P3 · Listas y diccionarios

**Objetivo:** guardar muchos datos juntos: listas (secuencias) y diccionarios (fichas clave → valor).
"""),
        md("## 1. Listas"),
        code("""
personajes = ["don Quijote", "Sancho Panza", "Dulcinea", "el cura", "el barbero"]
print(personajes)
print(len(personajes))
print(personajes[0], "y", personajes[1])
print(personajes[-1])
"""),
        code("""
personajes.append("Rocinante")       # añadir al final
print(personajes)
print(sorted(personajes))            # ordenar alfabéticamente
print("Dulcinea" in personajes)      # ¿está en la lista?
"""),
        md("## 2. Diccionarios: como una ficha bibliográfica\nUn **diccionario** asocia **claves** con **valores**."),
        code("""
ficha = {
    "título": "El ingenioso hidalgo don Quijote de la Mancha",
    "autor": "Miguel de Cervantes",
    "año": 1605,
    "género": "novela",
}
print(ficha["autor"])
ficha["lugar"] = "Madrid"       # añadir una entrada
print(ficha)
"""),
        code("""
obras = {
    "Cervantes": ["Don Quijote", "Novelas ejemplares", "La Galatea"],
    "Lope de Vega": ["Fuenteovejuna", "El perro del hortelano"],
    "Calderón": ["La vida es sueño"],
}
for autor, lista in obras.items():
    print(f"{autor}: {len(lista)} obras -> {', '.join(lista)}")
"""),
        md("## 3. Contar con un diccionario\nLa idea básica de la lingüística de corpus: contar cuántas veces aparece cada palabra."),
        code("""
texto = "el caballero y el escudero y el caballo del caballero"
conteo = {}
for palabra in texto.split():
    if palabra in conteo:
        conteo[palabra] += 1
    else:
        conteo[palabra] = 1
print(conteo)
"""),
        md("""
## 🖐️ Tu turno
1. Crea una lista con tres obras que te gusten y añade una cuarta.
2. Crea una ficha (diccionario) para el *Lazarillo de Tormes* (autor: «anónimo», año: 1554).
3. Cuenta las palabras de `"yo por bien tengo que cosas tan señaladas vengan a noticia de muchos y no se entierren"`.
"""),
        code("""
# Escribe aquí tu código

"""),
        solucion("""
mis_obras = ["Cien años de soledad", "La Regenta", "Niebla"]
mis_obras.append("Fortunata y Jacinta")
lazarillo = {"título": "Lazarillo de Tormes", "autor": "anónimo", "año": 1554}
conteo = {}
for p in "yo por bien tengo que cosas tan señaladas vengan a noticia de muchos y no se entierren".split():
    conteo[p] = conteo.get(p, 0) + 1
print(conteo)
"""),
    ]

    # ------------------------------------------------------------ P4
    P4 = [
        md("""
# P4 · Bucles, condiciones y funciones

**Objetivo:** repetir acciones (`for`, `while`), tomar decisiones (`if`) y empaquetar código en **funciones**.

> ⚠️ En Python, la **sangría** (los espacios al principio de la línea) es obligatoria: indica qué
> líneas están «dentro» de un bucle o una condición.
"""),
        md("## 1. `if` / `elif` / `else`"),
        code("""
anio = 1554
if anio < 1500:
    print("siglo XV o antes")
elif anio < 1600:
    print("siglo XVI")
else:
    print("siglo XVII o después")
"""),
        md("## 2. Bucle `for`: repetir para cada elemento"),
        code("""
versos = ["Yo sé quién soy", "y sé que puedo ser", "no solo los que he dicho"]
for verso in versos:
    print(verso.upper(), "-", len(verso.split()), "palabras")
"""),
        code("""
palabras = "el que lee mucho y anda mucho ve mucho y sabe mucho".split()
largas = []
for p in palabras:
    if len(p) > 3:
        largas.append(p)
print(largas)
"""),
        md("## 3. Bucle `while`: repetir mientras se cumpla algo"),
        code("""
cuenta = 3
while cuenta > 0:
    print("Faltan", cuenta)
    cuenta -= 1
print("¡Al ataque de los molinos!")
"""),
        md("## 4. (Opcional) Comprensiones de listas\nUna forma corta de crear listas:"),
        code("""
largas = [p for p in palabras if len(p) > 3]
longitudes = [len(p) for p in palabras]
print(largas, longitudes)
"""),
        md("## 5. Funciones: dar nombre a una receta\n`def` define una función; `return` devuelve el resultado."),
        code("""
def contar_palabras(texto):
    \"\"\"Devuelve el número de palabras de un texto.\"\"\"
    return len(texto.split())

def es_largo(texto, limite=10):
    return contar_palabras(texto) > limite

print(contar_palabras("En un lugar de la Mancha"))
print(es_largo("En un lugar de la Mancha"))
print(es_largo("En un lugar de la Mancha", limite=3))
"""),
        md("""
## 🖐️ Tu turno
1. Escribe una función `iniciales(nombre)` que devuelva las iniciales: `iniciales("Miguel de Cervantes Saavedra")` → `"MdCS"`.
2. Escribe una función `palabras_con(texto, letra)` que devuelva las palabras que empiezan por `letra`.
"""),
        code("""
# Escribe aquí tu código

"""),
        solucion("""
def iniciales(nombre):
    return "".join(p[0] for p in nombre.split())

def palabras_con(texto, letra):
    return [p for p in texto.split() if p.lower().startswith(letra.lower())]

print(iniciales("Miguel de Cervantes Saavedra"))
print(palabras_con("Sancho se sentó sobre su asno", "s"))
"""),
    ]

    # ------------------------------------------------------------ P5
    P5 = [
        md("""
# P5 · Ficheros: leer el *Quijote* entero

**Objetivo:** leer un archivo de texto, entender la codificación (UTF-8) y descargar un libro de Project Gutenberg.

Usaremos *Don Quijote* (Project Gutenberg, libro n.º 2000, dominio público). Si la descarga falla
(Gutenberg a veces no responde), usamos la **copia local** que hay en el repositorio.
"""),
        code(SETUP_CODE_DATA_ONLY),
        md("## 1. Descargar de Project Gutenberg (con plan B)"),
        code("""
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
COPIA_LOCAL = Path("data/gutenberg/pg2000_don_quijote.txt")

try:
    with urllib.request.urlopen(URL, timeout=15) as respuesta:
        texto = respuesta.read().decode("utf-8")
    if "Quijote" not in texto:
        raise ValueError("La página descargada no es el libro")
    print("Descargado de Gutenberg ✅")
except Exception as e:
    print("No se pudo descargar (", e, ") → usamos la copia local")
    texto = COPIA_LOCAL.read_text(encoding="utf-8")

print(f"{len(texto):,} caracteres".replace(",", "."))
"""),
        md("## 2. Leer un archivo con `open`\nAsí se lee cualquier archivo `.txt`. `encoding=\"utf-8\"` evita problemas con los acentos."),
        code("""
with open(COPIA_LOCAL, encoding="utf-8") as f:
    texto_local = f.read()
print(texto_local[:300])
"""),
        md("## 3. Quitar la cabecera y el pie de Gutenberg\nLos libros de Gutenberg llevan avisos en inglés al principio y al final. Los buscamos con `find`."),
        code("""
inicio = texto.find("*** START OF")
inicio = texto.find("\\n", inicio) + 1          # saltamos esa línea
fin = texto.find("*** END OF")
libro = texto[inicio:fin].strip() if fin != -1 else texto[inicio:].strip()
print(libro[:500])
print("...")
print(libro[-300:])
"""),
        md("## 4. Dividir en partes y capítulos"),
        code("""
corte = libro.find("Segunda parte del ingenioso caballero", 20000)
parte1, parte2 = libro[:corte], libro[corte:]
print(f"Parte I: {len(parte1.split()):,} palabras".replace(",", "."))
print(f"Parte II: {len(parte2.split()):,} palabras".replace(",", "."))
print("Capítulos (aprox.):", libro.count("Capítulo"))
"""),
        md("## 5. Guardar un archivo"),
        code("""
Path("outputs").mkdir(exist_ok=True)
with open("outputs/quijote_limpio.txt", "w", encoding="utf-8") as f:
    f.write(libro)
print("Guardado en outputs/quijote_limpio.txt")
"""),
        md("""
## 🖐️ Tu turno
1. Lee el archivo `data/sample/lazarillo_de_tormes.txt` y cuenta sus palabras.
2. ¿Cuántas veces aparece «Sancho» en la Parte I y en la Parte II?
"""),
        code("""
# Escribe aquí tu código

"""),
        solucion("""
laz = Path("data/sample/lazarillo_de_tormes.txt").read_text(encoding="utf-8")
print(len(laz.split()))
print(parte1.count("Sancho"), parte2.count("Sancho"))
"""),
    ]

    # ------------------------------------------------------------ P6
    P6 = [
        md("""
# P6 · Análisis sencillo de texto: frecuencias, KWIC y bigramas

**Objetivo:** contar palabras, buscar en contexto, hacer gráficos... y construir un primer «modelo de lenguaje» con bigramas.
"""),
        code(SETUP_CODE_DATA_ONLY),
        md("## 1. Cargar y limpiar\n> 🔁 ¿Se ha desconectado Colab durante la pausa? Ejecuta primero la celda ⚙️ del principio del cuaderno. Esta celda vuelve a cargar el texto."),
        code("""
import re
from pathlib import Path
texto = Path("data/gutenberg/pg2000_don_quijote.txt").read_text(encoding="utf-8")
texto = texto[texto.find("\\n", texto.find("*** START OF")) + 1 : texto.find("*** END OF")]

def tokenizar(texto):
    \"\"\"Minúsculas y solo letras (incluye acentos y ñ).\"\"\"
    return re.findall(r"[a-záéíóúüñ]+", texto.lower())

palabras = tokenizar(texto)
print(len(palabras), "palabras;", len(set(palabras)), "distintas")
print(palabras[:20])
"""),
        md("## 2. Frecuencias con `Counter`"),
        code("""
from collections import Counter
frecuencias = Counter(palabras)
frecuencias.most_common(15)
"""),
        md("## 3. Quitar palabras vacías (*stopwords*)"),
        code("""
VACIAS = set(\"\"\"de la que el en y a los se del las un por con no una su para es al lo como
más mas pero sus le ya o este sí si porque esta entre cuando muy sin sobre también me hasta hay donde
quien desde todo nos ni ese eso yo él mi tan así había era fue dijo había lo le les me te tu vuestra
merced os ha he está estaba ser sino pues aunque cual otro otra esto qué todos todas\"\"\".split())
contenido = Counter(p for p in palabras if p not in VACIAS and len(p) > 2)
contenido.most_common(20)
"""),
        md("## 4. Gráfico de barras"),
        code("""
import matplotlib.pyplot as plt
top = contenido.most_common(20)
plt.figure(figsize=(10, 4))
plt.bar([p for p, n in top], [n for p, n in top], color="#8c2d19")
plt.xticks(rotation=60, ha="right")
plt.title("Palabras más frecuentes del Quijote (sin palabras vacías)")
plt.tight_layout()
plt.show()
"""),
        md("## 5. Concordancia (KWIC: *keyword in context*)"),
        code("""
def kwic(texto, palabra, ancho=40, maximo=10):
    plano = texto.replace("\\n", " ")
    for i, m in enumerate(re.finditer(rf"\\b{palabra}\\b", plano, flags=re.IGNORECASE)):
        if i >= maximo:
            break
        izq = plano[max(0, m.start() - ancho): m.start()]
        der = plano[m.end(): m.end() + ancho]
        print(f"{izq:>{ancho}} [{m.group()}] {der}")

kwic(texto, "molinos")
"""),
        md("## 6. Comparar la Parte I y la Parte II"),
        code("""
corte = texto.find("Segunda parte del ingenioso caballero", 20000)
p1, p2 = Counter(tokenizar(texto[:corte])), Counter(tokenizar(texto[corte:]))
n1, n2 = sum(p1.values()), sum(p2.values())
for w in ["dulcinea", "sancho", "ínsula", "duque", "encantadores", "rocinante"]:
    print(f"{w:>12}: Parte I {1000 * p1[w] / n1:5.2f} ‰   Parte II {1000 * p2[w] / n2:5.2f} ‰")
"""),
        md("""
## 7. Del recuento al modelo de lenguaje: bigramas
Un **modelo de lenguaje** predice la siguiente palabra. El más simple: contar qué palabra sigue a cada palabra
(**bigramas**) y elegir al azar según esas frecuencias. ¡Esto es lo que hacen los LLM, pero con mucho más contexto!
"""),
        code("""
import random
from collections import defaultdict

siguientes = defaultdict(Counter)
for a, b in zip(palabras, palabras[1:]):
    siguientes[a][b] += 1

print("Después de 'don':", siguientes["don"].most_common(5))
print("Después de 'sancho':", siguientes["sancho"].most_common(5))
"""),
        code("""
def generar(inicio="don", n=25, semilla=None):
    random.seed(semilla)
    palabra, salida = inicio, [inicio]
    for _ in range(n):
        opciones = siguientes.get(palabra)
        if not opciones:
            break
        palabra = random.choices(list(opciones), weights=opciones.values())[0]
        salida.append(palabra)
    return " ".join(salida)

for s in range(3):
    print(generar("sancho", semilla=s))
"""),
        md("""
## 🖐️ Tu turno
1. Busca en contexto (KWIC) «Dulcinea», «libros» o «gigantes».
2. Haz el gráfico de las palabras más frecuentes de la **Parte II**.
3. Genera texto con bigramas empezando por otras palabras. ¿Por qué pierde el sentido tan rápido?
4. **Reto:** haz un modelo de **trigramas** (las dos palabras anteriores predicen la siguiente).
   ¿Mejora? ¿Copia frases enteras del original? 👉 Esto lo retomamos en la clase sobre LLM.
"""),
        code("""
# Escribe aquí tu código

"""),
        solucion("""
trigramas = defaultdict(Counter)
for a, b, c in zip(palabras, palabras[1:], palabras[2:]):
    trigramas[(a, b)][c] += 1

def generar3(a="dijo", b="don", n=25, semilla=0):
    random.seed(semilla)
    salida = [a, b]
    for _ in range(n):
        opciones = trigramas.get((a, b))
        if not opciones:
            break
        c = random.choices(list(opciones), weights=opciones.values())[0]
        salida.append(c)
        a, b = b, c
    return " ".join(salida)

print(generar3())
"""),
    ]

    # ------------------------------------------------------------ cuadernos combinados
    write_notebook(f"{DIA1}/P1_colab_y_textos.ipynb", [
        md("""
# P1 · Colab y textos

⏱️ **≈ 75 min** (lunes 10:00–11:15)

| Parte | Contenido | Tiempo |
|---|---|---|
| A | Primeros pasos en Colab: celdas, `print`, variables, errores | ≈ 30 min |
| B | Cadenas de texto: longitud, trozos, mayúsculas, dividir, reemplazar, acentos | ≈ 45 min |
"""),
        *part("Parte A", P1),
        *part("Parte B", P2),
    ])

    write_notebook(f"{DIA1}/P2_listas_bucles_funciones.ipynb", [
        md("""
# P2 · Listas, bucles y funciones

⏱️ **≈ 90 min** (lunes 11:45–13:15)

| Parte | Contenido | Tiempo |
|---|---|---|
| A | Listas y diccionarios: guardar muchos datos juntos | ≈ 40 min |
| B | Bucles, condiciones y funciones: repetir, decidir y reutilizar código | ≈ 50 min |
"""),
        *part("Parte A", P3),
        *part("Parte B", P4),
    ])

    write_notebook(f"{DIA1}/P3_leer_y_analizar_el_quijote.ipynb", [
        md("""
# P3 · Leer y analizar el *Quijote*

⏱️ **≈ 105 min** (lunes 13:15–14:00 la parte A, 15:00–16:00 la parte B)

| Parte | Contenido | Tiempo |
|---|---|---|
| A | Ficheros: descargar el *Quijote* de Gutenberg (con copia local), leer, limpiar, guardar | ≈ 45 min |
| B | Análisis: frecuencias, gráficos, palabra clave en contexto (KWIC) y un primer «modelo de lenguaje» con bigramas | ≈ 60 min |
"""),
        code(SETUP_CODE_DATA_ONLY),
        *part("Parte A", P5, drop_sources=(SETUP_CODE_DATA_ONLY,)),
        *part("Parte B", P6, drop_sources=(SETUP_CODE_DATA_ONLY,)),
    ])


if __name__ == "__main__":
    main()
