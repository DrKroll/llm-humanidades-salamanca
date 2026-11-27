# Fuentes de los textos

Todos los textos son de **dominio público** (autores fallecidos hace siglos) y proceden de
[Project Gutenberg](https://www.gutenberg.org). Se descargaron el 23/11/2026 desde el espejo
oficial `gutenberg.pglaf.org`.

| Archivo | Obra | Gutenberg | Tratamiento |
|---|---|---|---|
| `sample/quijote_primera_parte.txt` | Miguel de Cervantes, *El ingenioso hidalgo don Quijote de la Mancha*, Primera parte (1605) | [n.º 2000](https://www.gutenberg.org/ebooks/2000) | Sin cabecera/pie de Gutenberg; solo la Primera parte |
| `sample/lazarillo_de_tormes.txt` | Anónimo, *La vida de Lazarillo de Tormes y de sus fortunas y adversidades* (Burgos, 1554) | [n.º 320](https://www.gutenberg.org/ebooks/320) | Sin cabecera/pie de Gutenberg |
| `gutenberg/pg2000_don_quijote.txt` | *Don Quijote* completo (Partes I y II) | [n.º 2000](https://www.gutenberg.org/ebooks/2000) | Archivo tal como lo ofrece Gutenberg (con marcas START/END), para el cuaderno P3 (parte A) |

**Nota sobre la licencia de Project Gutenberg:** los textos son de dominio público. La marca
«Project Gutenberg» está protegida; por eso los archivos de `sample/` se distribuyen sin la
cabecera ni el pie de Gutenberg, como permite su licencia. Más información:
<https://www.gutenberg.org/policy/license.html>.

`qa/ejemplo_preguntas_respuestas.jsonl`: 30 pares pregunta-respuesta redactados para este taller (licencia MIT, como el resto del repositorio).

---

# Teatro del Siglo de Oro (`calderon/tei/` y `teatro/`)

**688 obras de 26 autores**, sin duplicados, reunidas el 27/11/2026 de los repositorios de Simon Kroll:

| Repositorio de origen | Destino en este repo | Obras | Formato |
|---|---|---:|---|
| `DrKroll/Calderon_TEI` | `calderon/tei/` | 150 | TEI-XML (copiado sin cambios) |
| `DrKroll/Calderon_TEI` | `teatro/calderon-de-la-barca/` | 8 | texto con personajes (convertido del formato txt2tei) |
| `DrKroll/corpus` | `teatro/<autor>/` | 514 | texto con personajes (txt2tei) o solo texto de los parlamentos |
| `DrKroll/corpus` | `calderon/tei/` | 2 | TEI-XML |
| selección anterior de este repo (ver tabla de abajo) | `calderon/tei/` | 14 | TEI-XML |

También se revisaron `DrKroll/Computational-stylometry` (solo copias o fragmentos de obras ya incluidas),
`DrKroll/machine-learning-rhythm-authorship` (sin textos) y `DrKroll/Curso-UAM` (vacío): no aportan obras nuevas.

**Licencia:** las obras son de los siglos XVI y XVII y están en el **dominio público**. Así lo ha decidido
Simon Kroll (27/11/2026) para todo el teatro áureo de sus repositorios. Las ediciones digitales de las que proceden se
citan en la cabecera TEI de cada obra o en la columna `edicion` de **`teatro_indice.csv`**, que también da para cada
obra el repositorio y la ruta de origen. Dos archivos llevan un aviso de copyright moderno de la edición digital:
*El divino Narciso* y su loa (Sor Juana; «© UNAM 2020»).

**Sin duplicados:** cuando una obra estaba en varios repositorios, formatos o ediciones, se conserva **una sola versión**:
la de este repositorio si ya estaba; si no, la TEI; si no, la versión con personajes, y entre ellas la más completa.
Las copias se detectaron por autor y título normalizado y por comparación del texto (copias con otro nombre de archivo,
copias parciales y fragmentos). Se conservan por separado obras distintas con el mismo título (auto y comedia de
*El pintor de su deshonra* y *El jardín de Falerina*; primera y segunda parte de *La hija del aire*, *Don Juan de Castro*,
*El príncipe perfecto*; las dos versiones de *El divino Orfeo*, 1634 y 1663, y de *La vida es sueño* auto).
No se incluyen los romances y la poesía de Góngora ni dos archivos ilegibles (`Gongora/all.txt`, `lope_padrino_desposado.txt`).

Formatos en `teatro/`: «texto con personajes» = cada parlamento con el nombre del personaje (`NOMBRE:`), versos con sangría,
acotaciones entre corchetes y `JORNADA n`; «solo texto» = solo el texto de los parlamentos (versiones preparadas para
estilometría), cuando no había otra versión.

## Las 14 obras de la primera selección del curso

| Archivo | Obra | Edición / fuente (según la cabecera TEI) | Licencia (cabecera TEI) |
|---|---|---|---|
| `auto_el-gran-teatro-del-mundo.xml` | *El gran teatro del mundo* (auto) | Pando y Mier, *Autos sacramentales…*, Madrid, 1717; base TESO (autorizada por E. Rodríguez Cuadros); TEI DraCor | CC0 1.0 |
| `auto_la-humildad-coronada-de-las-plantas.xml` | *La humildad coronada* (auto) | Pando y Mier, 1717; TESO; TEI DraCor | CC0 1.0 |
| `auto_psiquis-y-cupido.xml` | *Psiquis y Cupido, que escribió para esta villa de Madrid* (auto) | Pando y Mier, 1717; TESO; TEI DraCor | CC0 1.0 |
| `EMOTHE0020_LaVidaEsSueno.xml` | *La vida es sueño* | Texto base: F. Antonucci (ed.), Barcelona, Crítica, 2008; ARTELOPE/EMOTHE | ARTELOPE/CTCE: «libres de derechos de lectura y de reproducción», citar la procedencia (\*) |
| `EMOTHE0312_LaDamaDuende.xml` | *La dama duende* | L. Iglesias Feijoo; Canon 60 – TC/12; digitalización BVMC; ARTELOPE | ARTELOPE/CTCE (\*) |
| `EMOTHE0322_CasaConDosPuertasMalaEsDeGuardar.xml` | *Casa con dos puertas mala es de guardar* | L. Iglesias Feijoo; Canon 60; BVMC 2014; ARTELOPE | ARTELOPE/CTCE (\*) |
| `EMOTHE0321_ElMedicoDeSuHonra.xml` | *El médico de su honra* | S. Fernández Mosquera; Canon 60; BVMC 2014; ARTELOPE | ARTELOPE/CTCE (\*) |
| `EMOTHE0357_ASecretoAgravioSecretaVenganza.xml` | *A secreto agravio, secreta venganza* | S. Fernández Mosquera; Canon 60; BVMC 2014; ARTELOPE | ARTELOPE/CTCE (\*) |
| `EMOTHE0319_ElAlcaldeDeZalamea.xml` | *El alcalde de Zalamea* | I. Arellano; Canon 60; BVMC 2014; ARTELOPE | ARTELOPE/CTCE (\*) |
| `EMOTHE0386_LosCabellosDeAbsalon.xml` | *Los cabellos de Absalón* | E. Rodríguez Cuadros; Canon 60; BVMC 2014; ARTELOPE | ARTELOPE/CTCE (\*) |
| `EMOTHE0537_ElPrincipeConstante.xml` | *El príncipe constante* | L. Iglesias Feijoo; Canon 60; BVMC; EMOTHE | **CC BY-NC-ND 4.0** (EMOTHE, Universitat de València) |
| `kroll_los-empenos-de-un-acaso.xml` | *Los empeños de un acaso* | Edición de **Simon Kroll**, Kassel, Reichenberger, 2022; TEI generado por F. Sanz-Lázaro | CC BY (cabecera TEI) |
| `kroll_eco-y-narciso.xml` | *Eco y Narciso* | «Edición digital a partir de la edición de Margaret Greer, Fernando Rodríguez-Gallego y **Simon Kroll**, Kassel, Reichenberger, en prensa»; TEI generado automáticamente por F. Sanz-Lázaro (txt2tei, DraCor, 2023) | CC BY (cabecera TEI: `<ab>CC BY</ab>`, enlace `https://creativecommons.org/publicdomain/BY/3.0/`) (\*\*) |
| `kroll_el-secreto-a-voces.xml` | *El secreto a voces* | «Edición digital a partir de la edición de Wolfram Aichinger, **Simon Kroll** y Fernando Rodríguez-Gallego, Kassel, Reichenberger, 2015»; TEI generado automáticamente por F. Sanz-Lázaro (txt2tei, DraCor, 2023) | CC BY (cabecera TEI, mismo enlace) (\*\*) |

(\*) Cita obligatoria: «Oleza, Joan et al. *Base de Datos y Argumentos del teatro de Lope de Vega. ARTELOPE*.
Publicación en web: http://artelope.uv.es» — para EMOTHE: «Oleza, Joan et al. *EMOTHE: The Classics of Early
Modern European Theatre*. Valencia: ARTELOPE/EMOTHE Universitat de València, 2010-. https://emothe.uv.es/biblioteca/».

(\*\*) Ediciones en coautoría; incluidas con autorización de Simon Kroll (coeditor). La cabecera indica «CC BY»
sin versión clara (el enlace `publicdomain/BY/3.0` no es una URL válida de Creative Commons; probablemente CC BY 3.0).
*Eco y Narciso* procede de una edición **en prensa** (Reichenberger). Citar siempre la edición impresa y a los coeditores.

**Uso en el taller:** solo con fines docentes y sin ánimo de lucro. *El príncipe constante* (CC BY-NC-ND) se
distribuye sin cambios; los adaptadores entrenados no deben publicarse fuera de este curso.

(Las demás obras de Calderón proceden de las ediciones citadas en sus cabeceras TEI: GRISO/Universidad de
Navarra–Reichenberger, Biblioteca Castro, Iberoamericana/Vervuert, CANON60, DraCor, ediciones de Simon Kroll y otras.)
