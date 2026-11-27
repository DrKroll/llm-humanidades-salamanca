# Tus textos

Copia aquí tus propios archivos `.txt` (codificación UTF-8) o `.xml` en TEI, por ejemplo
cartas, poemas, actas, crónicas, transcripciones...

Después cambia en `config.yaml`:

```yaml
data:
  folders: ["data/mis_textos"]
```

Consejos:
- Cuanto más texto, mejor: con menos de ~100 000 caracteres el GPT diminuto
  apenas aprende. Para el ajuste fino (cuaderno 03) bastan ~50 000 caracteres.
- Comprueba que tienes derecho a usar los textos (derechos de autor, datos personales).
- En Colab, los archivos subidos se borran al cerrar la sesión.
