# Aplicacion TFG - Analisis defensivo con tracking y eventing

Aplicacion web en Streamlit para analizar la estructura defensiva de un equipo a partir de datos BePro:
tracking, eventing, secuencias ofensivas rivales, tipologias, desorganizacion defensiva, peligrosidad,
momentum e informe final.

La app esta preparada como plataforma general: se inicia sesion, se elige un equipo registrado y despues
se abre el espacio del club con sus partidos analizados. En esta version el equipo cargado es CD Subiza.

## Estructura del repositorio

```text
app/                         Interfaz Streamlit
src/tfg_analysis/            Pipeline de analisis y modulos reutilizables
scripts/                     Scripts de generacion, descarga y automatizacion
assets/team_logos/           Escudos usados por la app
eventing_partidos/           Eventing descargado de BePro
sequences_partidos/          Secuencias descargadas de BePro
tracking_partidos/           Carpeta local para tracking crudo
outputs/app_data/            Datos ligeros que lee la app
outputs/global_clusters/     Modelo y tablas de clustering global
.github/workflows/           Automatizacion con GitHub Actions
```

Los CSV de tracking crudo no se suben por defecto porque cada partido puede superar los 300 MB y GitHub
bloquea archivos de mas de 100 MB. La carpeta `tracking_partidos/` se mantiene para ejecucion local o para
la descarga temporal en GitHub Actions. Los resultados ligeros de la app si quedan versionados en
`outputs/app_data/`.

## Ejecutar la app en local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
streamlit run app/streamlit_app.py
```

Credenciales demo:

```text
Usuario: entrenador
Contrasena: subiza2026
Codigo CD Subiza: subiza2026
```

## Regenerar analisis ya descargados

Si tienes tracking, eventing y sequences locales:

```powershell
$env:PYTHONPATH="src"
python scripts/build_app_data.py --team-id 12987 --force-analysis
```

Para un unico partido:

```powershell
python scripts/build_app_data.py --team-id 12987 --match-id 199542 --force-analysis
```

## Automatizacion BePro

El script principal es:

```powershell
python scripts/auto_update_bepro.py --team-id 12987
```

Funcionamiento:

1. Lee las credenciales BePro desde variables de entorno.
2. Consulta los partidos disponibles en BePro.
3. Compara contra los partidos ya generados en `outputs/app_data/`.
4. Si hay un partido nuevo con tracking disponible, descarga tracking, eventing y sequences.
5. Ejecuta el pipeline completo.
6. Genera los datos ligeros de la web en `outputs/app_data/`.

Variables admitidas:

```text
BEPRO_API_KEY       Token BePro recomendado
BEPRO_DATA_TOKEN    Alternativa al token principal
BEPRO_AUTH_TOKEN    Alternativa al token principal
BEPRO_TEAM_ID       ID del equipo propio, por defecto 12987
BEPRO_SEASON_IDS    ID o IDs usados para listar partidos en BePro, separados por coma
```

## GitHub Actions

El workflow esta en `.github/workflows/update_app_data.yml`.

Por defecto se ejecuta cada lunes a las 05:00 UTC y tambien se puede lanzar manualmente desde la pestana
`Comportamiento` / `Actions` de GitHub.

Antes de activarlo en GitHub:

1. Entra en `Settings` > `Secrets and variables` > `Actions`.
2. Crea el secreto `BEPRO_API_KEY` con el token de BePro.
3. En `Variables`, crea `BEPRO_TEAM_ID` con `12987`.
4. Si tu endpoint de BePro lista partidos por otro identificador, crea `BEPRO_SEASON_IDS`.
5. En `Actions`, ejecuta manualmente `Actualizar analisis BePro` para probar.

La accion descarga datos, regenera la app y hace commit automaticamente de:

```text
outputs/app_data/
eventing_partidos/
sequences_partidos/
outputs/global_clusters/
```

No commitea los tracking crudos por tamano. Si necesitas guardar tracking historico dentro de GitHub, usa
Git LFS o un almacenamiento externo y adapta el workflow.

## Subir a GitHub

```powershell
git init
git add .
git commit -m "Initial Streamlit defensive analysis app"
git branch -M main
git remote add origin https://github.com/juanhpdc03/App-TFG-AnalisisDefensivo.git
git push -u origin main
```
