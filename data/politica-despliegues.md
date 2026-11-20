# Política de despliegues

Esta política define cuándo y cómo se despliega código a producción en la plataforma Lumen. Su objetivo es que los cambios lleguen rápido, pero sin poner en riesgo el procesamiento de pagos.

## Ventanas de despliegue

Los despliegues a producción se hacen de lunes a jueves, entre las 10:00 y las 16:00 (hora de Buenos Aires). Fuera de esa ventana no hay suficiente gente disponible para reaccionar si algo sale mal.

Hay dos períodos de congelamiento en los que no se despliega nada, salvo hotfixes:

- **Los viernes**, todo el día.
- **Los últimos 3 días hábiles de cada mes**, por el cierre contable: los comercios concilian sus cobros del mes y cualquier error tiene un impacto mayor.

Además, durante eventos de alto tráfico anunciados con anticipación (como Hot Sale o Black Friday) el congelamiento se extiende desde dos días antes hasta un día después del evento.

## Requisitos antes de desplegar

Un cambio puede desplegarse a producción cuando cumple todos estos requisitos:

- El pull request tiene la aprobación de al menos 2 personas del equipo dueño del servicio.
- Pasaron todos los tests automáticos del pipeline de integración continua.
- El cambio estuvo desplegado en el entorno de staging durante al menos 24 horas sin errores nuevos.
- Si modifica el esquema de la base de datos, la migración es compatible hacia atrás: la versión anterior del servicio tiene que poder seguir funcionando con el esquema nuevo.

## Despliegue canary

Ningún cambio llega a todos los usuarios de una vez. El despliegue sigue tres etapas:

1. **Canary al 5%**: la versión nueva recibe el 5% del tráfico durante 30 minutos.
2. **50%**: si el canary no mostró problemas, pasa a la mitad del tráfico durante otros 30 minutos.
3. **100%**: se completa el despliegue y la versión anterior queda disponible durante 24 horas para un rollback rápido.

Durante cada etapa, la persona que despliega tiene que estar mirando el panel "Lumen - Salud general" en Datadog.

## Criterios de rollback

Se hace rollback inmediato, sin esperar a investigar la causa, si en cualquier etapa se cumple alguna de estas condiciones durante más de 5 minutos:

- La tasa de errores de la versión nueva supera el 1%.
- La latencia p99 supera los 800 milisegundos.
- La tasa de aprobación de pagos cae más de 2 puntos respecto de la versión anterior.

El rollback se ejecuta con el comando `lumen deploy rollback <servicio>`, que devuelve el 100% del tráfico a la versión anterior en menos de un minuto. Después del rollback se abre un ticket para analizar qué falló antes de volver a intentar.

## Hotfixes

Un hotfix es un cambio urgente para resolver un incidente SEV1 o SEV2 en curso. Es la única excepción a las ventanas y a los períodos de congelamiento.

Un hotfix necesita la aprobación de 1 persona en lugar de 2, y puede saltear las 24 horas en staging. Sigue siendo obligatorio el canary al 5%, aunque se reduce a 10 minutos. Todo hotfix se menciona en el postmortem del incidente que lo motivó.
