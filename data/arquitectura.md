# Arquitectura de la plataforma Lumen

Lumen es la plataforma de cobros de Lumen Pagos. Procesa pagos con tarjeta y transferencias para unos 4.000 comercios. Este documento describe los componentes, cómo se comunican y los límites que hay que respetar al integrarse.

## Componentes principales

La plataforma está dividida en cuatro servicios. Todos corren como contenedores en un cluster de Kubernetes administrado, en dos zonas de disponibilidad de la misma región.

- **api-gateway**: escrito en Go. Es la única puerta de entrada desde internet. Valida el token del comercio, aplica los límites de tráfico y enruta cada request al servicio que corresponde.
- **servicio-pagos**: escrito en Java 21 con Spring Boot 3. Crea, autoriza y captura los pagos. Es el único servicio que habla con los procesadores de tarjetas externos.
- **servicio-conciliacion**: escrito en Python 3.12. Todas las noches compara los pagos registrados con los archivos de liquidación que envían los procesadores y marca las diferencias para revisión manual.
- **servicio-notificaciones**: escrito en Node.js 22. Envía webhooks a los comercios cuando un pago cambia de estado y reintenta los envíos fallidos hasta 8 veces a lo largo de 24 horas.

## Datos y almacenamiento

La base de datos principal es PostgreSQL 16, con una instancia primaria y dos réplicas de lectura. Solo servicio-pagos escribe en la primaria. Los reportes y el panel de comercios leen siempre de las réplicas para no cargar la primaria.

Los backups completos de PostgreSQL se toman todos los días a las 02:00 y se conservan 35 días. Además, el log de transacciones se archiva de forma continua, lo que permite restaurar la base a cualquier minuto de los últimos 7 días.

Redis se usa para dos cosas: guardar las claves de idempotencia de los pagos (con un TTL de 24 horas) y cachear la configuración de cada comercio (con un TTL de 5 minutos). Si Redis se cae, los pagos siguen funcionando, pero más lentos, porque la configuración se lee de PostgreSQL en cada request.

## Comunicación entre servicios

Los servicios se comunican de dos formas. Las consultas que necesitan respuesta inmediata usan HTTP interno con un timeout de 2 segundos. Los cambios de estado se publican como eventos en Kafka.

Los tópicos de Kafka más importantes son:

- `pagos.creados`: lo publica servicio-pagos cada vez que se crea un pago.
- `pagos.estado-cambiado`: lo consume servicio-notificaciones para avisar a los comercios.
- `conciliacion.diferencias`: lo publica servicio-conciliacion cuando encuentra un pago que no coincide con la liquidación.

Los eventos se retienen 7 días en Kafka. Pasado ese plazo, la única fuente de verdad es PostgreSQL.

## Límites de la API pública

Cada comercio puede hacer hasta 300 requests por minuto contra la API pública. Si supera ese límite, el api-gateway responde con un error 429 e incluye el header `Retry-After` con los segundos que tiene que esperar.

Todas las operaciones que crean un pago exigen el header `Idempotency-Key`. Si un comercio repite un request con la misma clave dentro de las 24 horas, recibe la respuesta original en lugar de crear un pago duplicado.

El tiempo máximo de respuesta de la API pública es de 10 segundos. Si un procesador de tarjetas no responde en ese tiempo, el pago queda en estado `pendiente` y se resuelve más tarde por el proceso de conciliación.

## Observabilidad

Cada servicio envía sus logs y métricas a Datadog. Los paneles principales son "Lumen - Salud general", con la tasa de errores y la latencia de cada servicio, y "Lumen - Pagos", con la cantidad de pagos por minuto y la tasa de aprobación.

Las alertas de producción se envían a PagerDuty, que avisa a la persona de guardia según el procedimiento descrito en el runbook de incidentes.
