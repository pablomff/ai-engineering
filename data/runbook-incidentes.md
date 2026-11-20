# Runbook de incidentes

Este runbook define cómo se clasifica, se atiende y se cierra un incidente de producción en la plataforma Lumen. Lo tiene que conocer toda persona que haga guardias.

## Niveles de severidad

Cada incidente se clasifica en uno de cuatro niveles. La severidad la decide quien atiende primero la alerta y se puede cambiar durante el incidente.

- **SEV1**: la plataforma no puede procesar pagos, o hay riesgo de pérdida o exposición de datos. Se reconoce la alerta en menos de 5 minutos y se abre una sala de crisis.
- **SEV2**: una funcionalidad importante está degradada (por ejemplo, la aprobación de pagos cae más de 10 puntos o las notificaciones a comercios se atrasan más de 30 minutos), pero los pagos siguen funcionando. Se reconoce en menos de 15 minutos.
- **SEV3**: un problema que afecta a pocos comercios o tiene un workaround conocido. Se atiende dentro del horario laboral del mismo día.
- **SEV4**: errores menores o mejoras detectadas durante la operación. Se registran como ticket y se priorizan en la planificación semanal.

## Guardias

Hay una guardia primaria y una secundaria. La rotación es semanal y cambia los lunes a las 10:00. La persona de guardia primaria recibe todas las alertas de PagerDuty. Si no reconoce una alerta SEV1 en 5 minutos, PagerDuty escala automáticamente a la guardia secundaria, y a los 15 minutos a la líder técnica del equipo de plataforma.

Las guardias se compensan con un día libre por cada semana de guardia primaria. Nadie hace dos semanas seguidas de guardia primaria.

## Cómo se atiende un SEV1 o un SEV2

1. Reconocer la alerta en PagerDuty para que no siga escalando.
2. Crear el canal de Slack del incidente con el formato `#inc-AAAAMMDD-tema`, por ejemplo `#inc-20261110-pagos`.
3. Asignar los roles: una persona coordina (no toca sistemas, solo organiza y comunica) y otra investiga y aplica los cambios.
4. Publicar el estado en la página de estado pública. En un SEV1 se actualiza al menos cada 30 minutos, aunque no haya novedades.
5. Mitigar primero y entender después: si un despliegue reciente es sospechoso, se hace rollback antes de buscar la causa.
6. Declarar el incidente resuelto cuando las métricas vuelven a lo normal durante al menos 30 minutos.

## Procedimientos frecuentes

### Cola de conciliación atrasada

Síntoma: la alerta "conciliacion lag" indica que el tópico `conciliacion.diferencias` tiene más de 50.000 mensajes sin consumir. Pasos: revisar en Datadog si servicio-conciliacion está reiniciándose; si el problema es de capacidad, escalar el servicio a 6 réplicas. No se borran mensajes del tópico bajo ninguna circunstancia, porque cada mensaje es una diferencia de dinero que hay que revisar.

### Réplica de PostgreSQL con retraso

Síntoma: una réplica de lectura tiene más de 60 segundos de retraso respecto de la primaria. Pasos: sacar la réplica del balanceador de lectura para que los reportes usen la otra, y revisar si hay una consulta pesada corriendo. Si el retraso supera los 10 minutos, se reconstruye la réplica desde el último backup.

## Después del incidente

Todo SEV1 y SEV2 requiere un postmortem escrito dentro de los 5 días hábiles posteriores a la resolución. El postmortem es sin culpables: describe qué pasó, por qué, cómo se detectó y qué acciones evitan que se repita. Cada acción tiene un responsable y una fecha, y se carga como ticket.

Los postmortems se presentan en la reunión de operaciones de los jueves, y se publican en el espacio de Confluence del equipo de plataforma.
