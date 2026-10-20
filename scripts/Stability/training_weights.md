### Los que persiguen algo (positivos)

| Término | Peso | Qué premia | Subir el peso | Bajar el peso |
|---|---|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | Que la velocidad real (adelante/lateral) se parezca al comando pedido | Prioriza caminar rápido/preciso por encima de todo lo demás | Menos urgencia por seguir el comando, más conservador |
| `track_ang_vel_z_exp` | +0.5 | Lo mismo pero para girar (yaw) | Más prioridad en girar según se pide | Menos prioridad en girar |
| `feet_air_time` | +0.5 | Que cada pata pase tiempo real en el aire antes de tocar (el fix anti-arrastre) | Paso más marcado/alto | Riesgo de volver a arrastrar si otros rewards lo compensan |

### Los que castigan algo (negativos)

| Término | Peso | Qué castiga | Subir el castigo (más negativo) | Bajar el castigo |
|---|---|---|---|---|
| `flat_orientation_l2` | **-5.0** | Que el cuerpo NO esté nivelado/vertical — el más grande de todos | Prioriza casi obsesivamente mantenerse derecho (puede volverse rígido/tieso) | Más libertad para bambolearse/absorber el empujón del brazo con el cuerpo |
| `lin_vel_z_l2` | -2.0 | Rebotar verticalmente | Marcha más "pegada al piso" | Permite más rebote (puede ayudar a amortiguar el empujón) |
| `undesired_contacts` | -1.0 | Que el MUSLO toque el suelo (señal de caída/arrastre con la pierna) | Más estricto contra posturas caídas | Más tolerante a esa postura |
| `arm_leg_contact_penalty` | **-2.0** (era -1.0) | Que el brazo/gripper realmente toque una pata | Más prioridad en evitar la colisión, aunque sacrifique velocidad | Tolera más contacto brazo-pata |
| `ang_vel_xy_l2` | -0.05 | Bamboleo angular (roll/pitch) no deseado — justo lo que el brazo provoca | Exige más quietud angular | Más tolerancia al bamboleo |
| `action_rate_l2` | -0.01 | Que la acción cambie mucho de un paso a otro (temblor/jitter) | Acciones más suaves y consistentes | Permite reacciones más rápidas/erráticas ante el swing del brazo |
| `dof_torques_l2` | **-1.0e-05** (corregido; antes decía mal -2.5e-05) | Usar mucho torque (gasto de energía) | Patas más "perezosas" | Más fuerza libre para reaccionar |
| `dof_acc_l2` | -2.5e-07 | Aceleración brusca de las articulaciones | Movimientos más suaves/lentos | Permite reacciones más bruscas |
| `dof_pos_limits` | **-2.0** (era 0.0/apagado) | Forzar una articulación cerca de su límite mecánico real | Ya activado — sigue empeorando en el run de 8000 iteraciones pese a -2.0, candidato a subir más si no mejora | Volver a 0.0 lo dejaría ciego otra vez, como antes |

### Rango del comando de velocidad (no es un peso de reward, pero afecta la exigencia igual)

`lin_vel_x`, `lin_vel_y`, `ang_vel_z`: heredados en `(-1.0, 1.0)`, **halved a `(-0.5, 0.5)` el 2026-10-20** — no es un currículum (no vuelve a ensancharse después), es un techo permanente para esta tarea: el objetivo real es una plataforma estable para el brazo, no locomoción rápida. Causa raíz atacada: `dof_torques_l2`/`dof_acc_l2`/`action_rate_l2` empeoraban sin parar porque se le pedía llegar a velocidades más exigentes de lo que esta tarea necesita.

**Estado actual (2026-10-20), resumen de lo que cambió desde la tabla original:** `arm_leg_contact_penalty` y `dof_pos_limits` ya se subieron/activaron; el rango de velocidad ya se achicó. Pendiente de ver en la próxima corrida: si `dof_pos_limits` sigue empeorando pese a -2.0, y si el rango más chico por fin frena el empeoramiento de `dof_torques_l2`/`dof_acc_l2`/`action_rate_l2`. Ver `daily_log.md` para el detalle con datos de cada cambio.