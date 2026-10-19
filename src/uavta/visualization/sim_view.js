// Shared mission-replay helpers for the end-user page (simple.html).
// Pure functions of a replay payload from /api/replay and a time t (seconds); no network code.
"use strict";

const SimView = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const nf = (v, d = 0) => (v == null || Number.isNaN(v)) ? "–" : Number(v).toLocaleString("tr-TR", { minimumFractionDigits: d, maximumFractionDigits: d });
  const mmss = (s) => { s = Math.max(0, Math.round(s || 0)); return s < 60 ? `${s} sn` : s % 60 ? `${Math.floor(s / 60)} dk ${s % 60} sn` : `${s / 60} dk`; };
  const clock = (s) => { s = Math.max(0, Math.round(s || 0)); return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; };
  const SENSOR_TR = { VL: "kamerası", IR: "kızılötesi kamerası", RD: "radarı" };

  /** Add derived lookup tables to a replay payload (call once after fetching). */
  function prepare(r) {
    r.tmax = Math.max(1, ...r.activities.map((a) => a.t1 || 0));
    r.byUav = r.uavs.map((_, i) => r.activities.filter((a) => a.uav === i));
    r.slotToId = Object.fromEntries(r.targets.map((t) => [t.slot, t.id]));
    r.tdead = r.byUav.map((acts) => (acts.find((a) => a.kind === "failed" || a.kind === "depleted")?.t0) ?? Infinity);
    r.vmax = Math.max(...r.targets.map((x) => x.initial_value ?? x.value), 1);
    return r;
  }

  function uavAt(r, i, t) {
    const acts = r.byUav[i];
    let cur = acts[0];
    for (const a of acts) { if (a.t0 <= t) cur = a; else break; }
    let pos = cur.p0, heading = null;
    if (cur.kind === "transit" && cur.p1 && cur.t1 > cur.t0) {
      const f = Math.min(1, Math.max(0, (t - cur.t0) / (cur.t1 - cur.t0)));
      pos = cur.p0.map((v, k) => v + f * (cur.p1[k] - v));
      heading = Math.atan2(cur.p1[1] - cur.p0[1], cur.p1[0] - cur.p0[0]);
    }
    if (heading == null) {
      for (let k = acts.indexOf(cur); k >= 0; k--) {
        const a = acts[k];
        if (a.kind === "transit" && a.p1 && (a.p1[0] !== a.p0[0] || a.p1[1] !== a.p0[1])) { heading = Math.atan2(a.p1[1] - a.p0[1], a.p1[0] - a.p0[0]); break; }
      }
    }
    const battery = Math.max(0, 1 - Math.min(t, r.tdead[i]) / r.uavs[i].endurance);
    const dead = cur.kind === "failed" || cur.kind === "depleted";
    return { act: cur, pos, heading: heading ?? Math.PI / 2, battery, dead };
  }
  function targetPosAt(r, tg, t) {
    let pos = tg.initial_position;
    for (const e of r.events) if (e.type === "target_move" && e.target === tg.id && e.effect === "applied" && e.time <= t) pos = e.position;
    return pos;
  }
  function targetValueAt(r, tg, t) {
    let v = tg.initial_value ?? tg.value;
    for (const e of r.events) if (e.type === "value_change" && e.target === tg.id && e.effect === "applied" && e.time <= t) v = e.value;
    return v;
  }
  function weatherAt(r, t) {
    let w = r.initial_weather;
    for (const e of r.events) if (e.type === "weather" && e.effect === "applied" && e.time <= t) w = e.omega;
    return w;
  }
  /** Observed targets, appeared targets and collected information share at time t. */
  function progressAt(r, t) {
    let v = 0, n = 0;
    for (const tg of r.targets) if (tg.completed && tg.completion_time <= t) { v += tg.collected_value; n++; }
    const appeared = r.targets.filter((q) => q.appear_time <= t).length;
    const alive = r.uavs.filter((_, i) => !uavAt(r, i, t).dead).length;
    return { observed: n, appeared, info: v / Math.max(r.metrics.appeared_value, 1e-9), alive };
  }

  /** One plain-language sentence per event; null for events that should not be announced. */
  function eventSentence(e) {
    if (e.effect !== "applied") return null;
    switch (e.type) {
      case "uav_crash": return { icon: "⚠", tone: "bad", text: `${e.uav} drone'u düştü! ${e.released_target ? "Yarım kalan işini diğer dronelar devralacak." : "Kalan dronelar devam ediyor."}` };
      case "sensor_damage": return { icon: "🔧", tone: "warn", text: `${e.uav} drone'unun ${SENSOR_TR[e.sensor] || "sensörü"} bozuldu; artık daha kötü görüyor.` };
      case "weather": return { icon: "☁", tone: "warn", text: "Hava bozuldu! Bütün dronelar daha az net görüyor." };
      case "target_move": return { icon: "➜", tone: "warn", text: "Bir hedef yer değiştirdi; ona giden drone rotasını değiştiriyor." };
      case "value_change": return { icon: "↓", tone: "info", text: "Bir hedefin önemi azaldı; artık o kadar acil değil." };
      case "new_target": return { icon: "📍", tone: "good", text: "Yeni bir hedef ortaya çıktı! Biri oraya gitmeli." };
      default: return null;
    }
  }

  /**
   * Draw the map of replay r at time t on canvas cv. Returns hover hit-boxes
   * [{x, y, r, html}]. opts: {trails: bool}
   */
  function draw(cv, r, t, opts = {}) {
    const ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 500;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(W * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css("--map"); ctx.fillRect(0, 0, W, W);
    const hits = [];
    if (!r) { ctx.fillStyle = css("--muted"); ctx.font = "17px system-ui"; ctx.fillText("Görev hazırlanıyor…", 22, 38); return hits; }
    t = Math.min(t, r.tmax);
    const pad = 16, L = r.airspace, S = W - 2 * pad, scale = S / L[0];
    const X = (x) => pad + S * x / L[0], Y = (y) => pad + S * (1 - y / L[1]);
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    for (let k = 0; k <= 8; k++) {
      const v = k * L[0] / 8;
      ctx.beginPath(); ctx.moveTo(X(v), Y(0)); ctx.lineTo(X(v), Y(L[1])); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(L[0]), Y(v)); ctx.stroke();
    }
    // base
    const starts = r.byUav.map((acts) => acts[0].p0);
    const bx0 = Math.max(0, Math.min(...starts.map((p) => p[0])) - 250), bx1 = Math.max(...starts.map((p) => p[0])) + 250;
    const by0 = Math.max(0, Math.min(...starts.map((p) => p[1])) - 250), by1 = Math.max(...starts.map((p) => p[1])) + 250;
    ctx.fillStyle = css("--base"); ctx.beginPath(); ctx.roundRect(X(bx0), Y(by1), (bx1 - bx0) * scale, (by1 - by0) * scale, 8); ctx.fill();
    ctx.fillStyle = css("--text-secondary"); ctx.font = "bold 12px system-ui"; ctx.fillText("🏠 Üs", X(bx0) + 5, Y(by1) + 15);

    const uavs = r.uavs.map((_, i) => uavAt(r, i, t));
    const color = (i) => css(`--drone-${(i % 6) + 1}`);
    if (opts.trails !== false) {
      uavs.forEach((u, i) => {
        ctx.strokeStyle = color(i); ctx.globalAlpha = 0.28; ctx.lineWidth = 2.5;
        for (const a of r.byUav[i]) {
          if (a.kind !== "transit" || a.t0 > t || !a.p1) continue;
          const end = a.t1 <= t ? a.p1 : u.pos;
          ctx.beginPath(); ctx.moveTo(X(a.p0[0]), Y(a.p0[1])); ctx.lineTo(X(end[0]), Y(end[1])); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      });
    }
    const going = {}, watching = {};
    uavs.forEach((u, i) => {
      if (u.act.slot == null) return;
      if (u.act.kind === "transit") going[u.act.slot] = i;
      if (u.act.kind === "observe") watching[u.act.slot] = i;
    });
    const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 170);
    for (const tg of r.targets) {
      if (tg.appear_time > t) continue;
      const p = targetPosAt(r, tg, t), value = targetValueAt(r, tg, t);
      const rad = 6 + 9 * Math.sqrt(Math.max(value, 1) / r.vmax);
      const x = X(p[0]), y = Y(p[1]);
      const done = tg.completed && tg.completion_time <= t;
      ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI);
      let status;
      if (done) {
        ctx.fillStyle = css("--done"); ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.2; ctx.beginPath();
        ctx.moveTo(x - rad * .45, y); ctx.lineTo(x - rad * .1, y + rad * .4); ctx.lineTo(x + rad * .5, y - rad * .4); ctx.stroke();
        status = "Gözlendi ✓";
      } else if (watching[tg.slot] != null) {
        const i = watching[tg.slot];
        ctx.fillStyle = color(i); ctx.globalAlpha = 0.5; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = color(i); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, rad + 3 + 4 * pulse, 0, 2 * Math.PI); ctx.stroke();
        status = `${r.uavs[i].id} şu an burayı gözlüyor`;
      } else {
        ctx.fillStyle = css("--map"); ctx.fill();
        const i = going[tg.slot];
        ctx.strokeStyle = i != null ? color(i) : css("--target"); ctx.lineWidth = i != null ? 3 : 2; ctx.stroke();
        status = i != null ? `${r.uavs[i].id} buraya geliyor` : "Sırasını bekliyor";
      }
      const imp = value >= 0.85 * r.vmax ? "Çok önemli" : value >= 0.65 * r.vmax ? "Önemli" : "Normal";
      hits.push({ x, y, r: rad + 5, html: `<b>📍 Gözlenecek yer</b><br>Önem: <b>${imp}</b><br>Durum: <b>${status}</b>` });
    }
    uavs.forEach((u, i) => {
      const x = X(u.pos[0]), y = Y(u.pos[1]);
      if (u.act.kind === "transit" && u.act.slot != null) {
        const tg = r.targets.find((q) => q.slot === u.act.slot);
        if (tg) {
          const p = targetPosAt(r, tg, t);
          ctx.strokeStyle = color(i); ctx.setLineDash([6, 5]); ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(X(p[0]), Y(p[1])); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      if (u.dead) {
        ctx.strokeStyle = u.act.kind === "failed" ? css("--bad") : css("--muted"); ctx.lineWidth = 4; ctx.beginPath();
        ctx.moveTo(x - 9, y - 9); ctx.lineTo(x + 9, y + 9); ctx.moveTo(x + 9, y - 9); ctx.lineTo(x - 9, y + 9); ctx.stroke();
      } else {
        ctx.save(); ctx.translate(x, y); ctx.rotate(-u.heading + Math.PI / 2);
        ctx.fillStyle = color(i); ctx.strokeStyle = css("--map"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, -13); ctx.lineTo(10, 10); ctx.lineTo(0, 5); ctx.lineTo(-10, 10); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.restore();
        const bw = 24, bx = x - bw / 2, byy = y + 15;
        ctx.fillStyle = css("--grid"); ctx.fillRect(bx, byy, bw, 4);
        ctx.fillStyle = u.battery < 0.15 ? css("--bad") : u.battery < 0.3 ? css("--warn") : css("--done");
        ctx.fillRect(bx, byy, bw * u.battery, 4);
      }
      ctx.fillStyle = css("--text-primary"); ctx.font = "bold 12px system-ui"; ctx.fillText(r.uavs[i].id, x + 12, y - 8);
      const doing = u.dead ? (u.act.kind === "failed" ? "Düştü" : "Pili bitti")
        : u.act.kind === "transit" ? "Bir hedefe uçuyor" : u.act.kind === "observe" ? "Gözlem yapıyor" : "Yeni görev bekliyor";
      hits.push({ x, y, r: 16, html: `<b>🛩 Drone ${r.uavs[i].id}</b><br>Ne yapıyor: <b>${doing}</b><br>Pil: <b>%${nf(100 * u.battery)}</b>` });
    });
    const w = weatherAt(r, t), wmean = w.reduce((a, b) => a + b, 0) / w.length;
    if (wmean < 0.97) {
      ctx.fillStyle = `rgba(${css("--cloud")},${Math.min(0.22, (1 - wmean) * 0.4)})`; ctx.fillRect(pad, pad, S, S);
      ctx.font = "28px system-ui"; ctx.globalAlpha = 0.5;
      for (const [cx, cy] of [[.2, .25], [.55, .12], [.82, .4], [.35, .62], [.72, .8]]) ctx.fillText("☁", pad + S * cx, pad + S * cy);
      ctx.globalAlpha = 1;
    }
    return hits;
  }

  return { prepare, uavAt, progressAt, eventSentence, draw, nf, mmss, clock };
})();
