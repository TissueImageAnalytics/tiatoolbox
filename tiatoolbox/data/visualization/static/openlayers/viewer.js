//#region node_modules/ol/MapProperty.js
var e = {
	LAYERGROUP: "layergroup",
	SIZE: "size",
	TARGET: "target",
	VIEW: "view"
}, t = "ol-hidden", n = "ol-selectable", r = "ol-unselectable", i = "ol-unsupported", a = "ol-control", o = "ol-collapsed", s = new RegExp([
	"^\\s*(?=(?:(?:[-a-z]+\\s*){0,2}(italic|oblique))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(small-caps))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)",
	"(?:(?:normal|\\1|\\2|\\3)\\s*){0,3}((?:xx?-)?",
	"(?:small|large)|medium|smaller|larger|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx]))",
	"(?:\\s*\\/\\s*(normal|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx])?))",
	"?\\s*([-,\\\"\\'\\sa-z0-9]+?)\\s*$"
].join(""), "i"), c = [
	"style",
	"variant",
	"weight",
	"size",
	"lineHeight",
	"family"
], l = {
	normal: 400,
	bold: 700
}, u = function(e) {
	let t = e.match(s);
	if (!t) return null;
	let n = {
		lineHeight: "normal",
		size: "1.2em",
		style: "normal",
		weight: "400",
		variant: "normal"
	};
	for (let e = 0, r = c.length; e < r; ++e) {
		let r = t[e + 1];
		r !== void 0 && (n[c[e]] = typeof r == "string" ? r.trim() : r);
	}
	return isNaN(Number(n.weight)) && n.weight in l && (n.weight = l[n.weight]), n.families = n.family.split(/,\s?/).map((e) => e.trim().replace(/^['"]|['"]$/g, "")), n;
}, d = typeof navigator < "u" && navigator.userAgent !== void 0 ? navigator.userAgent.toLowerCase() : "", f = d.includes("safari") && !d.includes("chrom") && (d.includes("version/15.4") || /cpu (os|iphone os) 15_4 like mac os x/.test(d)), p = d.includes("webkit") && !d.includes("edge"), m = d.includes("macintosh"), h = typeof devicePixelRatio < "u" ? devicePixelRatio : 1, g = typeof WorkerGlobalScope < "u" && typeof OffscreenCanvas < "u" && self instanceof WorkerGlobalScope, _ = typeof Image < "u" && Image.prototype.decode, v = (function() {
	let e = !1;
	try {
		let t = Object.defineProperty({}, "passive", { get: function() {
			e = !0;
		} });
		window.addEventListener("_", null, t), window.removeEventListener("_", null, t);
	} catch {}
	return e;
})();
//#endregion
//#region node_modules/ol/dom.js
function y(e, t, n, r) {
	let i;
	return i = n && n.length ? n.shift() : g ? new class extends OffscreenCanvas {
		style = {};
	}(e ?? 300, t ?? 150) : document.createElement("canvas"), e && (i.width = e), t && (i.height = t), i.getContext("2d", r);
}
var b;
function x() {
	return b ||= y(1, 1), b;
}
function S(e) {
	let t = e.canvas;
	t.width = 1, t.height = 1, e.clearRect(0, 0, 1, 1);
}
function C(e) {
	let t = e.offsetWidth, n = getComputedStyle(e);
	return t += parseInt(n.marginLeft, 10) + parseInt(n.marginRight, 10), t;
}
function w(e) {
	let t = e.offsetHeight, n = getComputedStyle(e);
	return t += parseInt(n.marginTop, 10) + parseInt(n.marginBottom, 10), t;
}
function T(e, t) {
	let n = t.parentNode;
	n && n.replaceChild(e, t);
}
function E(e) {
	for (; e.lastChild;) e.lastChild.remove();
}
function ee(e, t) {
	let n = e.childNodes;
	for (let r = 0;; ++r) {
		let i = n[r], a = t[r];
		if (!i && !a) break;
		if (i !== a) {
			if (!i) {
				e.appendChild(a);
				continue;
			}
			if (!a) {
				e.removeChild(i), --r;
				continue;
			}
			e.insertBefore(a, i);
		}
	}
}
function D() {
	return new Proxy({
		childNodes: [],
		appendChild: function(e) {
			return this.childNodes.push(e), e;
		},
		remove: function() {},
		removeChild: function(e) {
			let t = this.childNodes.indexOf(e);
			if (t === -1) throw Error("Node to remove was not found");
			return this.childNodes.splice(t, 1), e;
		},
		insertBefore: function(e, t) {
			let n = this.childNodes.indexOf(t);
			if (n === -1) throw Error("Reference node not found");
			return this.childNodes.splice(n, 0, e), e;
		},
		style: {}
	}, { get(e, t, n) {
		return t === "firstElementChild" ? e.childNodes.length > 0 ? e.childNodes[0] : null : Reflect.get(e, t, n);
	} });
}
function O(e) {
	return typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof OffscreenCanvas < "u" && e instanceof OffscreenCanvas;
}
//#endregion
//#region node_modules/ol/obj.js
function k(e) {
	for (let t in e) delete e[t];
}
function te(e) {
	let t;
	for (t in e) return !1;
	return !t;
}
//#endregion
//#region node_modules/ol/events.js
function A(e, t, n, r, i) {
	if (i) {
		let i = n;
		n = function(a) {
			return e.removeEventListener(t, n), i.call(r ?? this, a);
		};
	} else r && r !== e && (n = n.bind(r));
	let a = {
		target: e,
		type: t,
		listener: n
	};
	return e.addEventListener(t, n), a;
}
function ne(e, t, n, r) {
	return A(e, t, n, r, !0);
}
function j(e) {
	e && e.target && (e.target.removeEventListener(e.type, e.listener), k(e));
}
//#endregion
//#region node_modules/ol/events/EventType.js
var M = {
	CHANGE: "change",
	ERROR: "error",
	BLUR: "blur",
	CLEAR: "clear",
	CONTEXTMENU: "contextmenu",
	CLICK: "click",
	DBLCLICK: "dblclick",
	DRAGENTER: "dragenter",
	DRAGOVER: "dragover",
	DROP: "drop",
	FOCUS: "focus",
	KEYDOWN: "keydown",
	KEYPRESS: "keypress",
	LOAD: "load",
	RESIZE: "resize",
	TOUCHMOVE: "touchmove",
	WHEEL: "wheel"
}, N = {
	POSTRENDER: "postrender",
	MOVESTART: "movestart",
	MOVEEND: "moveend",
	LOADSTART: "loadstart",
	LOADEND: "loadend"
}, re = { PROPERTYCHANGE: "propertychange" }, ie = class {
	constructor() {
		this.disposed = !1;
	}
	dispose() {
		this.disposed || (this.disposed = !0, this.disposeInternal());
	}
	disposeInternal() {}
};
//#endregion
//#region node_modules/ol/array.js
function ae(e, t, n) {
	let r, i;
	n ||= oe;
	let a = 0, o = e.length, s = !1;
	for (; a < o;) r = a + (o - a >> 1), i = +n(e[r], t), i < 0 ? a = r + 1 : (o = r, s = !i);
	return s ? a : ~a;
}
function oe(e, t) {
	return e > t ? 1 : e < t ? -1 : 0;
}
function se(e, t) {
	return e < t ? 1 : e > t ? -1 : 0;
}
function ce(e, t, n) {
	if (e[0] <= t) return 0;
	let r = e.length;
	if (t <= e[r - 1]) return r - 1;
	if (typeof n == "function") {
		for (let i = 1; i < r; ++i) {
			let r = e[i];
			if (r === t) return i;
			if (r < t) return n(t, e[i - 1], r) > 0 ? i - 1 : i;
		}
		return r - 1;
	}
	if (n > 0) {
		for (let n = 1; n < r; ++n) if (e[n] < t) return n - 1;
		return r - 1;
	}
	if (n < 0) {
		for (let n = 1; n < r; ++n) if (e[n] <= t) return n;
		return r - 1;
	}
	for (let n = 1; n < r; ++n) {
		if (e[n] == t) return n;
		if (e[n] < t) return e[n - 1] - t < t - e[n] ? n - 1 : n;
	}
	return r - 1;
}
function le(e, t, n) {
	for (; t < n;) {
		let r = e[t];
		e[t] = e[n], e[n] = r, ++t, --n;
	}
}
function ue(e, t) {
	let n = Array.isArray(t) ? t : [t], r = n.length;
	for (let t = 0; t < r; t++) e[e.length] = n[t];
}
function de(e, t) {
	let n = e.length;
	if (n !== t.length) return !1;
	for (let r = 0; r < n; r++) if (e[r] !== t[r]) return !1;
	return !0;
}
function fe(e, t, n) {
	let r = t || oe;
	return e.every(function(t, i) {
		if (i === 0) return !0;
		let a = r(e[i - 1], t);
		return !(a > 0 || n && a === 0);
	});
}
//#endregion
//#region node_modules/ol/functions.js
function pe() {
	return !0;
}
function me() {
	return !1;
}
function he() {}
function ge(e) {
	let t, n, r;
	return function() {
		let i = Array.prototype.slice.call(arguments);
		return (!n || this !== r || !de(i, n)) && (r = this, n = i, t = e.apply(this, arguments)), t;
	};
}
function _e(e) {
	function t() {
		let t;
		try {
			t = e();
		} catch (e) {
			return Promise.reject(e);
		}
		return t instanceof Promise ? t : Promise.resolve(t);
	}
	return t();
}
//#endregion
//#region node_modules/ol/events/Event.js
var ve = class {
	constructor(e) {
		this.propagationStopped, this.defaultPrevented, this.type = e, this.target = null;
	}
	preventDefault() {
		this.defaultPrevented = !0;
	}
	stopPropagation() {
		this.propagationStopped = !0;
	}
}, ye = class extends ie {
	constructor(e) {
		super(), this.eventTarget_ = e, this.pendingRemovals_ = null, this.dispatching_ = null, this.listeners_ = null;
	}
	addEventListener(e, t) {
		if (!e || !t) return;
		let n = this.listeners_ ||= {}, r = n[e] || (n[e] = []);
		r.includes(t) || r.push(t);
	}
	dispatchEvent(e) {
		let t = typeof e == "string", n = t ? e : e.type, r = this.listeners_ && this.listeners_[n];
		if (!r) return;
		let i = t ? new ve(e) : e;
		i.target ||= this.eventTarget_ || this;
		let a = this.dispatching_ ||= {}, o = this.pendingRemovals_ ||= {};
		n in a || (a[n] = 0, o[n] = 0), ++a[n];
		let s;
		for (let e = 0, t = r.length; e < t; ++e) if (s = "handleEvent" in r[e] ? r[e].handleEvent(i) : r[e].call(this, i), s === !1 || i.propagationStopped) {
			s = !1;
			break;
		}
		if (--a[n] === 0) {
			let e = o[n];
			for (delete o[n]; e--;) this.removeEventListener(n, he);
			delete a[n];
		}
		return s;
	}
	disposeInternal() {
		this.listeners_ && k(this.listeners_);
	}
	getListeners(e) {
		return this.listeners_ && this.listeners_[e] || void 0;
	}
	hasListener(e) {
		return this.listeners_ ? e ? e in this.listeners_ : Object.keys(this.listeners_).length > 0 : !1;
	}
	removeEventListener(e, t) {
		if (!this.listeners_) return;
		let n = this.listeners_[e];
		if (!n) return;
		let r = n.indexOf(t);
		r !== -1 && (this.pendingRemovals_ && e in this.pendingRemovals_ ? (n[r] = he, ++this.pendingRemovals_[e]) : (n.splice(r, 1), n.length === 0 && delete this.listeners_[e]));
	}
}, be = class extends ye {
	constructor() {
		super(), this.on = this.onInternal, this.once = this.onceInternal, this.un = this.unInternal, this.revision_ = 0;
	}
	changed() {
		++this.revision_, this.dispatchEvent(M.CHANGE);
	}
	getRevision() {
		return this.revision_;
	}
	onInternal(e, t) {
		if (Array.isArray(e)) {
			let n = e.length, r = Array(n);
			for (let i = 0; i < n; ++i) r[i] = A(this, e[i], t);
			return r;
		}
		return A(this, e, t);
	}
	onceInternal(e, t) {
		let n;
		if (Array.isArray(e)) {
			let r = e.length;
			n = Array(r);
			for (let i = 0; i < r; ++i) n[i] = ne(this, e[i], t);
		} else n = ne(this, e, t);
		return t.ol_key = n, n;
	}
	unInternal(e, t) {
		let n = t.ol_key;
		if (n) xe(n);
		else if (Array.isArray(e)) for (let n = 0, r = e.length; n < r; ++n) this.removeEventListener(e[n], t);
		else this.removeEventListener(e, t);
	}
};
be.prototype.on, be.prototype.once, be.prototype.un;
function xe(e) {
	if (Array.isArray(e)) for (let t = 0, n = e.length; t < n; ++t) j(e[t]);
	else j(e);
}
//#endregion
//#region node_modules/ol/util.js
function P() {
	throw Error("Unimplemented abstract method.");
}
var Se = 0;
function F(e) {
	return e.ol_uid ||= String(++Se);
}
//#endregion
//#region node_modules/ol/Object.js
var Ce = class extends ve {
	constructor(e, t, n) {
		super(e), this.key = t, this.oldValue = n;
	}
}, we = class extends be {
	constructor(e) {
		super(), this.on, this.once, this.un, F(this), this.values_ = null, e !== void 0 && this.setProperties(e);
	}
	get(e) {
		let t;
		return this.values_ && this.values_.hasOwnProperty(e) && (t = this.values_[e]), t;
	}
	getKeys() {
		return this.values_ && Object.keys(this.values_) || [];
	}
	getProperties() {
		return this.values_ && Object.assign({}, this.values_) || {};
	}
	getPropertiesInternal() {
		return this.values_;
	}
	hasProperties() {
		return !!this.values_;
	}
	notify(e, t) {
		let n;
		n = `change:${e}`, this.hasListener(n) && this.dispatchEvent(new Ce(n, e, t)), n = re.PROPERTYCHANGE, this.hasListener(n) && this.dispatchEvent(new Ce(n, e, t));
	}
	addChangeListener(e, t) {
		this.addEventListener(`change:${e}`, t);
	}
	removeChangeListener(e, t) {
		this.removeEventListener(`change:${e}`, t);
	}
	set(e, t, n) {
		let r = this.values_ ||= {};
		if (n) r[e] = t;
		else {
			let n = r[e];
			r[e] = t, n !== t && this.notify(e, n);
		}
	}
	setProperties(e, t) {
		for (let n in e) this.set(n, e[n], t);
	}
	applyProperties(e) {
		e.values_ && Object.assign(this.values_ ||= {}, e.values_);
	}
	unset(e, t) {
		if (this.values_ && e in this.values_) {
			let n = this.values_[e];
			delete this.values_[e], te(this.values_) && (this.values_ = null), t || this.notify(e, n);
		}
	}
}, Te = class extends we {
	constructor(e) {
		super();
		let t = e.element;
		t && !e.target && !t.style.pointerEvents && (t.style.pointerEvents = "auto"), this.element = t || null, this.target_ = null, this.map_ = null, this.listenerKeys = [], e.render && (this.render = e.render), e.target && this.setTarget(e.target);
	}
	disposeInternal() {
		this.element?.remove(), super.disposeInternal();
	}
	getMap() {
		return this.map_;
	}
	setMap(e) {
		this.map_ && this.element?.remove();
		for (let e = 0, t = this.listenerKeys.length; e < t; ++e) j(this.listenerKeys[e]);
		if (this.listenerKeys.length = 0, this.map_ = e, e) {
			let t = this.target_ ?? e.getOverlayContainerStopEvent();
			this.element && t.appendChild(this.element), this.render !== he && this.listenerKeys.push(A(e, N.POSTRENDER, this.render, this)), e.render();
		}
	}
	render(e) {}
	setTarget(e) {
		this.target_ = typeof e == "string" ? document.getElementById(e) : e;
	}
}, Ee = ["fullscreenchange", "webkitfullscreenchange"], De = {
	ENTERFULLSCREEN: "enterfullscreen",
	LEAVEFULLSCREEN: "leavefullscreen"
}, Oe = class extends Te {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			target: e.target
		}), this.on, this.once, this.un, this.keys_ = e.keys !== void 0 && e.keys, this.source_ = e.source, this.isInFullscreen_ = !1, this.boundHandleMapTargetChange_ = this.handleMapTargetChange_.bind(this), this.cssClassName_ = e.className === void 0 ? "ol-full-screen" : e.className, this.documentListeners_ = [], this.activeClassName_ = e.activeClassName === void 0 ? [this.cssClassName_ + "-true"] : e.activeClassName.split(" "), this.inactiveClassName_ = e.inactiveClassName === void 0 ? [this.cssClassName_ + "-false"] : e.inactiveClassName.split(" ");
		let t = e.label === void 0 ? "⤢" : e.label;
		this.labelNode_ = typeof t == "string" ? document.createTextNode(t) : t;
		let n = e.labelActive === void 0 ? "×" : e.labelActive;
		this.labelActiveNode_ = typeof n == "string" ? document.createTextNode(n) : n;
		let i = e.tipLabel ? e.tipLabel : "Toggle full-screen";
		this.button_ = document.createElement("button"), this.button_.title = i, this.button_.setAttribute("type", "button"), this.button_.appendChild(this.labelNode_), this.button_.addEventListener(M.CLICK, this.handleClick_.bind(this), !1), this.setClassName_(this.button_, this.isInFullscreen_), this.element.className = `${this.cssClassName_} ${r} ${a}`, this.element.appendChild(this.button_);
	}
	handleClick_(e) {
		e.preventDefault(), this.handleFullScreen_();
	}
	handleFullScreen_() {
		let e = this.getMap();
		if (!e) return;
		let t = e.getOwnerDocument();
		if (ke(t)) {
			if (Ae(t)) Ne(t);
			else {
				let n;
				n = this.source_ ? typeof this.source_ == "string" ? t.getElementById(this.source_) : this.source_ : e.getTargetElement(), this.keys_ ? Me(n) : je(n);
			}
		}
	}
	handleFullScreenChange_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.isInFullscreen_;
		this.isInFullscreen_ = Ae(e.getOwnerDocument()), t !== this.isInFullscreen_ && (this.setClassName_(this.button_, this.isInFullscreen_), this.isInFullscreen_ ? (T(this.labelActiveNode_, this.labelNode_), this.dispatchEvent(De.ENTERFULLSCREEN)) : (T(this.labelNode_, this.labelActiveNode_), this.dispatchEvent(De.LEAVEFULLSCREEN)), e.updateSize());
	}
	setClassName_(e, t) {
		t ? (e.classList.remove(...this.inactiveClassName_), e.classList.add(...this.activeClassName_)) : (e.classList.remove(...this.activeClassName_), e.classList.add(...this.inactiveClassName_));
	}
	setMap(t) {
		let n = this.getMap();
		n && n.removeChangeListener(e.TARGET, this.boundHandleMapTargetChange_), super.setMap(t), this.handleMapTargetChange_(), t && t.addChangeListener(e.TARGET, this.boundHandleMapTargetChange_);
	}
	handleMapTargetChange_() {
		let e = this.documentListeners_;
		for (let t = 0, n = e.length; t < n; ++t) j(e[t]);
		e.length = 0;
		let t = this.getMap();
		if (t) {
			let n = t.getOwnerDocument();
			ke(n) ? this.element.classList.remove(i) : this.element.classList.add(i);
			for (let t = 0, r = Ee.length; t < r; ++t) e.push(A(n, Ee[t], this.handleFullScreenChange_, this));
			this.handleFullScreenChange_();
		}
	}
};
function ke(e) {
	let t = e.body;
	return !!(t.webkitRequestFullscreen || t.requestFullscreen && e.fullscreenEnabled);
}
function Ae(e) {
	return !!(e.webkitIsFullScreen || e.fullscreenElement);
}
function je(e) {
	e.requestFullscreen ? e.requestFullscreen() : e.webkitRequestFullscreen && e.webkitRequestFullscreen();
}
function Me(e) {
	e.webkitRequestFullscreen ? e.webkitRequestFullscreen() : je(e);
}
function Ne(e) {
	e.exitFullscreen ? e.exitFullscreen() : e.webkitExitFullscreen && e.webkitExitFullscreen();
}
//#endregion
//#region node_modules/ol/extent/Relationship.js
var Pe = {
	UNKNOWN: 0,
	INTERSECTING: 1,
	ABOVE: 2,
	RIGHT: 4,
	BELOW: 8,
	LEFT: 16
};
//#endregion
//#region node_modules/ol/extent.js
function Fe(e) {
	let t = We();
	for (let n = 0, r = e.length; n < r; ++n) Ze(t, e[n]);
	return t;
}
function Ie(e, t, n) {
	return Ge(Math.min.apply(null, e), Math.min.apply(null, t), Math.max.apply(null, e), Math.max.apply(null, t), n);
}
function Le(e, t, n) {
	return n ? (n[0] = e[0] - t, n[1] = e[1] - t, n[2] = e[2] + t, n[3] = e[3] + t, n) : [
		e[0] - t,
		e[1] - t,
		e[2] + t,
		e[3] + t
	];
}
function Re(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e.slice();
}
function ze(e, t, n) {
	let r, i;
	return r = t < e[0] ? e[0] - t : e[2] < t ? t - e[2] : 0, i = n < e[1] ? e[1] - n : e[3] < n ? n - e[3] : 0, r * r + i * i;
}
function Be(e, t) {
	return He(e, t[0], t[1]);
}
function Ve(e, t) {
	return e[0] <= t[0] && t[2] <= e[2] && e[1] <= t[1] && t[3] <= e[3];
}
function He(e, t, n) {
	return e[0] <= t && t <= e[2] && e[1] <= n && n <= e[3];
}
function Ue(e, t) {
	let n = e[0], r = e[1], i = e[2], a = e[3], o = t[0], s = t[1], c = Pe.UNKNOWN;
	return o < n ? c |= Pe.LEFT : o > i && (c |= Pe.RIGHT), s < r ? c |= Pe.BELOW : s > a && (c |= Pe.ABOVE), c === Pe.UNKNOWN && (c = Pe.INTERSECTING), c;
}
function We() {
	return [
		Infinity,
		Infinity,
		-Infinity,
		-Infinity
	];
}
function Ge(e, t, n, r, i) {
	return i ? (i[0] = e, i[1] = t, i[2] = n, i[3] = r, i) : [
		e,
		t,
		n,
		r
	];
}
function Ke(e) {
	return Ge(Infinity, Infinity, -Infinity, -Infinity, e);
}
function qe(e, t) {
	let n = e[0], r = e[1];
	return Ge(n, r, n, r, t);
}
function Je(e, t, n, r, i) {
	return Qe(Ke(i), e, t, n, r);
}
function Ye(e, t) {
	return e[0] == t[0] && e[2] == t[2] && e[1] == t[1] && e[3] == t[3];
}
function Xe(e, t) {
	return t[0] < e[0] && (e[0] = t[0]), t[2] > e[2] && (e[2] = t[2]), t[1] < e[1] && (e[1] = t[1]), t[3] > e[3] && (e[3] = t[3]), e;
}
function Ze(e, t) {
	t[0] < e[0] && (e[0] = t[0]), t[0] > e[2] && (e[2] = t[0]), t[1] < e[1] && (e[1] = t[1]), t[1] > e[3] && (e[3] = t[1]);
}
function Qe(e, t, n, r, i) {
	for (; n < r; n += i) $e(e, t[n], t[n + 1]);
	return e;
}
function $e(e, t, n) {
	e[0] = Math.min(e[0], t), e[1] = Math.min(e[1], n), e[2] = Math.max(e[2], t), e[3] = Math.max(e[3], n);
}
function et(e, t) {
	let n;
	return n = t(nt(e)), n || (n = t(rt(e)), n) || (n = t(ft(e)), n) || (n = t(dt(e)), n) ? n : !1;
}
function tt(e) {
	let t = 0;
	return mt(e) || (t = I(e) * ct(e)), t;
}
function nt(e) {
	return [e[0], e[1]];
}
function rt(e) {
	return [e[2], e[1]];
}
function it(e) {
	return [(e[0] + e[2]) / 2, (e[1] + e[3]) / 2];
}
function at(e, t) {
	let n;
	if (t === "bottom-left") n = nt(e);
	else if (t === "bottom-right") n = rt(e);
	else if (t === "top-left") n = dt(e);
	else if (t === "top-right") n = ft(e);
	else throw Error("Invalid corner");
	return n;
}
function ot(e, t, n, r, i) {
	let [a, o, s, c, l, u, d, f] = st(e, t, n, r);
	return Ge(Math.min(a, s, l, d), Math.min(o, c, u, f), Math.max(a, s, l, d), Math.max(o, c, u, f), i);
}
function st(e, t, n, r) {
	let i = t * r[0] / 2, a = t * r[1] / 2, o = Math.cos(n), s = Math.sin(n), c = i * o, l = i * s, u = a * o, d = a * s, f = e[0], p = e[1];
	return [
		f - c + d,
		p - l - u,
		f - c - d,
		p - l + u,
		f + c - d,
		p + l + u,
		f + c + d,
		p + l - u,
		f - c + d,
		p - l - u
	];
}
function ct(e) {
	return e[3] - e[1];
}
function lt(e, t, n) {
	let r = n || We();
	return pt(e, t) ? (r[0] = e[0] > t[0] ? e[0] : t[0], r[1] = e[1] > t[1] ? e[1] : t[1], r[2] = e[2] < t[2] ? e[2] : t[2], r[3] = e[3] < t[3] ? e[3] : t[3]) : Ke(r), r;
}
function ut(e, t) {
	if (!pt(e, t)) return [e.slice()];
	if (Ve(t, e)) return [];
	let [n, r, i, a] = e, o = Math.max(n, t[0]), s = Math.max(r, t[1]), c = Math.min(i, t[2]), l = Math.min(a, t[3]), u = [];
	return o > n && u.push([
		n,
		r,
		o,
		a
	]), c < i && u.push([
		c,
		r,
		i,
		a
	]), s > r && u.push([
		o,
		r,
		c,
		s
	]), l < a && u.push([
		o,
		l,
		c,
		a
	]), u;
}
function dt(e) {
	return [e[0], e[3]];
}
function ft(e) {
	return [e[2], e[3]];
}
function I(e) {
	return e[2] - e[0];
}
function pt(e, t) {
	return e[0] <= t[2] && e[2] >= t[0] && e[1] <= t[3] && e[3] >= t[1];
}
function mt(e) {
	return e[2] < e[0] || e[3] < e[1];
}
function ht(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e;
}
function gt(e, t) {
	let n = (e[2] - e[0]) / 2 * (t - 1), r = (e[3] - e[1]) / 2 * (t - 1);
	e[0] -= n, e[2] += n, e[1] -= r, e[3] += r;
}
function _t(e, t, n) {
	let r = !1, i = Ue(e, t), a = Ue(e, n);
	if (i === Pe.INTERSECTING || a === Pe.INTERSECTING) r = !0;
	else {
		let o = e[0], s = e[1], c = e[2], l = e[3], u = t[0], d = t[1], f = n[0], p = n[1], m = (p - d) / (f - u), h, g;
		a & Pe.ABOVE && !(i & Pe.ABOVE) && (h = f - (p - l) / m, r = h >= o && h <= c), !r && a & Pe.RIGHT && !(i & Pe.RIGHT) && (g = p - (f - c) * m, r = g >= s && g <= l), !r && a & Pe.BELOW && !(i & Pe.BELOW) && (h = f - (p - s) / m, r = h >= o && h <= c), !r && a & Pe.LEFT && !(i & Pe.LEFT) && (g = p - (f - o) * m, r = g >= s && g <= l);
	}
	return r;
}
function vt(e, t, n, r) {
	if (mt(e)) return Ke(n);
	let i = [];
	if (r > 1) {
		let t = e[2] - e[0], n = e[3] - e[1];
		for (let a = 0; a < r; ++a) i.push(e[0] + t * a / r, e[1], e[2], e[1] + n * a / r, e[2] - t * a / r, e[3], e[0], e[3] - n * a / r);
	} else i = [
		e[0],
		e[1],
		e[2],
		e[1],
		e[2],
		e[3],
		e[0],
		e[3]
	];
	t(i, i, 2);
	let a = [], o = [];
	for (let e = 0, t = i.length; e < t; e += 2) a.push(i[e]), o.push(i[e + 1]);
	return Ie(a, o, n);
}
function yt(e, t) {
	let n = t.getExtent(), r = it(e);
	if (t.canWrapX() && (r[0] < n[0] || r[0] >= n[2])) {
		let t = I(n), i = Math.floor((r[0] - n[0]) / t) * t;
		e[0] -= i, e[2] -= i;
	}
	return e;
}
function bt(e, t, n) {
	if (t.canWrapX()) {
		let r = t.getExtent();
		if (!isFinite(e[0]) || !isFinite(e[2])) return [[
			r[0],
			e[1],
			r[2],
			e[3]
		]];
		yt(e, t);
		let i = I(r);
		if (I(e) > i && !n) return [[
			r[0],
			e[1],
			r[2],
			e[3]
		]];
		if (e[0] < r[0]) return [[
			e[0] + i,
			e[1],
			r[2],
			e[3]
		], [
			r[0],
			e[1],
			e[2],
			e[3]
		]];
		if (e[2] > r[2]) return [[
			e[0],
			e[1],
			r[2],
			e[3]
		], [
			r[0],
			e[1],
			e[2] - i,
			e[3]
		]];
	}
	return [e];
}
function xt(e, t) {
	let n = [e];
	for (let e = 0, r = t.length; e < r && n.length > 0; ++e) {
		let r = [];
		for (let i = 0, a = n.length; i < a; ++i) r.push(...ut(n[i], t[e]));
		n = r;
	}
	return n;
}
//#endregion
//#region node_modules/ol/math.js
function St(e, t, n) {
	return Math.min(Math.max(e, t), n);
}
function Ct(e, t, n, r, i, a) {
	let o = i - n, s = a - r;
	if (o !== 0 || s !== 0) {
		let c = ((e - n) * o + (t - r) * s) / (o * o + s * s);
		c > 1 ? (n = i, r = a) : c > 0 && (n += o * c, r += s * c);
	}
	return wt(e, t, n, r);
}
function wt(e, t, n, r) {
	let i = n - e, a = r - t;
	return i * i + a * a;
}
function Tt(e) {
	let t = e.length;
	for (let n = 0; n < t; n++) {
		let r = n, i = Math.abs(e[n][n]);
		for (let a = n + 1; a < t; a++) {
			let t = Math.abs(e[a][n]);
			t > i && (i = t, r = a);
		}
		if (i === 0) return null;
		let a = e[r];
		e[r] = e[n], e[n] = a;
		for (let r = n + 1; r < t; r++) {
			let i = -e[r][n] / e[n][n];
			for (let a = n; a < t + 1; a++) n == a ? e[r][a] = 0 : e[r][a] += i * e[n][a];
		}
	}
	let n = Array(t);
	for (let r = t - 1; r >= 0; r--) {
		n[r] = e[r][t] / e[r][r];
		for (let i = r - 1; i >= 0; i--) e[i][t] -= e[i][r] * n[r];
	}
	return n;
}
function Et(e) {
	return e * 180 / Math.PI;
}
function Dt(e) {
	return e * Math.PI / 180;
}
function Ot(e, t) {
	let n = e % t;
	return n * t < 0 ? n + t : n;
}
function kt(e, t, n) {
	return e + n * (t - e);
}
function At(e, t) {
	let n = 10 ** t;
	return Math.round(e * n) / n;
}
function jt(e, t) {
	return Math.floor(At(e, t));
}
function Mt(e, t) {
	return Math.ceil(At(e, t));
}
function Nt(e, t, n) {
	if (e >= t && e < n) return e;
	let r = n - t;
	return ((e - t) % r + r) % r + t;
}
//#endregion
//#region node_modules/ol/coordinate.js
function Pt(e, t) {
	return e[0] += +t[0], e[1] += +t[1], e;
}
function Ft(e, t, n) {
	return e ? t.replace("{x}", e[0].toFixed(n)).replace("{y}", e[1].toFixed(n)) : "";
}
function It(e, t) {
	let n = !0;
	for (let r = e.length - 1; r >= 0; --r) if (e[r] != t[r]) {
		n = !1;
		break;
	}
	return n;
}
function Lt(e, t) {
	let n = Math.cos(t), r = Math.sin(t), i = e[0] * n - e[1] * r, a = e[1] * n + e[0] * r;
	return e[0] = i, e[1] = a, e;
}
function Rt(e, t) {
	return e[0] *= t, e[1] *= t, e;
}
function zt(e, t) {
	if (t.canWrapX()) {
		let n = I(t.getExtent()), r = Bt(e, t, n);
		r && (e[0] -= r * n);
	}
	return e;
}
function Bt(e, t, n) {
	let r = t.getExtent(), i = 0;
	return t.canWrapX() && (e[0] < r[0] || e[0] > r[2]) && (n ||= I(r), i = Math.floor((e[0] - r[0]) / n)), i;
}
function Vt(e, t, n) {
	let r = Math.sqrt((t[0] - e[0]) * (t[0] - e[0]) + (t[1] - e[1]) * (t[1] - e[1])), i = [(t[0] - e[0]) / r, (t[1] - e[1]) / r], a = [-i[1], i[0]], o = Math.sqrt((n[0] - e[0]) * (n[0] - e[0]) + (n[1] - e[1]) * (n[1] - e[1])), s = [(n[0] - e[0]) / o, (n[1] - e[1]) / o], c = r === 0 || o === 0 ? 0 : Math.acos(St(s[0] * i[0] + s[1] * i[1], -1, 1));
	return c = Math.max(c, 1e-5), s[0] * a[0] + s[1] * a[1] > 0 ? c : Math.PI * 2 - c;
}
//#endregion
//#region node_modules/ol/pointer/EventType.js
var Ht = {
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, Ut = {
	info: 1,
	warn: 2,
	error: 3,
	none: 4
}, Wt = Ut.info;
function Gt(...e) {
	Wt > Ut.warn || console.warn(...e);
}
//#endregion
//#region node_modules/ol/proj/Units.js
var Kt = {
	radians: 6370997 / (2 * Math.PI),
	degrees: 2 * Math.PI * 6370997 / 360,
	ft: .3048,
	m: 1,
	"us-ft": 1200 / 3937
}, qt = class {
	constructor(e) {
		this.code_ = e.code, this.units_ = e.units, this.extent_ = e.extent === void 0 ? null : e.extent, this.worldExtent_ = e.worldExtent === void 0 ? null : e.worldExtent, this.axisOrientation_ = e.axisOrientation === void 0 ? "enu" : e.axisOrientation, this.global_ = e.global !== void 0 && e.global, this.canWrapX_ = !!(this.global_ && this.extent_), this.getPointResolutionFunc_ = e.getPointResolution, this.defaultTileGrid_ = null, this.metersPerUnit_ = e.metersPerUnit;
	}
	canWrapX() {
		return this.canWrapX_;
	}
	getCode() {
		return this.code_;
	}
	getExtent() {
		return this.extent_;
	}
	getUnits() {
		return this.units_;
	}
	getMetersPerUnit() {
		return this.metersPerUnit_ || Kt[this.units_];
	}
	getWorldExtent() {
		return this.worldExtent_;
	}
	getAxisOrientation() {
		return this.axisOrientation_;
	}
	isGlobal() {
		return this.global_;
	}
	setGlobal(e) {
		this.global_ = e, this.canWrapX_ = !!(e && this.extent_);
	}
	getDefaultTileGrid() {
		return this.defaultTileGrid_;
	}
	setDefaultTileGrid(e) {
		this.defaultTileGrid_ = e;
	}
	setExtent(e) {
		this.extent_ = e, this.canWrapX_ = !!(this.global_ && e);
	}
	setWorldExtent(e) {
		this.worldExtent_ = e;
	}
	setGetPointResolution(e) {
		this.getPointResolutionFunc_ = e;
	}
	getPointResolutionFunc() {
		return this.getPointResolutionFunc_;
	}
}, Jt = 6378137, Yt = Math.PI * Jt, Xt = [
	-Yt,
	-Yt,
	Yt,
	Yt
], Zt = [
	-180,
	-85,
	180,
	85
], Qt = Jt * Math.log(Math.tan(Math.PI / 2)), $t = class extends qt {
	constructor(e) {
		super({
			code: e,
			units: "m",
			extent: Xt,
			global: !0,
			worldExtent: Zt,
			getPointResolution: function(e, t) {
				return e / Math.cosh(t[1] / Jt);
			}
		});
	}
}, en = [
	new $t("EPSG:3857"),
	new $t("EPSG:102100"),
	new $t("EPSG:102113"),
	new $t("EPSG:900913"),
	new $t("http://www.opengis.net/def/crs/EPSG/0/3857"),
	new $t("http://www.opengis.net/gml/srs/epsg.xml#3857")
];
function tn(e, t, n, r) {
	let i = e.length;
	n = n > 1 ? n : 2, r ??= n, t === void 0 && (t = n > 2 ? e.slice() : Array(i));
	for (let n = 0; n < i; n += r) {
		t[n] = Yt * e[n] / 180;
		let r = Jt * Math.log(Math.tan(Math.PI * (+e[n + 1] + 90) / 360));
		r > Qt ? r = Qt : r < -Qt && (r = -Qt), t[n + 1] = r;
	}
	return t;
}
function nn(e, t, n, r) {
	let i = e.length;
	n = n > 1 ? n : 2, r ??= n, t === void 0 && (t = n > 2 ? e.slice() : Array(i));
	for (let n = 0; n < i; n += r) t[n] = 180 * e[n] / Yt, t[n + 1] = 360 * Math.atan(Math.exp(e[n + 1] / Jt)) / Math.PI - 90;
	return t;
}
//#endregion
//#region node_modules/ol/proj/epsg4326.js
var rn = 6378137, an = [
	-180,
	-90,
	180,
	90
], on = Math.PI * rn / 180, sn = class extends qt {
	constructor(e, t) {
		super({
			code: e,
			units: "degrees",
			extent: an,
			axisOrientation: t,
			global: !0,
			metersPerUnit: on,
			worldExtent: an
		});
	}
}, cn = [
	new sn("CRS:84"),
	new sn("EPSG:4326", "neu"),
	new sn("urn:ogc:def:crs:OGC:1.3:CRS84"),
	new sn("urn:ogc:def:crs:OGC:2:84"),
	new sn("http://www.opengis.net/def/crs/OGC/1.3/CRS84"),
	new sn("http://www.opengis.net/gml/srs/epsg.xml#4326", "neu"),
	new sn("http://www.opengis.net/def/crs/EPSG/0/4326", "neu")
], ln = {};
function un(e) {
	return ln[e] || ln[e.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\w+)$/, "EPSG:$3")] || null;
}
function dn(e, t) {
	ln[e] = t;
}
//#endregion
//#region node_modules/ol/proj/transforms.js
var fn = {};
function pn(e, t, n) {
	let r = e.getCode(), i = t.getCode();
	r in fn || (fn[r] = {}), fn[r][i] = n;
}
function mn(e, t) {
	return e in fn && t in fn[e] ? fn[e][t] : null;
}
//#endregion
//#region node_modules/ol/proj/utm.js
var hn = .9996, gn = .00669438, _n = gn * gn, vn = _n * gn, yn = gn / .99330562, bn = Math.sqrt(.99330562), xn = (1 - bn) / (1 + bn), Sn = xn * xn, Cn = Sn * xn, wn = Cn * xn, Tn = wn * xn, En = 1 - gn / 4 - 3 * _n / 64 - 5 * vn / 256, Dn = .002514607064228144, On = 26390466021299826e-22, kn = 35 * vn / 3072, An = 3 / 2 * xn - 27 / 32 * Cn + 269 / 512 * Tn, jn = 21 / 16 * Sn - 55 / 32 * wn, Mn = 151 / 96 * Cn - 417 / 128 * Tn, Nn = 1097 / 512 * wn, Pn = 6378137;
function Fn(e, t, n) {
	let r = e - 5e5, i = (n.north ? t : t - 1e7) / hn / (Pn * En), a = i + An * Math.sin(2 * i) + jn * Math.sin(4 * i) + Mn * Math.sin(6 * i) + Nn * Math.sin(8 * i), o = Math.sin(a), s = o * o, c = Math.cos(a), l = o / c, u = l * l, d = u * u, f = 1 - gn * s, p = Pn / Math.sqrt(1 - gn * s), m = .99330562 / f, h = yn * c ** 2, g = h * h, _ = r / (p * hn), v = _ * _, y = v * _, b = y * _, x = b * _, S = x * _, C = a - l / m * (v / 2 - b / 24 * (5 + 3 * u + 10 * h - 4 * g - 9 * yn)) + S / 720 * (61 + 90 * u + 298 * h + 45 * d - 252 * yn - 3 * g), w = (_ - y / 6 * (1 + 2 * u + h) + x / 120 * (5 - 2 * h + 28 * u - 3 * g + 8 * yn + 24 * d)) / c;
	return w = Nt(w + Dt(Vn(n.number)), -Math.PI, Math.PI), [Et(w), Et(C)];
}
var In = -80, Ln = 84, Rn = -180, zn = 180;
function Bn(e, t, n) {
	e = Nt(e, Rn, zn), t < In ? t = In : t > Ln && (t = Ln);
	let r = Dt(t), i = Math.sin(r), a = Math.cos(r), o = i / a, s = o * o, c = s * s, l = Dt(e), u = Dt(Vn(n.number)), d = Pn / Math.sqrt(1 - gn * i ** 2), f = yn * a ** 2, p = a * Nt(l - u, -Math.PI, Math.PI), m = p * p, h = m * p, g = h * p, _ = g * p, v = _ * p, y = Pn * (En * r - Dn * Math.sin(2 * r) + On * Math.sin(4 * r) - kn * Math.sin(6 * r)), b = hn * d * (p + h / 6 * (1 - s + f) + _ / 120 * (5 - 18 * s + c + 72 * f - 58 * yn)) + 5e5, x = hn * (y + d * o * (m / 2 + g / 24 * (5 - s + 9 * f + 4 * f ** 2) + v / 720 * (61 - 58 * s + c + 600 * f - 330 * yn)));
	return n.north || (x += 1e7), [b, x];
}
function Vn(e) {
	return (e - 1) * 6 - 180 + 3;
}
var Hn = [
	/^EPSG:(\d+)$/,
	/^urn:ogc:def:crs:EPSG::(\d+)$/,
	/^http:\/\/www\.opengis\.net\/def\/crs\/EPSG\/0\/(\d+)$/
];
function Un(e) {
	let t = 0;
	for (let n of Hn) {
		let r = e.match(n);
		if (r) {
			t = parseInt(r[1]);
			break;
		}
	}
	if (!t) return null;
	let n = 0, r = !1;
	return t > 32700 && t < 32761 ? n = t - 32700 : t > 32600 && t < 32661 && (r = !0, n = t - 32600), n ? {
		number: n,
		north: r
	} : null;
}
function Wn(e, t) {
	return function(n, r, i, a) {
		let o = n.length;
		i = i > 1 ? i : 2, a ??= i, r ||= i > 2 ? n.slice() : Array(o);
		for (let i = 0; i < o; i += a) {
			let a = n[i], o = n[i + 1], s = e(a, o, t);
			r[i] = s[0], r[i + 1] = s[1];
		}
		return r;
	};
}
function Gn(e) {
	return Un(e) ? new qt({
		code: e,
		units: "m"
	}) : null;
}
function Kn(e) {
	let t = Un(e.getCode());
	return t ? {
		forward: Wn(Bn, t),
		inverse: Wn(Fn, t)
	} : null;
}
function qn(e, t, n) {
	n ||= 6371008.8;
	let r = Dt(e[1]), i = Dt(t[1]), a = (i - r) / 2, o = Dt(t[0] - e[0]) / 2, s = Math.sin(a) * Math.sin(a) + Math.sin(o) * Math.sin(o) * Math.cos(r) * Math.cos(i);
	return 2 * n * Math.atan2(Math.sqrt(s), Math.sqrt(1 - s));
}
//#endregion
//#region node_modules/ol/proj.js
var Jn = [Kn], Yn = [Gn], Xn = !0;
function Zn(e) {
	Xn = !(e === void 0 || e);
}
function Qn(e, t) {
	if (t !== void 0) {
		for (let n = 0, r = e.length; n < r; ++n) t[n] = e[n];
		t = t;
	} else t = e.slice();
	return t;
}
function $n(e, t) {
	if (t !== void 0 && e !== t) {
		for (let n = 0, r = e.length; n < r; ++n) t[n] = e[n];
		e = t;
	}
	return e;
}
function er(e) {
	dn(e.getCode(), e), pn(e, e, Qn);
}
function tr(e) {
	e.forEach(er);
}
function nr(e) {
	if (typeof e != "string") return e;
	let t = un(e);
	if (t) return t;
	for (let t of Yn) {
		let n = t(e);
		if (n) return n;
	}
	return null;
}
function rr(e, t, n, r) {
	e = nr(e);
	let i, a = e.getPointResolutionFunc();
	if (a) {
		if (i = a(t, n), r && r !== e.getUnits()) {
			let t = e.getMetersPerUnit();
			t && (i = i * t / Kt[r]);
		}
	} else {
		let a = e.getUnits();
		if (a == "degrees" && !r || r == "degrees") i = t;
		else {
			let o = lr(e, nr("EPSG:4326"));
			if (!o && a !== "degrees") i = t * e.getMetersPerUnit();
			else {
				let e = [
					n[0] - t / 2,
					n[1],
					n[0] + t / 2,
					n[1],
					n[0],
					n[1] - t / 2,
					n[0],
					n[1] + t / 2
				];
				e = o(e, e, 2), i = (qn(e.slice(0, 2), e.slice(2, 4)) + qn(e.slice(4, 6), e.slice(6, 8))) / 2;
			}
			let s = r ? Kt[r] : e.getMetersPerUnit();
			s !== void 0 && (i /= s);
		}
	}
	return i;
}
function ir(e) {
	tr(e), e.forEach(function(t) {
		e.forEach(function(e) {
			t !== e && pn(t, e, Qn);
		});
	});
}
function ar(e, t, n, r) {
	e.forEach(function(e) {
		t.forEach(function(t) {
			pn(e, t, n), pn(t, e, r);
		});
	});
}
function or(e, t) {
	return e ? typeof e == "string" ? nr(e) : e : nr(t);
}
function sr(e) {
	return (function(t, n, r, i) {
		let a = t.length;
		r = r === void 0 ? 2 : r, i ??= r, n = n === void 0 ? Array(a) : n;
		for (let o = 0; o < a; o += i) {
			let a = e(t.slice(o, o + r)), s = a.length;
			for (let e = 0, r = i; e < r; ++e) n[o + e] = e >= s ? t[o + e] : a[e];
		}
		return n;
	});
}
function cr(e, t) {
	if (e === t) return !0;
	let n = e.getUnits() === t.getUnits();
	return (e.getCode() === t.getCode() || lr(e, t) === Qn) && n;
}
function lr(e, t) {
	let n = e.getCode(), r = t.getCode(), i = mn(n, r);
	if (i) return i;
	let a = null, o = null;
	for (let n of Jn) a ||= n(e), o ||= n(t);
	if (!a && !o) return null;
	let s = "EPSG:4326";
	if (!o) {
		let e = mn(s, r);
		e && (i = ur(a.inverse, e));
	} else if (a) i = ur(a.inverse, o.forward);
	else {
		let e = mn(n, s);
		e && (i = ur(e, o.forward));
	}
	return i && (er(e), er(t), pn(e, t, i)), i;
}
function ur(e, t) {
	return function(n, r, i, a) {
		return r = e(n, r, i, a), t(r, r, i, a);
	};
}
function dr(e, t) {
	return lr(nr(e), nr(t));
}
function fr(e, t, n) {
	let r = dr(t, n);
	if (!r) {
		let e = nr(t).getCode(), r = nr(n).getCode();
		throw Error(`No transform available between ${e} and ${r}`);
	}
	return r(e, void 0, e.length);
}
function pr(e, t, n, r) {
	return vt(e, dr(t, n), void 0, r);
}
var mr = null;
function hr() {
	return mr;
}
function gr(e, t) {
	return e;
}
function _r(e, t) {
	return Xn && !It(e, [0, 0]) && e[0] >= -180 && e[0] <= 180 && e[1] >= -90 && e[1] <= 90 && (Xn = !1, Gt("Call useGeographic() from ol/proj once to work with [longitude, latitude] coordinates.")), e;
}
function vr(e, t) {
	return e;
}
function yr(e, t) {
	return e;
}
function br(e, t) {
	return e;
}
function xr() {
	ir(en), ir(cn), ar(cn, en, tn, nn);
}
xr();
//#endregion
//#region node_modules/ol/control/MousePosition.js
var Sr = "projection", Cr = "coordinateFormat", wr = class extends Te {
	constructor(e) {
		e ||= {};
		let t = document.createElement("div");
		t.className = e.className === void 0 ? "ol-mouse-position" : e.className, super({
			element: t,
			render: e.render,
			target: e.target
		}), this.on, this.once, this.un, this.addChangeListener(Sr, this.handleProjectionChanged_), e.coordinateFormat && this.setCoordinateFormat(e.coordinateFormat), e.projection && this.setProjection(e.projection), this.renderOnMouseOut_ = e.placeholder !== void 0, this.placeholder_ = this.renderOnMouseOut_ ? e.placeholder : "&#160;", this.renderedHTML_ = t.innerHTML, this.mapProjection_ = null, this.transform_ = null, this.wrapX_ = e.wrapX !== !1;
	}
	handleProjectionChanged_() {
		this.transform_ = null;
	}
	getCoordinateFormat() {
		return this.get(Cr);
	}
	getProjection() {
		return this.get(Sr);
	}
	handleMouseMove(e) {
		let t = this.getMap();
		this.updateHTML_(t.getEventPixel(e));
	}
	handleMouseOut(e) {
		this.updateHTML_(null);
	}
	setMap(e) {
		if (super.setMap(e), e) {
			let t = e.getViewport();
			this.listenerKeys.push(A(t, Ht.POINTERMOVE, this.handleMouseMove, this)), this.renderOnMouseOut_ && this.listenerKeys.push(A(t, Ht.POINTEROUT, this.handleMouseOut, this)), this.updateHTML_(null);
		}
	}
	setCoordinateFormat(e) {
		this.set(Cr, e);
	}
	setProjection(e) {
		this.set(Sr, nr(e));
	}
	updateHTML_(e) {
		let t = this.placeholder_;
		if (e && this.mapProjection_) {
			if (!this.transform_) {
				let e = this.getProjection();
				this.transform_ = e ? lr(this.mapProjection_, e) : $n;
			}
			let n = this.getMap().getCoordinateFromPixelInternal(e);
			if (n) {
				let e = hr();
				e && (this.transform_ = lr(this.mapProjection_, e)), this.transform_(n, n), this.wrapX_ && zt(n, e || this.getProjection() || this.mapProjection_);
				let r = this.getCoordinateFormat();
				t = r ? r(n) : n.toString();
			}
		}
		(!this.renderedHTML_ || t !== this.renderedHTML_) && (this.element.innerHTML = t, this.renderedHTML_ = t);
	}
	render(e) {
		let t = e.frameState;
		t ? this.mapProjection_ != t.viewState.projection && (this.mapProjection_ = t.viewState.projection, this.transform_ = null) : this.mapProjection_ = null;
	}
}, Tr = {
	ADD: "add",
	REMOVE: "remove"
}, Er = { LENGTH: "length" }, Dr = class extends ve {
	constructor(e, t, n) {
		super(e), this.element = t, this.index = n;
	}
}, Or = class extends we {
	constructor(e, t) {
		if (super(), this.on, this.once, this.un, t ||= {}, this.unique_ = !!t.unique, this.array_ = e ?? [], this.unique_) for (let e = 1, t = this.array_.length; e < t; ++e) this.assertUnique_(this.array_[e], e);
		this.updateLength_();
	}
	clear() {
		for (; this.getLength() > 0;) this.pop();
	}
	extend(e) {
		for (let t = 0, n = e.length; t < n; ++t) this.push(e[t]);
		return this;
	}
	forEach(e) {
		let t = this.array_;
		for (let n = 0, r = t.length; n < r; ++n) e(t[n], n, t);
	}
	getArray() {
		return this.array_;
	}
	item(e) {
		return this.array_[e];
	}
	getLength() {
		return this.get(Er.LENGTH);
	}
	insertAt(e, t) {
		if (e < 0 || e > this.getLength()) throw Error("Index out of bounds: " + e);
		this.unique_ && this.assertUnique_(t), this.array_.splice(e, 0, t), this.updateLength_(), this.dispatchEvent(new Dr(Tr.ADD, t, e));
	}
	pop() {
		return this.removeAt(this.getLength() - 1);
	}
	push(e) {
		let t = this.getLength();
		return this.insertAt(t, e), this.getLength();
	}
	remove(e) {
		let t = this.array_;
		for (let n = 0, r = t.length; n < r; ++n) if (t[n] === e) return this.removeAt(n);
	}
	removeAt(e) {
		if (e < 0 || e >= this.getLength()) return;
		let t = this.array_[e];
		return this.array_.splice(e, 1), this.updateLength_(), this.dispatchEvent(new Dr(Tr.REMOVE, t, e)), t;
	}
	setAt(e, t) {
		if (e >= this.getLength()) {
			this.insertAt(e, t);
			return;
		}
		if (e < 0) throw Error("Index out of bounds: " + e);
		this.unique_ && this.assertUnique_(t, e);
		let n = this.array_[e];
		this.array_[e] = t, this.dispatchEvent(new Dr(Tr.REMOVE, n, e)), this.dispatchEvent(new Dr(Tr.ADD, t, e));
	}
	updateLength_() {
		this.set(Er.LENGTH, this.array_.length);
	}
	assertUnique_(e, t) {
		let n = this.array_;
		for (let r = 0, i = n.length; r < i; ++r) if (n[r] === e && r !== t) throw Error("Duplicate item added to a unique collection");
	}
}, kr = class extends ve {
	constructor(e, t, n) {
		super(e), this.map = t, this.frameState = n === void 0 ? null : n;
	}
}, Ar = class extends kr {
	constructor(e, t, n, r, i, a) {
		super(e, t, i), this.originalEvent = n, this.pixel_ = null, this.coordinate_ = null, this.dragging = r !== void 0 && r, this.activePointers = a;
	}
	get pixel() {
		return this.pixel_ ||= this.map.getEventPixel(this.originalEvent), this.pixel_;
	}
	set pixel(e) {
		this.pixel_ = e;
	}
	get coordinate() {
		return this.coordinate_ ||= this.map.getCoordinateFromPixel(this.pixel), this.coordinate_;
	}
	set coordinate(e) {
		this.coordinate_ = e;
	}
	preventDefault() {
		super.preventDefault(), "preventDefault" in this.originalEvent && this.originalEvent.preventDefault();
	}
	stopPropagation() {
		super.stopPropagation(), "stopPropagation" in this.originalEvent && this.originalEvent.stopPropagation();
	}
}, jr = {
	SINGLECLICK: "singleclick",
	CLICK: M.CLICK,
	DBLCLICK: M.DBLCLICK,
	POINTERDRAG: "pointerdrag",
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, Mr = class extends ye {
	constructor(e, t) {
		super(e), this.map_ = e, this.clickTimeoutId_, this.emulateClicks_ = !1, this.dragging_ = !1, this.dragListenerKeys_ = [], this.moveTolerance_ = t === void 0 ? 1 : t, this.down_ = null;
		let n = this.map_.getViewport();
		this.activePointers_ = [], this.trackedTouches_ = {}, this.element_ = n, this.pointerdownListenerKey_ = A(n, Ht.POINTERDOWN, this.handlePointerDown_, this), this.originalPointerMoveEvent_, this.relayedListenerKey_ = A(n, Ht.POINTERMOVE, this.relayMoveEvent_, this), this.boundHandleTouchMove_ = this.handleTouchMove_.bind(this), this.element_.addEventListener(M.TOUCHMOVE, this.boundHandleTouchMove_, v ? { passive: !1 } : !1);
	}
	emulateClick_(e) {
		let t = new Ar(jr.CLICK, this.map_, e);
		this.dispatchEvent(t), this.clickTimeoutId_ === void 0 ? this.clickTimeoutId_ = setTimeout(() => {
			this.clickTimeoutId_ = void 0;
			let t = new Ar(jr.SINGLECLICK, this.map_, e);
			this.dispatchEvent(t);
		}, 250) : (clearTimeout(this.clickTimeoutId_), this.clickTimeoutId_ = void 0, t = new Ar(jr.DBLCLICK, this.map_, e), this.dispatchEvent(t));
	}
	updateActivePointers_(e) {
		let t = e, n = t.pointerId;
		if (t.type == jr.POINTERUP || t.type == jr.POINTERCANCEL) {
			delete this.trackedTouches_[n];
			for (let e in this.trackedTouches_) if (this.trackedTouches_[e].target !== t.target) {
				delete this.trackedTouches_[e];
				break;
			}
		} else (t.type == jr.POINTERDOWN || t.type == jr.POINTERMOVE) && (this.trackedTouches_[n] = t);
		this.activePointers_ = Object.values(this.trackedTouches_);
	}
	handlePointerUp_(e) {
		this.updateActivePointers_(e);
		let t = new Ar(jr.POINTERUP, this.map_, e, void 0, void 0, this.activePointers_);
		this.dispatchEvent(t), this.emulateClicks_ && !t.defaultPrevented && !this.dragging_ && this.isMouseActionButton_(e) && this.emulateClick_(this.down_), this.activePointers_.length === 0 && (this.dragListenerKeys_.forEach(j), this.dragListenerKeys_.length = 0, this.dragging_ = !1, this.down_ = null);
	}
	isMouseActionButton_(e) {
		return e.button === 0;
	}
	handlePointerDown_(e) {
		this.emulateClicks_ = this.activePointers_.length === 0, this.updateActivePointers_(e);
		let t = new Ar(jr.POINTERDOWN, this.map_, e, void 0, void 0, this.activePointers_);
		if (this.dispatchEvent(t), this.down_ = new PointerEvent(e.type, e), Object.defineProperty(this.down_, "target", {
			writable: !1,
			value: e.target
		}), this.dragListenerKeys_.length === 0) {
			let e = this.map_.getOwnerDocument();
			this.dragListenerKeys_.push(A(e, jr.POINTERMOVE, this.handlePointerMove_, this), A(e, jr.POINTERUP, this.handlePointerUp_, this), A(this.element_, jr.POINTERCANCEL, this.handlePointerUp_, this)), this.element_.getRootNode && this.element_.getRootNode() !== e && this.dragListenerKeys_.push(A(this.element_.getRootNode(), jr.POINTERUP, this.handlePointerUp_, this));
		}
	}
	handlePointerMove_(e) {
		if (this.isMoving_(e)) {
			this.updateActivePointers_(e), this.dragging_ = !0;
			let t = new Ar(jr.POINTERDRAG, this.map_, e, this.dragging_, void 0, this.activePointers_);
			this.dispatchEvent(t);
		}
	}
	relayMoveEvent_(e) {
		this.originalPointerMoveEvent_ = e;
		let t = !!(this.down_ && this.isMoving_(e));
		this.dispatchEvent(new Ar(jr.POINTERMOVE, this.map_, e, t));
	}
	handleTouchMove_(e) {
		let t = this.originalPointerMoveEvent_;
		(!t || t.defaultPrevented) && (typeof e.cancelable != "boolean" || e.cancelable === !0) && e.preventDefault();
	}
	isMoving_(e) {
		return this.dragging_ || Math.abs(e.clientX - this.down_.clientX) > this.moveTolerance_ || Math.abs(e.clientY - this.down_.clientY) > this.moveTolerance_;
	}
	disposeInternal() {
		this.relayedListenerKey_ &&= (j(this.relayedListenerKey_), null), this.element_.removeEventListener(M.TOUCHMOVE, this.boundHandleTouchMove_), this.pointerdownListenerKey_ &&= (j(this.pointerdownListenerKey_), null), this.dragListenerKeys_.forEach(j), this.dragListenerKeys_.length = 0, this.element_ = null, super.disposeInternal();
	}
}, L = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
};
//#endregion
//#region node_modules/ol/asserts.js
function R(e, t) {
	if (!e) throw Error(t);
}
//#endregion
//#region node_modules/ol/structs/PriorityQueue.js
var Nr = Infinity, Pr = class {
	constructor(e, t) {
		this.priorityFunction_ = e, this.keyFunction_ = t, this.elements_ = [], this.priorities_ = [], this.queuedElements_ = {};
	}
	clear() {
		this.elements_.length = 0, this.priorities_.length = 0, k(this.queuedElements_);
	}
	dequeue() {
		let e = this.elements_, t = this.priorities_, n = e[0];
		e.length == 1 ? (e.length = 0, t.length = 0) : (e[0] = e.pop(), t[0] = t.pop(), this.siftUp_(0));
		let r = this.keyFunction_(n);
		return delete this.queuedElements_[r], n;
	}
	enqueue(e) {
		R(!(this.keyFunction_(e) in this.queuedElements_), "Tried to enqueue an `element` that was already added to the queue");
		let t = this.priorityFunction_(e);
		return t != Infinity && (this.elements_.push(e), this.priorities_.push(t), this.queuedElements_[this.keyFunction_(e)] = !0, this.siftDown_(0, this.elements_.length - 1), !0);
	}
	getCount() {
		return this.elements_.length;
	}
	getLeftChildIndex_(e) {
		return e * 2 + 1;
	}
	getRightChildIndex_(e) {
		return e * 2 + 2;
	}
	getParentIndex_(e) {
		return e - 1 >> 1;
	}
	heapify_() {
		let e;
		for (e = (this.elements_.length >> 1) - 1; e >= 0; e--) this.siftUp_(e);
	}
	isEmpty() {
		return this.elements_.length === 0;
	}
	isKeyQueued(e) {
		return e in this.queuedElements_;
	}
	isQueued(e) {
		return this.isKeyQueued(this.keyFunction_(e));
	}
	siftUp_(e) {
		let t = this.elements_, n = this.priorities_, r = t.length, i = t[e], a = n[e], o = e;
		for (; e < r >> 1;) {
			let i = this.getLeftChildIndex_(e), a = this.getRightChildIndex_(e), o = a < r && n[a] < n[i] ? a : i;
			t[e] = t[o], n[e] = n[o], e = o;
		}
		t[e] = i, n[e] = a, this.siftDown_(o, e);
	}
	siftDown_(e, t) {
		let n = this.elements_, r = this.priorities_, i = n[t], a = r[t];
		for (; t > e;) {
			let e = this.getParentIndex_(t);
			if (r[e] > a) n[t] = n[e], r[t] = r[e], t = e;
			else break;
		}
		n[t] = i, r[t] = a;
	}
	reprioritize() {
		let e = this.priorityFunction_, t = this.elements_, n = this.priorities_, r = 0, i = t.length, a, o, s;
		for (o = 0; o < i; ++o) a = t[o], s = e(a), s == Infinity ? delete this.queuedElements_[this.keyFunction_(a)] : (n[r] = s, t[r++] = a);
		t.length = r, n.length = r, this.heapify_();
	}
}, Fr = class extends Pr {
	constructor(e, t) {
		super((t) => e.apply(null, t), (e) => e[0].getKey()), this.boundHandleTileChange_ = this.handleTileChange.bind(this), this.tileChangeCallback_ = t, this.tilesLoading_ = 0, this.tilesLoadingKeys_ = {};
	}
	enqueue(e) {
		let t = super.enqueue(e);
		return t && e[0].addEventListener(M.CHANGE, this.boundHandleTileChange_), t;
	}
	getTilesLoading() {
		return this.tilesLoading_;
	}
	handleTileChange(e) {
		let t = e.target, n = t.getState();
		if (n === L.LOADED || n === L.ERROR || n === L.EMPTY) {
			n !== L.ERROR && t.removeEventListener(M.CHANGE, this.boundHandleTileChange_);
			let e = t.getKey();
			e in this.tilesLoadingKeys_ && (delete this.tilesLoadingKeys_[e], --this.tilesLoading_), this.tileChangeCallback_();
		}
	}
	loadMoreTiles(e, t) {
		let n = 0;
		for (; this.tilesLoading_ < e && n < t && this.getCount() > 0;) {
			let e = this.dequeue()[0], t = e.getKey();
			e.getState() === L.IDLE && !(t in this.tilesLoadingKeys_) && (this.tilesLoadingKeys_[t] = !0, ++this.tilesLoading_, ++n, e.load());
		}
	}
};
function Ir(e, t, n, r, i) {
	if (!e || !(n in e.wantedTiles) || !e.wantedTiles[n][t.getKey()]) return Nr;
	let a = e.viewState.center, o = r[0] - a[0], s = r[1] - a[1];
	return 65536 * Math.log(i) + Math.sqrt(o * o + s * s) / i;
}
//#endregion
//#region node_modules/ol/ViewHint.js
var Lr = {
	ANIMATING: 0,
	INTERACTING: 1
}, Rr = {
	CENTER: "center",
	RESOLUTION: "resolution",
	ROTATION: "rotation"
};
//#endregion
//#region node_modules/ol/centerconstraint.js
function zr(e, t, n) {
	return (function(r, i, a, o, s) {
		if (!r) return;
		if (!i && !t) return r;
		let c = t ? 0 : a[0] * i, l = t ? 0 : a[1] * i, u = s ? s[0] : 0, d = s ? s[1] : 0, f = e[0] + c / 2 + u, p = e[2] - c / 2 + u, m = e[1] + l / 2 + d, h = e[3] - l / 2 + d;
		f > p && (f = (p + f) / 2, p = f), m > h && (m = (h + m) / 2, h = m);
		let g = St(r[0], f, p), _ = St(r[1], m, h);
		if (o && n && i) {
			let e = 30 * i;
			g += -e * Math.log(1 + Math.max(0, f - r[0]) / e) + e * Math.log(1 + Math.max(0, r[0] - p) / e), _ += -e * Math.log(1 + Math.max(0, m - r[1]) / e) + e * Math.log(1 + Math.max(0, r[1] - h) / e);
		}
		return [g, _];
	});
}
function Br(e) {
	return e;
}
//#endregion
//#region node_modules/ol/easing.js
function Vr(e) {
	return e ** 3;
}
function Hr(e) {
	return 1 - Vr(1 - e);
}
function Ur(e) {
	return 3 * e * e - 2 * e * e * e;
}
function Wr(e) {
	return e;
}
//#endregion
//#region node_modules/ol/transform.js
var Gr = [
	1,
	0,
	0,
	1,
	0,
	0
], Kr = [
	,
	,
	,
	,
	,
	,
];
function qr() {
	return Gr.slice(0);
}
function Jr(e) {
	return Xr(e, 1, 0, 0, 1, 0, 0);
}
function Yr(e, t) {
	let n = e[0], r = e[1], i = e[2], a = e[3], o = e[4], s = e[5], c = t[0], l = t[1], u = t[2], d = t[3], f = t[4], p = t[5];
	return e[0] = n * c + i * l, e[1] = r * c + a * l, e[2] = n * u + i * d, e[3] = r * u + a * d, e[4] = n * f + i * p + o, e[5] = r * f + a * p + s, e;
}
function Xr(e, t, n, r, i, a, o) {
	return e[0] = t, e[1] = n, e[2] = r, e[3] = i, e[4] = a, e[5] = o, e;
}
function Zr(e, t) {
	return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e;
}
function z(e, t) {
	let n = t[0], r = t[1];
	return t[0] = e[0] * n + e[2] * r + e[4], t[1] = e[1] * n + e[3] * r + e[5], t;
}
function Qr(e, t, n) {
	return Yr(e, Xr(Kr, t, 0, 0, n, 0, 0));
}
function $r(e, t, n) {
	return Yr(e, Xr(Kr, 1, 0, 0, 1, t, n));
}
function ei(e, t, n, r, i, a, o, s) {
	let c = Math.sin(a), l = Math.cos(a);
	return e[0] = r * l, e[1] = i * c, e[2] = -r * c, e[3] = i * l, e[4] = o * r * l - s * r * c + t, e[5] = o * i * c + s * i * l + n, e;
}
function ti(e, t) {
	let n = ni(t);
	R(n !== 0, "Transformation matrix cannot be inverted");
	let r = t[0], i = t[1], a = t[2], o = t[3], s = t[4], c = t[5];
	return e[0] = o / n, e[1] = -i / n, e[2] = -a / n, e[3] = r / n, e[4] = (a * c - o * s) / n, e[5] = -(r * c - i * s) / n, e;
}
function ni(e) {
	return e[0] * e[3] - e[1] * e[2];
}
var ri = [
	1e5,
	1e5,
	1e5,
	1e5,
	2,
	2
];
function ii(e) {
	return "matrix(" + e.join(", ") + ")";
}
function ai(e) {
	return e.substring(7, e.length - 1).split(",").map(parseFloat);
}
function oi(e, t) {
	let n = ai(e), r = ai(t);
	for (let e = 0; e < 6; ++e) if (Math.round((n[e] - r[e]) * ri[e]) !== 0) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/transform.js
function si(e, t, n, r, i, a, o) {
	a ||= [], o ||= 2;
	let s = 0;
	for (let c = t; c < n; c += r) {
		let t = e[c], n = e[c + 1];
		a[s++] = i[0] * t + i[2] * n + i[4], a[s++] = i[1] * t + i[3] * n + i[5];
		for (let t = 2; t < o; t++) a[s++] = e[c + t];
	}
	return a && a.length != s && (a.length = s), a;
}
function ci(e, t, n, r, i, a, o) {
	o ||= [];
	let s = Math.cos(i), c = Math.sin(i), l = a[0], u = a[1], d = 0;
	for (let i = t; i < n; i += r) {
		let t = e[i] - l, n = e[i + 1] - u;
		o[d++] = l + t * s - n * c, o[d++] = u + t * c + n * s;
		for (let t = i + 2; t < i + r; ++t) o[d++] = e[t];
	}
	return o && o.length != d && (o.length = d), o;
}
function li(e, t, n, r, i, a, o, s) {
	s ||= [];
	let c = o[0], l = o[1], u = 0;
	for (let o = t; o < n; o += r) {
		let t = e[o] - c, n = e[o + 1] - l;
		s[u++] = c + i * t, s[u++] = l + a * n;
		for (let t = o + 2; t < o + r; ++t) s[u++] = e[t];
	}
	return s && s.length != u && (s.length = u), s;
}
function ui(e, t, n, r, i, a, o) {
	o ||= [];
	let s = 0;
	for (let c = t; c < n; c += r) {
		o[s++] = e[c] + i, o[s++] = e[c + 1] + a;
		for (let t = c + 2; t < c + r; ++t) o[s++] = e[t];
	}
	return o && o.length != s && (o.length = s), o;
}
//#endregion
//#region node_modules/ol/geom/Geometry.js
var di = qr(), fi = [NaN, NaN], pi = class extends we {
	constructor() {
		super(), this.extent_ = We(), this.extentRevision_ = -1, this.simplifiedGeometryMaxMinSquaredTolerance = 0, this.simplifiedGeometryRevision = 0, this.simplifyTransformedInternal = ge((e, t, n) => {
			if (!n) return this.getSimplifiedGeometry(t);
			let r = this.clone();
			return r.applyTransform(n), r.getSimplifiedGeometry(t);
		});
	}
	simplifyTransformed(e, t) {
		return this.simplifyTransformedInternal(this.getRevision(), e, t);
	}
	clone() {
		return P();
	}
	closestPointXY(e, t, n, r) {
		return P();
	}
	containsXY(e, t) {
		return this.closestPointXY(e, t, fi, Number.MIN_VALUE) === 0;
	}
	getClosestPoint(e, t) {
		return t ||= [NaN, NaN], this.closestPointXY(e[0], e[1], t, Infinity), t;
	}
	intersectsCoordinate(e) {
		return this.containsXY(e[0], e[1]);
	}
	computeExtent(e) {
		return P();
	}
	getExtent(e) {
		if (this.extentRevision_ != this.getRevision()) {
			let e = this.computeExtent(this.extent_);
			(isNaN(e[0]) || isNaN(e[1])) && Ke(e), this.extentRevision_ = this.getRevision();
		}
		return ht(this.extent_, e);
	}
	rotate(e, t) {
		P();
	}
	scale(e, t, n) {
		P();
	}
	simplify(e) {
		return this.getSimplifiedGeometry(e * e);
	}
	getSimplifiedGeometry(e) {
		return P();
	}
	getType() {
		return P();
	}
	applyTransform(e) {
		P();
	}
	intersectsExtent(e) {
		return P();
	}
	translate(e, t) {
		P();
	}
	transform(e, t) {
		let n = nr(e), r = n.getUnits() == "tile-pixels" ? function(e, r, i) {
			let a = n.getExtent(), o = n.getWorldExtent(), s = ct(o) / ct(a);
			ei(di, o[0], o[3], s, -s, 0, 0, 0);
			let c = si(e, 0, e.length, i, di, r), l = dr(n, t);
			return l ? l(c, c, i) : c;
		} : dr(n, t);
		return this.applyTransform(r), this;
	}
}, mi = class extends pi {
	constructor() {
		super(), this.layout = "XY", this.stride = 2, this.flatCoordinates;
	}
	computeExtent(e) {
		return Je(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	getCoordinates() {
		return P();
	}
	getFirstCoordinate() {
		return this.flatCoordinates.slice(0, this.stride);
	}
	getFlatCoordinates() {
		return this.flatCoordinates;
	}
	getLastCoordinate() {
		return this.flatCoordinates.slice(this.flatCoordinates.length - this.stride);
	}
	getLayout() {
		return this.layout;
	}
	getSimplifiedGeometry(e) {
		if (this.simplifiedGeometryRevision !== this.getRevision() && (this.simplifiedGeometryMaxMinSquaredTolerance = 0, this.simplifiedGeometryRevision = this.getRevision()), e < 0 || this.simplifiedGeometryMaxMinSquaredTolerance !== 0 && e <= this.simplifiedGeometryMaxMinSquaredTolerance) return this;
		let t = this.getSimplifiedGeometryInternal(e);
		return t.getFlatCoordinates().length < this.flatCoordinates.length ? t : (this.simplifiedGeometryMaxMinSquaredTolerance = e, this);
	}
	getSimplifiedGeometryInternal(e) {
		return this;
	}
	getStride() {
		return this.stride;
	}
	setFlatCoordinates(e, t) {
		this.stride = gi(e), this.layout = e, this.flatCoordinates = t;
	}
	setCoordinates(e, t) {
		P();
	}
	setLayout(e, t, n) {
		let r;
		if (e) r = gi(e);
		else {
			for (let e = 0; e < n; ++e) {
				if (t.length === 0) {
					this.layout = "XY", this.stride = 2;
					return;
				}
				t = t[0];
			}
			r = t.length, e = hi(r);
		}
		this.layout = e, this.stride = r;
	}
	applyTransform(e) {
		this.flatCoordinates && (e(this.flatCoordinates, this.flatCoordinates, this.layout.startsWith("XYZ") ? 3 : 2, this.stride), this.changed());
	}
	rotate(e, t) {
		let n = this.getFlatCoordinates();
		if (n) {
			let r = this.getStride();
			ci(n, 0, n.length, r, e, t, n), this.changed();
		}
	}
	scale(e, t, n) {
		t === void 0 && (t = e), n ||= it(this.getExtent());
		let r = this.getFlatCoordinates();
		if (r) {
			let i = this.getStride();
			li(r, 0, r.length, i, e, t, n, r), this.changed();
		}
	}
	translate(e, t) {
		let n = this.getFlatCoordinates();
		if (n) {
			let r = this.getStride();
			ui(n, 0, n.length, r, e, t, n), this.changed();
		}
	}
};
function hi(e) {
	let t;
	return e == 2 ? t = "XY" : e == 3 ? t = "XYZ" : e == 4 && (t = "XYZM"), t;
}
function gi(e) {
	let t;
	return e == "XY" ? t = 2 : e == "XYZ" || e == "XYM" ? t = 3 : e == "XYZM" && (t = 4), t;
}
function _i(e, t, n) {
	let r = e.getFlatCoordinates();
	if (!r) return null;
	let i = e.getStride();
	return si(r, 0, r.length, i, t, n);
}
//#endregion
//#region node_modules/ol/geom/flat/area.js
function vi(e, t, n, r) {
	let i = 0, a = e[n - r], o = e[n - r + 1], s = 0, c = 0;
	for (; t < n; t += r) {
		let n = e[t] - a, r = e[t + 1] - o;
		i += c * n - s * r, s = n, c = r;
	}
	return i / 2;
}
function yi(e, t, n, r) {
	let i = 0;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i += vi(e, t, o, r), t = o;
	}
	return i;
}
//#endregion
//#region node_modules/ol/geom/flat/closest.js
function bi(e, t, n, r, i, a, o) {
	let s = e[t], c = e[t + 1], l = e[n] - s, u = e[n + 1] - c, d;
	if (l === 0 && u === 0) d = t;
	else {
		let f = ((i - s) * l + (a - c) * u) / (l * l + u * u);
		if (f > 1) d = n;
		else if (f > 0) {
			for (let i = 0; i < r; ++i) o[i] = kt(e[t + i], e[n + i], f);
			o.length = r;
			return;
		} else d = t;
	}
	for (let t = 0; t < r; ++t) o[t] = e[d + t];
	o.length = r;
}
function xi(e, t, n, r, i) {
	let a = e[t], o = e[t + 1];
	for (t += r; t < n; t += r) {
		let n = e[t], r = e[t + 1], s = wt(a, o, n, r);
		s > i && (i = s), a = n, o = r;
	}
	return i;
}
function Si(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i = xi(e, t, o, r, i), t = o;
	}
	return i;
}
function Ci(e, t, n, r, i, a, o, s, c, l, u) {
	if (t == n) return l;
	let d, f;
	if (i === 0) {
		if (f = wt(o, s, e[t], e[t + 1]), f < l) {
			for (d = 0; d < r; ++d) c[d] = e[t + d];
			return c.length = r, f;
		}
		return l;
	}
	u ||= [NaN, NaN];
	let p = t + r;
	for (; p < n;) if (bi(e, p - r, p, r, o, s, u), f = wt(o, s, u[0], u[1]), f < l) {
		for (l = f, d = 0; d < r; ++d) c[d] = u[d];
		c.length = r, p += r;
	} else p += r * Math.max((Math.sqrt(f) - Math.sqrt(l)) / i | 0, 1);
	if (a && (bi(e, n - r, t, r, o, s, u), f = wt(o, s, u[0], u[1]), f < l)) {
		for (l = f, d = 0; d < r; ++d) c[d] = u[d];
		c.length = r;
	}
	return l;
}
function wi(e, t, n, r, i, a, o, s, c, l, u) {
	u ||= [NaN, NaN];
	for (let d = 0, f = n.length; d < f; ++d) {
		let f = n[d];
		l = Ci(e, t, f, r, i, a, o, s, c, l, u), t = f;
	}
	return l;
}
//#endregion
//#region node_modules/ol/geom/flat/deflate.js
function Ti(e, t, n, r) {
	for (let r = 0, i = n.length; r < i; ++r) e[t++] = n[r];
	return t;
}
function Ei(e, t, n, r) {
	for (let i = 0, a = n.length; i < a; ++i) {
		let a = n[i];
		for (let n = 0; n < r; ++n) e[t++] = a[n];
	}
	return t;
}
function Di(e, t, n, r, i) {
	i ||= [];
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = Ei(e, t, n[o], r);
		i[a++] = s, t = s;
	}
	return i.length = a, i;
}
//#endregion
//#region node_modules/ol/geom/flat/inflate.js
function Oi(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = t; o < n; o += r) i[a++] = e.slice(o, o + r);
	return i.length = a, i;
}
function ki(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		i[a++] = Oi(e, t, s, r, i[a]), t = s;
	}
	return i.length = a, i;
}
function Ai(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		i[a++] = s.length === 1 && s[0] === t ? [] : ki(e, t, s, r, i[a]), t = s[s.length - 1];
	}
	return i.length = a, i;
}
//#endregion
//#region node_modules/ol/geom/flat/contains.js
function ji(e, t, n, r, i) {
	return !et(i, function(i) {
		return !Mi(e, t, n, r, i[0], i[1]);
	});
}
function Mi(e, t, n, r, i, a) {
	let o = 0, s = e[n - r], c = e[n - r + 1];
	for (; t < n; t += r) {
		let n = e[t], r = e[t + 1];
		c <= a ? r > a && (n - s) * (a - c) - (i - s) * (r - c) > 0 && o++ : r <= a && (n - s) * (a - c) - (i - s) * (r - c) < 0 && o--, s = n, c = r;
	}
	return o !== 0;
}
function Ni(e, t, n, r, i, a) {
	if (n.length === 0 || !Mi(e, t, n[0], r, i, a)) return !1;
	for (let t = 1, o = n.length; t < o; ++t) if (Mi(e, n[t - 1], n[t], r, i, a)) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/segments.js
function Pi(e, t, n, r, i) {
	let a;
	for (t += r; t < n; t += r) if (a = i(e.slice(t - r, t), e.slice(t, t + r)), a) return a;
	return !1;
}
//#endregion
//#region node_modules/ol/geom/flat/intersectsextent.js
function Fi(e, t, n, r, i, a) {
	return a ??= Qe(We(), e, t, n, r), pt(i, a) ? a[0] >= i[0] && a[2] <= i[2] || a[1] >= i[1] && a[3] <= i[3] || Pi(e, t, n, r, function(e, t) {
		return _t(i, e, t);
	}) : !1;
}
function Ii(e, t, n, r, i) {
	return !!(Fi(e, t, n, r, i) || Mi(e, t, n, r, i[0], i[1]) || Mi(e, t, n, r, i[0], i[3]) || Mi(e, t, n, r, i[2], i[1]) || Mi(e, t, n, r, i[2], i[3]));
}
function Li(e, t, n, r, i) {
	if (!Ii(e, t, n[0], r, i)) return !1;
	if (n.length === 1) return !0;
	for (let t = 1, a = n.length; t < a; ++t) if (ji(e, n[t - 1], n[t], r, i) && !Fi(e, n[t - 1], n[t], r, i)) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/simplify.js
function Ri(e, t, n, r, i, a, o) {
	let s = (n - t) / r;
	if (s < 3) {
		for (; t < n; t += r) a[o++] = e[t], a[o++] = e[t + 1];
		return o;
	}
	let c = Array(s);
	c[0] = 1, c[s - 1] = 1;
	let l = [t, n - r], u = 0;
	for (; l.length > 0;) {
		let n = l.pop(), a = l.pop(), o = 0, s = e[a], d = e[a + 1], f = e[n], p = e[n + 1];
		for (let t = a + r; t < n; t += r) {
			let n = e[t], r = e[t + 1], i = Ct(n, r, s, d, f, p);
			i > o && (u = t, o = i);
		}
		o > i && (c[(u - t) / r] = 1, a + r < u && l.push(a, u), u + r < n && l.push(u, n));
	}
	for (let n = 0; n < s; ++n) c[n] && (a[o++] = e[t + n * r], a[o++] = e[t + n * r + 1]);
	return o;
}
function zi(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c];
		o = Ri(e, t, l, r, i, a, o), s.push(o), t = l;
	}
	return o;
}
function Bi(e, t) {
	return t * Math.round(e / t);
}
function Vi(e, t, n, r, i, a, o) {
	if (t == n) return o;
	let s = Bi(e[t], i), c = Bi(e[t + 1], i);
	t += r, a[o++] = s, a[o++] = c;
	let l, u;
	do
		if (l = Bi(e[t], i), u = Bi(e[t + 1], i), t += r, t == n) return a[o++] = l, a[o++] = u, o;
	while (l == s && u == c);
	for (; t < n;) {
		let n = Bi(e[t], i), d = Bi(e[t + 1], i);
		if (t += r, n == l && d == u) continue;
		let f = l - s, p = u - c, m = n - s, h = d - c;
		if (f * h == p * m && (f < 0 && m < f || f == m || f > 0 && m > f) && (p < 0 && h < p || p == h || p > 0 && h > p)) {
			l = n, u = d;
			continue;
		}
		a[o++] = l, a[o++] = u, s = l, c = u, l = n, u = d;
	}
	return a[o++] = l, a[o++] = u, o;
}
function Hi(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c];
		o = Vi(e, t, l, r, i, a, o), s.push(o), t = l;
	}
	return o;
}
//#endregion
//#region node_modules/ol/geom/LinearRing.js
var Ui = class e extends mi {
	constructor(e, t) {
		super(), this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, t !== void 0 && !Array.isArray(e[0]) ? this.setFlatCoordinates(t, e) : this.setCoordinates(e, t);
	}
	clone() {
		return new e(this.flatCoordinates.slice(), this.layout);
	}
	closestPointXY(e, t, n, r) {
		return r < ze(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(xi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), Ci(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	getArea() {
		return vi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getCoordinates() {
		return Oi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getSimplifiedGeometryInternal(t) {
		let n = [];
		return n.length = Ri(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, t, n, 0), new e(n, "XY");
	}
	getType() {
		return "LinearRing";
	}
	intersectsExtent(e) {
		return Fi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = Ei(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
}, Wi = class e extends mi {
	constructor(e, t) {
		super(), this.setCoordinates(e, t);
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout);
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		let i = this.flatCoordinates, a = wt(e, t, i[0], i[1]);
		if (a < r) {
			let e = this.stride;
			for (let t = 0; t < e; ++t) n[t] = i[t];
			return n.length = e, a;
		}
		return r;
	}
	getCoordinates() {
		return this.flatCoordinates.slice();
	}
	computeExtent(e) {
		return qe(this.flatCoordinates, e);
	}
	getType() {
		return "Point";
	}
	intersectsExtent(e) {
		return He(e, this.flatCoordinates[0], this.flatCoordinates[1]);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 0), this.flatCoordinates ||= [], this.flatCoordinates.length = Ti(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
};
//#endregion
//#region node_modules/ol/geom/flat/interiorpoint.js
function Gi(e, t, n, r, i, a, o) {
	let s, c, l, u, d, f, p, m = i[a + 1], h = [];
	for (let i = 0, a = n.length; i < a; ++i) {
		let a = n[i];
		for (u = e[a - r], f = e[a - r + 1], s = t; s < a; s += r) d = e[s], p = e[s + 1], (m <= f && p <= m || f <= m && m <= p) && (l = (m - f) / (p - f) * (d - u) + u, h.push(l)), u = d, f = p;
	}
	let g = NaN, _ = -Infinity;
	for (h.sort(oe), u = h[0], s = 1, c = h.length; s < c; ++s) {
		d = h[s];
		let i = Math.abs(d - u);
		i > _ && (l = (u + d) / 2, Ni(e, t, n, r, l, m) && (g = l, _ = i)), u = d;
	}
	return isNaN(g) && (g = i[a]), o ? (o.push(g, m, _), o) : [
		g,
		m,
		_
	];
}
function Ki(e, t, n, r, i) {
	let a = [];
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		a = Gi(e, t, s, r, i, 2 * o, a), t = s[s.length - 1];
	}
	return a;
}
//#endregion
//#region node_modules/ol/geom/flat/reverse.js
function qi(e, t, n, r) {
	for (; t < n - r;) {
		for (let i = 0; i < r; ++i) {
			let a = e[t + i];
			e[t + i] = e[n - r + i], e[n - r + i] = a;
		}
		t += r, n -= r;
	}
}
//#endregion
//#region node_modules/ol/geom/flat/orient.js
function Ji(e, t, n, r) {
	let i = 0, a = e[n - r], o = e[n - r + 1];
	for (; t < n; t += r) {
		let n = e[t], r = e[t + 1];
		i += (n - a) * (r + o), a = n, o = r;
	}
	return i === 0 ? void 0 : i > 0;
}
function Yi(e, t, n, r, i) {
	i = i !== void 0 && i;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a], s = Ji(e, t, o, r);
		if (a === 0) {
			if (i && s || !i && !s) return !1;
		} else if (i && !s || !i && s) return !1;
		t = o;
	}
	return !0;
}
function Xi(e, t, n, r, i) {
	i = i !== void 0 && i;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a], s = Ji(e, t, o, r);
		(a === 0 ? i && s || !i && !s : i && !s || !i && s) && qi(e, t, o, r), t = o;
	}
	return t;
}
function Zi(e, t) {
	let n = [], r = 0, i = 0, a;
	for (let o = 0, s = t.length; o < s; ++o) {
		let s = t[o], c = Ji(e, r, s, 2);
		if (a === void 0 && (a = c), c === a) n.push(t.slice(i, o + 1));
		else {
			if (n.length === 0) continue;
			n[n.length - 1].push(t[i]);
		}
		i = o + 1, r = s;
	}
	return n;
}
//#endregion
//#region node_modules/ol/geom/Polygon.js
var Qi = class e extends mi {
	constructor(e, t, n) {
		super(), this.ends_ = [], this.flatInteriorPointRevision_ = -1, this.flatInteriorPoint_ = null, this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, this.orientedRevision_ = -1, this.orientedFlatCoordinates_ = null, t !== void 0 && n ? (this.setFlatCoordinates(t, e), this.ends_ = n) : this.setCoordinates(e, t);
	}
	appendLinearRing(e) {
		this.flatCoordinates ? ue(this.flatCoordinates, e.getFlatCoordinates()) : this.flatCoordinates = e.getFlatCoordinates().slice(), this.ends_.push(this.flatCoordinates.length), this.changed();
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout, this.ends_.slice());
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		return r < ze(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(Si(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), wi(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	containsXY(e, t) {
		return Ni(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, t);
	}
	getArea() {
		return yi(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride);
	}
	getCoordinates(e) {
		let t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), Xi(t, 0, this.ends_, this.stride, e)), ki(t, 0, this.ends_, this.stride);
	}
	getEnds() {
		return this.ends_;
	}
	getFlatInteriorPoint() {
		if (this.flatInteriorPointRevision_ != this.getRevision()) {
			let e = it(this.getExtent());
			this.flatInteriorPoint_ = Gi(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, 0), this.flatInteriorPointRevision_ = this.getRevision();
		}
		return this.flatInteriorPoint_;
	}
	getInteriorPoint() {
		return new Wi(this.getFlatInteriorPoint(), "XYM");
	}
	getLinearRingCount() {
		return this.ends_.length;
	}
	getLinearRing(e) {
		return e < 0 || this.ends_.length <= e ? null : new Ui(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}
	getLinearRings() {
		let e = this.layout, t = this.flatCoordinates, n = this.ends_, r = [], i = 0;
		for (let a = 0, o = n.length; a < o; ++a) {
			let o = n[a], s = new Ui(t.slice(i, o), e);
			r.push(s), i = o;
		}
		return r;
	}
	getOrientedFlatCoordinates() {
		if (this.orientedRevision_ != this.getRevision()) {
			let e = this.flatCoordinates;
			Yi(e, 0, this.ends_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = Xi(this.orientedFlatCoordinates_, 0, this.ends_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = Hi(this.flatCoordinates, 0, this.ends_, this.stride, Math.sqrt(t), n, 0, r), new e(n, "XY", r);
	}
	getType() {
		return "Polygon";
	}
	intersectsExtent(e) {
		return Li(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		let n = Di(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}
};
function $i(e) {
	if (mt(e)) throw Error("Cannot create polygon from empty extent");
	let t = e[0], n = e[1], r = e[2], i = e[3], a = [
		t,
		n,
		t,
		i,
		r,
		i,
		r,
		n,
		t,
		n
	];
	return new Qi(a, "XY", [a.length]);
}
//#endregion
//#region node_modules/ol/resolutionconstraint.js
function ea(e, t, n, r) {
	let i = I(t) / n[0], a = ct(t) / n[1];
	return r ? Math.min(e, Math.max(i, a)) : Math.min(e, Math.min(i, a));
}
function ta(e, t, n) {
	let r = Math.min(e, t);
	return r *= Math.log(1 + 50 * Math.max(0, e / t - 1)) / 50 + 1, n && (r = Math.max(r, n), r /= Math.log(1 + 50 * Math.max(0, n / e - 1)) / 50 + 1), St(r, n / 2, t * 2);
}
function na(e, t, n, r) {
	return t = t === void 0 || t, (function(i, a, o, s) {
		if (i !== void 0) {
			let c = e[0], l = e[e.length - 1], u = n ? ea(c, n, o, r) : c;
			if (s) return t ? ta(i, u, l) : St(i, l, u);
			let d = Math.floor(ce(e, Math.min(u, i), a));
			return e[d] > u && d < e.length - 1 ? e[d + 1] : e[d];
		}
	});
}
function ra(e, t, n, r, i, a) {
	return r = r === void 0 || r, n = n === void 0 ? 0 : n, (function(o, s, c, l) {
		if (o !== void 0) {
			let u = i ? ea(t, i, c, a) : t;
			if (l) return r ? ta(o, u, n) : St(o, n, u);
			let d = Math.ceil(Math.log(t / u) / Math.log(e) - 1e-9), f = -s * .499999999 + .5, p = Math.floor(Math.log(t / Math.min(u, o)) / Math.log(e) + f);
			return St(t / e ** +Math.max(d, p), n, u);
		}
	});
}
function ia(e, t, n, r, i) {
	return n = n === void 0 || n, (function(a, o, s, c) {
		if (a !== void 0) {
			let o = r ? ea(e, r, s, i) : e;
			return !n || !c ? St(a, t, o) : ta(a, o, t);
		}
	});
}
//#endregion
//#region node_modules/ol/rotationconstraint.js
function aa(e) {
	if (e !== void 0) return 0;
}
function oa(e) {
	if (e !== void 0) return e;
}
function sa(e) {
	let t = 2 * Math.PI / e;
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return e = Math.floor(e / t + .5) * t, e;
	});
}
function ca(e) {
	let t = e === void 0 ? Dt(5) : e;
	return (function(e, n) {
		return n || e === void 0 ? e : Math.abs(e) <= t ? 0 : e;
	});
}
//#endregion
//#region node_modules/ol/View.js
var la = 0, ua = class extends we {
	constructor(e) {
		super(), this.on, this.once, this.un, e = Object.assign({}, e), this.hints_ = [0, 0], this.animations_ = [], this.updateAnimationKey_, this.projection_ = or(e.projection, "EPSG:3857"), this.viewportSize_ = [100, 100], this.targetCenter_ = null, this.targetResolution_, this.targetRotation_, this.nextCenter_ = null, this.nextResolution_, this.nextRotation_, this.cancelAnchor_ = void 0, e.projection && Zn(), e.center && (e.center = _r(e.center, this.projection_)), e.extent && (e.extent = yr(e.extent, this.projection_)), this.applyOptions_(e);
	}
	applyOptions_(e) {
		let t = Object.assign({}, e);
		for (let e in Rr) delete t[e];
		this.setProperties(t, !0);
		let n = pa(e);
		this.maxResolution_ = n.maxResolution, this.minResolution_ = n.minResolution, this.zoomFactor_ = n.zoomFactor, this.resolutions_ = e.resolutions, this.padding_ = e.padding, this.minZoom_ = n.minZoom;
		let r = fa(e), i = n.constraint, a = ma(e);
		this.constraints_ = {
			center: r,
			resolution: i,
			rotation: a
		}, this.setRotation(e.rotation === void 0 ? 0 : e.rotation), this.setCenterInternal(e.center === void 0 ? null : e.center), e.resolution === void 0 ? e.zoom !== void 0 && this.setZoom(e.zoom) : this.setResolution(e.resolution);
	}
	get padding() {
		return this.padding_;
	}
	set padding(e) {
		let t = this.padding_;
		this.padding_ = e;
		let n = this.getCenterInternal();
		if (n) {
			let r = e || [
				0,
				0,
				0,
				0
			];
			t ||= [
				0,
				0,
				0,
				0
			];
			let i = this.getResolution(), a = i / 2 * (r[3] - t[3] + t[1] - r[1]), o = i / 2 * (r[0] - t[0] + t[2] - r[2]);
			this.setCenterInternal([n[0] + a, n[1] - o]);
		}
	}
	getUpdatedOptions_(e) {
		let t = this.getProperties();
		return t.resolution === void 0 ? t.zoom = this.getZoom() : t.resolution = this.getResolution(), t.center = this.getCenterInternal(), t.rotation = this.getRotation(), Object.assign({}, t, e);
	}
	animate(e) {
		this.isDef() && !this.getAnimating() && this.resolveConstraints(0);
		let t = Array(arguments.length);
		for (let e = 0; e < t.length; ++e) {
			let n = arguments[e];
			n.center && (n = Object.assign({}, n), n.center = _r(n.center, this.getProjection())), n.anchor && (n = Object.assign({}, n), n.anchor = _r(n.anchor, this.getProjection())), t[e] = n;
		}
		this.animateInternal.apply(this, t);
	}
	animateInternal(e) {
		let t = arguments.length, n;
		t > 1 && typeof arguments[t - 1] == "function" && (n = arguments[t - 1], --t);
		let r = 0;
		for (; r < t && !this.isDef(); ++r) {
			let e = arguments[r];
			e.center && this.setCenterInternal(e.center), e.zoom === void 0 ? e.resolution && this.setResolution(e.resolution) : this.setZoom(e.zoom), e.rotation !== void 0 && this.setRotation(e.rotation);
		}
		if (r === t) {
			n && da(n, !0);
			return;
		}
		let i = Date.now(), a = this.targetCenter_.slice(), o = this.targetResolution_, s = this.targetRotation_, c = [];
		for (; r < t; ++r) {
			let e = arguments[r], t = {
				start: i,
				complete: !1,
				anchor: e.anchor,
				duration: e.duration === void 0 ? 1e3 : e.duration,
				easing: e.easing || Ur,
				callback: n
			};
			if (e.center && (t.sourceCenter = a, t.targetCenter = e.center.slice(), a = t.targetCenter), e.zoom === void 0 ? e.resolution && (t.sourceResolution = o, t.targetResolution = e.resolution, o = t.targetResolution) : (t.sourceResolution = o, t.targetResolution = this.getResolutionForZoom(e.zoom), o = t.targetResolution), e.rotation !== void 0) {
				t.sourceRotation = s;
				let n = Ot(e.rotation - s + Math.PI, 2 * Math.PI) - Math.PI;
				t.targetRotation = s + n, s = t.targetRotation;
			}
			ha(t) ? t.complete = !0 : i += t.duration, c.push(t);
		}
		this.animations_.push(c), this.setHint(Lr.ANIMATING, 1), this.updateAnimations_();
	}
	getAnimating() {
		return this.hints_[Lr.ANIMATING] > 0;
	}
	getInteracting() {
		return this.hints_[Lr.INTERACTING] > 0;
	}
	cancelAnimations() {
		this.setHint(Lr.ANIMATING, -this.hints_[Lr.ANIMATING]);
		let e;
		for (let t = 0, n = this.animations_.length; t < n; ++t) {
			let n = this.animations_[t];
			if (n[0].callback && da(n[0].callback, !1), !e) for (let t = 0, r = n.length; t < r; ++t) {
				let r = n[t];
				if (!r.complete) {
					e = r.anchor;
					break;
				}
			}
		}
		this.animations_.length = 0, this.cancelAnchor_ = e, this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
	}
	updateAnimations_() {
		if (this.updateAnimationKey_ !== void 0 && (cancelAnimationFrame(this.updateAnimationKey_), this.updateAnimationKey_ = void 0), !this.getAnimating()) return;
		let e = Date.now(), t = !1;
		for (let n = this.animations_.length - 1; n >= 0; --n) {
			let r = this.animations_[n], i = !0;
			for (let n = 0, a = r.length; n < a; ++n) {
				let a = r[n];
				if (a.complete) continue;
				let o = e - a.start, s = a.duration > 0 ? o / a.duration : 1;
				s >= 1 ? (a.complete = !0, s = 1) : i = !1;
				let c = a.easing(s);
				if (a.sourceCenter) {
					let e = a.sourceCenter[0], t = a.sourceCenter[1], n = a.targetCenter[0], r = a.targetCenter[1];
					this.nextCenter_ = a.targetCenter;
					let i = e + c * (n - e), o = t + c * (r - t);
					this.targetCenter_ = [i, o];
				}
				if (a.sourceResolution && a.targetResolution) {
					let e = c === 1 ? a.targetResolution : a.sourceResolution + c * (a.targetResolution - a.sourceResolution);
					if (a.anchor) {
						let t = this.getViewportSize_(this.getRotation()), n = this.constraints_.resolution(e, 0, t, !0);
						this.targetCenter_ = this.calculateCenterZoom(n, a.anchor);
					}
					this.nextResolution_ = a.targetResolution, this.targetResolution_ = e, this.applyTargetState_(!0);
				}
				if (a.sourceRotation !== void 0 && a.targetRotation !== void 0) {
					let e = c === 1 ? Ot(a.targetRotation + Math.PI, 2 * Math.PI) - Math.PI : a.sourceRotation + c * (a.targetRotation - a.sourceRotation);
					if (a.anchor) {
						let t = this.constraints_.rotation(e, !0);
						this.targetCenter_ = this.calculateCenterRotate(t, a.anchor);
					}
					this.nextRotation_ = a.targetRotation, this.targetRotation_ = e;
				}
				if (this.applyTargetState_(!0), t = !0, !a.complete) break;
			}
			if (i) {
				this.animations_[n] = null, this.setHint(Lr.ANIMATING, -1), this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
				let e = r[0].callback;
				e && da(e, !0);
			}
		}
		this.animations_ = this.animations_.filter(Boolean), t && this.updateAnimationKey_ === void 0 && (this.updateAnimationKey_ = requestAnimationFrame(this.updateAnimations_.bind(this)));
	}
	calculateCenterRotate(e, t) {
		let n, r = this.getCenterInternal();
		return r !== void 0 && (n = [r[0] - t[0], r[1] - t[1]], Lt(n, e - this.getRotation()), Pt(n, t)), n;
	}
	calculateCenterZoom(e, t) {
		let n, r = this.getCenterInternal(), i = this.getResolution();
		return r !== void 0 && i !== void 0 && (n = [t[0] - e * (t[0] - r[0]) / i, t[1] - e * (t[1] - r[1]) / i]), n;
	}
	getViewportSize_(e) {
		let t = this.viewportSize_;
		if (e) {
			let n = t[0], r = t[1];
			return [Math.abs(n * Math.cos(e)) + Math.abs(r * Math.sin(e)), Math.abs(n * Math.sin(e)) + Math.abs(r * Math.cos(e))];
		}
		return t;
	}
	setViewportSize(e) {
		this.viewportSize_ = Array.isArray(e) ? e.slice() : [100, 100], this.getAnimating() || this.resolveConstraints(0);
	}
	getCenter() {
		let e = this.getCenterInternal();
		return e && gr(e, this.getProjection());
	}
	getCenterInternal() {
		return this.get(Rr.CENTER);
	}
	getConstraints() {
		return this.constraints_;
	}
	getConstrainResolution() {
		return this.get("constrainResolution");
	}
	getHints(e) {
		return e === void 0 ? this.hints_.slice() : (e[0] = this.hints_[0], e[1] = this.hints_[1], e);
	}
	calculateExtent(e) {
		return vr(this.calculateExtentInternal(e), this.getProjection());
	}
	calculateExtentInternal(e) {
		e ||= this.getViewportSizeMinusPadding_();
		let t = this.getCenterInternal();
		R(t, "The view center is not defined");
		let n = this.getResolution();
		R(n !== void 0, "The view resolution is not defined");
		let r = this.getRotation();
		return R(r !== void 0, "The view rotation is not defined"), ot(t, n, r, e);
	}
	getMaxResolution() {
		return this.maxResolution_;
	}
	getMinResolution() {
		return this.minResolution_;
	}
	getMaxZoom() {
		return this.getZoomForResolution(this.minResolution_);
	}
	setMaxZoom(e) {
		this.applyOptions_(this.getUpdatedOptions_({ maxZoom: e }));
	}
	getMinZoom() {
		return this.getZoomForResolution(this.maxResolution_);
	}
	setMinZoom(e) {
		this.applyOptions_(this.getUpdatedOptions_({ minZoom: e }));
	}
	setConstrainResolution(e) {
		this.applyOptions_(this.getUpdatedOptions_({ constrainResolution: e }));
	}
	getProjection() {
		return this.projection_;
	}
	getResolution() {
		return this.get(Rr.RESOLUTION);
	}
	getResolutions() {
		return this.resolutions_;
	}
	getResolutionForExtent(e, t) {
		return this.getResolutionForExtentInternal(yr(e, this.getProjection()), t);
	}
	getResolutionForExtentInternal(e, t) {
		t ||= this.getViewportSizeMinusPadding_();
		let n = I(e) / t[0], r = ct(e) / t[1];
		return Math.max(n, r);
	}
	getResolutionForValueFunction(e) {
		e ||= 2;
		let t = this.getConstrainedResolution(this.maxResolution_), n = this.minResolution_, r = Math.log(t / n) / Math.log(e);
		return (function(n) {
			return t / e ** +(n * r);
		});
	}
	getRotation() {
		return this.get(Rr.ROTATION);
	}
	getValueForResolutionFunction(e) {
		let t = Math.log(e || 2), n = this.getConstrainedResolution(this.maxResolution_), r = this.minResolution_, i = Math.log(n / r) / t;
		return (function(e) {
			return Math.log(n / e) / t / i;
		});
	}
	getViewportSizeMinusPadding_(e) {
		let t = this.getViewportSize_(e), n = this.padding_;
		return n && (t = [t[0] - n[1] - n[3], t[1] - n[0] - n[2]]), t;
	}
	getState() {
		let e = this.getProjection(), t = this.getResolution(), n = this.getRotation(), r = this.getCenterInternal(), i = this.padding_;
		if (i) {
			let e = this.getViewportSizeMinusPadding_();
			r = ga(r, this.getViewportSize_(), [e[0] / 2 + i[3], e[1] / 2 + i[0]], t, n);
		}
		return {
			center: r.slice(0),
			projection: e === void 0 ? null : e,
			resolution: t,
			nextCenter: this.nextCenter_,
			nextResolution: this.nextResolution_,
			nextRotation: this.nextRotation_,
			rotation: n,
			zoom: this.getZoom()
		};
	}
	getViewStateAndExtent() {
		return {
			viewState: this.getState(),
			extent: this.calculateExtent()
		};
	}
	getZoom() {
		let e, t = this.getResolution();
		return t !== void 0 && (e = this.getZoomForResolution(t)), e;
	}
	getZoomForResolution(e) {
		let t = this.minZoom_ || 0, n, r;
		if (this.resolutions_) {
			let i = ce(this.resolutions_, e, 1);
			t = i, n = this.resolutions_[i], r = i == this.resolutions_.length - 1 ? 2 : n / this.resolutions_[i + 1];
		} else n = this.maxResolution_, r = this.zoomFactor_;
		return t + Math.log(n / e) / Math.log(r);
	}
	getResolutionForZoom(e) {
		if (this.resolutions_?.length) {
			if (this.resolutions_.length === 1) return this.resolutions_[0];
			let t = St(Math.floor(e), 0, this.resolutions_.length - 2), n = this.resolutions_[t] / this.resolutions_[t + 1];
			return this.resolutions_[t] / n ** +St(e - t, 0, 1);
		}
		return this.maxResolution_ / this.zoomFactor_ ** +(e - this.minZoom_);
	}
	fit(e, t) {
		let n;
		if (R(Array.isArray(e) || typeof e.getSimplifiedGeometry == "function", "Invalid extent or geometry provided as `geometry`"), Array.isArray(e)) R(!mt(e), "Cannot fit empty extent provided as `geometry`"), n = $i(yr(e, this.getProjection()));
		else if (e.getType() === "Circle") {
			let t = yr(e.getExtent(), this.getProjection());
			n = $i(t), n.rotate(this.getRotation(), it(t));
		} else {
			let t = hr();
			n = t ? e.clone().transform(t, this.getProjection()) : e;
		}
		this.fitInternal(n, t);
	}
	rotatedExtentForGeometry(e) {
		let t = this.getRotation(), n = Math.cos(t), r = Math.sin(-t), i = e.getFlatCoordinates(), a = e.getStride(), o = Infinity, s = Infinity, c = -Infinity, l = -Infinity;
		for (let e = 0, t = i.length; e < t; e += a) {
			let t = i[e] * n - i[e + 1] * r, a = i[e] * r + i[e + 1] * n;
			o = Math.min(o, t), s = Math.min(s, a), c = Math.max(c, t), l = Math.max(l, a);
		}
		return [
			o,
			s,
			c,
			l
		];
	}
	fitInternal(e, t) {
		t ||= {};
		let n = t.size;
		n ||= this.getViewportSizeMinusPadding_();
		let r = t.padding === void 0 ? [
			0,
			0,
			0,
			0
		] : t.padding, i = t.nearest !== void 0 && t.nearest, a;
		a = t.minResolution === void 0 ? t.maxZoom === void 0 ? 0 : this.getResolutionForZoom(t.maxZoom) : t.minResolution;
		let o = this.rotatedExtentForGeometry(e), s = this.getResolutionForExtentInternal(o, [n[0] - r[1] - r[3], n[1] - r[0] - r[2]]);
		s = isNaN(s) ? a : Math.max(s, a), s = this.getConstrainedResolution(s, +!i);
		let c = this.getRotation(), l = Math.sin(c), u = Math.cos(c), d = it(o);
		d[0] += (r[1] - r[3]) / 2 * s, d[1] += (r[0] - r[2]) / 2 * s;
		let f = d[0] * u - d[1] * l, p = d[1] * u + d[0] * l, m = this.getConstrainedCenter([f, p], s), h = t.callback ? t.callback : he;
		t.duration === void 0 ? (this.targetResolution_ = s, this.targetCenter_ = m, this.applyTargetState_(!1, !0), da(h, !0)) : this.animateInternal({
			resolution: s,
			center: m,
			duration: t.duration,
			easing: t.easing
		}, h);
	}
	centerOn(e, t, n) {
		this.centerOnInternal(_r(e, this.getProjection()), t, n);
	}
	centerOnInternal(e, t, n) {
		this.setCenterInternal(ga(e, t, n, this.getResolution(), this.getRotation()));
	}
	calculateCenterShift(e, t, n, r) {
		let i, a = this.padding_;
		if (a && e) {
			let o = this.getViewportSizeMinusPadding_(-n), s = ga(e, r, [o[0] / 2 + a[3], o[1] / 2 + a[0]], t, n);
			i = [e[0] - s[0], e[1] - s[1]];
		}
		return i;
	}
	isDef() {
		return !!this.getCenterInternal() && this.getResolution() !== void 0;
	}
	adjustCenter(e) {
		let t = gr(this.targetCenter_, this.getProjection());
		this.setCenter([t[0] + e[0], t[1] + e[1]]);
	}
	adjustCenterInternal(e) {
		let t = this.targetCenter_;
		this.setCenterInternal([t[0] + e[0], t[1] + e[1]]);
	}
	adjustResolution(e, t) {
		t &&= _r(t, this.getProjection()), this.adjustResolutionInternal(e, t);
	}
	adjustResolutionInternal(e, t) {
		let n = this.getAnimating() || this.getInteracting(), r = this.getViewportSize_(this.getRotation()), i = this.constraints_.resolution(this.targetResolution_ * e, 0, r, n);
		t && (this.targetCenter_ = this.calculateCenterZoom(i, t)), this.targetResolution_ *= e, this.applyTargetState_();
	}
	adjustZoom(e, t) {
		this.adjustResolution(this.zoomFactor_ ** +-e, t);
	}
	adjustRotation(e, t) {
		t &&= _r(t, this.getProjection()), this.adjustRotationInternal(e, t);
	}
	adjustRotationInternal(e, t) {
		let n = this.getAnimating() || this.getInteracting(), r = this.constraints_.rotation(this.targetRotation_ + e, n);
		t && (this.targetCenter_ = this.calculateCenterRotate(r, t)), this.targetRotation_ += e, this.applyTargetState_();
	}
	setCenter(e) {
		this.setCenterInternal(e && _r(e, this.getProjection()));
	}
	setCenterInternal(e) {
		this.targetCenter_ = e, this.applyTargetState_();
	}
	setHint(e, t) {
		return this.hints_[e] += t, this.changed(), this.hints_[e];
	}
	setResolution(e) {
		this.targetResolution_ = e, this.applyTargetState_();
	}
	setRotation(e) {
		this.targetRotation_ = e, this.applyTargetState_();
	}
	setZoom(e) {
		this.setResolution(this.getResolutionForZoom(e));
	}
	applyTargetState_(e, t) {
		let n = this.getAnimating() || this.getInteracting() || t, r = this.constraints_.rotation(this.targetRotation_, n), i = this.getViewportSize_(r), a = this.constraints_.resolution(this.targetResolution_, 0, i, n), o = this.constraints_.center(this.targetCenter_, a, i, n, this.calculateCenterShift(this.targetCenter_, a, r, i));
		this.get(Rr.ROTATION) !== r && this.set(Rr.ROTATION, r), this.get(Rr.RESOLUTION) !== a && (this.set(Rr.RESOLUTION, a), this.set("zoom", this.getZoom(), !0)), (!o || !this.get(Rr.CENTER) || !It(this.get(Rr.CENTER), o)) && this.set(Rr.CENTER, o), this.getAnimating() && !e && this.cancelAnimations(), this.cancelAnchor_ = void 0;
	}
	resolveConstraints(e, t, n) {
		e = e === void 0 ? 200 : e;
		let r = t || 0, i = this.constraints_.rotation(this.targetRotation_), a = this.getViewportSize_(i), o = this.constraints_.resolution(this.targetResolution_, r, a), s = this.constraints_.center(this.targetCenter_, o, a, !1, this.calculateCenterShift(this.targetCenter_, o, i, a));
		if (e === 0 && !this.cancelAnchor_) {
			this.targetResolution_ = o, this.targetRotation_ = i, this.targetCenter_ = s, this.applyTargetState_();
			return;
		}
		n ||= e === 0 ? this.cancelAnchor_ : void 0, this.cancelAnchor_ = void 0, (this.getResolution() !== o || this.getRotation() !== i || !this.getCenterInternal() || !It(this.getCenterInternal(), s)) && (this.getAnimating() && this.cancelAnimations(), this.animateInternal({
			rotation: i,
			center: s,
			resolution: o,
			duration: e,
			easing: Hr,
			anchor: n
		}));
	}
	beginInteraction() {
		this.resolveConstraints(0), this.setHint(Lr.INTERACTING, 1);
	}
	endInteraction(e, t, n) {
		n &&= _r(n, this.getProjection()), this.endInteractionInternal(e, t, n);
	}
	endInteractionInternal(e, t, n) {
		this.getInteracting() && (this.setHint(Lr.INTERACTING, -1), this.resolveConstraints(e, t, n));
	}
	getConstrainedCenter(e, t) {
		let n = this.getViewportSize_(this.getRotation());
		return this.constraints_.center(e, t || this.getResolution(), n);
	}
	getConstrainedZoom(e, t) {
		let n = this.getResolutionForZoom(e);
		return this.getZoomForResolution(this.getConstrainedResolution(n, t));
	}
	getConstrainedResolution(e, t) {
		t ||= 0;
		let n = this.getViewportSize_(this.getRotation());
		return this.constraints_.resolution(e, t, n);
	}
};
function da(e, t) {
	setTimeout(function() {
		e(t);
	}, 0);
}
function fa(e) {
	if (e.extent !== void 0) {
		let t = e.smoothExtentConstraint === void 0 || e.smoothExtentConstraint;
		return zr(e.extent, e.constrainOnlyCenter, t);
	}
	let t = or(e.projection, "EPSG:3857");
	if (e.multiWorld !== !0 && t.isGlobal()) {
		let e = t.getExtent().slice();
		return e[0] = -Infinity, e[2] = Infinity, zr(e, !1, !1);
	}
	return Br;
}
function pa(e) {
	let t, n, r, i = e.minZoom === void 0 ? la : e.minZoom, a = e.maxZoom === void 0 ? 28 : e.maxZoom, o = e.zoomFactor === void 0 ? 2 : e.zoomFactor, s = e.multiWorld !== void 0 && e.multiWorld, c = e.smoothResolutionConstraint === void 0 || e.smoothResolutionConstraint, l = e.showFullExtent !== void 0 && e.showFullExtent, u = or(e.projection, "EPSG:3857"), d = u.getExtent(), f = e.constrainOnlyCenter, p = e.extent;
	if (!s && !p && u.isGlobal() && (f = !1, p = d), e.resolutions !== void 0) {
		let o = e.resolutions;
		n = o[i], r = o[a] === void 0 ? o[o.length - 1] : o[a], t = e.constrainResolution ? na(o, c, !f && p, l) : ia(n, r, c, !f && p, l);
	} else {
		let s = (d ? Math.max(I(d), ct(d)) : 360 * Kt.degrees / u.getMetersPerUnit()) / 256 / 2 ** la, m = s / 2 ** 28;
		n = e.maxResolution, n === void 0 ? n = s / o ** +i : i = 0, r = e.minResolution, r === void 0 && (r = e.maxZoom === void 0 ? m : e.maxResolution === void 0 ? s / o ** +a : n / o ** +a), a = i + Math.floor(Math.log(n / r) / Math.log(o)), r = n / o ** +(a - i), t = e.constrainResolution ? ra(o, n, r, c, !f && p, l) : ia(n, r, c, !f && p, l);
	}
	return {
		constraint: t,
		maxResolution: n,
		minResolution: r,
		minZoom: i,
		zoomFactor: o
	};
}
function ma(e) {
	if (e.enableRotation === void 0 || e.enableRotation) {
		let t = e.constrainRotation;
		return t === void 0 || t === !0 ? ca() : t === !1 ? oa : typeof t == "number" ? sa(t) : oa;
	}
	return aa;
}
function ha(e) {
	return !(e.sourceCenter && e.targetCenter && !It(e.sourceCenter, e.targetCenter) || e.sourceResolution !== e.targetResolution || e.sourceRotation !== e.targetRotation);
}
function ga(e, t, n, r, i) {
	let a = Math.cos(-i), o = Math.sin(-i), s = e[0] * a - e[1] * o, c = e[1] * a + e[0] * o;
	return s += (t[0] / 2 - n[0]) * r, c += (n[1] - t[1] / 2) * r, o = -o, [s * a - c * o, c * a + s * o];
}
//#endregion
//#region node_modules/ol/control/Attribution.js
var _a = class extends Te {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		}), this.ulElement_ = document.createElement("ul"), this.collapsed_ = e.collapsed === void 0 || e.collapsed, this.userCollapsed_ = this.collapsed_, this.overrideCollapsible_ = e.collapsible !== void 0, this.collapsible_ = e.collapsible === void 0 || e.collapsible, this.collapsible_ || (this.collapsed_ = !1), this.attributions_ = e.attributions;
		let t = e.className === void 0 ? "ol-attribution" : e.className, n = e.tipLabel === void 0 ? "Attributions" : e.tipLabel, i = e.expandClassName === void 0 ? t + "-expand" : e.expandClassName, s = e.collapseLabel === void 0 ? "›" : e.collapseLabel, c = e.collapseClassName === void 0 ? t + "-collapse" : e.collapseClassName;
		typeof s == "string" ? (this.collapseLabel_ = document.createElement("span"), this.collapseLabel_.textContent = s, this.collapseLabel_.className = c) : this.collapseLabel_ = s;
		let l = e.label === void 0 ? "i" : e.label;
		typeof l == "string" ? (this.label_ = document.createElement("span"), this.label_.textContent = l, this.label_.className = i) : this.label_ = l;
		let u = this.collapsible_ && !this.collapsed_ ? this.collapseLabel_ : this.label_;
		this.toggleButton_ = document.createElement("button"), this.toggleButton_.setAttribute("type", "button"), this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_)), this.toggleButton_.title = n, this.toggleButton_.appendChild(u), this.toggleButton_.addEventListener(M.CLICK, this.handleClick_.bind(this), !1);
		let d = t + " " + r + " " + a + (this.collapsed_ && this.collapsible_ ? " " + o : "") + (this.collapsible_ ? "" : " ol-uncollapsible"), f = this.element;
		f.className = d, f.appendChild(this.toggleButton_), f.appendChild(this.ulElement_), this.renderedAttributions_ = [], this.renderedVisible_ = !0;
	}
	collectSourceAttributions_(e) {
		let t = this.getMap().getAllLayers(), n = new Set(t.flatMap((t) => t.getAttributions(e)));
		if (this.attributions_ !== void 0 && (Array.isArray(this.attributions_) ? this.attributions_.forEach((e) => n.add(e)) : n.add(this.attributions_)), !this.overrideCollapsible_) {
			let e = !t.some((e) => e.getSource()?.getAttributionsCollapsible() === !1);
			this.setCollapsible(e);
		}
		return Array.from(n);
	}
	async updateElement_(e) {
		if (!e) {
			this.renderedVisible_ &&= (this.element.style.display = "none", !1);
			return;
		}
		let t = await Promise.all(this.collectSourceAttributions_(e).map((e) => _e(() => e))), n = t.length > 0;
		if (this.renderedVisible_ != n && (this.element.style.display = n ? "" : "none", this.renderedVisible_ = n), !de(t, this.renderedAttributions_)) {
			E(this.ulElement_);
			for (let e = 0, n = t.length; e < n; ++e) {
				let n = document.createElement("li");
				n.innerHTML = t[e], this.ulElement_.appendChild(n);
			}
			this.renderedAttributions_ = t;
		}
	}
	handleClick_(e) {
		e.preventDefault(), this.handleToggle_(), this.userCollapsed_ = this.collapsed_;
	}
	handleToggle_() {
		this.element.classList.toggle(o), this.collapsed_ ? T(this.collapseLabel_, this.label_) : T(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_, this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_));
	}
	getCollapsible() {
		return this.collapsible_;
	}
	setCollapsible(e) {
		this.collapsible_ !== e && (this.collapsible_ = e, this.element.classList.toggle("ol-uncollapsible"), this.userCollapsed_ && this.handleToggle_());
	}
	setCollapsed(e) {
		this.userCollapsed_ = e, !(!this.collapsible_ || this.collapsed_ === e) && this.handleToggle_();
	}
	getCollapsed() {
		return this.collapsed_;
	}
	render(e) {
		this.updateElement_(e.frameState);
	}
}, va = class extends Te {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		});
		let n = e.className === void 0 ? "ol-rotate" : e.className, i = e.label === void 0 ? "⇧" : e.label, o = e.compassClassName === void 0 ? "ol-compass" : e.compassClassName;
		this.label_ = null, typeof i == "string" ? (this.label_ = document.createElement("span"), this.label_.className = o, this.label_.textContent = i) : (this.label_ = i, this.label_.classList.add(o));
		let s = e.tipLabel ? e.tipLabel : "Reset rotation", c = document.createElement("button");
		c.className = n + "-reset", c.setAttribute("type", "button"), c.title = s, c.appendChild(this.label_), c.addEventListener(M.CLICK, this.handleClick_.bind(this), !1);
		let l = n + " " + r + " " + a, u = this.element;
		u.className = l, u.appendChild(c), this.callResetNorth_ = e.resetNorth ? e.resetNorth : void 0, this.duration_ = e.duration === void 0 ? 250 : e.duration, this.autoHide_ = e.autoHide === void 0 || e.autoHide, this.rotation_ = void 0, this.autoHide_ && this.element.classList.add(t);
	}
	handleClick_(e) {
		e.preventDefault(), this.callResetNorth_ === void 0 ? this.resetNorth_() : this.callResetNorth_();
	}
	resetNorth_() {
		let e = this.getMap().getView();
		if (!e) return;
		let t = e.getRotation();
		t !== void 0 && (this.duration_ > 0 && t % (2 * Math.PI) != 0 ? e.animate({
			rotation: 0,
			duration: this.duration_,
			easing: Hr
		}) : e.setRotation(0));
	}
	render(e) {
		let n = e.frameState;
		if (!n) return;
		let r = n.viewState.rotation;
		if (r != this.rotation_) {
			let e = "rotate(" + r + "rad)";
			if (this.autoHide_) {
				let e = this.element.classList.contains(t);
				!e && r === 0 ? this.element.classList.add(t) : e && r !== 0 && this.element.classList.remove(t);
			}
			this.label_.style.transform = e;
		}
		this.rotation_ = r;
	}
}, ya = class extends Te {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			target: e.target
		});
		let t = e.className === void 0 ? "ol-zoom" : e.className, n = e.delta === void 0 ? 1 : e.delta, i = e.zoomInClassName === void 0 ? t + "-in" : e.zoomInClassName, o = e.zoomOutClassName === void 0 ? t + "-out" : e.zoomOutClassName, s = e.zoomInLabel === void 0 ? "+" : e.zoomInLabel, c = e.zoomOutLabel === void 0 ? "–" : e.zoomOutLabel, l = e.zoomInTipLabel === void 0 ? "Zoom in" : e.zoomInTipLabel, u = e.zoomOutTipLabel === void 0 ? "Zoom out" : e.zoomOutTipLabel, d = document.createElement("button");
		d.className = i, d.setAttribute("type", "button"), d.title = l, d.appendChild(typeof s == "string" ? document.createTextNode(s) : s), d.addEventListener(M.CLICK, this.handleClick_.bind(this, n), !1);
		let f = document.createElement("button");
		f.className = o, f.setAttribute("type", "button"), f.title = u, f.appendChild(typeof c == "string" ? document.createTextNode(c) : c), f.addEventListener(M.CLICK, this.handleClick_.bind(this, -n), !1);
		let p = t + " " + r + " " + a, m = this.element;
		m.className = p, m.appendChild(d), m.appendChild(f), this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleClick_(e, t) {
		t.preventDefault(), this.zoomByDelta_(e);
	}
	zoomByDelta_(e) {
		let t = this.getMap().getView();
		if (!t) return;
		let n = t.getZoom();
		if (n !== void 0) {
			let r = t.getConstrainedZoom(n + e);
			this.duration_ > 0 ? (t.getAnimating() && t.cancelAnimations(), t.animate({
				zoom: r,
				duration: this.duration_,
				easing: Hr
			})) : t.setZoom(r);
		}
	}
};
//#endregion
//#region node_modules/ol/control/defaults.js
function ba(e) {
	e ||= {};
	let t = new Or();
	return (e.zoom === void 0 || e.zoom) && t.push(new ya(e.zoomOptions)), (e.rotate === void 0 || e.rotate) && t.push(new va(e.rotateOptions)), (e.attribution === void 0 || e.attribution) && t.push(new _a(e.attributionOptions)), t;
}
//#endregion
//#region node_modules/ol/Kinetic.js
var xa = class {
	constructor(e, t, n) {
		this.decay_ = e, this.minVelocity_ = t, this.delay_ = n, this.points_ = [], this.angle_ = 0, this.initialVelocity_ = 0;
	}
	begin() {
		this.points_.length = 0, this.angle_ = 0, this.initialVelocity_ = 0;
	}
	update(e, t) {
		this.points_.push(e, t, Date.now());
	}
	end() {
		if (this.points_.length < 6) return !1;
		let e = Date.now() - this.delay_, t = this.points_.length - 3;
		if (this.points_[t + 2] < e) return !1;
		let n = t - 3;
		for (; n > 0 && this.points_[n + 2] > e;) n -= 3;
		let r = this.points_[t + 2] - this.points_[n + 2];
		if (r < 1e3 / 60) return !1;
		let i = this.points_[t] - this.points_[n], a = this.points_[t + 1] - this.points_[n + 1];
		return this.angle_ = Math.atan2(a, i), this.initialVelocity_ = Math.sqrt(i * i + a * a) / r, this.initialVelocity_ > this.minVelocity_;
	}
	getDistance() {
		return (this.minVelocity_ - this.initialVelocity_) / this.decay_;
	}
	getAngle() {
		return this.angle_;
	}
}, Sa = { ACTIVE: "active" }, Ca = class extends we {
	constructor(e) {
		super(), this.on, this.once, this.un, e && e.handleEvent && (this.handleEvent = e.handleEvent), this.map_ = null, this.setActive(!0);
	}
	getActive() {
		return this.get(Sa.ACTIVE);
	}
	getMap() {
		return this.map_;
	}
	handleEvent(e) {
		return !0;
	}
	setActive(e) {
		this.set(Sa.ACTIVE, e);
	}
	setMap(e) {
		this.map_ = e;
	}
};
function wa(e, t, n) {
	let r = e.getCenterInternal();
	if (r) {
		let i = [r[0] + t[0], r[1] + t[1]];
		e.animateInternal({
			duration: n === void 0 ? 250 : n,
			easing: Wr,
			center: e.getConstrainedCenter(i)
		});
	}
}
function Ta(e, t, n, r) {
	let i = e.getZoom();
	if (i === void 0) return;
	let a = e.getConstrainedZoom(i + t), o = e.getResolutionForZoom(a);
	e.getAnimating() && e.cancelAnimations(), e.animate({
		resolution: o,
		anchor: n,
		duration: r === void 0 ? 250 : r,
		easing: Hr
	});
}
//#endregion
//#region node_modules/ol/interaction/DoubleClickZoom.js
var Ea = class extends Ca {
	constructor(e) {
		super(), e ||= {}, this.delta_ = e.delta ? e.delta : 1, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == jr.DBLCLICK) {
			let n = e.originalEvent, r = e.map, i = e.coordinate, a = n.shiftKey ? -this.delta_ : this.delta_;
			Ta(r.getView(), a, i, this.duration_), n.preventDefault(), t = !0;
		}
		return !t;
	}
};
//#endregion
//#region node_modules/ol/events/condition.js
function Da(e) {
	let t = arguments;
	return function(e) {
		let n = !0;
		for (let r = 0, i = t.length; r < i && (n &&= t[r](e), n); ++r);
		return n;
	};
}
var Oa = function(e) {
	let t = e.originalEvent;
	return t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, ka = function(e) {
	let t = e.map.getTargetElement(), n = t.getRootNode(), r = e.map.getOwnerDocument().activeElement;
	return n instanceof ShadowRoot ? n.host.contains(r) : t.contains(r);
}, Aa = function(e) {
	let t = e.map.getTargetElement(), n = t.getRootNode();
	return !(n instanceof ShadowRoot ? n.host : t).hasAttribute("tabindex") || ka(e);
}, ja = pe, Ma = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.button == 0 && !(p && m && t.ctrlKey);
}, Na = function(e) {
	let t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && !t.shiftKey;
}, Pa = function(e) {
	let t = e.originalEvent;
	return m ? t.metaKey : t.ctrlKey;
}, Fa = function(e) {
	let t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, Ia = function(e) {
	let t = e.originalEvent, n = t.target.tagName;
	return n !== "INPUT" && n !== "SELECT" && n !== "TEXTAREA" && !t.target.isContentEditable;
}, La = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.pointerType == "mouse";
}, Ra = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.isPrimary && t.button === 0;
}, za = class extends Ca {
	constructor(e) {
		e ||= {}, super(e), e.handleDownEvent && (this.handleDownEvent = e.handleDownEvent), e.handleDragEvent && (this.handleDragEvent = e.handleDragEvent), e.handleMoveEvent && (this.handleMoveEvent = e.handleMoveEvent), e.handleUpEvent && (this.handleUpEvent = e.handleUpEvent), e.stopDown && (this.stopDown = e.stopDown), this.handlingDownUpSequence = !1, this.targetPointers = [];
	}
	getPointerCount() {
		return this.targetPointers.length;
	}
	handleDownEvent(e) {
		return !1;
	}
	handleDragEvent(e) {}
	handleEvent(e) {
		if (!e.originalEvent) return !0;
		let t = !1;
		if (this.updateTrackedPointers_(e), this.handlingDownUpSequence) {
			if (e.type == jr.POINTERDRAG) this.handleDragEvent(e), e.originalEvent.preventDefault();
			else if (e.type == jr.POINTERUP) {
				let t = this.handleUpEvent(e);
				this.handlingDownUpSequence = t && this.targetPointers.length > 0;
			}
		} else if (e.type == jr.POINTERDOWN) {
			let n = this.handleDownEvent(e);
			this.handlingDownUpSequence = n, t = this.stopDown(n);
		} else e.type == jr.POINTERMOVE && this.handleMoveEvent(e);
		return !t;
	}
	handleMoveEvent(e) {}
	handleUpEvent(e) {
		return !1;
	}
	stopDown(e) {
		return e;
	}
	updateTrackedPointers_(e) {
		e.activePointers && (this.targetPointers = e.activePointers);
	}
};
function Ba(e) {
	let t = e.length, n = 0, r = 0;
	for (let i = 0; i < t; i++) n += e[i].clientX, r += e[i].clientY;
	return {
		clientX: n / t,
		clientY: r / t
	};
}
//#endregion
//#region node_modules/ol/interaction/DragPan.js
var Va = class extends za {
	constructor(e) {
		super({ stopDown: me }), e ||= {}, this.kinetic_ = e.kinetic, this.lastCentroid = null, this.lastPointersCount_, this.panning_ = !1;
		let t = e.condition ? e.condition : Da(Na, Ra);
		this.condition_ = e.onFocusOnly ? Da(Aa, t) : t, this.noKinetic_ = !1;
	}
	handleDragEvent(e) {
		let t = e.map;
		this.panning_ || (this.panning_ = !0, t.getView().beginInteraction());
		let n = this.targetPointers, r = t.getEventPixel(Ba(n));
		if (n.length == this.lastPointersCount_) {
			if (this.kinetic_ && this.kinetic_.update(r[0], r[1]), this.lastCentroid) {
				let t = [this.lastCentroid[0] - r[0], r[1] - this.lastCentroid[1]], n = e.map.getView();
				Rt(t, n.getResolution()), Lt(t, n.getRotation()), n.adjustCenterInternal(t);
			}
		} else this.kinetic_ && this.kinetic_.begin();
		this.lastCentroid = r, this.lastPointersCount_ = n.length, e.originalEvent.preventDefault();
	}
	handleUpEvent(e) {
		let t = e.map, n = t.getView();
		if (this.targetPointers.length === 0) {
			if (!this.noKinetic_ && this.kinetic_ && this.kinetic_.end()) {
				let e = this.kinetic_.getDistance(), r = this.kinetic_.getAngle(), i = n.getCenterInternal(), a = t.getPixelFromCoordinateInternal(i), o = t.getCoordinateFromPixelInternal([a[0] - e * Math.cos(r), a[1] - e * Math.sin(r)]);
				n.animateInternal({
					center: n.getConstrainedCenter(o),
					duration: 500,
					easing: Hr
				});
			}
			return this.panning_ && (this.panning_ = !1, n.endInteraction()), !1;
		}
		return this.kinetic_ && this.kinetic_.begin(), this.lastCentroid = null, !0;
	}
	handleDownEvent(e) {
		if (this.targetPointers.length > 0 && this.condition_(e)) {
			let t = e.map.getView();
			return this.lastCentroid = null, t.getAnimating() && t.cancelAnimations(), this.kinetic_ && this.kinetic_.begin(), this.noKinetic_ = this.targetPointers.length > 1, !0;
		}
		return !1;
	}
}, Ha = class extends za {
	constructor(e) {
		e ||= {}, super({ stopDown: me }), this.condition_ = e.condition ? e.condition : Oa, this.lastAngle_ = void 0, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleDragEvent(e) {
		if (!La(e)) return;
		let t = e.map, n = t.getView();
		if (n.getConstraints().rotation === aa) return;
		let r = t.getSize(), i = e.pixel, a = Math.atan2(r[1] / 2 - i[1], i[0] - r[0] / 2);
		if (this.lastAngle_ !== void 0) {
			let e = a - this.lastAngle_;
			n.adjustRotationInternal(-e);
		}
		this.lastAngle_ = a;
	}
	handleUpEvent(e) {
		return !La(e) || (e.map.getView().endInteraction(this.duration_), !1);
	}
	handleDownEvent(e) {
		return La(e) && Ma(e) && this.condition_(e) ? (e.map.getView().beginInteraction(), this.lastAngle_ = void 0, !0) : !1;
	}
}, Ua = class extends ie {
	constructor(e) {
		super(), this.geometry_ = null, this.element_ = document.createElement("div"), this.element_.style.position = "absolute", this.element_.style.pointerEvents = "auto", this.element_.className = "ol-box " + e, this.map_ = null, this.startPixel_ = null, this.endPixel_ = null;
	}
	disposeInternal() {
		this.setMap(null);
	}
	render_() {
		let e = this.startPixel_, t = this.endPixel_, n = this.element_.style;
		n.left = Math.min(e[0], t[0]) + "px", n.top = Math.min(e[1], t[1]) + "px", n.width = Math.abs(t[0] - e[0]) + "px", n.height = Math.abs(t[1] - e[1]) + "px";
	}
	setMap(e) {
		if (this.map_) {
			this.map_.getOverlayContainer().removeChild(this.element_);
			let e = this.element_.style;
			e.left = "inherit", e.top = "inherit", e.width = "inherit", e.height = "inherit";
		}
		this.map_ = e, this.map_ && this.map_.getOverlayContainer().appendChild(this.element_);
	}
	setPixels(e, t) {
		this.startPixel_ = e, this.endPixel_ = t, this.createOrUpdateGeometry(), this.render_();
	}
	createOrUpdateGeometry() {
		if (!this.map_) return;
		let e = this.startPixel_, t = this.endPixel_, n = [
			e,
			[e[0], t[1]],
			t,
			[t[0], e[1]]
		].map(this.map_.getCoordinateFromPixelInternal, this.map_);
		n[4] = n[0].slice(), this.geometry_ ? this.geometry_.setCoordinates([n]) : this.geometry_ = new Qi([n]);
	}
	getGeometry() {
		return this.geometry_;
	}
}, Wa = {
	BOXSTART: "boxstart",
	BOXDRAG: "boxdrag",
	BOXEND: "boxend",
	BOXCANCEL: "boxcancel"
}, Ga = class extends ve {
	constructor(e, t, n) {
		super(e), this.coordinate = t, this.mapBrowserEvent = n;
	}
}, Ka = class extends za {
	constructor(e) {
		super(), this.on, this.once, this.un, e ??= {}, this.box_ = new Ua(e.className || "ol-dragbox"), this.minArea_ = e.minArea ?? 64, e.onBoxEnd && (this.onBoxEnd = e.onBoxEnd), this.startPixel_ = null, this.condition_ = e.condition ?? Ma, this.boxEndCondition_ = e.boxEndCondition ?? this.defaultBoxEndCondition;
	}
	defaultBoxEndCondition(e, t, n) {
		let r = n[0] - t[0], i = n[1] - t[1];
		return r * r + i * i >= this.minArea_;
	}
	getGeometry() {
		return this.box_.getGeometry();
	}
	handleDragEvent(e) {
		this.startPixel_ && (this.box_.setPixels(this.startPixel_, e.pixel), this.dispatchEvent(new Ga(Wa.BOXDRAG, e.coordinate, e)));
	}
	handleUpEvent(e) {
		if (!this.startPixel_) return !1;
		let t = this.boxEndCondition_(e, this.startPixel_, e.pixel);
		return t && this.onBoxEnd(e), this.dispatchEvent(new Ga(t ? Wa.BOXEND : Wa.BOXCANCEL, e.coordinate, e)), this.box_.setMap(null), this.startPixel_ = null, !1;
	}
	handleDownEvent(e) {
		return this.condition_(e) ? (this.startPixel_ = e.pixel, this.box_.setMap(e.map), this.box_.setPixels(this.startPixel_, this.startPixel_), this.dispatchEvent(new Ga(Wa.BOXSTART, e.coordinate, e)), !0) : !1;
	}
	onBoxEnd(e) {}
	setActive(e) {
		e || (this.box_.setMap(null), this.startPixel_ &&= (this.dispatchEvent(new Ga(Wa.BOXCANCEL, this.startPixel_, null)), null)), super.setActive(e);
	}
	setMap(e) {
		this.getMap() && (this.box_.setMap(null), this.startPixel_ &&= (this.dispatchEvent(new Ga(Wa.BOXCANCEL, this.startPixel_, null)), null)), super.setMap(e);
	}
}, qa = class extends Ka {
	constructor(e) {
		e ||= {};
		let t = e.condition ? e.condition : Fa;
		super({
			condition: t,
			className: e.className || "ol-dragzoom",
			minArea: e.minArea
		}), this.duration_ = e.duration === void 0 ? 200 : e.duration, this.out_ = e.out !== void 0 && e.out;
	}
	onBoxEnd(e) {
		let t = this.getMap().getView(), n = this.getGeometry();
		if (this.out_) {
			let e = t.rotatedExtentForGeometry(n), r = t.getResolutionForExtentInternal(e), i = t.getResolution() / r;
			n = n.clone(), n.scale(i * i);
		}
		t.fitInternal(n, {
			duration: this.duration_,
			easing: Hr
		});
	}
}, Ja = {
	LEFT: "ArrowLeft",
	UP: "ArrowUp",
	RIGHT: "ArrowRight",
	DOWN: "ArrowDown"
}, Ya = class extends Ca {
	constructor(e) {
		super(), e ||= {}, this.defaultCondition_ = function(e) {
			return Na(e) && Ia(e);
		}, this.condition_ = e.condition === void 0 ? this.defaultCondition_ : e.condition, this.duration_ = e.duration === void 0 ? 100 : e.duration, this.pixelDelta_ = e.pixelDelta === void 0 ? 128 : e.pixelDelta;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == M.KEYDOWN) {
			let n = e.originalEvent, r = n.key;
			if (this.condition_(e) && (r == Ja.DOWN || r == Ja.LEFT || r == Ja.RIGHT || r == Ja.UP)) {
				let i = e.map.getView(), a = i.getResolution() * this.pixelDelta_, o = 0, s = 0;
				r == Ja.DOWN ? s = -a : r == Ja.LEFT ? o = -a : r == Ja.RIGHT ? o = a : s = a;
				let c = [o, s];
				Lt(c, i.getRotation()), wa(i, c, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}
}, Xa = class extends Ca {
	constructor(e) {
		super(), e ||= {}, this.condition_ = e.condition ? e.condition : function(e) {
			return !Pa(e) && Ia(e);
		}, this.delta_ = e.delta ? e.delta : 1, this.duration_ = e.duration === void 0 ? 100 : e.duration;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == M.KEYDOWN || e.type == M.KEYPRESS) {
			let n = e.originalEvent, r = n.key;
			if (this.condition_(e) && (r === "+" || r === "-")) {
				let i = e.map, a = r === "+" ? this.delta_ : -this.delta_;
				Ta(i.getView(), a, void 0, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}
}, Za = 40, Qa = 300, $a = 3, eo = class extends Ca {
	constructor(e) {
		e ||= {}, super(e), this.totalDelta_ = 0, this.lastDelta_ = 0, this.maxDelta_ = e.maxDelta === void 0 ? 1 : e.maxDelta, this.duration_ = e.duration === void 0 ? 250 : e.duration, this.timeout_ = e.timeout === void 0 ? 80 : e.timeout, this.useAnchor_ = e.useAnchor === void 0 || e.useAnchor, this.constrainResolution_ = e.constrainResolution !== void 0 && e.constrainResolution;
		let t = e.condition ? e.condition : ja;
		this.condition_ = e.onFocusOnly ? Da(Aa, t) : t, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_, this.mode_ = void 0, this.trackpadEventGap_ = 400, this.trackpadTimeoutId_, this.deltaPerZoom_ = 300, this.ctrlKeyPressed_ = !1, this.ctrlKeyListenerKeys_ = [];
	}
	setMap(e) {
		if (this.ctrlKeyListenerKeys_.forEach(j), this.ctrlKeyListenerKeys_.length = 0, this.ctrlKeyPressed_ = !1, super.setMap(e), e) {
			let t = e.getOwnerDocument();
			this.ctrlKeyListenerKeys_.push(A(t, "keydown", (e) => {
				e.key === "Control" && (this.ctrlKeyPressed_ = !0);
			}), A(t, "keyup", (e) => {
				e.key === "Control" && (this.ctrlKeyPressed_ = !1);
			}));
		}
	}
	endInteraction_() {
		this.trackpadTimeoutId_ = void 0;
		let e = this.getMap();
		if (!e) return;
		let t = e.getView(), n = this.lastDelta_ ? this.lastDelta_ > 0 ? 1 : -1 : 0;
		t.endInteraction(this.constrainResolution_ || t.getConstrainResolution() ? 100 : void 0, n, this.lastAnchor_ ? e.getCoordinateFromPixel(this.lastAnchor_) : null);
	}
	handleEvent(e) {
		if (!this.condition_(e) || e.type !== M.WHEEL) return !0;
		let t = e.map, n = e.originalEvent;
		n.preventDefault();
		let r = n.ctrlKey && !this.ctrlKeyPressed_;
		n.ctrlKey || (this.ctrlKeyPressed_ = !1), this.useAnchor_ && (this.lastAnchor_ = e.pixel);
		let i = n.deltaY;
		switch (n.deltaMode) {
			case WheelEvent.DOM_DELTA_LINE:
				i *= Za;
				break;
			case WheelEvent.DOM_DELTA_PAGE: i *= Qa;
		}
		if (i === 0) return !1;
		this.lastDelta_ = i;
		let a = Date.now();
		this.startTime_ === void 0 && (this.startTime_ = a), (!this.mode_ || a - this.startTime_ > this.trackpadEventGap_) && (this.mode_ = Math.abs(i) < 4 ? "trackpad" : "wheel");
		let o = t.getView();
		if (this.mode_ === "trackpad") return this.trackpadTimeoutId_ ? clearTimeout(this.trackpadTimeoutId_) : (o.getAnimating() && o.cancelAnimations(), o.beginInteraction()), this.trackpadTimeoutId_ = setTimeout(this.endInteraction_.bind(this), this.timeout_), r && (i *= $a), o.adjustZoom(-i / this.deltaPerZoom_, this.lastAnchor_ ? t.getCoordinateFromPixel(this.lastAnchor_) : null), this.startTime_ = a, !1;
		this.totalDelta_ += i;
		let s = Math.max(this.timeout_ - (a - this.startTime_), 0);
		return clearTimeout(this.timeoutId_), this.timeoutId_ = setTimeout(this.handleWheelZoom_.bind(this, t), s), !1;
	}
	handleWheelZoom_(e) {
		let t = e.getView();
		t.getAnimating() && t.cancelAnimations();
		let n = -St(this.totalDelta_, -this.maxDelta_ * this.deltaPerZoom_, this.maxDelta_ * this.deltaPerZoom_) / this.deltaPerZoom_;
		(t.getConstrainResolution() || this.constrainResolution_) && (n = n ? n > 0 ? 1 : -1 : 0), Ta(t, n, this.lastAnchor_ ? e.getCoordinateFromPixel(this.lastAnchor_) : null, this.duration_), this.mode_ = void 0, this.totalDelta_ = 0, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_ = void 0;
	}
	setMouseAnchor(e) {
		this.useAnchor_ = e, e || (this.lastAnchor_ = null);
	}
}, to = class extends za {
	constructor(e) {
		e ||= {};
		let t = e;
		t.stopDown ||= me, super(t), this.anchor_ = null, this.lastAngle_ = void 0, this.rotating_ = !1, this.rotationDelta_ = 0, this.threshold_ = e.threshold === void 0 ? .3 : e.threshold, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleDragEvent(e) {
		let t = 0, n = this.targetPointers[0], r = this.targetPointers[1], i = Math.atan2(r.clientY - n.clientY, r.clientX - n.clientX);
		if (this.lastAngle_ !== void 0) {
			let e = i - this.lastAngle_;
			this.rotationDelta_ += e, !this.rotating_ && Math.abs(this.rotationDelta_) > this.threshold_ && (this.rotating_ = !0), t = e;
		}
		this.lastAngle_ = i;
		let a = e.map, o = a.getView();
		o.getConstraints().rotation !== aa && (this.anchor_ = a.getCoordinateFromPixelInternal(a.getEventPixel(Ba(this.targetPointers))), this.rotating_ && (a.render(), o.adjustRotationInternal(t, this.anchor_)));
	}
	handleUpEvent(e) {
		return this.targetPointers.length < 2 ? (e.map.getView().endInteraction(this.duration_), !1) : !0;
	}
	handleDownEvent(e) {
		if (this.targetPointers.length >= 2) {
			let t = e.map;
			return this.anchor_ = null, this.lastAngle_ = void 0, this.rotating_ = !1, this.rotationDelta_ = 0, this.handlingDownUpSequence || t.getView().beginInteraction(), !0;
		}
		return !1;
	}
}, no = class extends za {
	constructor(e) {
		e ||= {};
		let t = e;
		t.stopDown ||= me, super(t), this.anchor_ = null, this.duration_ = e.duration === void 0 ? 400 : e.duration, this.lastDistance_ = void 0, this.lastScaleDelta_ = 1;
	}
	handleDragEvent(e) {
		let t = 1, n = this.targetPointers[0], r = this.targetPointers[1], i = n.clientX - r.clientX, a = n.clientY - r.clientY, o = Math.sqrt(i * i + a * a);
		this.lastDistance_ !== void 0 && (t = this.lastDistance_ / o), this.lastDistance_ = o;
		let s = e.map, c = s.getView();
		t != 1 && (this.lastScaleDelta_ = t), this.anchor_ = s.getCoordinateFromPixelInternal(s.getEventPixel(Ba(this.targetPointers))), s.render(), c.adjustResolutionInternal(t, this.anchor_);
	}
	handleUpEvent(e) {
		if (this.targetPointers.length < 2) {
			let t = e.map.getView(), n = this.lastScaleDelta_ > 1 ? 1 : -1;
			return t.endInteraction(this.duration_, n), !1;
		}
		return !0;
	}
	handleDownEvent(e) {
		if (this.targetPointers.length >= 2) {
			let t = e.map;
			return this.anchor_ = null, this.lastDistance_ = void 0, this.lastScaleDelta_ = 1, this.handlingDownUpSequence || t.getView().beginInteraction(), !0;
		}
		return !1;
	}
};
//#endregion
//#region node_modules/ol/interaction/defaults.js
function ro(e) {
	e ||= {};
	let t = new Or(), n = new xa(-.005, .05, 100);
	return (e.altShiftDragRotate === void 0 || e.altShiftDragRotate) && t.push(new Ha()), (e.doubleClickZoom === void 0 || e.doubleClickZoom) && t.push(new Ea({
		delta: e.zoomDelta,
		duration: e.zoomDuration
	})), (e.dragPan === void 0 || e.dragPan) && t.push(new Va({
		onFocusOnly: e.onFocusOnly,
		kinetic: n
	})), (e.pinchRotate === void 0 || e.pinchRotate) && t.push(new to()), (e.pinchZoom === void 0 || e.pinchZoom) && t.push(new no({ duration: e.zoomDuration })), (e.keyboard === void 0 || e.keyboard) && (t.push(new Ya()), t.push(new Xa({
		delta: e.zoomDelta,
		duration: e.zoomDuration
	}))), (e.mouseWheelZoom === void 0 || e.mouseWheelZoom) && t.push(new eo({
		onFocusOnly: e.onFocusOnly,
		duration: e.zoomDuration
	})), (e.shiftDragZoom === void 0 || e.shiftDragZoom) && t.push(new qa({ duration: e.zoomDuration })), t;
}
//#endregion
//#region node_modules/ol/layer/Property.js
var B = {
	OPACITY: "opacity",
	VISIBLE: "visible",
	EXTENT: "extent",
	Z_INDEX: "zIndex",
	MAX_RESOLUTION: "maxResolution",
	MIN_RESOLUTION: "minResolution",
	MAX_ZOOM: "maxZoom",
	MIN_ZOOM: "minZoom",
	SOURCE: "source",
	MAP: "map"
}, io = class extends we {
	constructor(e) {
		super(), this.on, this.once, this.un, this.background_ = e.background;
		let t = Object.assign({}, e);
		typeof e.properties == "object" && (delete t.properties, Object.assign(t, e.properties)), t[B.OPACITY] = e.opacity === void 0 ? 1 : e.opacity, R(typeof t[B.OPACITY] == "number", "Layer opacity must be a number"), t[B.VISIBLE] = e.visible === void 0 || e.visible, t[B.Z_INDEX] = e.zIndex, t[B.MAX_RESOLUTION] = e.maxResolution === void 0 ? Infinity : e.maxResolution, t[B.MIN_RESOLUTION] = e.minResolution === void 0 ? 0 : e.minResolution, t[B.MIN_ZOOM] = e.minZoom === void 0 ? -Infinity : e.minZoom, t[B.MAX_ZOOM] = e.maxZoom === void 0 ? Infinity : e.maxZoom, this.className_ = t.className === void 0 ? "ol-layer" : t.className, delete t.className, this.setProperties(t), this.state_ = null;
	}
	getBackground() {
		return this.background_;
	}
	getClassName() {
		return this.className_;
	}
	getLayerState(e) {
		let t = this.state_ || {
			layer: this,
			managed: e === void 0 || e
		}, n = this.getZIndex();
		return t.opacity = St(Math.round(this.getOpacity() * 100) / 100, 0, 1), t.visible = this.getVisible(), t.extent = this.getExtent(), t.zIndex = n === void 0 && !t.managed ? Infinity : n, t.maxResolution = this.getMaxResolution(), t.minResolution = Math.max(this.getMinResolution(), 0), t.minZoom = this.getMinZoom(), t.maxZoom = this.getMaxZoom(), this.state_ = t, t;
	}
	getLayersArray(e) {
		return P();
	}
	getLayerStatesArray(e) {
		return P();
	}
	getExtent() {
		return this.get(B.EXTENT);
	}
	getMaxResolution() {
		return this.get(B.MAX_RESOLUTION);
	}
	getMinResolution() {
		return this.get(B.MIN_RESOLUTION);
	}
	getMinZoom() {
		return this.get(B.MIN_ZOOM);
	}
	getMaxZoom() {
		return this.get(B.MAX_ZOOM);
	}
	getOpacity() {
		return this.get(B.OPACITY);
	}
	getSourceState() {
		return P();
	}
	getVisible() {
		return this.get(B.VISIBLE);
	}
	getZIndex() {
		return this.get(B.Z_INDEX);
	}
	setBackground(e) {
		this.background_ = e, this.changed();
	}
	setExtent(e) {
		this.set(B.EXTENT, e);
	}
	setMaxResolution(e) {
		this.set(B.MAX_RESOLUTION, e);
	}
	setMinResolution(e) {
		this.set(B.MIN_RESOLUTION, e);
	}
	setMaxZoom(e) {
		this.set(B.MAX_ZOOM, e);
	}
	setMinZoom(e) {
		this.set(B.MIN_ZOOM, e);
	}
	setOpacity(e) {
		R(typeof e == "number", "Layer opacity must be a number"), this.set(B.OPACITY, e);
	}
	setVisible(e) {
		this.set(B.VISIBLE, e);
	}
	setZIndex(e) {
		this.set(B.Z_INDEX, e);
	}
	disposeInternal() {
		this.state_ &&= (this.state_.layer = null, null), super.disposeInternal();
	}
}, ao = {
	ADDLAYER: "addlayer",
	REMOVELAYER: "removelayer"
}, oo = class extends ve {
	constructor(e, t) {
		super(e), this.layer = t;
	}
}, so = { LAYERS: "layers" }, co = class e extends io {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.layers;
		let n = e.layers;
		super(t), this.on, this.once, this.un, this.layersListenerKeys_ = [], this.listenerKeys_ = {}, this.addChangeListener(so.LAYERS, this.handleLayersChanged_), n ? Array.isArray(n) ? n = new Or(n.slice(), { unique: !0 }) : R(typeof n.getArray == "function", "Expected `layers` to be an array or a `Collection`") : n = new Or(void 0, { unique: !0 }), this.setLayers(n);
	}
	handleLayerChange_() {
		this.changed();
	}
	handleLayersChanged_() {
		this.layersListenerKeys_.forEach(j), this.layersListenerKeys_.length = 0;
		let e = this.getLayers();
		this.layersListenerKeys_.push(A(e, Tr.ADD, this.handleLayersAdd_, this), A(e, Tr.REMOVE, this.handleLayersRemove_, this));
		for (let e in this.listenerKeys_) this.listenerKeys_[e].forEach(j);
		k(this.listenerKeys_);
		let t = e.getArray();
		for (let e = 0, n = t.length; e < n; e++) {
			let n = t[e];
			this.registerLayerListeners_(n), this.dispatchEvent(new oo(ao.ADDLAYER, n));
		}
		this.changed();
	}
	registerLayerListeners_(t) {
		let n = [A(t, re.PROPERTYCHANGE, this.handleLayerChange_, this), A(t, M.CHANGE, this.handleLayerChange_, this)];
		t instanceof e && n.push(A(t, ao.ADDLAYER, this.handleLayerGroupAdd_, this), A(t, ao.REMOVELAYER, this.handleLayerGroupRemove_, this)), this.listenerKeys_[F(t)] = n;
	}
	handleLayerGroupAdd_(e) {
		this.dispatchEvent(new oo(ao.ADDLAYER, e.layer));
	}
	handleLayerGroupRemove_(e) {
		this.dispatchEvent(new oo(ao.REMOVELAYER, e.layer));
	}
	handleLayersAdd_(e) {
		let t = e.element;
		this.registerLayerListeners_(t), this.dispatchEvent(new oo(ao.ADDLAYER, t)), this.changed();
	}
	handleLayersRemove_(e) {
		let t = e.element, n = F(t);
		this.listenerKeys_[n].forEach(j), delete this.listenerKeys_[n], this.dispatchEvent(new oo(ao.REMOVELAYER, t)), this.changed();
	}
	getLayers() {
		return this.get(so.LAYERS);
	}
	setLayers(e) {
		let t = this.getLayers();
		if (t) {
			let e = t.getArray();
			for (let t = 0, n = e.length; t < n; ++t) this.dispatchEvent(new oo(ao.REMOVELAYER, e[t]));
		}
		this.set(so.LAYERS, e);
	}
	getLayersArray(e) {
		return e = e === void 0 ? [] : e, this.getLayers().forEach(function(t) {
			t.getLayersArray(e);
		}), e;
	}
	getLayerStatesArray(e) {
		let t = e === void 0 ? [] : e, n = t.length;
		this.getLayers().forEach(function(e) {
			e.getLayerStatesArray(t);
		});
		let r = this.getLayerState(), i = r.zIndex;
		!e && r.zIndex === void 0 && (i = 0);
		for (let e = n, a = t.length; e < a; e++) {
			let n = t[e];
			n.opacity *= r.opacity, n.visible = n.visible && r.visible, n.maxResolution = Math.min(n.maxResolution, r.maxResolution), n.minResolution = Math.max(n.minResolution, r.minResolution), n.minZoom = Math.max(n.minZoom, r.minZoom), n.maxZoom = Math.min(n.maxZoom, r.maxZoom), r.extent !== void 0 && (n.extent = n.extent === void 0 ? r.extent : lt(n.extent, r.extent)), n.zIndex === void 0 && (n.zIndex = i);
		}
		return t;
	}
	getSourceState() {
		return "ready";
	}
}, lo = {
	PRERENDER: "prerender",
	POSTRENDER: "postrender",
	PRECOMPOSE: "precompose",
	POSTCOMPOSE: "postcompose",
	RENDERCOMPLETE: "rendercomplete"
}, uo = class extends io {
	constructor(e) {
		let t = Object.assign({}, e);
		delete t.source, super(t), this.on, this.once, this.un, this.mapPrecomposeKey_ = null, this.mapRenderKey_ = null, this.sourceChangeKey_ = null, this.renderer_ = null, this.sourceReady_ = !1, this.rendered = !1, e.render && (this.render = e.render), e.map && this.setMap(e.map), this.addChangeListener(B.SOURCE, this.handleSourcePropertyChange_);
		let n = e.source ? e.source : null;
		this.setSource(n);
	}
	getLayersArray(e) {
		return e ||= [], e.push(this), e;
	}
	getLayerStatesArray(e) {
		return e ||= [], e.push(this.getLayerState()), e;
	}
	getSource() {
		return this.get(B.SOURCE) || null;
	}
	getRenderSource() {
		return this.getSource();
	}
	getSourceState() {
		let e = this.getSource();
		return e ? e.getState() : "undefined";
	}
	handleSourceChange_() {
		this.changed(), !(this.sourceReady_ || this.getSource().getState() !== "ready") && (this.sourceReady_ = !0, this.dispatchEvent("sourceready"));
	}
	handleSourcePropertyChange_() {
		this.sourceChangeKey_ &&= (j(this.sourceChangeKey_), null), this.sourceReady_ = !1;
		let e = this.getSource();
		e && (this.sourceChangeKey_ = A(e, M.CHANGE, this.handleSourceChange_, this), e.getState() === "ready" && (this.sourceReady_ = !0, setTimeout(() => {
			this.dispatchEvent("sourceready");
		}, 0))), this.changed();
	}
	getFeatures(e) {
		return this.renderer_ ? this.renderer_.getFeatures(e) : Promise.resolve([]);
	}
	getData(e) {
		return !this.renderer_ || !this.rendered ? null : this.renderer_.getData(e);
	}
	isVisible(e) {
		let t, n = this.getMapInternal();
		!e && n && (e = n.getView()), t = e instanceof ua ? {
			viewState: e.getState(),
			extent: e.calculateExtent()
		} : e, !t.layerStatesArray && n && (t.layerStatesArray = n.getLayerGroup().getLayerStatesArray());
		let r;
		if (t.layerStatesArray) {
			if (r = t.layerStatesArray.find((e) => e.layer === this), !r) return !1;
		} else r = this.getLayerState();
		let i = this.getExtent();
		return fo(r, t.viewState) && (!i || pt(i, t.extent));
	}
	getAttributions(e) {
		if (!this.isVisible(e)) return [];
		let t = this.getSource()?.getAttributions();
		if (!t) return [];
		let n = t(e instanceof ua ? e.getViewStateAndExtent() : e);
		return Array.isArray(n) || (n = [n]), n;
	}
	render(e, t) {
		let n = this.getRenderer();
		return n.prepareFrame(e) ? (this.rendered = !0, n.renderFrame(e, t)) : null;
	}
	unrender() {
		this.rendered = !1;
	}
	getDeclutter() {}
	renderDeclutter(e, t) {}
	renderDeferred(e) {
		let t = this.getRenderer();
		t && t.renderDeferred(e);
	}
	setMapInternal(e) {
		e || this.unrender(), this.set(B.MAP, e);
	}
	getMapInternal() {
		return this.get(B.MAP);
	}
	setMap(e) {
		this.mapPrecomposeKey_ &&= (j(this.mapPrecomposeKey_), null), e || this.changed(), this.mapRenderKey_ &&= (j(this.mapRenderKey_), null), e && (this.mapPrecomposeKey_ = A(e, lo.PRECOMPOSE, this.handlePrecompose_, this), this.mapRenderKey_ = A(this, M.CHANGE, e.render, e), this.changed());
	}
	handlePrecompose_(e) {
		let t = e.frameState.layerStatesArray, n = this.getLayerState(!1);
		R(!t.some((e) => e.layer === n.layer), "A layer can only be added to the map once. Use either `layer.setMap()` or `map.addLayer()`, not both."), t.push(n);
	}
	setSource(e) {
		this.set(B.SOURCE, e);
	}
	getRenderer() {
		return this.renderer_ ||= this.createRenderer(), this.renderer_;
	}
	hasRenderer() {
		return !!this.renderer_;
	}
	createRenderer() {
		return null;
	}
	clearRenderer() {
		this.renderer_ && (this.renderer_.dispose(), delete this.renderer_);
	}
	disposeInternal() {
		this.clearRenderer(), this.setSource(null), super.disposeInternal();
	}
};
function fo(e, t) {
	if (!e.visible) return !1;
	let n = t.resolution;
	if (n < e.minResolution || n >= e.maxResolution) return !1;
	let r = t.zoom;
	return r > e.minZoom && r <= e.maxZoom;
}
//#endregion
//#region node_modules/quickselect/index.js
function po(e, t, n = 0, r = e.length - 1, i = ho) {
	for (; r > n;) {
		if (r - n > 600) {
			let a = r - n + 1, o = t - n + 1, s = Math.log(a), c = .5 * Math.exp(2 * s / 3), l = .5 * Math.sqrt(s * c * (a - c) / a) * (o - a / 2 < 0 ? -1 : 1);
			po(e, t, Math.max(n, Math.floor(t - o * c / a + l)), Math.min(r, Math.floor(t + (a - o) * c / a + l)), i);
		}
		let a = e[t], o = n, s = r;
		for (mo(e, n, t), i(e[r], a) > 0 && mo(e, n, r); o < s;) {
			for (mo(e, o, s), o++, s--; i(e[o], a) < 0;) o++;
			for (; i(e[s], a) > 0;) s--;
		}
		i(e[n], a) === 0 ? mo(e, n, s) : (s++, mo(e, s, r)), s <= t && (n = s + 1), t <= s && (r = s - 1);
	}
}
function mo(e, t, n) {
	let r = e[t];
	e[t] = e[n], e[n] = r;
}
function ho(e, t) {
	return e < t ? -1 : +(e > t);
}
//#endregion
//#region node_modules/rbush/index.js
var go = class {
	constructor(e = 9) {
		this._maxEntries = Math.max(4, e), this._minEntries = Math.max(2, Math.ceil(this._maxEntries * .4)), this.clear();
	}
	all() {
		return this._all(this.data, []);
	}
	search(e) {
		let t = this.data, n = [];
		if (!Oo(e, t)) return n;
		let r = this.toBBox, i = [];
		for (; t;) {
			for (let a = 0; a < t.children.length; a++) {
				let o = t.children[a], s = t.leaf ? r(o) : o;
				Oo(e, s) && (t.leaf ? n.push(o) : Do(e, s) ? this._all(o, n) : i.push(o));
			}
			t = i.pop();
		}
		return n;
	}
	collides(e) {
		let t = this.data;
		if (!Oo(e, t)) return !1;
		let n = [];
		for (; t;) {
			for (let r = 0; r < t.children.length; r++) {
				let i = t.children[r], a = t.leaf ? this.toBBox(i) : i;
				if (Oo(e, a)) {
					if (t.leaf || Do(e, a)) return !0;
					n.push(i);
				}
			}
			t = n.pop();
		}
		return !1;
	}
	load(e) {
		if (!(e && e.length)) return this;
		if (e.length < this._minEntries) {
			for (let t = 0; t < e.length; t++) this.insert(e[t]);
			return this;
		}
		let t = this._build(e.slice(), 0, e.length - 1, 0);
		if (!this.data.children.length) this.data = t;
		else if (this.data.height === t.height) this._splitRoot(this.data, t);
		else {
			if (this.data.height < t.height) {
				let e = this.data;
				this.data = t, t = e;
			}
			this._insert(t, this.data.height - t.height - 1, !0);
		}
		return this;
	}
	insert(e) {
		return e && this._insert(e, this.data.height - 1), this;
	}
	clear() {
		return this.data = ko([]), this;
	}
	remove(e, t) {
		if (!e) return this;
		let n = this.data, r = this.toBBox(e), i = [], a = [], o, s, c;
		for (; n || i.length;) {
			if (n || (n = i.pop(), s = i[i.length - 1], o = a.pop(), c = !0), n.leaf) {
				let r = _o(e, n.children, t);
				if (r !== -1) return n.children.splice(r, 1), i.push(n), this._condense(i), this;
			}
			!c && !n.leaf && Do(n, r) ? (i.push(n), a.push(o), o = 0, s = n, n = n.children[0]) : s ? (o++, n = s.children[o], c = !1) : n = null;
		}
		return this;
	}
	toBBox(e) {
		return e;
	}
	compareMinX(e, t) {
		return e.minX - t.minX;
	}
	compareMinY(e, t) {
		return e.minY - t.minY;
	}
	toJSON() {
		return this.data;
	}
	fromJSON(e) {
		return this.data = e, this;
	}
	_all(e, t) {
		let n = [];
		for (; e;) e.leaf ? t.push(...e.children) : n.push(...e.children), e = n.pop();
		return t;
	}
	_build(e, t, n, r) {
		let i = n - t + 1, a = this._maxEntries, o;
		if (i <= a) return o = ko(e.slice(t, n + 1)), vo(o, this.toBBox), o;
		r || (r = Math.ceil(Math.log(i) / Math.log(a)), a = Math.ceil(i / a ** (r - 1))), o = ko([]), o.leaf = !1, o.height = r;
		let s = Math.ceil(i / a), c = s * Math.ceil(Math.sqrt(a));
		Ao(e, t, n, c, this.compareMinX);
		for (let i = t; i <= n; i += c) {
			let t = Math.min(i + c - 1, n);
			Ao(e, i, t, s, this.compareMinY);
			for (let n = i; n <= t; n += s) {
				let i = Math.min(n + s - 1, t);
				o.children.push(this._build(e, n, i, r - 1));
			}
		}
		return vo(o, this.toBBox), o;
	}
	_chooseSubtree(e, t, n, r) {
		for (; r.push(t), !(t.leaf || r.length - 1 === n);) {
			let n = Infinity, r = Infinity, i;
			for (let a = 0; a < t.children.length; a++) {
				let o = t.children[a], s = Co(o), c = To(e, o) - s;
				c < r ? (r = c, n = s < n ? s : n, i = o) : c === r && s < n && (n = s, i = o);
			}
			t = i || t.children[0];
		}
		return t;
	}
	_insert(e, t, n) {
		let r = n ? e : this.toBBox(e), i = [], a = this._chooseSubtree(r, this.data, t, i);
		for (a.children.push(e), bo(a, r); t >= 0 && i[t].children.length > this._maxEntries;) this._split(i, t), t--;
		this._adjustParentBBoxes(r, i, t);
	}
	_split(e, t) {
		let n = e[t], r = n.children.length, i = this._minEntries;
		this._chooseSplitAxis(n, i, r);
		let a = this._chooseSplitIndex(n, i, r), o = ko(n.children.splice(a, n.children.length - a));
		o.height = n.height, o.leaf = n.leaf, vo(n, this.toBBox), vo(o, this.toBBox), t ? e[t - 1].children.push(o) : this._splitRoot(n, o);
	}
	_splitRoot(e, t) {
		this.data = ko([e, t]), this.data.height = e.height + 1, this.data.leaf = !1, vo(this.data, this.toBBox);
	}
	_chooseSplitIndex(e, t, n) {
		let r, i = Infinity, a = Infinity;
		for (let o = t; o <= n - t; o++) {
			let t = yo(e, 0, o, this.toBBox), s = yo(e, o, n, this.toBBox), c = Eo(t, s), l = Co(t) + Co(s);
			c < i ? (i = c, r = o, a = l < a ? l : a) : c === i && l < a && (a = l, r = o);
		}
		return r || n - t;
	}
	_chooseSplitAxis(e, t, n) {
		let r = e.leaf ? this.compareMinX : xo, i = e.leaf ? this.compareMinY : So;
		this._allDistMargin(e, t, n, r) < this._allDistMargin(e, t, n, i) && e.children.sort(r);
	}
	_allDistMargin(e, t, n, r) {
		e.children.sort(r);
		let i = this.toBBox, a = yo(e, 0, t, i), o = yo(e, n - t, n, i), s = wo(a) + wo(o);
		for (let r = t; r < n - t; r++) {
			let t = e.children[r];
			bo(a, e.leaf ? i(t) : t), s += wo(a);
		}
		for (let r = n - t - 1; r >= t; r--) {
			let t = e.children[r];
			bo(o, e.leaf ? i(t) : t), s += wo(o);
		}
		return s;
	}
	_adjustParentBBoxes(e, t, n) {
		for (let r = n; r >= 0; r--) bo(t[r], e);
	}
	_condense(e) {
		for (let t = e.length - 1, n; t >= 0; t--) e[t].children.length === 0 ? t > 0 ? (n = e[t - 1].children, n.splice(n.indexOf(e[t]), 1)) : this.clear() : vo(e[t], this.toBBox);
	}
};
function _o(e, t, n) {
	if (!n) return t.indexOf(e);
	for (let r = 0; r < t.length; r++) if (n(e, t[r])) return r;
	return -1;
}
function vo(e, t) {
	yo(e, 0, e.children.length, t, e);
}
function yo(e, t, n, r, i) {
	i ||= ko(null), i.minX = Infinity, i.minY = Infinity, i.maxX = -Infinity, i.maxY = -Infinity;
	for (let a = t; a < n; a++) {
		let t = e.children[a];
		bo(i, e.leaf ? r(t) : t);
	}
	return i;
}
function bo(e, t) {
	return e.minX = Math.min(e.minX, t.minX), e.minY = Math.min(e.minY, t.minY), e.maxX = Math.max(e.maxX, t.maxX), e.maxY = Math.max(e.maxY, t.maxY), e;
}
function xo(e, t) {
	return e.minX - t.minX;
}
function So(e, t) {
	return e.minY - t.minY;
}
function Co(e) {
	return (e.maxX - e.minX) * (e.maxY - e.minY);
}
function wo(e) {
	return e.maxX - e.minX + (e.maxY - e.minY);
}
function To(e, t) {
	return (Math.max(t.maxX, e.maxX) - Math.min(t.minX, e.minX)) * (Math.max(t.maxY, e.maxY) - Math.min(t.minY, e.minY));
}
function Eo(e, t) {
	let n = Math.max(e.minX, t.minX), r = Math.max(e.minY, t.minY), i = Math.min(e.maxX, t.maxX), a = Math.min(e.maxY, t.maxY);
	return Math.max(0, i - n) * Math.max(0, a - r);
}
function Do(e, t) {
	return e.minX <= t.minX && e.minY <= t.minY && t.maxX <= e.maxX && t.maxY <= e.maxY;
}
function Oo(e, t) {
	return t.minX <= e.maxX && t.minY <= e.maxY && t.maxX >= e.minX && t.maxY >= e.minY;
}
function ko(e) {
	return {
		children: e,
		height: 1,
		leaf: !0,
		minX: Infinity,
		minY: Infinity,
		maxX: -Infinity,
		maxY: -Infinity
	};
}
function Ao(e, t, n, r, i) {
	let a = [t, n];
	for (; a.length;) {
		if (n = a.pop(), t = a.pop(), n - t <= r) continue;
		let o = t + Math.ceil((n - t) / r / 2) * r;
		po(e, o, t, n, i), a.push(t, o, o, n);
	}
}
//#endregion
//#region node_modules/ol/color.js
var jo = [
	NaN,
	NaN,
	NaN,
	0
], Mo;
function No() {
	return Mo ||= y(1, 1, void 0, {
		willReadFrequently: !0,
		desynchronized: !0
	}), Mo;
}
var Po = /^rgba?\(\s*(\d+%?)\s+(\d+%?)\s+(\d+%?)(?:\s*\/\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, Fo = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, Io = /^rgba?\(\s*(\d+%)\s*,\s*(\d+%)\s*,\s*(\d+%)(?:\s*,\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, Lo = /^#([\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$/i;
function Ro(e, t) {
	return e.endsWith("%") ? Number(e.substring(0, e.length - 1)) / t : Number(e);
}
function zo(e) {
	throw Error("failed to parse \"" + e + "\" as color");
}
function Bo(e) {
	if (e.toLowerCase().startsWith("rgb")) {
		let t = e.match(Fo) || e.match(Po) || e.match(Io);
		if (t) {
			let e = t[4], n = 100 / 255;
			return [
				St(Ro(t[1], n) + .5 | 0, 0, 255),
				St(Ro(t[2], n) + .5 | 0, 0, 255),
				St(Ro(t[3], n) + .5 | 0, 0, 255),
				e === void 0 ? 1 : St(Ro(e, 100), 0, 1)
			];
		}
		zo(e);
	}
	if (e.startsWith("#")) {
		if (Lo.test(e)) {
			let t = e.substring(1), n = t.length <= 4 ? 1 : 2, r = [
				0,
				0,
				0,
				255
			];
			for (let e = 0, i = t.length; e < i; e += n) {
				let i = parseInt(t.substring(e, e + n), 16);
				n === 1 && (i += i << 4), r[e / n] = i;
			}
			return r[3] /= 255, r;
		}
		zo(e);
	}
	let t = No();
	t.fillStyle = "#abcdef";
	let n = t.fillStyle;
	t.fillStyle = e, t.fillStyle === n && (t.fillStyle = "#fedcba", n = t.fillStyle, t.fillStyle = e, t.fillStyle === n && zo(e));
	let r = t.fillStyle;
	if (r.startsWith("#") || r.startsWith("rgba")) return Bo(r);
	t.clearRect(0, 0, 1, 1), t.fillRect(0, 0, 1, 1);
	let i = Array.from(t.getImageData(0, 0, 1, 1).data);
	return i[3] = At(i[3] / 255, 3), i;
}
function Vo(e) {
	return typeof e == "string" ? e : es(e);
}
var Ho = 1024, Uo = {}, Wo = 0;
function Go(e) {
	if (e.length === 4) return e;
	let t = e.slice();
	return t[3] = 1, t;
}
function Ko(e) {
	return e > .0031308 ? e ** (1 / 2.4) * 269.025 - 14.025 : e * 3294.6;
}
function qo(e) {
	return e > .2068965 ? e ** 3 : (e - 4 / 29) * (108 / 841);
}
function Jo(e) {
	return e > 10.314724 ? ((e + 14.025) / 269.025) ** 2.4 : e / 3294.6;
}
function Yo(e) {
	return e > .0088564 ? e ** (1 / 3) : e / (108 / 841) + 4 / 29;
}
function Xo(e) {
	let t = Jo(e[0]), n = Jo(e[1]), r = Jo(e[2]), i = Yo(t * .222488403 + n * .716873169 + r * .06060791), a = 500 * (Yo(t * .452247074 + n * .399439023 + r * .148375274) - i), o = 200 * (i - Yo(t * .016863605 + n * .117638439 + r * .865350722)), s = 180 / Math.PI * Math.atan2(o, a);
	return [
		116 * i - 16,
		Math.sqrt(a * a + o * o),
		s < 0 ? s + 360 : s,
		e[3]
	];
}
function Zo(e) {
	let t = (e[0] + 16) / 116, n = e[1], r = e[2] * Math.PI / 180, i = qo(t), a = qo(t + n / 500 * Math.cos(r)), o = qo(t - n / 200 * Math.sin(r)), s = Ko(a * 3.021973625 - i * 1.617392459 - o * .404875592), c = Ko(a * -.943766287 + i * 1.916279586 + o * .027607165), l = Ko(a * .069407491 - i * .22898585 + o * 1.159737864);
	return [
		St(s + .5 | 0, 0, 255),
		St(c + .5 | 0, 0, 255),
		St(l + .5 | 0, 0, 255),
		e[3]
	];
}
function Qo(e) {
	if (e === "none") return jo;
	if (Uo.hasOwnProperty(e)) return Uo[e];
	if (Wo >= Ho) {
		let e = 0;
		for (let t in Uo) e++ & 3 || (delete Uo[t], --Wo);
	}
	let t = Bo(e);
	t.length !== 4 && zo(e);
	for (let n of t) isNaN(n) && zo(e);
	return Uo[e] = t, ++Wo, t;
}
function $o(e) {
	return Array.isArray(e) ? e : Qo(e);
}
function es(e) {
	let t = e[0];
	t != (t | 0) && (t = t + .5 | 0);
	let n = e[1];
	n != (n | 0) && (n = n + .5 | 0);
	let r = e[2];
	r != (r | 0) && (r = r + .5 | 0);
	let i = e[3] === void 0 ? 1 : Math.round(e[3] * 1e3) / 1e3;
	return "rgba(" + t + "," + n + "," + r + "," + i + ")";
}
//#endregion
//#region node_modules/ol/size.js
function ts(e) {
	return e[0] > 0 && e[1] > 0;
}
function ns(e, t, n) {
	return n === void 0 && (n = [0, 0]), n[0] = e[0] * t + .5 | 0, n[1] = e[1] * t + .5 | 0, n;
}
function rs(e, t) {
	return Array.isArray(e) ? e : (t === void 0 ? t = [e, e] : (t[0] = e, t[1] = e), t);
}
//#endregion
//#region node_modules/ol/expr/expression.js
var is = 0, as = 1 << is++, V = 1 << is++, H = 1 << is++, U = 1 << is++, os = 1 << is++, ss = 1 << is++, cs = 2 ** is - 1, ls = {
	[as]: "boolean",
	[V]: "number",
	[H]: "string",
	[U]: "color",
	[os]: "number[]",
	[ss]: "size"
}, us = Object.keys(ls).map(Number).sort(oe);
function ds(e) {
	return e in ls;
}
function fs(e) {
	let t = [];
	for (let n of us) ps(e, n) && t.push(ls[n]);
	return t.length === 0 ? "untyped" : t.length < 3 ? t.join(" or ") : t.slice(0, -1).join(", ") + ", or " + t[t.length - 1];
}
function ps(e, t) {
	return (e & t) === t;
}
function ms(e, t) {
	return !!(e & t);
}
function hs(e, t) {
	return e === t;
}
var gs = class {
	constructor(e, t) {
		if (!ds(e)) throw Error(`literal expressions must have a specific type, got ${fs(e)}`);
		this.type = e, this.value = t;
	}
}, _s = class {
	constructor(e, t, ...n) {
		this.type = e, this.operator = t, this.args = n;
	}
};
function vs(e) {
	return {
		variables: /* @__PURE__ */ new Map(),
		properties: /* @__PURE__ */ new Map(),
		featureId: !1,
		geometryType: !1,
		mCoordinate: !1,
		mapState: !1,
		inputVariables: e
	};
}
function ys(e, t, n) {
	switch (typeof e) {
		case "boolean":
			if (hs(t, H)) return new gs(H, e ? "true" : "false");
			if (!ps(t, as)) throw Error(`got a boolean, but expected ${fs(t)}`);
			return new gs(as, e);
		case "number":
			if (hs(t, ss)) return new gs(ss, rs(e));
			if (hs(t, as)) return new gs(as, !!e);
			if (hs(t, H)) return new gs(H, e.toString());
			if (!ps(t, V)) throw Error(`got a number, but expected ${fs(t)}`);
			return new gs(V, e);
		case "string":
			if (hs(t, U)) return new gs(U, Qo(e));
			if (hs(t, as)) return new gs(as, !!e);
			if (!ps(t, H)) throw Error(`got a string, but expected ${fs(t)}`);
			return new gs(H, e);
	}
	if (!Array.isArray(e)) throw Error("expression must be an array or a primitive value");
	if (e.length === 0) throw Error("empty expression");
	if (typeof e[0] == "string") return Ls(e, t, n);
	for (let t of e) if (typeof t != "number") throw Error("expected an array of numbers");
	if (hs(t, ss)) {
		if (e.length !== 2) throw Error(`expected an array of two values for a size, got ${e.length}`);
		return new gs(ss, e);
	}
	if (hs(t, U)) {
		if (e.length === 3) return new gs(U, [...e, 1]);
		if (e.length === 4) return new gs(U, e);
		throw Error(`expected an array of 3 or 4 values for a color, got ${e.length}`);
	}
	if (!ps(t, os)) throw Error(`got an array of numbers, but expected ${fs(t)}`);
	return new gs(os, e);
}
var W = {
	Get: "get",
	Var: "var",
	Concat: "concat",
	GeometryType: "geometry-type",
	LineMetric: "line-metric",
	Any: "any",
	All: "all",
	Not: "!",
	Resolution: "resolution",
	Zoom: "zoom",
	Time: "time",
	Equal: "==",
	NotEqual: "!=",
	GreaterThan: ">",
	GreaterThanOrEqualTo: ">=",
	LessThan: "<",
	LessThanOrEqualTo: "<=",
	Multiply: "*",
	Divide: "/",
	Add: "+",
	Subtract: "-",
	Clamp: "clamp",
	Mod: "%",
	Pow: "^",
	Abs: "abs",
	Floor: "floor",
	Ceil: "ceil",
	Round: "round",
	Sin: "sin",
	Cos: "cos",
	Atan: "atan",
	Sqrt: "sqrt",
	Match: "match",
	Between: "between",
	Interpolate: "interpolate",
	Coalesce: "coalesce",
	Case: "case",
	In: "in",
	Number: "number",
	String: "string",
	Array: "array",
	Color: "color",
	Id: "id",
	Band: "band",
	Palette: "palette",
	ToString: "to-string",
	Has: "has"
}, bs = {
	[W.Get]: q(G(1, Infinity), xs),
	[W.Var]: Ss(),
	[W.Has]: q(G(1, Infinity), xs),
	[W.Id]: q(Cs, Ds),
	[W.Concat]: q(G(2, Infinity), K(H)),
	[W.GeometryType]: q(ws, Ds),
	[W.LineMetric]: q(Ts, Ds),
	[W.Resolution]: q(Es, Ds),
	[W.Zoom]: q(Es, Ds),
	[W.Time]: q(Es, Ds),
	[W.Any]: q(G(2, Infinity), K(as)),
	[W.All]: q(G(2, Infinity), K(as)),
	[W.Not]: q(G(1, 1), K(as)),
	[W.Equal]: q(G(2, 2), ks()),
	[W.NotEqual]: q(G(2, 2), ks()),
	[W.GreaterThan]: q(G(2, 2), K(V)),
	[W.GreaterThanOrEqualTo]: q(G(2, 2), K(V)),
	[W.LessThan]: q(G(2, 2), K(V)),
	[W.LessThanOrEqualTo]: q(G(2, 2), K(V)),
	[W.Multiply]: q(G(2, Infinity), Os),
	[W.Coalesce]: q(G(2, Infinity), Os),
	[W.Divide]: q(G(2, 2), K(V)),
	[W.Add]: q(G(2, Infinity), K(V)),
	[W.Subtract]: q(G(2, 2), K(V)),
	[W.Clamp]: q(G(3, 3), K(V)),
	[W.Mod]: q(G(2, 2), K(V)),
	[W.Pow]: q(G(2, 2), K(V)),
	[W.Abs]: q(G(1, 1), K(V)),
	[W.Floor]: q(G(1, 1), K(V)),
	[W.Ceil]: q(G(1, 1), K(V)),
	[W.Round]: q(G(1, 1), K(V)),
	[W.Sin]: q(G(1, 1), K(V)),
	[W.Cos]: q(G(1, 1), K(V)),
	[W.Atan]: q(G(1, 2), K(V)),
	[W.Sqrt]: q(G(1, 1), K(V)),
	[W.Match]: q(G(4, Infinity), js, Ms),
	[W.Between]: q(G(3, 3), K(V)),
	[W.Interpolate]: q(G(6, Infinity), js, Ns),
	[W.Case]: q(G(3, Infinity), As, Ps),
	[W.In]: q(G(2, 2), Fs),
	[W.Number]: q(G(1, Infinity), K(cs)),
	[W.String]: q(G(1, Infinity), K(cs)),
	[W.Array]: q(G(1, Infinity), K(V)),
	[W.Color]: q(G(1, 4), K(V)),
	[W.Band]: q(G(1, 3), K(V)),
	[W.Palette]: q(G(2, 2), Is),
	[W.ToString]: q(G(1, 1), K(as | V | H | U))
};
function xs(e, t, n) {
	let r = e.length - 1, i = Array(r);
	for (let a = 0; a < r; ++a) {
		let r = e[a + 1];
		switch (typeof r) {
			case "number":
				i[a] = new gs(V, r);
				break;
			case "string":
				i[a] = new gs(H, r);
				break;
			default: throw Error(`expected a string key or numeric array index for a get operation, got ${r}`);
		}
		a === 0 && n.properties.set(String(r), t);
	}
	return i;
}
function Ss() {
	return function(e, t, n) {
		let r = e[1];
		if (typeof r != "string") throw Error("expected a string argument for var operation");
		let i = t, a = n.inputVariables?.[r];
		if (a !== void 0) {
			let e = ys(a, cs, n);
			if (!(e instanceof gs)) throw Error(`style variables should only be literal values (no expressions!), variable name: ${r}`);
			let o = e.type;
			if (typeof a == "string" && ms(i, U) && !ms(i, H) ? o = U : Array.isArray(a) && a.length === 2 && ms(i, ss) && !ms(i, os) && (o = ss), i &= o, i === 0) throw Error(`the type expected from the var operator (${fs(t)}) did not have any overlap with the type of the corresponding style variables (${fs(o)}), variable name: ${r}`);
		}
		if (n.variables.has(r)) {
			let e = n.variables.get(r);
			if (i &= e, i === 0) throw Error(`a new type expected from the var operator (${fs(t)}) did not have any overlap with the previous type expected for it (${fs(e)}), variable name: ${r}`);
		}
		return n.variables.set(r, i), new _s(i, "var", new gs(H, r));
	};
}
function Cs(e, t, n) {
	n.featureId = !0;
}
function ws(e, t, n) {
	n.geometryType = !0;
}
function Ts(e, t, n) {
	n.mCoordinate = !0;
}
function Es(e, t, n) {
	n.mapState = !0;
}
function Ds(e, t, n) {
	let r = e[0];
	if (e.length !== 1) throw Error(`expected no arguments for ${r} operation`);
	return [];
}
function G(e, t) {
	return function(n, r, i) {
		let a = n[0], o = n.length - 1;
		if (e === t) {
			if (o !== e) throw Error(`expected ${e} argument${e === 1 ? "" : "s"} for ${a}, got ${o}`);
		} else if (o < e || o > t) {
			let n = t === Infinity ? `${e} or more` : `${e} to ${t}`;
			throw Error(`expected ${n} arguments for ${a}, got ${o}`);
		}
	};
}
function Os(e, t, n) {
	let r = e.length - 1, i = Array(r);
	for (let a = 0; a < r; ++a) {
		let r = ys(e[a + 1], t, n);
		i[a] = r;
	}
	return i;
}
function K(e) {
	return function(t, n, r) {
		let i = t.length - 1, a = Array(i);
		for (let n = 0; n < i; ++n) {
			let i = ys(t[n + 1], e, r);
			a[n] = i;
		}
		return a;
	};
}
function ks() {
	return function(e, t, n) {
		let r = e[0], i = e.length - 1, a = Array(i), o = cs;
		for (let t = 0; t < i; ++t) {
			let r = ys(e[t + 1], o, n);
			o &= r.type;
		}
		if (o === 0) throw Error(`no common type was found among the arguments of ${r}`);
		for (let t = 0; t < i; ++t) {
			let r = ys(e[t + 1], o, n);
			a[t] = r;
		}
		return a;
	};
}
function As(e, t, n) {
	let r = e[0], i = e.length - 1;
	if (i % 2 == 0) throw Error(`expected an odd number of arguments for ${r}, got ${i} instead`);
}
function js(e, t, n) {
	let r = e[0], i = e.length - 1;
	if (i % 2 == 1) throw Error(`expected an even number of arguments for operation ${r}, got ${i} instead`);
}
function Ms(e, t, n) {
	let r = e.length - 1, i = ys(e[e.length - 1], t, n), a = H | V | as, o = Array(r - 2);
	for (let t = 0; t < r - 2; t += 2) {
		try {
			let r = ys(e[t + 2], a, n);
			a &= r.type;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of match expression: ${e.message}`);
		}
		if (a === 0) throw Error("no common type was found among the arguments of match expression");
	}
	for (let t = 0; t < r - 2; t += 2) {
		try {
			let r = ys(e[t + 2], a, n);
			o[t] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of match expression: ${e.message}`);
		}
		try {
			let r = ys(e[t + 3], i.type, n);
			o[t + 1] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 2} of match expression: ${e.message}`);
		}
	}
	return [
		ys(e[1], a, n),
		...o,
		i
	];
}
function Ns(e, t, n) {
	let r = e[1], i;
	switch (r[0]) {
		case "linear":
			i = 1;
			break;
		case "exponential":
			let e = r[1];
			if (typeof e != "number" || e <= 0) throw Error(`expected a number base for exponential interpolation, got ${JSON.stringify(e)} instead`);
			i = e;
			break;
		default: throw Error(`invalid interpolation type: ${JSON.stringify(r)}`);
	}
	let a = new gs(V, i), o;
	try {
		o = ys(e[2], V, n);
	} catch (e) {
		throw Error(`failed to parse argument 1 in interpolate expression: ${e.message}`);
	}
	let s = Array(e.length - 3);
	for (let r = 0; r < s.length; r += 2) {
		try {
			let t = ys(e[r + 3], V, n);
			s[r] = t;
		} catch (e) {
			throw Error(`failed to parse argument ${r + 2} for interpolate expression: ${e.message}`);
		}
		try {
			let i = ys(e[r + 4], t, n);
			s[r + 1] = i;
		} catch (e) {
			throw Error(`failed to parse argument ${r + 3} for interpolate expression: ${e.message}`);
		}
	}
	return [
		a,
		o,
		...s
	];
}
function Ps(e, t, n) {
	let r = ys(e[e.length - 1], t, n), i = Array(e.length - 1);
	for (let t = 0; t < i.length - 1; t += 2) {
		try {
			let r = ys(e[t + 1], as, n);
			i[t] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t} of case expression: ${e.message}`);
		}
		try {
			let a = ys(e[t + 2], r.type, n);
			i[t + 1] = a;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of case expression: ${e.message}`);
		}
	}
	return i[i.length - 1] = r, i;
}
function Fs(e, t, n) {
	let r = e[2];
	if (!Array.isArray(r)) throw Error("the second argument for the \"in\" operator must be an array");
	let i;
	if (r[0] === "literal") {
		if (r = r[1], !Array.isArray(r)) throw Error("failed to parse \"in\" expression: the literal operator must be followed by an array");
	} else if (typeof r[0] == "string") throw Error("for the \"in\" operator, a string array should be wrapped in a \"literal\" operator to disambiguate from expressions");
	i = typeof r[0] == "string" ? H : V;
	let a = Array(r.length);
	for (let e = 0; e < a.length; e++) try {
		let t = ys(r[e], i, n);
		a[e] = t;
	} catch (t) {
		throw Error(`failed to parse haystack item ${e} for "in" expression: ${t.message}`);
	}
	return [ys(e[1], i, n), ...a];
}
function Is(e, t, n) {
	let r;
	try {
		r = ys(e[1], V, n);
	} catch (e) {
		throw Error(`failed to parse first argument in palette expression: ${e.message}`);
	}
	let i = e[2];
	if (!Array.isArray(i)) throw Error("the second argument of palette must be an array");
	let a = Array(i.length);
	for (let e = 0; e < a.length; e++) {
		let t;
		try {
			t = ys(i[e], U, n);
		} catch (t) {
			throw Error(`failed to parse color at index ${e} in palette expression: ${t.message}`);
		}
		if (!(t instanceof gs)) throw Error(`the palette color at index ${e} must be a literal value`);
		a[e] = t;
	}
	return [r, ...a];
}
function q(...e) {
	return function(t, n, r) {
		let i = t[0], a;
		for (let i = 0; i < e.length; i++) {
			let o = e[i](t, n, r);
			if (i == e.length - 1) {
				if (!o) throw Error("expected last argument validator to return the parsed args");
				a = o;
			}
		}
		return new _s(n, i, ...a);
	};
}
function Ls(e, t, n) {
	let r = e[0], i = bs[r];
	if (!i) throw Error(`unknown operator: ${r}`);
	return i(e, t, n);
}
function Rs(e) {
	if (!e) return "";
	let t = e.getType();
	switch (t) {
		case "Point":
		case "LineString":
		case "Polygon": return t;
		case "MultiPoint":
		case "MultiLineString":
		case "MultiPolygon": return t.substring(5);
		case "Circle": return "Polygon";
		case "GeometryCollection": return Rs(e.getGeometries()[0]);
		default: return "";
	}
}
//#endregion
//#region node_modules/ol/expr/cpu.js
function zs() {
	return {
		variables: {},
		properties: {},
		resolution: NaN,
		featureId: null,
		geometryType: ""
	};
}
function Bs(e, t, n) {
	return Vs(ys(e, t, n), n);
}
function Vs(e, t) {
	if (e instanceof gs) {
		if (e.type === U && typeof e.value == "string") {
			let t = Qo(e.value);
			return function() {
				return t;
			};
		}
		return function() {
			return e.value;
		};
	}
	let n = e.operator;
	switch (n) {
		case W.Number:
		case W.String:
		case W.Coalesce: return Hs(e, t);
		case W.Get:
		case W.Var:
		case W.Has: return Us(e, t);
		case W.Id: return (e) => e.featureId;
		case W.GeometryType: return (e) => e.geometryType;
		case W.Concat: {
			let n = e.args.map((e) => Vs(e, t));
			return (e) => "".concat(...n.map((t) => t(e).toString()));
		}
		case W.Resolution: return (e) => e.resolution;
		case W.Any:
		case W.All:
		case W.Between:
		case W.In:
		case W.Not: return Gs(e, t);
		case W.Equal:
		case W.NotEqual:
		case W.LessThan:
		case W.LessThanOrEqualTo:
		case W.GreaterThan:
		case W.GreaterThanOrEqualTo: return Ws(e, t);
		case W.Multiply:
		case W.Divide:
		case W.Add:
		case W.Subtract:
		case W.Clamp:
		case W.Mod:
		case W.Pow:
		case W.Abs:
		case W.Floor:
		case W.Ceil:
		case W.Round:
		case W.Sin:
		case W.Cos:
		case W.Atan:
		case W.Sqrt: return Ks(e, t);
		case W.Case: return qs(e, t);
		case W.Match: return Js(e, t);
		case W.Interpolate: return Ys(e, t);
		case W.ToString: return Xs(e, t);
		default: throw Error(`Unsupported operator ${n}`);
	}
}
function Hs(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = Vs(e.args[n], t);
	switch (n) {
		case W.Coalesce: return (e) => {
			for (let t = 0; t < r; ++t) {
				let n = i[t](e);
				if (n != null) return n;
			}
			throw Error("Expected one of the values to be non-null");
		};
		case W.Number:
		case W.String: return (e) => {
			for (let t = 0; t < r; ++t) {
				let r = i[t](e);
				if (typeof r === n) return r;
			}
			throw Error(`Expected one of the values to be a ${n}`);
		};
		default: throw Error(`Unsupported assertion operator ${n}`);
	}
}
function Us(e, t) {
	let n = e.args[0].value;
	switch (e.operator) {
		case W.Get: return (t) => {
			let r = e.args, i = t.properties[n];
			for (let e = 1, t = r.length; e < t; ++e) {
				let t = r[e].value;
				i = i[t];
			}
			return i;
		};
		case W.Var: return (e) => e.variables[n];
		case W.Has: return (t) => {
			let r = e.args;
			if (!(n in t.properties)) return !1;
			let i = t.properties[n];
			for (let e = 1, t = r.length; e < t; ++e) {
				let t = r[e].value;
				if (!i || !Object.hasOwn(i, t)) return !1;
				i = i[t];
			}
			return !0;
		};
		default: throw Error(`Unsupported accessor operator ${e.operator}`);
	}
}
function Ws(e, t) {
	let n = e.operator, r = Vs(e.args[0], t), i = Vs(e.args[1], t);
	switch (n) {
		case W.Equal: return (e) => r(e) === i(e);
		case W.NotEqual: return (e) => r(e) !== i(e);
		case W.LessThan: return (e) => r(e) < i(e);
		case W.LessThanOrEqualTo: return (e) => r(e) <= i(e);
		case W.GreaterThan: return (e) => r(e) > i(e);
		case W.GreaterThanOrEqualTo: return (e) => r(e) >= i(e);
		default: throw Error(`Unsupported comparison operator ${n}`);
	}
}
function Gs(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = Vs(e.args[n], t);
	switch (n) {
		case W.Any: return (e) => {
			for (let t = 0; t < r; ++t) if (i[t](e)) return !0;
			return !1;
		};
		case W.All: return (e) => {
			for (let t = 0; t < r; ++t) if (!i[t](e)) return !1;
			return !0;
		};
		case W.Between: return (e) => {
			let t = i[0](e), n = i[1](e), r = i[2](e);
			return t >= n && t <= r;
		};
		case W.In: return (e) => {
			let t = i[0](e);
			for (let n = 1; n < r; ++n) if (t === i[n](e)) return !0;
			return !1;
		};
		case W.Not: return (e) => !i[0](e);
		default: throw Error(`Unsupported logical operator ${n}`);
	}
}
function Ks(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = Vs(e.args[n], t);
	switch (n) {
		case W.Multiply: return (e) => {
			let t = 1;
			for (let n = 0; n < r; ++n) t *= i[n](e);
			return t;
		};
		case W.Divide: return (e) => i[0](e) / i[1](e);
		case W.Add: return (e) => {
			let t = 0;
			for (let n = 0; n < r; ++n) t += i[n](e);
			return t;
		};
		case W.Subtract: return (e) => i[0](e) - i[1](e);
		case W.Clamp: return (e) => {
			let t = i[0](e), n = i[1](e);
			if (t < n) return n;
			let r = i[2](e);
			return t > r ? r : t;
		};
		case W.Mod: return (e) => i[0](e) % i[1](e);
		case W.Pow: return (e) => i[0](e) ** +i[1](e);
		case W.Abs: return (e) => Math.abs(i[0](e));
		case W.Floor: return (e) => Math.floor(i[0](e));
		case W.Ceil: return (e) => Math.ceil(i[0](e));
		case W.Round: return (e) => Math.round(i[0](e));
		case W.Sin: return (e) => Math.sin(i[0](e));
		case W.Cos: return (e) => Math.cos(i[0](e));
		case W.Atan: return r === 2 ? (e) => Math.atan2(i[0](e), i[1](e)) : (e) => Math.atan(i[0](e));
		case W.Sqrt: return (e) => Math.sqrt(i[0](e));
		default: throw Error(`Unsupported numeric operator ${n}`);
	}
}
function qs(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Vs(e.args[i], t);
	return (e) => {
		for (let t = 0; t < n - 1; t += 2) if (r[t](e)) return r[t + 1](e);
		return r[n - 1](e);
	};
}
function Js(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Vs(e.args[i], t);
	return (e) => {
		let t = r[0](e);
		for (let i = 1; i < n - 1; i += 2) if (t === r[i](e)) return r[i + 1](e);
		return r[n - 1](e);
	};
}
function Ys(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Vs(e.args[i], t);
	return (e) => {
		let t = r[0](e), i = r[1](e), a, o;
		for (let s = 2; s < n; s += 2) {
			let n = r[s](e), c = r[s + 1](e), l = Array.isArray(c);
			if (l && (c = Go(c)), n >= i) return s === 2 ? c : l ? Qs(t, i, a, o, n, c) : Zs(t, i, a, o, n, c);
			a = n, o = c;
		}
		return o;
	};
}
function Xs(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = Vs(e.args[n], t);
	switch (n) {
		case W.ToString: return (t) => {
			let n = i[0](t);
			return e.args[0].type === U ? es(n) : n.toString();
		};
		default: throw Error(`Unsupported convert operator ${n}`);
	}
}
function Zs(e, t, n, r, i, a) {
	let o = i - n;
	if (o === 0) return r;
	let s = t - n;
	return r + (e === 1 ? s / o : (e ** +s - 1) / (e ** +o - 1)) * (a - r);
}
function Qs(e, t, n, r, i, a) {
	if (i - n === 0) return r;
	let o = Xo(r), s = Xo(a), c = s[2] - o[2];
	return c > 180 ? c -= 360 : c < -180 && (c += 360), Zo([
		Zs(e, t, n, o[0], i, s[0]),
		Zs(e, t, n, o[1], i, s[1]),
		o[2] + Zs(e, t, n, 0, i, c),
		Zs(e, t, n, r[3], i, a[3])
	]);
}
//#endregion
//#region node_modules/ol/ImageState.js
var J = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
};
//#endregion
//#region node_modules/ol/Image.js
function $s(e, t, n) {
	let r = e, i = !0, a = !1, o = !1, s = [ne(r, M.LOAD, function() {
		o = !0, a || t();
	})];
	return r.src && _ ? (a = !0, r.decode().then(function() {
		i && t();
	}).catch(function(e) {
		i && (o ? t() : n());
	})) : s.push(ne(r, M.ERROR, n)), function() {
		i = !1, s.forEach(j);
	};
}
function ec(e, t) {
	return new Promise((n, r) => {
		function i() {
			o(), n(e);
		}
		function a() {
			o(), r(/* @__PURE__ */ Error("Image load error"));
		}
		function o() {
			e.removeEventListener("load", i), e.removeEventListener("error", a);
		}
		e.addEventListener("load", i), e.addEventListener("error", a), t && (e.src = t);
	});
}
function tc(e, t) {
	return t && (e.src = t), e.src && _ ? new Promise((t, n) => e.decode().then(() => t(e)).catch((r) => e.complete && e.width ? t(e) : n(r))) : ec(e);
}
//#endregion
//#region node_modules/ol/style/IconImageCache.js
var nc = class {
	constructor() {
		this.cache_ = {}, this.patternCache_ = {}, this.cacheSize_ = 0, this.maxCacheSize_ = 1024;
	}
	clear() {
		this.cache_ = {}, this.patternCache_ = {}, this.cacheSize_ = 0;
	}
	canExpireCache() {
		return this.cacheSize_ > this.maxCacheSize_;
	}
	expire() {
		if (this.canExpireCache()) {
			let e = 0;
			for (let t in this.cache_) {
				let n = this.cache_[t];
				!(e++ & 3) && !n.hasListener() && (delete this.cache_[t], delete this.patternCache_[t], --this.cacheSize_);
			}
		}
	}
	get(e, t) {
		let n = rc(e, t);
		return n in this.cache_ ? this.cache_[n] : null;
	}
	getPattern(e, t) {
		let n = rc(e, t);
		return n in this.patternCache_ ? this.patternCache_[n] : null;
	}
	set(e, t, n, r) {
		let i = rc(e, t), a = i in this.cache_;
		this.cache_[i] = n, r && (n.getImageState() === J.IDLE && n.load(), n.getImageState() === J.LOADING ? n.ready().then(() => {
			this.patternCache_[i] = x().createPattern(n.getImage(1), "repeat");
		}) : this.patternCache_[i] = x().createPattern(n.getImage(1), "repeat")), a || ++this.cacheSize_;
	}
	setSize(e) {
		this.maxCacheSize_ = e, this.expire();
	}
};
function rc(e, t) {
	let n = t ? $o(t) : "null";
	return e + ":" + n;
}
var ic = new nc(), ac = null, oc = class extends ye {
	constructor(e, t, n, r, i) {
		super(), this.hitDetectionImage_ = null, this.image_ = e, this.crossOrigin_ = n?.crossOrigin, this.referrerPolicy_ = n?.referrerPolicy, this.canvas_ = {}, this.color_ = i, this.imageState_ = r === void 0 ? J.IDLE : r, this.size_ = e && e.width && e.height ? [e.width, e.height] : null, this.src_ = t, this.tainted_, this.ready_ = null;
	}
	initializeImage_() {
		this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_);
	}
	isTainted_() {
		if (this.tainted_ === void 0 && this.imageState_ === J.LOADED) {
			ac ||= y(1, 1, void 0, { willReadFrequently: !0 }), ac.drawImage(this.image_, 0, 0);
			try {
				ac.getImageData(0, 0, 1, 1), this.tainted_ = !1;
			} catch {
				ac = null, this.tainted_ = !0;
			}
		}
		return this.tainted_ === !0;
	}
	dispatchChangeEvent_() {
		this.dispatchEvent(M.CHANGE);
	}
	handleImageError_() {
		this.imageState_ = J.ERROR, this.dispatchChangeEvent_();
	}
	handleImageLoad_() {
		this.imageState_ = J.LOADED, this.size_ = [this.image_.width, this.image_.height], this.dispatchChangeEvent_();
	}
	getImage(e) {
		return this.image_ || this.initializeImage_(), this.replaceColor_(e), this.canvas_[e] ? this.canvas_[e] : this.image_;
	}
	setImage(e) {
		this.image_ = e;
	}
	getPixelRatio(e) {
		return this.replaceColor_(e), this.canvas_[e] ? e : 1;
	}
	getImageState() {
		return this.imageState_;
	}
	getHitDetectionImage() {
		if (this.image_ || this.initializeImage_(), !this.hitDetectionImage_) {
			if (this.isTainted_()) {
				let e = this.size_[0], t = this.size_[1], n = y(e, t);
				n.fillRect(0, 0, e, t), this.hitDetectionImage_ = n.canvas;
			} else this.hitDetectionImage_ = this.image_;
		}
		return this.hitDetectionImage_;
	}
	getSize() {
		return this.size_;
	}
	getSrc() {
		return this.src_;
	}
	load() {
		if (this.imageState_ === J.IDLE) {
			this.image_ || this.initializeImage_(), this.imageState_ = J.LOADING;
			try {
				this.src_ !== void 0 && (this.image_.src = this.src_);
			} catch {
				this.handleImageError_();
			}
			this.image_ instanceof HTMLImageElement && tc(this.image_, this.src_).then((e) => {
				this.image_ = e, this.handleImageLoad_();
			}).catch(this.handleImageError_.bind(this));
		}
	}
	replaceColor_(e) {
		if (!this.color_ || this.canvas_[e] || this.imageState_ !== J.LOADED) return;
		let t = this.image_, n = y(Math.ceil(t.width * e), Math.ceil(t.height * e)), r = n.canvas;
		n.scale(e, e), n.drawImage(t, 0, 0), n.globalCompositeOperation = "multiply", n.fillStyle = Vo(this.color_), n.fillRect(0, 0, r.width / e, r.height / e), n.globalCompositeOperation = "destination-in", n.drawImage(t, 0, 0), this.canvas_[e] = r;
	}
	ready() {
		return this.ready_ ||= new Promise((e) => {
			if (this.imageState_ === J.LOADED || this.imageState_ === J.ERROR) e();
			else {
				let t = () => {
					(this.imageState_ === J.LOADED || this.imageState_ === J.ERROR) && (this.removeEventListener(M.CHANGE, t), e());
				};
				this.addEventListener(M.CHANGE, t);
			}
		}), this.ready_;
	}
};
function sc(e, t, n, r, i, a) {
	let o = t === void 0 ? void 0 : ic.get(t, i);
	return o || (o = new oc(e, e && "src" in e ? e.src || void 0 : t, n, r, i), ic.set(t, i, o, a)), a && o && !ic.getPattern(t, i) && ic.set(t, i, o, a), o;
}
//#endregion
//#region node_modules/ol/colorlike.js
function cc(e) {
	return e ? Array.isArray(e) ? es(e) : typeof e == "object" && "src" in e ? lc(e) : e : null;
}
function lc(e) {
	if (!e.offset || !e.size) return ic.getPattern(e.src, e.color);
	let t = e.src + ":" + e.offset, n = ic.getPattern(t, e.color);
	if (n) return n;
	let r = ic.get(e.src, null);
	if (r.getImageState() !== J.LOADED) return null;
	let i = y(e.size[0], e.size[1]);
	return i.drawImage(r.getImage(1), e.offset[0], e.offset[1], e.size[0], e.size[1], 0, 0, e.size[0], e.size[1]), sc(i.canvas, t, void 0, J.LOADED, e.color, !0), ic.getPattern(t, e.color);
}
//#endregion
//#region node_modules/ol/render/canvas.js
var uc = "10px sans-serif", dc = "#000", fc = "round", pc = [], mc = "round", hc = "#000", gc = "center", _c = "middle", vc = [
	0,
	0,
	0,
	0
], yc = new we(), bc = null, xc, Sc = {}, Cc = /* @__PURE__ */ new Set([
	"serif",
	"sans-serif",
	"monospace",
	"cursive",
	"fantasy",
	"system-ui",
	"ui-serif",
	"ui-sans-serif",
	"ui-monospace",
	"ui-rounded",
	"emoji",
	"math",
	"fangsong"
]);
function wc(e, t, n) {
	return `${e} ${t} 16px "${n}"`;
}
var Tc = (function() {
	let e, t;
	async function n(e) {
		await t.ready;
		let n = u(e), r = n.families[0].toLowerCase(), i = n.weight, a = [];
		return t.forEach((e) => {
			let t = e.family.replace(/^['"]|['"]$/g, "").toLowerCase(), o = l[e.weight] || e.weight;
			t === r && e.style === n.style && o == i && a.push(e);
		}), a.length !== 0 && (await Promise.all(a.map((e) => e.load().then(() => !0, () => !1)))).some((e) => e);
	}
	async function r() {
		await t.ready;
		let i = !0, a = yc.getProperties(), o = Object.keys(a).filter((e) => a[e] < 100);
		for (let e = o.length - 1; e >= 0; --e) {
			let t = o[e], r = a[t];
			r < 100 && (await n(t) ? (k(Sc), yc.set(t, 100)) : (r += 10, yc.set(t, r, !0), r < 100 && (i = !1)));
		}
		e = void 0, i || (e = setTimeout(r, 100));
	}
	return async function(n) {
		t ||= g ? self.fonts : document.fonts;
		let i = u(n);
		if (!i) return;
		let a = i.families, o = !1;
		for (let e of a) {
			if (Cc.has(e)) continue;
			let t = wc(i.style, i.weight, e);
			yc.get(t) === void 0 && (yc.set(t, 0, !0), o = !0);
		}
		o && (clearTimeout(e), e = setTimeout(r, 100));
	};
})(), Ec = (function() {
	let e;
	return function(t) {
		let n = Sc[t];
		if (n == null) {
			if (g) {
				let e = u(t), r = Dc(t, "Žg");
				n = (isNaN(Number(e.lineHeight)) ? 1.2 : Number(e.lineHeight)) * (r.actualBoundingBoxAscent + r.actualBoundingBoxDescent);
			} else e || (e = document.createElement("div"), e.innerHTML = "M", e.style.minHeight = "0", e.style.maxHeight = "none", e.style.height = "auto", e.style.padding = "0", e.style.border = "none", e.style.position = "absolute", e.style.display = "block", e.style.left = "-99999px"), e.style.font = t, document.body.appendChild(e), n = e.offsetHeight, document.body.removeChild(e);
			Sc[t] = n;
		}
		return n;
	};
})();
function Dc(e, t) {
	return bc ||= y(1, 1), e != xc && (bc.font = e, xc = bc.font), bc.measureText(t);
}
function Oc(e, t) {
	return Dc(e, t).width;
}
function kc(e, t, n) {
	if (t in n) return n[t];
	let r = t.split("\n").reduce((t, n) => Math.max(t, Oc(e, n)), 0);
	return n[t] = r, r;
}
function Ac(e, t) {
	let n = [], r = [], i = [], a = 0, o = 0, s = 0, c = 0;
	for (let l = 0, u = t.length; l <= u; l += 2) {
		let d = t[l];
		if (d === "\n" || l === u) {
			a = Math.max(a, o), i.push(o), o = 0, s += c, c = 0;
			continue;
		}
		let f = t[l + 1] || e.font, p = Oc(f, d);
		n.push(p), o += p;
		let m = Ec(f);
		r.push(m), c = Math.max(c, m);
	}
	return {
		width: a,
		height: s,
		widths: n,
		heights: r,
		lineWidths: i
	};
}
function jc(e, t, n, r, i, a, o, s, c, l, u) {
	e.save(), n !== 1 && (e.globalAlpha === void 0 ? e.globalAlpha = (e) => e.globalAlpha *= n : e.globalAlpha *= n), t && e.transform.apply(e, t), r.contextInstructions ? (e.translate(c, l), e.scale(u[0], u[1]), Mc(r, e)) : u[0] < 0 || u[1] < 0 ? (e.translate(c, l), e.scale(u[0], u[1]), e.drawImage(r, i, a, o, s, 0, 0, o, s)) : e.drawImage(r, i, a, o, s, c, l, o * u[0], s * u[1]), e.restore();
}
function Mc(e, t) {
	let n = e.contextInstructions;
	for (let e = 0, r = n.length; e < r; e += 2) Array.isArray(n[e + 1]) ? t[n[e]].apply(t, n[e + 1]) : t[n[e]] = n[e + 1];
}
//#endregion
//#region node_modules/ol/style/Image.js
var Nc = class e {
	constructor(e) {
		this.opacity_ = e.opacity, this.rotateWithView_ = e.rotateWithView, this.rotation_ = e.rotation, this.scale_ = e.scale, this.scaleArray_ = rs(e.scale), this.displacement_ = e.displacement, this.declutterMode_ = e.declutterMode;
	}
	clone() {
		let t = this.getScale();
		return new e({
			opacity: this.getOpacity(),
			scale: Array.isArray(t) ? t.slice() : t,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			displacement: this.getDisplacement().slice(),
			declutterMode: this.getDeclutterMode()
		});
	}
	getOpacity() {
		return this.opacity_;
	}
	getRotateWithView() {
		return this.rotateWithView_;
	}
	getRotation() {
		return this.rotation_;
	}
	getScale() {
		return this.scale_;
	}
	getScaleArray() {
		return this.scaleArray_;
	}
	getDisplacement() {
		return this.displacement_;
	}
	getDeclutterMode() {
		return this.declutterMode_;
	}
	getAnchor() {
		return P();
	}
	getImage(e) {
		return P();
	}
	getHitDetectionImage() {
		return P();
	}
	getPixelRatio(e) {
		return 1;
	}
	getImageState() {
		return P();
	}
	getImageSize() {
		return P();
	}
	getOrigin() {
		return P();
	}
	getSize() {
		return P();
	}
	setDisplacement(e) {
		this.displacement_ = e;
	}
	setOpacity(e) {
		this.opacity_ = e;
	}
	setRotateWithView(e) {
		this.rotateWithView_ = e;
	}
	setRotation(e) {
		this.rotation_ = e;
	}
	setScale(e) {
		this.scale_ = e, this.scaleArray_ = rs(e);
	}
	listenImageChange(e) {
		P();
	}
	load() {
		P();
	}
	unlistenImageChange(e) {
		P();
	}
	ready() {
		return Promise.resolve();
	}
}, Pc = class e extends Nc {
	constructor(e) {
		super({
			opacity: 1,
			rotateWithView: e.rotateWithView !== void 0 && e.rotateWithView,
			rotation: e.rotation === void 0 ? 0 : e.rotation,
			scale: e.scale === void 0 ? 1 : e.scale,
			displacement: e.displacement === void 0 ? [0, 0] : e.displacement,
			declutterMode: e.declutterMode
		}), this.hitDetectionCanvas_ = null, this.fill_ = e.fill === void 0 ? null : e.fill, this.origin_ = [0, 0], this.points_ = e.points, this.radius = e.radius, this.radius2_ = e.radius2, this.angle_ = e.angle === void 0 ? 0 : e.angle, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.size_, this.renderOptions_, this.imageState_ = this.fill_ && this.fill_.loading() ? J.LOADING : J.LOADED, this.imageState_ === J.LOADING && this.ready().then(() => this.imageState_ = J.LOADED), this.render();
	}
	clone() {
		let t = this.getScale(), n = new e({
			fill: this.getFill() ? this.getFill().clone() : void 0,
			points: this.getPoints(),
			radius: this.getRadius(),
			radius2: this.getRadius2(),
			angle: this.getAngle(),
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			scale: Array.isArray(t) ? t.slice() : t,
			displacement: this.getDisplacement().slice(),
			declutterMode: this.getDeclutterMode()
		});
		return n.setOpacity(this.getOpacity()), n;
	}
	getAnchor() {
		let e = this.size_, t = this.getDisplacement(), n = this.getScaleArray();
		return [e[0] / 2 - t[0] / n[0], e[1] / 2 + t[1] / n[1]];
	}
	getAngle() {
		return this.angle_;
	}
	getFill() {
		return this.fill_;
	}
	setFill(e) {
		this.fill_ = e, this.render();
	}
	getHitDetectionImage() {
		return this.hitDetectionCanvas_ ||= this.createHitDetectionCanvas_(this.renderOptions_), this.hitDetectionCanvas_;
	}
	getImage(e) {
		let t = this.fill_?.getKey(), n = `${e},${this.angle_},${this.radius},${this.radius2_},${this.points_},${t}` + Object.values(this.renderOptions_).join(","), r = ic.get(n, null)?.getImage(1);
		if (!r) {
			let t = this.renderOptions_, i = Math.ceil(t.size * e), a = y(i, i);
			this.draw_(t, a, e), r = a.canvas;
			let o = new oc(r, void 0, null, J.LOADED, null);
			ic.set(n, null, o), createImageBitmap(r).then((e) => {
				o.setImage(e);
			});
		}
		return r;
	}
	getPixelRatio(e) {
		return e;
	}
	getImageSize() {
		return this.size_;
	}
	getImageState() {
		return this.imageState_;
	}
	getOrigin() {
		return this.origin_;
	}
	getPoints() {
		return this.points_;
	}
	getRadius() {
		return this.radius;
	}
	setRadius(e) {
		this.radius !== e && (this.radius = e, this.render());
	}
	getRadius2() {
		return this.radius2_;
	}
	setRadius2(e) {
		this.radius2_ !== e && (this.radius2_ = e, this.render());
	}
	getSize() {
		return this.size_;
	}
	getStroke() {
		return this.stroke_;
	}
	setStroke(e) {
		this.stroke_ = e, this.render();
	}
	listenImageChange(e) {}
	load() {}
	unlistenImageChange(e) {}
	calculateLineJoinSize_(e, t, n) {
		if (t === 0 || this.points_ === Infinity || e !== "bevel" && e !== "miter") return t;
		let r = this.radius, i = this.radius2_ === void 0 ? r : this.radius2_;
		if (r < i) {
			let e = r;
			r = i, i = e;
		}
		let a = this.radius2_ === void 0 ? this.points_ : this.points_ * 2, o = 2 * Math.PI / a, s = i * Math.sin(o), c = Math.sqrt(i * i - s * s), l = r - c, u = Math.sqrt(s * s + l * l), d = u / s;
		if (e === "miter" && d <= n) return d * t;
		let f = t / 2 / d, p = t / 2 * (l / u), m = Math.sqrt((r + f) * (r + f) + p * p) - r;
		if (this.radius2_ === void 0 || e === "bevel") return m * 2;
		let h = r * Math.sin(o), g = Math.sqrt(r * r - h * h), _ = i - g, v = Math.sqrt(h * h + _ * _) / h;
		if (v <= n) {
			let e = v * t / 2 - i - r;
			return 2 * Math.max(m, e);
		}
		return m * 2;
	}
	createRenderOptions() {
		let e = fc, t = mc, n = 0, r = null, i = 0, a, o = 0;
		this.stroke_ && (a = cc(this.stroke_.getColor() ?? "#000"), o = this.stroke_.getWidth() ?? 1, r = this.stroke_.getLineDash(), i = this.stroke_.getLineDashOffset() ?? 0, t = this.stroke_.getLineJoin() ?? "round", e = this.stroke_.getLineCap() ?? "round", n = this.stroke_.getMiterLimit() ?? 10);
		let s = this.calculateLineJoinSize_(t, o, n), c = Math.max(this.radius, this.radius2_ || 0), l = Math.ceil(2 * c + s);
		return {
			strokeStyle: a,
			strokeWidth: o,
			size: l,
			lineCap: e,
			lineDash: r,
			lineDashOffset: i,
			lineJoin: t,
			miterLimit: n
		};
	}
	render() {
		this.renderOptions_ = this.createRenderOptions();
		let e = this.renderOptions_.size;
		this.hitDetectionCanvas_ = null, this.size_ = [e, e];
	}
	draw_(e, t, n) {
		if (t.scale(n, n), t.translate(e.size / 2, e.size / 2), this.createPath_(t), this.fill_) {
			let e = this.fill_.getColor();
			e === null && (e = dc), t.fillStyle = cc(e), t.fill();
		}
		e.strokeStyle && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineCap = e.lineCap, t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}
	createHitDetectionCanvas_(e) {
		let t;
		if (this.fill_) {
			let n = this.fill_.getColor(), r = 0;
			typeof n == "string" && (n = $o(n)), n === null ? r = 1 : Array.isArray(n) && (r = n.length === 4 ? n[3] : 1), r === 0 && (t = y(e.size, e.size), this.drawHitDetectionCanvas_(e, t));
		}
		return t ? t.canvas : this.getImage(1);
	}
	createPath_(e) {
		let t = this.points_, n = this.radius;
		if (t === Infinity) e.arc(0, 0, n, 0, 2 * Math.PI);
		else {
			let r = this.radius2_ === void 0 ? n : this.radius2_;
			this.radius2_ !== void 0 && (t *= 2);
			let i = this.angle_ - Math.PI / 2, a = 2 * Math.PI / t;
			for (let o = 0; o < t; o++) {
				let t = i + o * a, s = o % 2 == 0 ? n : r;
				e.lineTo(s * Math.cos(t), s * Math.sin(t));
			}
			e.closePath();
		}
	}
	drawHitDetectionCanvas_(e, t) {
		t.translate(e.size / 2, e.size / 2), this.createPath_(t), t.fillStyle = dc, t.fill(), e.strokeStyle && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}
	ready() {
		return this.fill_ ? this.fill_.ready() : Promise.resolve();
	}
}, Fc = class e extends Pc {
	constructor(e) {
		e ||= { radius: 5 }, super({
			points: Infinity,
			fill: e.fill,
			radius: e.radius,
			stroke: e.stroke,
			scale: e.scale === void 0 ? 1 : e.scale,
			rotation: e.rotation === void 0 ? 0 : e.rotation,
			rotateWithView: e.rotateWithView !== void 0 && e.rotateWithView,
			displacement: e.displacement === void 0 ? [0, 0] : e.displacement,
			declutterMode: e.declutterMode
		});
	}
	clone() {
		let t = this.getScale(), n = new e({
			fill: this.getFill() ? this.getFill().clone() : void 0,
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			radius: this.getRadius(),
			scale: Array.isArray(t) ? t.slice() : t,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			displacement: this.getDisplacement().slice(),
			declutterMode: this.getDeclutterMode()
		});
		return n.setOpacity(this.getOpacity()), n;
	}
}, Ic = class e {
	constructor(e) {
		e ||= {}, this.patternImage_ = null, this.color_ = null, e.color !== void 0 && this.setColor(e.color);
	}
	clone() {
		let t = this.getColor();
		return new e({ color: Array.isArray(t) ? t.slice() : t || void 0 });
	}
	getColor() {
		return this.color_;
	}
	setColor(e) {
		if (typeof e == "object" && e && "src" in e) {
			let t = sc(null, e.src, { crossOrigin: "anonymous" }, void 0, e.offset ? null : e.color ? e.color : null, !(e.offset && e.size));
			t.ready().then(() => {
				this.patternImage_ = null;
			}), t.getImageState() === J.IDLE && t.load(), t.getImageState() === J.LOADING && (this.patternImage_ = t);
		}
		this.color_ = e;
	}
	getKey() {
		let e = this.getColor();
		return e ? e instanceof CanvasPattern || e instanceof CanvasGradient ? F(e) : typeof e == "object" && "src" in e ? e.src + ":" + e.offset : $o(e).toString() : "";
	}
	loading() {
		return !!this.patternImage_;
	}
	ready() {
		return this.patternImage_ ? this.patternImage_.ready() : Promise.resolve();
	}
};
//#endregion
//#region node_modules/ol/style/Icon.js
function Lc(e, t, n, r) {
	return n !== void 0 && r !== void 0 ? [n / e, r / t] : n === void 0 ? r === void 0 ? 1 : r / t : n / e;
}
var Rc = class e extends Nc {
	constructor(e) {
		e ||= {};
		let t = e.opacity === void 0 ? 1 : e.opacity, n = e.rotation === void 0 ? 0 : e.rotation, r = e.scale === void 0 ? 1 : e.scale, i = e.rotateWithView !== void 0 && e.rotateWithView;
		super({
			opacity: t,
			rotation: n,
			scale: r,
			displacement: e.displacement === void 0 ? [0, 0] : e.displacement,
			rotateWithView: i,
			declutterMode: e.declutterMode
		}), this.anchor_ = e.anchor === void 0 ? [.5, .5] : e.anchor, this.normalizedAnchor_ = null, this.anchorOrigin_ = e.anchorOrigin === void 0 ? "top-left" : e.anchorOrigin, this.anchorXUnits_ = e.anchorXUnits === void 0 ? "fraction" : e.anchorXUnits, this.anchorYUnits_ = e.anchorYUnits === void 0 ? "fraction" : e.anchorYUnits, this.crossOrigin_ = e.crossOrigin === void 0 ? null : e.crossOrigin, this.referrerPolicy_ = e.referrerPolicy;
		let a = e.img === void 0 ? null : e.img, o = e.src;
		R(!(o !== void 0 && a), "`image` and `src` cannot be provided at the same time"), (o === void 0 || o.length === 0) && a && (o = a.src || F(a)), R(o !== void 0 && o.length > 0, "A defined and non-empty `src` or `image` must be provided"), R(e.width === void 0 && e.height === void 0 || e.scale === void 0, "`width` or `height` cannot be provided together with `scale`");
		let s;
		if (e.src === void 0 ? a !== void 0 && (s = "complete" in a ? a.complete ? a.src ? J.LOADED : J.IDLE : J.LOADING : J.LOADED) : s = J.IDLE, this.color_ = e.color === void 0 ? null : $o(e.color), this.iconImage_ = sc(a, o, {
			crossOrigin: this.crossOrigin_,
			referrerPolicy: this.referrerPolicy_
		}, s, this.color_), this.offset_ = e.offset === void 0 ? [0, 0] : e.offset, this.offsetOrigin_ = e.offsetOrigin === void 0 ? "top-left" : e.offsetOrigin, this.origin_ = null, this.size_ = e.size === void 0 ? null : e.size, this.initialOptions_, e.width !== void 0 || e.height !== void 0) {
			let t, n;
			if (e.size) [t, n] = e.size;
			else {
				let r = this.getImage(1);
				if (r.width && r.height) t = r.width, n = r.height;
				else if (r instanceof HTMLImageElement) {
					this.initialOptions_ = e;
					let t = () => {
						if (this.unlistenImageChange(t), !this.initialOptions_) return;
						let n = this.iconImage_.getSize();
						this.setScale(Lc(n[0], n[1], e.width, e.height));
					};
					this.listenImageChange(t);
					return;
				}
			}
			t !== void 0 && this.setScale(Lc(t, n, e.width, e.height));
		}
	}
	clone() {
		let t, n, r;
		return this.initialOptions_ ? (n = this.initialOptions_.width, r = this.initialOptions_.height) : (t = this.getScale(), t = Array.isArray(t) ? t.slice() : t), new e({
			anchor: this.anchor_.slice(),
			anchorOrigin: this.anchorOrigin_,
			anchorXUnits: this.anchorXUnits_,
			anchorYUnits: this.anchorYUnits_,
			color: this.color_ && this.color_.slice ? this.color_.slice() : this.color_ || void 0,
			crossOrigin: this.crossOrigin_,
			referrerPolicy: this.referrerPolicy_,
			offset: this.offset_.slice(),
			offsetOrigin: this.offsetOrigin_,
			opacity: this.getOpacity(),
			rotateWithView: this.getRotateWithView(),
			rotation: this.getRotation(),
			scale: t,
			width: n,
			height: r,
			size: this.size_ === null ? void 0 : this.size_.slice(),
			src: this.getSrc(),
			displacement: this.getDisplacement().slice(),
			declutterMode: this.getDeclutterMode()
		});
	}
	getAnchor() {
		let e = this.normalizedAnchor_;
		if (!e) {
			e = this.anchor_;
			let t = this.getSize();
			if (this.anchorXUnits_ == "fraction" || this.anchorYUnits_ == "fraction") {
				if (!t) return null;
				e = this.anchor_.slice(), this.anchorXUnits_ == "fraction" && (e[0] *= t[0]), this.anchorYUnits_ == "fraction" && (e[1] *= t[1]);
			}
			if (this.anchorOrigin_ != "top-left") {
				if (!t) return null;
				e === this.anchor_ && (e = this.anchor_.slice()), (this.anchorOrigin_ == "top-right" || this.anchorOrigin_ == "bottom-right") && (e[0] = -e[0] + t[0]), (this.anchorOrigin_ == "bottom-left" || this.anchorOrigin_ == "bottom-right") && (e[1] = -e[1] + t[1]);
			}
			this.normalizedAnchor_ = e;
		}
		let t = this.getDisplacement(), n = this.getScaleArray();
		return [e[0] - t[0] / n[0], e[1] + t[1] / n[1]];
	}
	setAnchor(e) {
		this.anchor_ = e, this.normalizedAnchor_ = null;
	}
	getColor() {
		return this.color_;
	}
	setColor(e) {
		let t = e ? $o(e) : null;
		if (this.color_ === t || this.color_ && t && this.color_.length === t.length && this.color_.every((e, n) => e === t[n])) return;
		this.color_ = t;
		let n = this.getSrc(), r = n === void 0 ? this.getHitDetectionImage() : null, i = n === void 0 ? this.iconImage_.getImageState() : J.IDLE;
		this.iconImage_ = sc(r, n, {
			crossOrigin: this.crossOrigin_,
			referrerPolicy: this.referrerPolicy_
		}, i, this.color_);
	}
	getImage(e) {
		return this.iconImage_.getImage(e);
	}
	getPixelRatio(e) {
		return this.iconImage_.getPixelRatio(e);
	}
	getImageSize() {
		return this.iconImage_.getSize();
	}
	getImageState() {
		return this.iconImage_.getImageState();
	}
	getHitDetectionImage() {
		return this.iconImage_.getHitDetectionImage();
	}
	getOrigin() {
		if (this.origin_) return this.origin_;
		let e = this.offset_;
		if (this.offsetOrigin_ != "top-left") {
			let t = this.getSize(), n = this.iconImage_.getSize();
			if (!t || !n) return null;
			e = e.slice(), (this.offsetOrigin_ == "top-right" || this.offsetOrigin_ == "bottom-right") && (e[0] = n[0] - t[0] - e[0]), (this.offsetOrigin_ == "bottom-left" || this.offsetOrigin_ == "bottom-right") && (e[1] = n[1] - t[1] - e[1]);
		}
		return this.origin_ = e, this.origin_;
	}
	getSrc() {
		return this.iconImage_.getSrc();
	}
	setSrc(e) {
		this.iconImage_ = sc(null, e, {
			crossOrigin: this.crossOrigin_,
			referrerPolicy: this.referrerPolicy_
		}, J.IDLE, this.color_);
	}
	getSize() {
		return this.size_ ? this.size_ : this.iconImage_.getSize();
	}
	getWidth() {
		let e = this.getScaleArray();
		if (this.size_) return this.size_[0] * e[0];
		if (this.iconImage_.getImageState() == J.LOADED) return this.iconImage_.getSize()[0] * e[0];
	}
	getHeight() {
		let e = this.getScaleArray();
		if (this.size_) return this.size_[1] * e[1];
		if (this.iconImage_.getImageState() == J.LOADED) return this.iconImage_.getSize()[1] * e[1];
	}
	setScale(e) {
		delete this.initialOptions_, super.setScale(e);
	}
	listenImageChange(e) {
		this.iconImage_.addEventListener(M.CHANGE, e);
	}
	load() {
		this.iconImage_.load();
	}
	unlistenImageChange(e) {
		this.iconImage_.removeEventListener(M.CHANGE, e);
	}
	ready() {
		return this.iconImage_.ready();
	}
}, zc = class e {
	constructor(e) {
		e ||= {}, this.color_ = e.color === void 0 ? null : e.color, this.lineCap_ = e.lineCap, this.lineDash_ = e.lineDash === void 0 ? null : e.lineDash, this.lineDashOffset_ = e.lineDashOffset, this.lineJoin_ = e.lineJoin, this.miterLimit_ = e.miterLimit, this.offset_ = e.offset, this.width_ = e.width;
	}
	clone() {
		let t = this.getColor();
		return new e({
			color: Array.isArray(t) ? t.slice() : t || void 0,
			lineCap: this.getLineCap(),
			lineDash: this.getLineDash() ? this.getLineDash().slice() : void 0,
			lineDashOffset: this.getLineDashOffset(),
			lineJoin: this.getLineJoin(),
			miterLimit: this.getMiterLimit(),
			offset: this.getOffset(),
			width: this.getWidth()
		});
	}
	getColor() {
		return this.color_;
	}
	getLineCap() {
		return this.lineCap_;
	}
	getLineDash() {
		return this.lineDash_;
	}
	getLineDashOffset() {
		return this.lineDashOffset_;
	}
	getLineJoin() {
		return this.lineJoin_;
	}
	getMiterLimit() {
		return this.miterLimit_;
	}
	getOffset() {
		return this.offset_;
	}
	getWidth() {
		return this.width_;
	}
	setColor(e) {
		this.color_ = e;
	}
	setLineCap(e) {
		this.lineCap_ = e;
	}
	setLineDash(e) {
		this.lineDash_ = e;
	}
	setLineDashOffset(e) {
		this.lineDashOffset_ = e;
	}
	setLineJoin(e) {
		this.lineJoin_ = e;
	}
	setMiterLimit(e) {
		this.miterLimit_ = e;
	}
	setOffset(e) {
		this.offset_ = e;
	}
	setWidth(e) {
		this.width_ = e;
	}
}, Bc = class e {
	constructor(e) {
		e ||= {}, this.geometry_ = null, this.geometryFunction_ = Wc, e.geometry !== void 0 && this.setGeometry(e.geometry), this.fill_ = e.fill === void 0 ? null : e.fill, this.image_ = e.image === void 0 ? null : e.image, this.renderer_ = e.renderer === void 0 ? null : e.renderer, this.hitDetectionRenderer_ = e.hitDetectionRenderer === void 0 ? null : e.hitDetectionRenderer, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.text_ = e.text === void 0 ? null : e.text, this.zIndex_ = e.zIndex;
	}
	clone() {
		let t = this.getGeometry();
		return t && typeof t == "object" && (t = t.clone()), new e({
			geometry: t ?? void 0,
			fill: this.getFill() ? this.getFill().clone() : void 0,
			image: this.getImage() ? this.getImage().clone() : void 0,
			renderer: this.getRenderer() ?? void 0,
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			text: this.getText() ? this.getText().clone() : void 0,
			zIndex: this.getZIndex()
		});
	}
	getRenderer() {
		return this.renderer_;
	}
	setRenderer(e) {
		this.renderer_ = e;
	}
	setHitDetectionRenderer(e) {
		this.hitDetectionRenderer_ = e;
	}
	getHitDetectionRenderer() {
		return this.hitDetectionRenderer_;
	}
	getGeometry() {
		return this.geometry_;
	}
	getGeometryFunction() {
		return this.geometryFunction_;
	}
	getFill() {
		return this.fill_;
	}
	setFill(e) {
		this.fill_ = e;
	}
	getImage() {
		return this.image_;
	}
	setImage(e) {
		this.image_ = e;
	}
	getStroke() {
		return this.stroke_;
	}
	setStroke(e) {
		this.stroke_ = e;
	}
	getText() {
		return this.text_;
	}
	setText(e) {
		this.text_ = e;
	}
	getZIndex() {
		return this.zIndex_;
	}
	setGeometry(e) {
		typeof e == "function" ? this.geometryFunction_ = e : typeof e == "string" ? this.geometryFunction_ = function(t) {
			return t.get(e);
		} : e ? e !== void 0 && (this.geometryFunction_ = function() {
			return e;
		}) : this.geometryFunction_ = Wc, this.geometry_ = e;
	}
	setZIndex(e) {
		this.zIndex_ = e;
	}
};
function Vc(e) {
	let t;
	if (typeof e == "function") t = e;
	else {
		let n;
		Array.isArray(e) ? n = e : (R(typeof e.getZIndex == "function", "Expected an `Style` or an array of `Style`"), n = [e]), t = function() {
			return n;
		};
	}
	return t;
}
var Hc = null;
function Uc(e, t) {
	if (!Hc) {
		let e = new Ic({ color: "rgba(255,255,255,0.4)" }), t = new zc({
			color: "#3399CC",
			width: 1.25
		});
		Hc = [new Bc({
			image: new Fc({
				fill: e,
				stroke: t,
				radius: 5
			}),
			fill: e,
			stroke: t
		})];
	}
	return Hc;
}
function Wc(e) {
	return e.getGeometry();
}
//#endregion
//#region node_modules/ol/style/Text.js
var Gc = "#333", Kc = class e {
	constructor(e) {
		e ||= {}, this.font_ = e.font, this.rotation_ = e.rotation, this.rotateWithView_ = e.rotateWithView, this.keepUpright_ = e.keepUpright, this.scale_ = e.scale, this.scaleArray_ = rs(e.scale === void 0 ? 1 : e.scale), this.text_ = e.text, this.textAlign_ = e.textAlign, this.justify_ = e.justify, this.repeat_ = e.repeat, this.textBaseline_ = e.textBaseline, this.fill_ = e.fill === void 0 ? new Ic({ color: Gc }) : e.fill, this.maxAngle_ = e.maxAngle === void 0 ? Math.PI / 4 : e.maxAngle, this.placement_ = e.placement === void 0 ? "point" : e.placement, this.overflow_ = !!e.overflow, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.offsetX_ = e.offsetX === void 0 ? 0 : e.offsetX, this.offsetY_ = e.offsetY === void 0 ? 0 : e.offsetY, this.backgroundFill_ = e.backgroundFill ? e.backgroundFill : null, this.backgroundStroke_ = e.backgroundStroke ? e.backgroundStroke : null, this.padding_ = e.padding === void 0 ? null : e.padding, this.declutterMode_ = e.declutterMode;
	}
	clone() {
		let t = this.getScale();
		return new e({
			font: this.getFont(),
			placement: this.getPlacement(),
			repeat: this.getRepeat(),
			maxAngle: this.getMaxAngle(),
			overflow: this.getOverflow(),
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			keepUpright: this.getKeepUpright(),
			scale: Array.isArray(t) ? t.slice() : t,
			text: this.getText(),
			textAlign: this.getTextAlign(),
			justify: this.getJustify(),
			textBaseline: this.getTextBaseline(),
			fill: this.getFill() instanceof Ic ? this.getFill().clone() : this.getFill(),
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			offsetX: this.getOffsetX(),
			offsetY: this.getOffsetY(),
			backgroundFill: this.getBackgroundFill() ? this.getBackgroundFill().clone() : void 0,
			backgroundStroke: this.getBackgroundStroke() ? this.getBackgroundStroke().clone() : void 0,
			padding: this.getPadding() || void 0,
			declutterMode: this.getDeclutterMode()
		});
	}
	getOverflow() {
		return this.overflow_;
	}
	getFont() {
		return this.font_;
	}
	getMaxAngle() {
		return this.maxAngle_;
	}
	getPlacement() {
		return this.placement_;
	}
	getRepeat() {
		return this.repeat_;
	}
	getOffsetX() {
		return this.offsetX_;
	}
	getOffsetY() {
		return this.offsetY_;
	}
	getFill() {
		return this.fill_;
	}
	getRotateWithView() {
		return this.rotateWithView_;
	}
	getKeepUpright() {
		return this.keepUpright_;
	}
	getRotation() {
		return this.rotation_;
	}
	getScale() {
		return this.scale_;
	}
	getScaleArray() {
		return this.scaleArray_;
	}
	getStroke() {
		return this.stroke_;
	}
	getText() {
		return this.text_;
	}
	getTextAlign() {
		return this.textAlign_;
	}
	getJustify() {
		return this.justify_;
	}
	getTextBaseline() {
		return this.textBaseline_;
	}
	getBackgroundFill() {
		return this.backgroundFill_;
	}
	getBackgroundStroke() {
		return this.backgroundStroke_;
	}
	getPadding() {
		return this.padding_;
	}
	getDeclutterMode() {
		return this.declutterMode_;
	}
	setOverflow(e) {
		this.overflow_ = e;
	}
	setFont(e) {
		this.font_ = e;
	}
	setMaxAngle(e) {
		this.maxAngle_ = e;
	}
	setOffsetX(e) {
		this.offsetX_ = e;
	}
	setOffsetY(e) {
		this.offsetY_ = e;
	}
	setPlacement(e) {
		this.placement_ = e;
	}
	setRepeat(e) {
		this.repeat_ = e;
	}
	setRotateWithView(e) {
		this.rotateWithView_ = e;
	}
	setKeepUpright(e) {
		this.keepUpright_ = e;
	}
	setFill(e) {
		this.fill_ = e;
	}
	setRotation(e) {
		this.rotation_ = e;
	}
	setScale(e) {
		this.scale_ = e, this.scaleArray_ = rs(e === void 0 ? 1 : e);
	}
	setStroke(e) {
		this.stroke_ = e;
	}
	setText(e) {
		this.text_ = e;
	}
	setTextAlign(e) {
		this.textAlign_ = e;
	}
	setJustify(e) {
		this.justify_ = e;
	}
	setTextBaseline(e) {
		this.textBaseline_ = e;
	}
	setBackgroundFill(e) {
		this.backgroundFill_ = e;
	}
	setBackgroundStroke(e) {
		this.backgroundStroke_ = e;
	}
	setPadding(e) {
		this.padding_ = e;
	}
};
//#endregion
//#region node_modules/ol/render/canvas/style.js
function qc(e) {
	return !0;
}
function Jc(e, t) {
	t ??= vs();
	let n = Zc(e, t), r = zs();
	return function(e, i) {
		if (r.properties = e.getPropertiesInternal(), r.resolution = i, t.featureId) {
			let t = e.getId();
			t === void 0 ? r.featureId = null : r.featureId = t;
		}
		return t.geometryType && (r.geometryType = Rs(e.getGeometry())), n(r);
	};
}
function Yc(e, t) {
	t ??= vs();
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Qc(e[i], t);
	let i = zs(), a = Array(n);
	return function(e, o) {
		if (i.properties = e.getPropertiesInternal(), i.resolution = o, t.featureId) {
			let t = e.getId();
			t === void 0 ? i.featureId = null : i.featureId = t;
		}
		t.geometryType && (i.geometryType = Rs(e.getGeometry()));
		let s = 0;
		for (let e = 0; e < n; ++e) {
			let t = r[e](i);
			t && (a[s] = t, s += 1);
		}
		return a.length = s, a;
	};
}
function Xc(e, t) {
	if (t ??= vs(), !Array.isArray(e)) return Yc([e], t);
	let n = e.length;
	if ("style" in e[0]) {
		let r = Array(n);
		for (let t = 0; t < n; ++t) {
			let n = e[t];
			if (!("style" in n)) throw Error("Expected a list of rules with a style property");
			r[t] = n;
		}
		return Jc(r, t);
	}
	return Yc(e, t);
}
function Zc(e, t) {
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) {
		let n = e[i], a = "filter" in n ? Bs(n.filter, as, t) : qc, o;
		if (Array.isArray(n.style)) {
			let e = n.style.length;
			o = Array(e);
			for (let r = 0; r < e; ++r) o[r] = Qc(n.style[r], t);
		} else o = [Qc(n.style, t)];
		r[i] = {
			filter: a,
			styles: o
		};
	}
	return function(t) {
		let i = [], a = !1;
		for (let o = 0; o < n; ++o) {
			let n = r[o].filter;
			if (n(t) && !(e[o].else && a)) {
				a = !0;
				for (let e of r[o].styles) {
					let n = e(t);
					n && i.push(n);
				}
			}
		}
		return i;
	};
}
function Qc(e, t) {
	let n = $c(e, "", t), r = el(e, "", t), i = tl(e, t), a = nl(e, t), o = sl(e, "z-index", t);
	if (!n && !r && !i && !a && !te(e)) throw Error("No fill, stroke, point, or text symbolizer properties in style: " + JSON.stringify(e));
	let s = new Bc();
	return function(e) {
		let t = !0;
		if (n) {
			let r = n(e);
			r && (t = !1), s.setFill(r);
		}
		if (r) {
			let n = r(e);
			n && (t = !1), s.setStroke(n);
		}
		if (i) {
			let n = i(e);
			n && (t = !1), s.setText(n);
		}
		if (a) {
			let n = a(e);
			n && (t = !1), s.setImage(n);
		}
		return o && s.setZIndex(o(e)), t ? null : s;
	};
}
function $c(e, t, n) {
	let r;
	if (t + "fill-pattern-src" in e) r = ll(e, t + "fill-", n);
	else {
		if (e[t + "fill-color"] === "none") return (e) => null;
		r = dl(e, t + "fill-color", n);
	}
	if (!r) return null;
	let i = new Ic();
	return function(e) {
		let t = r(e);
		return t === jo ? null : (i.setColor(t), i);
	};
}
function el(e, t, n) {
	let r = sl(e, t + "stroke-width", n), i = dl(e, t + "stroke-color", n);
	if (!r && !i) return null;
	let a = cl(e, t + "stroke-line-cap", n), o = cl(e, t + "stroke-line-join", n), s = fl(e, t + "stroke-line-dash", n), c = sl(e, t + "stroke-line-dash-offset", n), l = sl(e, t + "stroke-miter-limit", n), u = sl(e, t + "stroke-offset", n), d = new zc();
	return function(e) {
		if (i) {
			let t = i(e);
			if (t === jo) return null;
			d.setColor(t);
		}
		if (r && d.setWidth(r(e)), a) {
			let t = a(e);
			if (t !== "butt" && t !== "round" && t !== "square") throw Error("Expected butt, round, or square line cap");
			d.setLineCap(t);
		}
		if (o) {
			let t = o(e);
			if (t !== "bevel" && t !== "round" && t !== "miter") throw Error("Expected bevel, round, or miter line join");
			d.setLineJoin(t);
		}
		return s && d.setLineDash(s(e)), c && d.setLineDashOffset(c(e)), l && d.setMiterLimit(l(e)), u && d.setOffset(u(e)), d;
	};
}
function tl(e, t) {
	let n = "text-", r = cl(e, "text-value", t);
	if (!r) return null;
	let i = $c(e, n, t), a = $c(e, "text-background-", t), o = el(e, n, t), s = el(e, "text-background-", t), c = cl(e, "text-font", t), l = sl(e, "text-max-angle", t), u = sl(e, "text-offset-x", t), d = sl(e, "text-offset-y", t), f = ul(e, "text-overflow", t), p = cl(e, "text-placement", t), m = sl(e, "text-repeat", t), h = hl(e, "text-scale", t), g = ul(e, "text-rotate-with-view", t), _ = sl(e, "text-rotation", t), v = cl(e, "text-align", t), y = cl(e, "text-justify", t), b = cl(e, "text-baseline", t), x = ul(e, "text-keep-upright", t), S = fl(e, "text-padding", t), C = new Kc({ declutterMode: Sl(e, "text-declutter-mode") });
	return function(e) {
		if (C.setText(r(e)), i && C.setFill(i(e)), a && C.setBackgroundFill(a(e)), o && C.setStroke(o(e)), s && C.setBackgroundStroke(s(e)), c && C.setFont(c(e)), l && C.setMaxAngle(l(e)), u && C.setOffsetX(u(e)), d && C.setOffsetY(d(e)), f && C.setOverflow(f(e)), p) {
			let t = p(e);
			if (t !== "point" && t !== "line") throw Error("Expected point or line for text-placement");
			C.setPlacement(t);
		}
		if (m && C.setRepeat(m(e)), h && C.setScale(h(e)), g && C.setRotateWithView(g(e)), _ && C.setRotation(_(e)), v) {
			let t = v(e);
			if (t !== "left" && t !== "center" && t !== "right" && t !== "end" && t !== "start") throw Error("Expected left, right, center, start, or end for text-align");
			C.setTextAlign(t);
		}
		if (y) {
			let t = y(e);
			if (t !== "left" && t !== "right" && t !== "center") throw Error("Expected left, right, or center for text-justify");
			C.setJustify(t);
		}
		if (b) {
			let t = b(e);
			if (t !== "bottom" && t !== "top" && t !== "middle" && t !== "alphabetic" && t !== "hanging") throw Error("Expected bottom, top, middle, alphabetic, or hanging for text-baseline");
			C.setTextBaseline(t);
		}
		return S && C.setPadding(S(e)), x && C.setKeepUpright(x(e)), C;
	};
}
function nl(e, t) {
	return "icon-src" in e ? rl(e, t) : "shape-points" in e ? il(e, t) : "circle-radius" in e ? al(e, t) : null;
}
function rl(e, t) {
	let n = "icon-src", r = wl(e[n], n), i = pl(e, "icon-anchor", t), a = hl(e, "icon-scale", t), o = sl(e, "icon-opacity", t), s = pl(e, "icon-displacement", t), c = sl(e, "icon-rotation", t), l = ul(e, "icon-rotate-with-view", t), u = yl(e, "icon-anchor-origin"), d = bl(e, "icon-anchor-x-units"), f = bl(e, "icon-anchor-y-units"), p = ol(e, "icon-color"), m, h = null;
	p !== void 0 && (Array.isArray(p) && p.length > 0 && typeof p[0] == "string" ? h = dl(e, "icon-color", t) : m = El(p, "icon-color"));
	let g = vl(e, "icon-cross-origin"), _ = xl(e, "icon-offset"), v = yl(e, "icon-offset-origin"), y = gl(e, "icon-width"), b = {
		src: r,
		anchorOrigin: u,
		anchorXUnits: d,
		anchorYUnits: f,
		crossOrigin: g,
		offset: _,
		offsetOrigin: v,
		height: gl(e, "icon-height"),
		width: y,
		size: _l(e, "icon-size"),
		declutterMode: Sl(e, "icon-declutter-mode")
	}, x = null;
	return function(e) {
		if (x) h && x.setColor(h(e));
		else {
			let t = h ? h(e) : m;
			x = new Rc(t === void 0 ? Object.assign({}, b) : Object.assign({}, b, { color: t }));
		}
		return o && x.setOpacity(o(e)), s && x.setDisplacement(s(e)), c && x.setRotation(c(e)), l && x.setRotateWithView(l(e)), a && x.setScale(a(e)), i && x.setAnchor(i(e)), x;
	};
}
function il(e, t) {
	let n = "shape-", r = "shape-points", i = "shape-radius", a = Tl(e[r], r);
	if (!(i in e)) throw Error(`Expected a number for ${i}`);
	let o = sl(e, i, t), s = typeof e[i] == "number" ? e[i] : 5, c = "shape-radius2", l = sl(e, c, t), u = typeof e[c] == "number" ? e[c] : void 0, d = $c(e, n, t), f = el(e, n, t), p = hl(e, "shape-scale", t), m = pl(e, "shape-displacement", t), h = sl(e, "shape-rotation", t), g = ul(e, "shape-rotate-with-view", t), _ = new Pc({
		points: a,
		radius: s,
		radius2: u,
		angle: gl(e, "shape-angle"),
		declutterMode: Sl(e, "shape-declutter-mode")
	});
	return function(e) {
		return o && _.setRadius(o(e)), l && _.setRadius2(l(e)), d && _.setFill(d(e)), f && _.setStroke(f(e)), m && _.setDisplacement(m(e)), h && _.setRotation(h(e)), g && _.setRotateWithView(g(e)), p && _.setScale(p(e)), _;
	};
}
function al(e, t) {
	let n = "circle-", r = $c(e, n, t), i = el(e, n, t), a = sl(e, "circle-radius", t), o = hl(e, "circle-scale", t), s = pl(e, "circle-displacement", t), c = sl(e, "circle-rotation", t), l = ul(e, "circle-rotate-with-view", t), u = new Fc({
		radius: 5,
		declutterMode: Sl(e, "circle-declutter-mode")
	});
	return function(e) {
		return a && u.setRadius(a(e)), r && u.setFill(r(e)), i && u.setStroke(i(e)), s && u.setDisplacement(s(e)), c && u.setRotation(c(e)), l && u.setRotateWithView(l(e)), o && u.setScale(o(e)), u;
	};
}
function ol(e, t) {
	if (!(t in e)) return;
	let n = e[t];
	return n === void 0 ? void 0 : n;
}
function sl(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return;
	let i = Bs(r, V, n);
	return function(e) {
		return Tl(i(e), t);
	};
}
function cl(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	let i = Bs(r, H, n);
	return function(e) {
		return wl(i(e), t);
	};
}
function ll(e, t, n) {
	let r = cl(e, t + "pattern-src", n), i = ml(e, t + "pattern-offset", n), a = ml(e, t + "pattern-size", n), o = dl(e, t + "color", n);
	return function(e) {
		return {
			src: r(e),
			offset: i && i(e),
			size: a && a(e),
			color: o && o(e)
		};
	};
}
function ul(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	let i = Bs(r, as, n);
	return function(e) {
		let n = i(e);
		if (typeof n != "boolean") throw Error(`Expected a boolean for ${t}`);
		return n;
	};
}
function dl(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	let i = Bs(r, U, n);
	return function(e) {
		return El(i(e), t);
	};
}
function fl(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	if (Array.isArray(r) && (r.length === 0 || typeof r[0] != "string")) {
		let e = r.map((e, r) => {
			if (typeof e == "number") return () => e;
			let i = Bs(e, V, n);
			return function(e) {
				return Tl(i(e), `${t}[${r}]`);
			};
		});
		return function(t) {
			let n = Array(e.length);
			for (let r = 0; r < e.length; ++r) n[r] = e[r](t);
			return n;
		};
	}
	let i = Bs(r, os, n);
	return function(e) {
		return Cl(i(e), t);
	};
}
function pl(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	let i = Bs(r, os, n);
	return function(e) {
		let n = Cl(i(e), t);
		if (n.length !== 2) throw Error(`Expected two numbers for ${t}`);
		return n;
	};
}
function ml(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	let i = Bs(r, os, n);
	return function(e) {
		return Dl(i(e), t);
	};
}
function hl(e, t, n) {
	let r = ol(e, t);
	if (r === void 0) return null;
	let i = Bs(r, os | V, n);
	return function(e) {
		return Ol(i(e), t);
	};
}
function gl(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n != "number") throw Error(`Expected a number for ${t}`);
		return n;
	}
}
function _l(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n == "number") return rs(n);
		if (!Array.isArray(n) || n.length !== 2 || typeof n[0] != "number" || typeof n[1] != "number") throw Error(`Expected a number or size array for ${t}`);
		return n;
	}
}
function vl(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n != "string") throw Error(`Expected a string for ${t}`);
		return n;
	}
}
function yl(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (n !== "bottom-left" && n !== "bottom-right" && n !== "top-left" && n !== "top-right") throw Error(`Expected bottom-left, bottom-right, top-left, or top-right for ${t}`);
		return n;
	}
}
function bl(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (n !== "pixels" && n !== "fraction") throw Error(`Expected pixels or fraction for ${t}`);
		return n;
	}
}
function xl(e, t) {
	let n = e[t];
	if (n !== void 0) return Cl(n, t);
}
function Sl(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n != "string") throw Error(`Expected a string for ${t}`);
		if (n !== "declutter" && n !== "obstacle" && n !== "none") throw Error(`Expected declutter, obstacle, or none for ${t}`);
		return n;
	}
}
function Cl(e, t) {
	if (!Array.isArray(e)) throw Error(`Expected an array for ${t}`);
	let n = e.length;
	for (let r = 0; r < n; ++r) if (typeof e[r] != "number") throw Error(`Expected an array of numbers for ${t}`);
	return e;
}
function wl(e, t) {
	if (typeof e != "string") throw Error(`Expected a string for ${t}`);
	return e;
}
function Tl(e, t) {
	if (typeof e != "number") throw Error(`Expected a number for ${t}`);
	return e;
}
function El(e, t) {
	if (typeof e == "string") return e;
	let n = Cl(e, t), r = n.length;
	if (r < 3 || r > 4) throw Error(`Expected a color with 3 or 4 values for ${t}`);
	return n;
}
function Dl(e, t) {
	let n = Cl(e, t);
	if (n.length !== 2) throw Error(`Expected an array of two numbers for ${t}`);
	return n;
}
function Ol(e, t) {
	return typeof e == "number" ? e : Dl(e, t);
}
//#endregion
//#region node_modules/ol/layer/BaseVector.js
var kl = { RENDER_ORDER: "renderOrder" }, Al = class extends uo {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.style, delete t.renderBuffer, delete t.updateWhileAnimating, delete t.updateWhileInteracting, super(t), this.declutter_ = e.declutter ? String(e.declutter) : void 0, this.renderBuffer_ = e.renderBuffer === void 0 ? 100 : e.renderBuffer, this.style_ = null, this.styleFunction_ = void 0, this.setStyle(e.style), this.updateWhileAnimating_ = e.updateWhileAnimating !== void 0 && e.updateWhileAnimating, this.updateWhileInteracting_ = e.updateWhileInteracting !== void 0 && e.updateWhileInteracting;
	}
	getDeclutter() {
		return this.declutter_;
	}
	getFeatures(e) {
		return super.getFeatures(e);
	}
	getRenderBuffer() {
		return this.renderBuffer_;
	}
	getRenderOrder() {
		return this.get(kl.RENDER_ORDER);
	}
	getStyle() {
		return this.style_;
	}
	getStyleFunction() {
		return this.styleFunction_;
	}
	getUpdateWhileAnimating() {
		return this.updateWhileAnimating_;
	}
	getUpdateWhileInteracting() {
		return this.updateWhileInteracting_;
	}
	renderDeclutter(e, t) {
		let n = this.getDeclutter();
		n in e.declutter || (e.declutter[n] = new go(9)), this.getRenderer().renderDeclutter(e, t);
	}
	setRenderOrder(e) {
		this.set(kl.RENDER_ORDER, e);
	}
	setStyle(e) {
		this.style_ = e === void 0 ? Uc : e;
		let t = jl(e);
		this.styleFunction_ = e === null ? void 0 : Vc(t), this.changed();
	}
	setDeclutter(e) {
		this.declutter_ = e ? String(e) : void 0, this.changed();
	}
};
function jl(e) {
	if (e === void 0) return Uc;
	if (!e) return null;
	if (typeof e == "function" || e instanceof Bc) return e;
	if (Array.isArray(e) && e.length === 0) return [];
	if (Array.isArray(e) && e[0] instanceof Bc) {
		let t = e.length, n = Array(t);
		for (let r = 0; r < t; ++r) {
			let t = e[r];
			if (!(t instanceof Bc)) throw Error("Expected a list of style instances");
			n[r] = t;
		}
		return n;
	}
	return Xc(e);
}
//#endregion
//#region node_modules/ol/render/Event.js
var Ml = class extends ve {
	constructor(e, t, n, r) {
		super(e), this.inversePixelTransform = t, this.frameState = n, this.context = r;
	}
}, Nl = class extends ie {
	constructor(e) {
		super(), this.map_ = e;
	}
	dispatchRenderEvent(e, t) {
		P();
	}
	calculateMatrices2D(e) {
		let t = e.viewState, n = e.coordinateToPixelTransform, r = e.pixelToCoordinateTransform;
		ei(n, e.size[0] / 2, e.size[1] / 2, 1 / t.resolution, -1 / t.resolution, -t.rotation, -t.center[0], -t.center[1]), ti(r, n);
	}
	forEachFeatureAtCoordinate(e, t, n, r, i, a, o, s) {
		let c, l = t.viewState;
		function u(e, t, n, r) {
			return i.call(a, t, e ? n : null, r);
		}
		let d = l.projection, f = zt(e.slice(), d), p = [[0, 0]];
		if (d.canWrapX() && r) {
			let e = I(d.getExtent());
			p.push([-e, 0], [e, 0]);
		}
		let m = t.layerStatesArray, h = m.length, g = [], _ = [];
		for (let r = 0; r < p.length; r++) for (let i = h - 1; i >= 0; --i) {
			let a = m[i], d = a.layer;
			if (d.hasRenderer() && fo(a, l) && o.call(s, d)) {
				let i = d.getRenderer(), o = d.getSource();
				if (i && o) {
					let s = o.getWrapX() ? f : e, l = u.bind(null, a.managed);
					_[0] = s[0] + p[r][0], _[1] = s[1] + p[r][1], c = i.forEachFeatureAtCoordinate(_, t, n, l, g);
				}
				if (c) return c;
			}
		}
		if (g.length === 0) return;
		let v = 1 / g.length;
		return g.forEach((e, t) => e.distanceSq += t * v), g.sort((e, t) => e.distanceSq - t.distanceSq), g.some((e) => c = e.callback(e.feature, e.layer, e.geometry)), c;
	}
	hasFeatureAtCoordinate(e, t, n, r, i, a) {
		return this.forEachFeatureAtCoordinate(e, t, n, r, pe, this, i, a) !== void 0;
	}
	getMap() {
		return this.map_;
	}
	renderFrame(e) {
		P();
	}
	scheduleExpireIconCache(e) {
		ic.canExpireCache() && e.postRenderFunctions.push(Pl);
	}
};
function Pl(e, t) {
	ic.expire();
}
//#endregion
//#region node_modules/ol/renderer/Composite.js
var Fl = class extends Nl {
	constructor(e) {
		super(e), this.fontChangeListenerKey_ = A(yc, re.PROPERTYCHANGE, e.redrawText, e), this.element_ = g ? D() : document.createElement("div");
		let t = this.element_.style;
		t.position = "absolute", t.width = "100%", t.height = "100%", t.zIndex = "0", this.element_.className = r + " ol-layers";
		let n = e.getViewport();
		n && n.insertBefore(this.element_, n.firstChild || null), this.children_ = [], this.renderedVisible_ = !0;
	}
	dispatchRenderEvent(e, t) {
		let n = this.getMap();
		if (n.hasListener(e)) {
			let r = new Ml(e, void 0, t);
			n.dispatchEvent(r);
		}
	}
	disposeInternal() {
		j(this.fontChangeListenerKey_), this.element_.remove(), super.disposeInternal();
	}
	renderFrame(e) {
		if (!e) {
			this.renderedVisible_ &&= (this.element_.style.display = "none", !1);
			return;
		}
		this.calculateMatrices2D(e), this.dispatchRenderEvent(lo.PRECOMPOSE, e);
		let t = e.layerStatesArray.sort((e, t) => e.zIndex - t.zIndex);
		t.some((e) => e.layer instanceof Al && e.layer.getDeclutter()) && (e.declutter = {});
		let n = e.viewState;
		this.children_.length = 0;
		let r = this.getMap().getTargetElement(), i;
		O(r) && (i = r.getContext("2d"), i.setTransform(1, 0, 0, 1, 0, 0), i.clearRect(0, 0, r.width, r.height));
		let a = [], o = i ? r : null;
		for (let r = 0, i = t.length; r < i; ++r) {
			let i = t[r];
			e.layerIndex = r;
			let s = i.layer, c = s.getSourceState();
			if (!fo(i, n) || c != "ready" && c != "undefined") {
				s.unrender();
				continue;
			}
			let l = s.render(e, o);
			l && (l !== o && (this.children_.push(l), o = l), a.push(i));
		}
		this.declutter(e, a), ee(this.element_, this.children_);
		for (let e of i ? this.children_ : []) {
			let t = e.firstElementChild || e, n = e.style.backgroundColor;
			if (n && (!O(t) || t.width > 0) && (i.fillStyle = n, i.fillRect(0, 0, i.canvas.width, i.canvas.height)), !O(t) || t.width === 0) continue;
			i.save();
			let r = e.style.opacity || t.style.opacity;
			i.globalAlpha = r === "" ? 1 : Number(r);
			let a = t.style.transform;
			if (a) i.transform(...ai(a));
			else {
				let e = parseFloat(t.style.width) / t.width, n = parseFloat(t.style.height) / t.height;
				i.transform(e, 0, 0, n, 0, 0);
			}
			i.drawImage(t, 0, 0), i.restore();
		}
		this.dispatchRenderEvent(lo.POSTCOMPOSE, e), this.renderedVisible_ ||= (this.element_.style.display = "", !0), this.scheduleExpireIconCache(e);
	}
	declutter(e, t) {
		if (e.declutter) {
			for (let n = t.length - 1; n >= 0; --n) {
				let r = t[n], i = r.layer;
				i.getDeclutter() && i.renderDeclutter(e, r);
			}
			t.forEach((t) => t.layer.renderDeferred(e));
		}
	}
};
//#endregion
//#region node_modules/ol/Map.js
function Il(e) {
	if (e instanceof uo) {
		e.setMapInternal(null);
		return;
	}
	e instanceof co && e.getLayers().forEach(Il);
}
function Ll(e, t) {
	if (e instanceof uo) {
		e.setMapInternal(t);
		return;
	}
	if (e instanceof co) {
		let n = e.getLayers().getArray();
		for (let e = 0, r = n.length; e < r; ++e) Ll(n[e], t);
	}
}
var Rl = class extends we {
	constructor(t) {
		super(), t ||= {}, this.on, this.once, this.un;
		let n = zl(t);
		this.renderComplete_ = !1, this.loaded_ = !0, this.boundHandleBrowserEvent_ = this.handleBrowserEvent.bind(this), this.maxTilesLoading_ = t.maxTilesLoading === void 0 ? 16 : t.maxTilesLoading, this.pixelRatio_ = t.pixelRatio === void 0 ? h : t.pixelRatio, this.postRenderTimeoutHandle_, this.animationDelayKey_, this.animationDelay_ = this.animationDelay_.bind(this), this.coordinateToPixelTransform_ = qr(), this.pixelToCoordinateTransform_ = qr(), this.frameIndex_ = 0, this.frameState_ = null, this.previousExtent_ = null, this.viewPropertyListenerKey_ = null, this.viewChangeListenerKey_ = null, this.layerGroupPropertyListenerKeys_ = null, g || (this.viewport_ = document.createElement("div"), this.viewport_.className = "ol-viewport" + ("ontouchstart" in window ? " ol-touch" : ""), this.viewport_.style.position = "relative", this.viewport_.style.overflow = "hidden", this.viewport_.style.width = "100%", this.viewport_.style.height = "100%", this.overlayContainer_ = document.createElement("div"), this.overlayContainer_.style.position = "absolute", this.overlayContainer_.style.zIndex = "0", this.overlayContainer_.style.width = "100%", this.overlayContainer_.style.height = "100%", this.overlayContainer_.style.pointerEvents = "none", this.overlayContainer_.className = "ol-overlaycontainer", this.viewport_.appendChild(this.overlayContainer_), this.overlayContainerStopEvent_ = document.createElement("div"), this.overlayContainerStopEvent_.style.position = "absolute", this.overlayContainerStopEvent_.style.zIndex = "0", this.overlayContainerStopEvent_.style.width = "100%", this.overlayContainerStopEvent_.style.height = "100%", this.overlayContainerStopEvent_.style.pointerEvents = "none", this.overlayContainerStopEvent_.className = "ol-overlaycontainer-stopevent", this.viewport_.appendChild(this.overlayContainerStopEvent_)), this.mapBrowserEventHandler_ = null, this.moveTolerance_ = t.moveTolerance, this.keyboardEventTarget_ = n.keyboardEventTarget, this.targetChangeHandlerKeys_ = null, this.targetElement_ = null, g || (this.resizeObserver_ = new ResizeObserver(() => this.updateSize())), this.controls = n.controls || (g ? new Or() : ba()), this.interactions = n.interactions || (g ? new Or() : ro({ onFocusOnly: !0 })), this.overlays_ = n.overlays, this.overlayIdIndex_ = {}, this.renderer_ = null, this.postRenderFunctions_ = [], this.tileQueue_ = new Fr(this.getTilePriority.bind(this), this.handleTileChange_.bind(this)), this.addChangeListener(e.LAYERGROUP, this.handleLayerGroupChanged_), this.addChangeListener(e.VIEW, this.handleViewChanged_), this.addChangeListener(e.SIZE, this.handleSizeChanged_), this.addChangeListener(e.TARGET, this.handleTargetChanged_), this.setProperties(n.values);
		let r = this;
		t.view && !(t.view instanceof ua) && t.view.then(function(e) {
			r.setView(new ua(e));
		}), this.controls.addEventListener(Tr.ADD, (e) => {
			e.element.setMap(this);
		}), this.controls.addEventListener(Tr.REMOVE, (e) => {
			e.element.setMap(null);
		}), this.interactions.addEventListener(Tr.ADD, (e) => {
			e.element.setMap(this);
		}), this.interactions.addEventListener(Tr.REMOVE, (e) => {
			e.element.setMap(null);
		}), this.overlays_.addEventListener(Tr.ADD, (e) => {
			this.addOverlayInternal_(e.element);
		}), this.overlays_.addEventListener(Tr.REMOVE, (e) => {
			let t = e.element.getId();
			t !== void 0 && delete this.overlayIdIndex_[t.toString()], e.element.setMap(null);
		}), this.controls.forEach((e) => {
			e.setMap(this);
		}), this.interactions.forEach((e) => {
			e.setMap(this);
		}), this.overlays_.forEach(this.addOverlayInternal_.bind(this));
	}
	addControl(e) {
		this.getControls().push(e);
	}
	addInteraction(e) {
		this.getInteractions().push(e);
	}
	addLayer(e) {
		this.getLayerGroup().getLayers().push(e);
	}
	handleLayerAdd_(e) {
		Ll(e.layer, this);
	}
	addOverlay(e) {
		this.getOverlays().push(e);
	}
	addOverlayInternal_(e) {
		let t = e.getId();
		t !== void 0 && (this.overlayIdIndex_[t.toString()] = e), e.setMap(this);
	}
	disposeInternal() {
		this.controls.clear(), this.interactions.clear(), this.overlays_.clear(), this.resizeObserver_?.disconnect(), this.setTarget(null), super.disposeInternal();
	}
	forEachFeatureAtPixel(e, t, n) {
		if (!this.frameState_ || !this.renderer_) return;
		let r = this.getCoordinateFromPixelInternal(e);
		n = n === void 0 ? {} : n;
		let i = n.hitTolerance === void 0 ? 0 : n.hitTolerance, a = n.layerFilter === void 0 ? pe : n.layerFilter, o = n.checkWrapped !== !1;
		return this.renderer_.forEachFeatureAtCoordinate(r, this.frameState_, i, o, t, null, a, null);
	}
	getFeaturesAtPixel(e, t) {
		let n = [];
		return this.forEachFeatureAtPixel(e, function(e) {
			n.push(e);
		}, t), n;
	}
	getAllLayers() {
		let e = [];
		function t(n) {
			n.forEach(function(n) {
				n instanceof co ? t(n.getLayers()) : e.push(n);
			});
		}
		return t(this.getLayers()), e;
	}
	hasFeatureAtPixel(e, t) {
		if (!this.frameState_ || !this.renderer_) return !1;
		let n = this.getCoordinateFromPixelInternal(e);
		t = t === void 0 ? {} : t;
		let r = t.layerFilter === void 0 ? pe : t.layerFilter, i = t.hitTolerance === void 0 ? 0 : t.hitTolerance, a = t.checkWrapped !== !1;
		return this.renderer_.hasFeatureAtCoordinate(n, this.frameState_, i, a, r, null);
	}
	getEventCoordinate(e) {
		return this.getCoordinateFromPixel(this.getEventPixel(e));
	}
	getEventCoordinateInternal(e) {
		return this.getCoordinateFromPixelInternal(this.getEventPixel(e));
	}
	getEventPixel(e) {
		let t = this.viewport_.getBoundingClientRect(), n = this.getSize(), r = t.width / n[0], i = t.height / n[1], a = "changedTouches" in e ? e.changedTouches[0] : e;
		return [(a.clientX - t.left) / r, (a.clientY - t.top) / i];
	}
	getTarget() {
		return this.get(e.TARGET);
	}
	getTargetElement() {
		return this.targetElement_;
	}
	getCoordinateFromPixel(e) {
		return gr(this.getCoordinateFromPixelInternal(e), this.getView().getProjection());
	}
	getCoordinateFromPixelInternal(e) {
		let t = this.frameState_;
		return t ? z(t.pixelToCoordinateTransform, e.slice()) : null;
	}
	getControls() {
		return this.controls;
	}
	getOverlays() {
		return this.overlays_;
	}
	getOverlayById(e) {
		let t = this.overlayIdIndex_[e.toString()];
		return t === void 0 ? null : t;
	}
	getInteractions() {
		return this.interactions;
	}
	getLayerGroup() {
		return this.get(e.LAYERGROUP);
	}
	setLayers(e) {
		let t = this.getLayerGroup();
		if (e instanceof Or) {
			t.setLayers(e);
			return;
		}
		let n = t.getLayers();
		n.clear(), n.extend(e);
	}
	getLayers() {
		return this.getLayerGroup().getLayers();
	}
	getLoadingOrNotReady() {
		let e = this.getLayerGroup().getLayerStatesArray();
		for (let t = 0, n = e.length; t < n; ++t) {
			let n = e[t];
			if (!n.visible) continue;
			let r = n.layer.getRenderer();
			if (r && !r.ready) return !0;
			let i = n.layer.getSource();
			if (i && i.loading) return !0;
		}
		return !1;
	}
	getPixelFromCoordinate(e) {
		let t = _r(e, this.getView().getProjection());
		return this.getPixelFromCoordinateInternal(t);
	}
	getPixelFromCoordinateInternal(e) {
		let t = this.frameState_;
		return t ? z(t.coordinateToPixelTransform, e.slice(0, 2)) : null;
	}
	getPixelRatio() {
		return this.pixelRatio_;
	}
	setPixelRatio(e) {
		this.pixelRatio_ !== e && (this.pixelRatio_ = e, this.render());
	}
	getRenderer() {
		return this.renderer_;
	}
	getSize() {
		return this.get(e.SIZE);
	}
	getView() {
		return this.get(e.VIEW);
	}
	getViewport() {
		return this.viewport_;
	}
	getOverlayContainer() {
		return this.overlayContainer_;
	}
	getOverlayContainerStopEvent() {
		return this.overlayContainerStopEvent_;
	}
	getOwnerDocument() {
		let e = this.getTargetElement();
		return e ? e.ownerDocument : document;
	}
	getTilePriority(e, t, n, r) {
		return Ir(this.frameState_, e, t, n, r);
	}
	handleBrowserEvent(e, t) {
		t ||= e.type;
		let n = new Ar(t, this, e);
		this.handleMapBrowserEvent(n);
	}
	handleMapBrowserEvent(e) {
		if (!this.frameState_) return;
		let t = e.originalEvent, n = t.type;
		if (n === Ht.POINTERDOWN || n === M.WHEEL || n === M.KEYDOWN) {
			let e = this.getOwnerDocument(), n = this.viewport_.getRootNode ? this.viewport_.getRootNode() : e, r = t.target, i = n instanceof ShadowRoot ? n.host === r ? n.host.ownerDocument : n : n === e ? e.documentElement : n;
			if (this.overlayContainerStopEvent_.contains(r) || !i.contains(r)) return;
		}
		if (e.frameState = this.frameState_, this.dispatchEvent(e) !== !1) {
			let t = this.getInteractions().getArray().slice();
			for (let n = t.length - 1; n >= 0; n--) {
				let r = t[n];
				if (!(r.getMap() !== this || !r.getActive() || !this.getTargetElement()) && (!r.handleEvent(e) || e.propagationStopped)) break;
			}
		}
	}
	handlePostRender() {
		let e = this.frameState_, t = this.tileQueue_;
		if (!t.isEmpty()) {
			let n = this.maxTilesLoading_, r = n, i = e ? e.viewHints : void 0, a = i ? i[Lr.ANIMATING] || i[Lr.INTERACTING] : !1;
			if (a) {
				let t = Date.now() - e.time > 8;
				n = t ? 0 : 8, r = t ? 0 : 2;
			}
			t.getTilesLoading() < n && (a && t.reprioritize(), t.loadMoreTiles(n, r));
		}
		e && this.renderer_ && !e.animate && (this.renderComplete_ ? (this.hasListener(lo.RENDERCOMPLETE) && this.renderer_.dispatchRenderEvent(lo.RENDERCOMPLETE, e), this.loaded_ === !1 && (this.loaded_ = !0, this.dispatchEvent(new kr(N.LOADEND, this, e)))) : this.loaded_ === !0 && (this.loaded_ = !1, this.dispatchEvent(new kr(N.LOADSTART, this, e))));
		let n = this.postRenderFunctions_;
		if (e) for (let t = 0, r = n.length; t < r; ++t) n[t](this, e);
		n.length = 0;
	}
	handleSizeChanged_() {
		this.getView() && !this.getView().getAnimating() && this.getView().resolveConstraints(0), this.render();
	}
	handleTargetChanged_() {
		if (this.mapBrowserEventHandler_) {
			for (let e = 0, t = this.targetChangeHandlerKeys_.length; e < t; ++e) j(this.targetChangeHandlerKeys_[e]);
			this.targetChangeHandlerKeys_ = null, this.viewport_.removeEventListener(M.CONTEXTMENU, this.boundHandleBrowserEvent_), this.viewport_.removeEventListener(M.WHEEL, this.boundHandleBrowserEvent_), this.mapBrowserEventHandler_.dispose(), this.mapBrowserEventHandler_ = null, this.viewport_.remove();
		}
		if (this.targetElement_ && !O(this.targetElement_)) {
			this.resizeObserver_?.unobserve(this.targetElement_);
			let e = this.targetElement_.getRootNode();
			e instanceof ShadowRoot && this.resizeObserver_.unobserve(e.host), this.setSize(void 0);
		}
		let e = this.getTarget(), t = typeof e == "string" ? document.getElementById(e) : e;
		if (this.targetElement_ = t, !t) this.renderer_ &&= (clearTimeout(this.postRenderTimeoutHandle_), this.postRenderTimeoutHandle_ = void 0, this.postRenderFunctions_.length = 0, this.renderer_.dispose(), null), this.animationDelayKey_ &&= (cancelAnimationFrame(this.animationDelayKey_), void 0);
		else {
			if (O(t) || t.appendChild(this.viewport_), this.renderer_ ||= new Fl(this), !O(t)) {
				this.mapBrowserEventHandler_ = new Mr(this, this.moveTolerance_);
				for (let e in jr) this.mapBrowserEventHandler_.addEventListener(jr[e], this.handleMapBrowserEvent.bind(this));
				this.viewport_.addEventListener(M.CONTEXTMENU, this.boundHandleBrowserEvent_, !1), this.viewport_.addEventListener(M.WHEEL, this.boundHandleBrowserEvent_, v ? { passive: !1 } : !1);
				let e;
				if (this.keyboardEventTarget_) e = this.keyboardEventTarget_;
				else {
					let n = t.getRootNode();
					e = n instanceof ShadowRoot ? n.host : t;
				}
				if (this.targetChangeHandlerKeys_ = [A(e, M.KEYDOWN, this.handleBrowserEvent, this), A(e, M.KEYPRESS, this.handleBrowserEvent, this)], !O(t)) {
					let e = t.getRootNode();
					e instanceof ShadowRoot && this.resizeObserver_.observe(e.host), this.resizeObserver_?.observe(t);
				}
			}
			this.updateSize();
		}
	}
	handleTileChange_() {
		this.render();
	}
	handleViewPropertyChanged_() {
		this.render();
	}
	handleViewChanged_() {
		this.viewPropertyListenerKey_ &&= (j(this.viewPropertyListenerKey_), null), this.viewChangeListenerKey_ &&= (j(this.viewChangeListenerKey_), null);
		let e = this.getView();
		e && (this.updateViewportSize_(this.getSize()), this.viewPropertyListenerKey_ = A(e, re.PROPERTYCHANGE, this.handleViewPropertyChanged_, this), this.viewChangeListenerKey_ = A(e, M.CHANGE, this.handleViewPropertyChanged_, this), e.resolveConstraints(0)), this.render();
	}
	handleLayerGroupChanged_() {
		this.layerGroupPropertyListenerKeys_ &&= (this.layerGroupPropertyListenerKeys_.forEach(j), null);
		let e = this.getLayerGroup();
		e && (this.handleLayerAdd_(new oo("addlayer", e)), this.layerGroupPropertyListenerKeys_ = [
			A(e, re.PROPERTYCHANGE, this.render, this),
			A(e, M.CHANGE, this.render, this),
			A(e, "addlayer", this.handleLayerAdd_, this),
			A(e, "removelayer", this.handleLayerRemove_, this)
		]), this.render();
	}
	isRendered() {
		return !!this.frameState_;
	}
	animationDelay_() {
		this.animationDelayKey_ = void 0, this.renderFrame_(Date.now());
	}
	renderSync() {
		this.animationDelayKey_ && cancelAnimationFrame(this.animationDelayKey_), this.animationDelay_();
	}
	redrawText() {
		if (!this.frameState_) return;
		let e = this.frameState_.layerStatesArray;
		for (let t = 0, n = e.length; t < n; ++t) {
			let n = e[t].layer;
			n.hasRenderer() && n.getRenderer().handleFontsChanged();
		}
	}
	render() {
		this.renderer_ && this.animationDelayKey_ === void 0 && (this.animationDelayKey_ = requestAnimationFrame(this.animationDelay_));
	}
	removeControl(e) {
		return this.getControls().remove(e);
	}
	removeInteraction(e) {
		return this.getInteractions().remove(e);
	}
	removeLayer(e) {
		return this.getLayerGroup().getLayers().remove(e);
	}
	handleLayerRemove_(e) {
		Il(e.layer);
	}
	removeOverlay(e) {
		return this.getOverlays().remove(e);
	}
	renderFrame_(e) {
		let t = this.getSize(), n = this.getView(), r = this.frameState_, i = null;
		if (t !== void 0 && ts(t) && n && n.isDef()) {
			let r = n.getHints(this.frameState_ ? this.frameState_.viewHints : void 0), a = n.getState();
			if (i = {
				animate: !1,
				coordinateToPixelTransform: this.coordinateToPixelTransform_,
				declutter: null,
				extent: ot(a.center, a.resolution, a.rotation, t),
				index: this.frameIndex_++,
				layerIndex: 0,
				layerStatesArray: this.getLayerGroup().getLayerStatesArray(),
				pixelRatio: this.pixelRatio_,
				pixelToCoordinateTransform: this.pixelToCoordinateTransform_,
				postRenderFunctions: [],
				size: t,
				tileQueue: this.tileQueue_,
				time: e,
				usedTiles: {},
				viewState: a,
				viewHints: r,
				wantedTiles: {},
				mapId: F(this),
				renderTargets: {}
			}, a.nextCenter && a.nextResolution) {
				let e = isNaN(a.nextRotation) ? a.rotation : a.nextRotation;
				i.nextExtent = ot(a.nextCenter, a.nextResolution, e, t);
			}
		}
		this.frameState_ = i, this.renderer_.renderFrame(i), i && (i.animate && this.render(), Array.prototype.push.apply(this.postRenderFunctions_, i.postRenderFunctions), r && (!this.previousExtent_ || !mt(this.previousExtent_) && !Ye(i.extent, this.previousExtent_)) && (this.dispatchEvent(new kr(N.MOVESTART, this, r)), this.previousExtent_ = Ke(this.previousExtent_)), this.previousExtent_ && !i.viewHints[Lr.ANIMATING] && !i.viewHints[Lr.INTERACTING] && !Ye(i.extent, this.previousExtent_) && (this.dispatchEvent(new kr(N.MOVEEND, this, i)), Re(i.extent, this.previousExtent_))), this.dispatchEvent(new kr(N.POSTRENDER, this, i)), this.renderComplete_ = (this.hasListener(N.LOADSTART) || this.hasListener(N.LOADEND) || this.hasListener(lo.RENDERCOMPLETE)) && !this.tileQueue_.getTilesLoading() && !this.tileQueue_.getCount() && !this.getLoadingOrNotReady(), this.postRenderTimeoutHandle_ ||= setTimeout(() => {
			this.postRenderTimeoutHandle_ = void 0, this.handlePostRender();
		}, 0);
	}
	setLayerGroup(t) {
		let n = this.getLayerGroup();
		n && this.handleLayerRemove_(new oo("removelayer", n)), this.set(e.LAYERGROUP, t);
	}
	setSize(t) {
		this.set(e.SIZE, t);
	}
	setTarget(t) {
		this.set(e.TARGET, t);
	}
	setView(t) {
		if (!t || t instanceof ua) {
			this.set(e.VIEW, t);
			return;
		}
		this.set(e.VIEW, new ua());
		let n = this;
		t.then(function(e) {
			n.setView(new ua(e));
		});
	}
	updateSize() {
		let e = this.getTargetElement(), t;
		if (e) {
			let n, r;
			if (O(e)) {
				let t = e.getContext("2d").getTransform();
				n = e.width / t.a, r = e.height / t.d;
			} else {
				let t = getComputedStyle(e);
				n = e.offsetWidth - parseFloat(t.borderLeftWidth) - parseFloat(t.paddingLeft) - parseFloat(t.paddingRight) - parseFloat(t.borderRightWidth), r = e.offsetHeight - parseFloat(t.borderTopWidth) - parseFloat(t.paddingTop) - parseFloat(t.paddingBottom) - parseFloat(t.borderBottomWidth);
			}
			!isNaN(n) && !isNaN(r) && (t = [Math.max(0, n), Math.max(0, r)], !ts(t) && (e.offsetWidth || e.offsetHeight || e.getClientRects().length) && Gt("No map visible because the map container's width or height are 0."));
		}
		let n = this.getSize();
		t && (!n || !de(t, n)) && (this.updateViewportSize_(t), this.setSize(t));
	}
	updateViewportSize_(e) {
		let t = this.getView();
		t && t.setViewportSize(e);
	}
};
function zl(t) {
	let n = null;
	t.keyboardEventTarget !== void 0 && (n = typeof t.keyboardEventTarget == "string" ? document.getElementById(t.keyboardEventTarget) : t.keyboardEventTarget);
	let r = {}, i = t.layers && typeof t.layers.getLayers == "function" ? t.layers : new co({ layers: t.layers });
	r[e.LAYERGROUP] = i, r[e.TARGET] = t.target, r[e.VIEW] = t.view instanceof ua ? t.view : new ua();
	let a;
	t.controls !== void 0 && (Array.isArray(t.controls) ? a = new Or(t.controls.slice()) : (R(typeof t.controls.getArray == "function", "Expected `controls` to be an array or an `ol/Collection.js`"), a = t.controls));
	let o;
	t.interactions !== void 0 && (Array.isArray(t.interactions) ? o = new Or(t.interactions.slice()) : (R(typeof t.interactions.getArray == "function", "Expected `interactions` to be an array or an `ol/Collection.js`"), o = t.interactions));
	let s;
	return t.overlays === void 0 ? s = new Or() : Array.isArray(t.overlays) ? s = new Or(t.overlays.slice()) : (R(typeof t.overlays.getArray == "function", "Expected `overlays` to be an array or an `ol/Collection.js`"), s = t.overlays), {
		controls: a,
		interactions: o,
		keyboardEventTarget: n,
		overlays: s,
		values: r
	};
}
//#endregion
//#region node_modules/ol/Overlay.js
var Bl = {
	ELEMENT: "element",
	MAP: "map",
	OFFSET: "offset",
	POSITION: "position",
	POSITIONING: "positioning"
}, Vl = class extends we {
	constructor(e) {
		super(), this.on, this.once, this.un, this.options = e, this.id = e.id, this.insertFirst = e.insertFirst === void 0 || e.insertFirst, this.stopEvent = e.stopEvent === void 0 || e.stopEvent, this.element = document.createElement("div"), this.element.className = e.className === void 0 ? "ol-overlay-container " + n : e.className, this.element.style.position = "absolute", this.element.style.pointerEvents = "auto", this.autoPan = e.autoPan === !0 ? {} : e.autoPan || void 0, this.rendered = {
			transform_: "",
			visible: !0
		}, this.mapPostrenderListenerKey = null, this.addChangeListener(Bl.ELEMENT, this.handleElementChanged), this.addChangeListener(Bl.MAP, this.handleMapChanged), this.addChangeListener(Bl.OFFSET, this.handleOffsetChanged), this.addChangeListener(Bl.POSITION, this.handlePositionChanged), this.addChangeListener(Bl.POSITIONING, this.handlePositioningChanged), e.element !== void 0 && this.setElement(e.element), this.setOffset(e.offset === void 0 ? [0, 0] : e.offset), this.setPositioning(e.positioning || "top-left"), e.position !== void 0 && this.setPosition(e.position);
	}
	getElement() {
		return this.get(Bl.ELEMENT);
	}
	getId() {
		return this.id;
	}
	getMap() {
		return this.get(Bl.MAP) || null;
	}
	getOffset() {
		return this.get(Bl.OFFSET);
	}
	getPosition() {
		return this.get(Bl.POSITION);
	}
	getPositioning() {
		return this.get(Bl.POSITIONING);
	}
	handleElementChanged() {
		E(this.element);
		let e = this.getElement();
		e && this.element.appendChild(e);
	}
	handleMapChanged() {
		this.mapPostrenderListenerKey &&= (this.element?.remove(), j(this.mapPostrenderListenerKey), null);
		let e = this.getMap();
		if (e) {
			this.mapPostrenderListenerKey = A(e, N.POSTRENDER, this.render, this), this.updatePixelPosition();
			let t = this.stopEvent ? e.getOverlayContainerStopEvent() : e.getOverlayContainer();
			this.insertFirst ? t.insertBefore(this.element, t.childNodes[0] || null) : t.appendChild(this.element), this.performAutoPan();
		}
	}
	render() {
		this.updatePixelPosition();
	}
	handleOffsetChanged() {
		this.updatePixelPosition();
	}
	handlePositionChanged() {
		this.updatePixelPosition(), this.performAutoPan();
	}
	handlePositioningChanged() {
		this.updatePixelPosition();
	}
	setElement(e) {
		this.set(Bl.ELEMENT, e);
	}
	setMap(e) {
		this.set(Bl.MAP, e);
	}
	setOffset(e) {
		this.set(Bl.OFFSET, e);
	}
	setPosition(e) {
		this.set(Bl.POSITION, e);
	}
	performAutoPan() {
		this.autoPan && this.panIntoView(this.autoPan);
	}
	panIntoView(e) {
		let t = this.getMap();
		if (!t || !t.getTargetElement() || !this.get(Bl.POSITION)) return;
		let n = this.getRect(t.getTargetElement(), t.getSize()), r = this.getElement(), i = this.getRect(r, [C(r), w(r)]);
		e ||= {};
		let a = e.margin === void 0 ? 20 : e.margin;
		if (!Ve(n, i)) {
			let r = i[0] - n[0], o = n[2] - i[2], s = i[1] - n[1], c = n[3] - i[3], l = [0, 0];
			if (r < 0 ? l[0] = r - a : o < 0 && (l[0] = Math.abs(o) + a), s < 0 ? l[1] = s - a : c < 0 && (l[1] = Math.abs(c) + a), l[0] !== 0 || l[1] !== 0) {
				let n = t.getView().getCenterInternal(), r = t.getPixelFromCoordinateInternal(n);
				if (!r) return;
				let i = [r[0] + l[0], r[1] + l[1]], a = e.animation || {};
				t.getView().animateInternal({
					center: t.getCoordinateFromPixelInternal(i),
					duration: a.duration,
					easing: a.easing
				});
			}
		}
	}
	getRect(e, t) {
		let n = e.getBoundingClientRect(), r = n.left + window.pageXOffset, i = n.top + window.pageYOffset;
		return [
			r,
			i,
			r + t[0],
			i + t[1]
		];
	}
	setPositioning(e) {
		this.set(Bl.POSITIONING, e);
	}
	setVisible(e) {
		this.rendered.visible !== e && (this.element.style.display = e ? "" : "none", this.rendered.visible = e);
	}
	updatePixelPosition() {
		let e = this.getMap(), t = this.getPosition();
		if (!e || !e.isRendered() || !t) {
			this.setVisible(!1);
			return;
		}
		let n = e.getPixelFromCoordinate(t), r = e.getSize();
		this.updateRenderedPosition(n, r);
	}
	updateRenderedPosition(e, t) {
		let n = this.element.style, r = this.getOffset(), i = this.getPositioning();
		this.setVisible(!0);
		let a = `${e[0] + r[0]}px`, o = `${e[1] + r[1]}px`, s = "0%", c = "0%";
		i == "bottom-right" || i == "center-right" || i == "top-right" ? s = "-100%" : (i == "bottom-center" || i == "center-center" || i == "top-center") && (s = "-50%"), i == "bottom-left" || i == "bottom-center" || i == "bottom-right" ? c = "-100%" : (i == "center-left" || i == "center-center" || i == "center-right") && (c = "-50%");
		let l = `translate(${s}, ${c}) translate(${a}, ${o})`;
		this.rendered.transform_ != l && (this.rendered.transform_ = l, n.transform = l);
	}
	getOptions() {
		return this.options;
	}
}, Hl = .75, Ul = .1, Wl = class extends Te {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		}), this.boundHandleRotationChanged_ = this.handleRotationChanged_.bind(this), this.collapsed_ = e.collapsed === void 0 || e.collapsed, this.collapsible_ = e.collapsible === void 0 || e.collapsible, this.collapsible_ || (this.collapsed_ = !1), this.rotateWithView_ = e.rotateWithView !== void 0 && e.rotateWithView, this.viewExtent_ = void 0;
		let t = e.className === void 0 ? "ol-overviewmap" : e.className, n = e.tipLabel === void 0 ? "Overview map" : e.tipLabel, i = e.collapseLabel === void 0 ? "‹" : e.collapseLabel;
		typeof i == "string" ? (this.collapseLabel_ = document.createElement("span"), this.collapseLabel_.textContent = i) : this.collapseLabel_ = i;
		let s = e.label === void 0 ? "›" : e.label;
		typeof s == "string" ? (this.label_ = document.createElement("span"), this.label_.textContent = s) : this.label_ = s;
		let c = this.collapsible_ && !this.collapsed_ ? this.collapseLabel_ : this.label_, l = document.createElement("button");
		l.setAttribute("type", "button"), l.title = n, l.appendChild(c), l.addEventListener(M.CLICK, this.handleClick_.bind(this), !1), this.ovmapDiv_ = document.createElement("div"), this.ovmapDiv_.className = "ol-overviewmap-map", this.view_ = e.view;
		let u = new Rl({
			view: e.view,
			controls: new Or(),
			interactions: new Or()
		});
		this.ovmap_ = u, e.layers && e.layers.forEach(function(e) {
			u.addLayer(e);
		});
		let d = document.createElement("div");
		d.className = "ol-overviewmap-box", d.style.boxSizing = "border-box", this.boxOverlay_ = new Vl({
			position: [0, 0],
			positioning: "center-center",
			element: d
		}), this.ovmap_.addOverlay(this.boxOverlay_);
		let f = t + " " + r + " " + a + (this.collapsed_ && this.collapsible_ ? " " + o : "") + (this.collapsible_ ? "" : " ol-uncollapsible"), p = this.element;
		p.className = f, p.appendChild(this.ovmapDiv_), p.appendChild(l);
		let m = this.boxOverlay_, h = this.boxOverlay_.getElement(), g = (e) => ({
			clientX: e.clientX,
			clientY: e.clientY
		}), _ = function(e) {
			let t = g(e), n = u.getEventCoordinate(t);
			m.setPosition(n);
		}, v = (e) => {
			let t = u.getEventCoordinateInternal(e), n = this.getMap();
			n.getView().setCenterInternal(t);
			let r = n.getOwnerDocument();
			r.removeEventListener("pointermove", _), r.removeEventListener("pointerup", v);
		};
		this.ovmapDiv_.addEventListener("pointerdown", (e) => {
			let t = this.getMap().getOwnerDocument();
			e.target === h && t.addEventListener("pointermove", _), t.addEventListener("pointerup", v);
		});
	}
	setMap(e) {
		let t = this.getMap();
		if (e !== t) {
			if (t) {
				let e = t.getView();
				e && this.unbindView_(e), this.ovmap_.setTarget(null);
			}
			if (super.setMap(e), e) {
				this.ovmap_.setTarget(this.ovmapDiv_), this.listenerKeys.push(A(e, re.PROPERTYCHANGE, this.handleMapPropertyChange_, this));
				let t = e.getView();
				t && this.bindView_(t), this.ovmap_.isRendered() || this.updateBoxAfterOvmapIsRendered_();
			}
		}
	}
	handleMapPropertyChange_(t) {
		if (t.key === e.VIEW) {
			let e = t.oldValue;
			e && this.unbindView_(e);
			let n = this.getMap().getView();
			this.bindView_(n);
		} else !this.ovmap_.isRendered() && (t.key === e.TARGET || t.key === e.SIZE) && this.ovmap_.updateSize();
	}
	bindView_(e) {
		if (!this.view_) {
			let t = new ua({ projection: e.getProjection() });
			this.ovmap_.setView(t);
		}
		e.addChangeListener(Rr.ROTATION, this.boundHandleRotationChanged_), this.handleRotationChanged_(), e.isDef() && (this.ovmap_.updateSize(), this.resetExtent_());
	}
	unbindView_(e) {
		e.removeChangeListener(Rr.ROTATION, this.boundHandleRotationChanged_);
	}
	handleRotationChanged_() {
		this.rotateWithView_ && this.ovmap_.getView().setRotation(this.getMap().getView().getRotation());
	}
	validateExtent_() {
		let e = this.getMap(), t = this.ovmap_;
		if (!e.isRendered() || !t.isRendered()) return;
		let n = e.getSize(), r = e.getView().calculateExtentInternal(n);
		if (this.viewExtent_ && Ye(r, this.viewExtent_)) return;
		this.viewExtent_ = r;
		let i = t.getSize(), a = t.getView().calculateExtentInternal(i), o = t.getPixelFromCoordinateInternal(dt(r)), s = t.getPixelFromCoordinateInternal(rt(r)), c = Math.abs(o[0] - s[0]), l = Math.abs(o[1] - s[1]), u = i[0], d = i[1];
		c < u * Ul || l < d * Ul || c > u * Hl || l > d * Hl ? this.resetExtent_() : Ve(a, r) || this.recenter_();
	}
	resetExtent_() {
		let e = this.getMap(), t = this.ovmap_, n = e.getSize(), r = e.getView().calculateExtentInternal(n), i = t.getView();
		gt(r, 1 / (2 ** (Math.log(Hl / Ul) / Math.LN2 / 2) * Ul)), i.fitInternal($i(r));
	}
	recenter_() {
		let e = this.getMap(), t = this.ovmap_, n = e.getView();
		t.getView().setCenterInternal(n.getCenterInternal());
	}
	updateBox_() {
		let e = this.getMap(), t = this.ovmap_;
		if (!e.isRendered() || !t.isRendered()) return;
		let n = e.getSize(), r = e.getView(), i = t.getView(), a = this.rotateWithView_ ? 0 : -r.getRotation(), o = this.boxOverlay_, s = this.boxOverlay_.getElement(), c = r.getCenter(), l = r.getResolution(), u = i.getResolution(), d = n[0] * l / u, f = n[1] * l / u;
		if (o.setPosition(c), s) {
			s.style.width = d + "px", s.style.height = f + "px";
			let e = "rotate(" + a + "rad)";
			s.style.transform = e;
		}
	}
	updateBoxAfterOvmapIsRendered_() {
		this.ovmapPostrenderKey_ ||= ne(this.ovmap_, N.POSTRENDER, (e) => {
			delete this.ovmapPostrenderKey_, this.updateBox_();
		});
	}
	handleClick_(e) {
		e.preventDefault(), this.handleToggle_();
	}
	handleToggle_() {
		this.element.classList.toggle(o), this.collapsed_ ? T(this.collapseLabel_, this.label_) : T(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_;
		let e = this.ovmap_;
		if (!this.collapsed_) {
			if (e.isRendered()) {
				this.viewExtent_ = void 0, e.render();
				return;
			}
			e.updateSize(), this.resetExtent_(), this.updateBoxAfterOvmapIsRendered_();
		}
	}
	getCollapsible() {
		return this.collapsible_;
	}
	setCollapsible(e) {
		this.collapsible_ !== e && (this.collapsible_ = e, this.element.classList.toggle("ol-uncollapsible"), !e && this.collapsed_ && this.handleToggle_());
	}
	setCollapsed(e) {
		!this.collapsible_ || this.collapsed_ === e || this.handleToggle_();
	}
	getCollapsed() {
		return this.collapsed_;
	}
	getRotateWithView() {
		return this.rotateWithView_;
	}
	setRotateWithView(e) {
		this.rotateWithView_ !== e && (this.rotateWithView_ = e, this.getMap().getView().getRotation() !== 0 && (this.rotateWithView_ ? this.handleRotationChanged_() : this.ovmap_.getView().setRotation(0), this.viewExtent_ = void 0, this.validateExtent_(), this.updateBox_()));
	}
	getOverviewMap() {
		return this.ovmap_;
	}
	render(e) {
		this.validateExtent_(), this.updateBox_();
	}
}, Gl = "units", Kl = [
	1,
	2,
	5
], ql = 25.4 / .28, Jl = class extends Te {
	constructor(e) {
		e ||= {};
		let t = document.createElement("div");
		t.style.pointerEvents = "none", super({
			element: t,
			render: e.render,
			target: e.target
		}), this.on, this.once, this.un;
		let n = e.className === void 0 ? e.bar ? "ol-scale-bar" : "ol-scale-line" : e.className;
		this.innerElement_ = document.createElement("div"), this.innerElement_.className = n + "-inner", this.element.className = n + " " + r, this.element.appendChild(this.innerElement_), this.viewState_ = null, this.minWidth_ = e.minWidth === void 0 ? 64 : e.minWidth, this.maxWidth_ = e.maxWidth, this.renderedVisible_ = !1, this.renderedWidth_ = void 0, this.renderedHTML_ = "", this.addChangeListener(Gl, this.handleUnitsChanged_), this.setUnits(e.units || "metric"), this.scaleBar_ = e.bar || !1, this.scaleBarSteps_ = e.steps || 4, this.scaleBarText_ = e.text || !1, this.dpi_ = e.dpi || void 0;
	}
	getUnits() {
		return this.get(Gl);
	}
	handleUnitsChanged_() {
		this.updateElement_();
	}
	setUnits(e) {
		this.set(Gl, e);
	}
	setDpi(e) {
		this.dpi_ = e;
	}
	updateElement_() {
		let e = this.viewState_;
		if (!e) {
			this.renderedVisible_ &&= (this.element.style.display = "none", !1);
			return;
		}
		let t = e.center, n = e.projection, r = this.getUnits(), i = r == "degrees" ? "degrees" : "m", a = rr(n, e.resolution, t, i), o = this.minWidth_ * (this.dpi_ || ql) / ql, s = this.maxWidth_ === void 0 ? void 0 : this.maxWidth_ * (this.dpi_ || ql) / ql, c = o * a, l = "";
		if (r == "degrees") {
			let e = Kt.degrees;
			c *= e, c < e / 60 ? (l = "″", a *= 3600) : c < e ? (l = "′", a *= 60) : l = "°";
		} else if (r == "imperial") c < .9144 ? (l = "in", a /= .0254) : c < 1609.344 ? (l = "ft", a /= .3048) : (l = "mi", a /= 1609.344);
		else if (r == "nautical") a /= 1852, l = "NM";
		else if (r == "metric") c < 1e-6 ? (l = "nm", a *= 1e9) : c < .001 ? (l = "μm", a *= 1e6) : c < 1 ? (l = "mm", a *= 1e3) : c < 1e3 ? l = "m" : (l = "km", a /= 1e3);
		else if (r == "us") c < .9144 ? (l = "in", a *= 39.37) : c < 1609.344 ? (l = "ft", a /= .30480061) : (l = "mi", a /= 1609.3472);
		else throw Error("Invalid units");
		let u = 3 * Math.floor(Math.log(o * a) / Math.log(10)), d, f, p, m = 0, h, g;
		for (;;) {
			p = Math.floor(u / 3);
			let e = 10 ** p;
			if (d = Kl[(u % 3 + 3) % 3] * e, f = Math.round(d / a), isNaN(f)) {
				this.element.style.display = "none", this.renderedVisible_ = !1;
				return;
			}
			if (s !== void 0 && f >= s) {
				d = m, f = h, p = g;
				break;
			}
			if (f >= o) break;
			m = d, h = f, g = p, ++u;
		}
		let _ = this.scaleBar_ ? this.createScaleBar(f, d, l) : d.toFixed(p < 0 ? -p : 0) + " " + l;
		this.renderedHTML_ != _ && (this.innerElement_.innerHTML = _, this.renderedHTML_ = _), this.renderedWidth_ != f && (this.innerElement_.style.width = f + "px", this.renderedWidth_ = f), this.renderedVisible_ ||= (this.element.style.display = "", !0);
	}
	createScaleBar(e, t, n) {
		let r = this.getScaleForResolution(), i = r < 1 ? Math.round(1 / r).toLocaleString() + " : 1" : "1 : " + Math.round(r).toLocaleString(), a = this.scaleBarSteps_, o = e / a, s = [this.createMarker("absolute")];
		for (let r = 0; r < a; ++r) {
			let i = r % 2 == 0 ? "ol-scale-singlebar-odd" : "ol-scale-singlebar-even";
			s.push(`<div><div class="ol-scale-singlebar ${i}" style="width: ${o}px;"></div>` + this.createMarker("relative") + (r % 2 == 0 || a === 2 ? this.createStepText(r, e, !1, t, n) : "") + "</div>");
		}
		return s.push(this.createStepText(a, e, !0, t, n)), (this.scaleBarText_ ? `<div class="ol-scale-text" style="width: ${e}px;">` + i + "</div>" : "") + s.join("");
	}
	createMarker(e) {
		return `<div class="ol-scale-step-marker" style="position: ${e}; top: ${e === "absolute" ? 3 : -10}px;"></div>`;
	}
	createStepText(e, t, n, r, i) {
		let a = (e === 0 ? 0 : Math.round(r / this.scaleBarSteps_ * e * 100) / 100) + (e === 0 ? "" : " " + i), o = e === 0 ? -3 : t / this.scaleBarSteps_ * -1, s = e === 0 ? 0 : t / this.scaleBarSteps_ * 2;
		return `<div class="ol-scale-step-text" style="margin-left: ${o}px;text-align: ${e === 0 ? "left" : "center"};min-width: ${s}px;left: ${n ? t + "px" : "unset"};">` + a + "</div>";
	}
	getScaleForResolution() {
		let e = rr(this.viewState_.projection, this.viewState_.resolution, this.viewState_.center, "m"), t = this.dpi_ || ql;
		return 1e3 / 25.4 * e * t;
	}
	render(e) {
		let t = e.frameState;
		this.viewState_ = t ? t.viewState : null, this.updateElement_();
	}
}, Yl = class extends ye {
	constructor(e, t, n) {
		super(), n ||= {}, this.tileCoord = e, this.state = t, this.key = "", this.transition_ = n.transition === void 0 ? 250 : n.transition, this.transitionStarts_ = {}, this.interpolate = !!n.interpolate;
	}
	changed() {
		this.dispatchEvent(M.CHANGE);
	}
	release() {
		this.setState(L.EMPTY);
	}
	getKey() {
		return this.key + "/" + this.tileCoord;
	}
	getTileCoord() {
		return this.tileCoord;
	}
	getState() {
		return this.state;
	}
	setState(e) {
		if (this.state !== L.EMPTY) {
			if (this.state !== L.ERROR && this.state > e) throw Error("Tile load sequence violation");
			this.state = e, this.changed();
		}
	}
	load() {
		P();
	}
	getAlpha(e, t) {
		if (!this.transition_) return 1;
		let n = this.transitionStarts_[e];
		if (!n) n = t, this.transitionStarts_[e] = n;
		else if (n === -1) return 1;
		let r = t - n + 1e3 / 60;
		return r >= this.transition_ ? 1 : Vr(r / this.transition_);
	}
	inTransition(e) {
		return this.transition_ ? this.transitionStarts_[e] !== -1 : !1;
	}
	endTransition(e) {
		this.transition_ && (this.transitionStarts_[e] = -1);
	}
	disposeInternal() {
		this.release(), super.disposeInternal();
	}
};
//#endregion
//#region node_modules/ol/DataTile.js
function Xl(e) {
	return e instanceof Image || e instanceof HTMLCanvasElement || e instanceof HTMLVideoElement || e instanceof ImageBitmap ? e : null;
}
var Zl = /* @__PURE__ */ Error("disposed"), Ql = [256, 256], $l = class extends Yl {
	constructor(e) {
		let t = L.IDLE;
		super(e.tileCoord, t, {
			transition: e.transition,
			interpolate: e.interpolate
		}), this.loader_ = e.loader, this.data_ = null, this.error_ = null, this.size_ = e.size || null, this.controller_ = e.controller || null;
	}
	getSize() {
		if (this.size_) return this.size_;
		let e = Xl(this.data_);
		return e ? [e.width, e.height] : Ql;
	}
	getData() {
		return this.data_;
	}
	getError() {
		return this.error_;
	}
	load() {
		if (this.state !== L.IDLE && this.state !== L.ERROR) return;
		this.state = L.LOADING, this.changed();
		let e = this;
		this.loader_().then(function(t) {
			e.data_ = t, e.state = L.LOADED, e.changed();
		}).catch(function(t) {
			e.error_ = t, e.state = L.ERROR, e.changed();
		});
	}
	disposeInternal() {
		this.controller_ &&= (this.controller_.abort(Zl), null), super.disposeInternal();
	}
}, eu = class extends Yl {
	constructor(e, t, n, r, i, a) {
		super(e, t, a), this.crossOrigin_ = r?.crossOrigin, this.referrerPolicy_ = r?.referrerPolicy, this.src_ = n, this.key = n, this.image_, g ? this.image_ = new OffscreenCanvas(1, 1) : (this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_)), this.unlisten_ = null, this.tileLoadFunction_ = i;
	}
	getImage() {
		return this.image_;
	}
	setImage(e) {
		this.image_ = e, this.state = L.LOADED, this.unlistenImage_(), this.changed();
	}
	getCrossOrigin() {
		return this.crossOrigin_;
	}
	getReferrerPolicy() {
		return this.referrerPolicy_;
	}
	handleImageError_() {
		this.state = L.ERROR, this.unlistenImage_(), this.image_ = tu(), this.changed();
	}
	handleImageLoad_() {
		if (g) this.state = L.LOADED;
		else {
			let e = this.image_;
			this.state = e.naturalWidth && e.naturalHeight ? L.LOADED : L.EMPTY;
		}
		this.unlistenImage_(), this.changed();
	}
	load() {
		this.state == L.ERROR && (this.state = L.IDLE, this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_)), this.state == L.IDLE && (this.state = L.LOADING, this.changed(), this.tileLoadFunction_(this, this.src_), this.unlisten_ = $s(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
	}
	unlistenImage_() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}
	disposeInternal() {
		this.unlistenImage_(), this.image_ = null, super.disposeInternal();
	}
};
function tu() {
	let e = y(1, 1);
	return e.fillStyle = "rgba(0,0,0,0)", e.fillRect(0, 0, 1, 1), e.canvas;
}
//#endregion
//#region node_modules/ol/TileRange.js
var nu = class {
	constructor(e, t, n, r) {
		this.minX = e, this.maxX = t, this.minY = n, this.maxY = r;
	}
	contains(e) {
		return this.containsXY(e[1], e[2]);
	}
	containsTileRange(e) {
		return this.minX <= e.minX && e.maxX <= this.maxX && this.minY <= e.minY && e.maxY <= this.maxY;
	}
	containsXY(e, t) {
		return this.minX <= e && e <= this.maxX && this.minY <= t && t <= this.maxY;
	}
	equals(e) {
		return this.minX == e.minX && this.minY == e.minY && this.maxX == e.maxX && this.maxY == e.maxY;
	}
	extend(e) {
		e.minX < this.minX && (this.minX = e.minX), e.maxX > this.maxX && (this.maxX = e.maxX), e.minY < this.minY && (this.minY = e.minY), e.maxY > this.maxY && (this.maxY = e.maxY);
	}
	getHeight() {
		return this.maxY - this.minY + 1;
	}
	getSize() {
		return [this.getWidth(), this.getHeight()];
	}
	getWidth() {
		return this.maxX - this.minX + 1;
	}
	intersects(e) {
		return this.minX <= e.maxX && this.maxX >= e.minX && this.minY <= e.maxY && this.maxY >= e.minY;
	}
};
function ru(e, t, n, r, i) {
	return i === void 0 ? new nu(e, t, n, r) : (i.minX = e, i.maxX = t, i.minY = n, i.maxY = r, i);
}
//#endregion
//#region node_modules/ol/reproj.js
var iu, au = [];
function ou(e, t, n, r, i) {
	e.beginPath(), e.moveTo(0, 0), e.lineTo(t, n), e.lineTo(r, i), e.closePath(), e.save(), e.clip(), e.fillRect(0, 0, Math.max(t, r) + 1, Math.max(n, i)), e.restore();
}
function su(e, t) {
	return Math.abs(e[t * 4] - 210) > 2 || Math.abs(e[t * 4 + 3] - 191.25) > 2;
}
function cu() {
	if (iu === void 0) {
		let e = y(6, 6, au);
		e.globalCompositeOperation = "lighter", e.fillStyle = "rgba(210, 0, 0, 0.75)", ou(e, 4, 5, 4, 0), ou(e, 4, 5, 0, 5);
		let t = e.getImageData(0, 0, 3, 3).data;
		iu = su(t, 0) || su(t, 4) || su(t, 8), S(e), au.push(e.canvas);
	}
	return iu;
}
function lu(e, t, n, r) {
	let i = fr(n, t, e), a = rr(t, r, n), o = t.getMetersPerUnit();
	o !== void 0 && (a *= o);
	let s = e.getMetersPerUnit();
	s !== void 0 && (a /= s);
	let c = e.getExtent();
	if (!c || Be(c, i)) {
		let t = rr(e, a, i) / a;
		isFinite(t) && t > 0 && (a /= t);
	}
	return a;
}
function uu(e, t, n, r) {
	let i = lu(e, t, it(n), r);
	return (!isFinite(i) || i <= 0) && et(n, function(n) {
		return i = lu(e, t, n, r), isFinite(i) && i > 0;
	}), i;
}
function du(e, t, n, r, i, a, o, s, c, l, u, d, f, p) {
	let m = y(Math.round(n * e), Math.round(n * t), au);
	if (d || (m.imageSmoothingEnabled = !1), c.length === 0) return m.canvas;
	m.scale(n, n);
	function h(e) {
		return Math.round(e * n) / n;
	}
	m.globalCompositeOperation = "lighter";
	let g = We();
	c.forEach(function(e, t, n) {
		Xe(g, e.extent);
	});
	let _, v = n / r, b = (d ? 1 : 1 + 2 ** -24) / v;
	if (!f || c.length !== 1 || l !== 0) {
		if (_ = y(Math.round(I(g) * v), Math.round(ct(g) * v), au), d || (_.imageSmoothingEnabled = !1), i && p) {
			let e = (i[0] - g[0]) * v, t = -(i[3] - g[3]) * v, n = I(i) * v, r = ct(i) * v;
			_.rect(e, t, n, r), _.clip();
		}
		c.forEach(function(e, t, n) {
			if (e.image.width > 0 && e.image.height > 0) {
				if (e.clipExtent) {
					_.save();
					let t = (e.clipExtent[0] - g[0]) * v, n = -(e.clipExtent[3] - g[3]) * v, r = I(e.clipExtent) * v, i = ct(e.clipExtent) * v;
					_.rect(d ? t : Math.round(t), d ? n : Math.round(n), d ? r : Math.round(t + r) - Math.round(t), d ? i : Math.round(n + i) - Math.round(n)), _.clip();
				}
				let t = (e.extent[0] - g[0]) * v, n = -(e.extent[3] - g[3]) * v, r = I(e.extent) * v, i = ct(e.extent) * v;
				_.drawImage(e.image, l, l, e.image.width - 2 * l, e.image.height - 2 * l, d ? t : Math.round(t), d ? n : Math.round(n), d ? r : Math.round(t + r) - Math.round(t), d ? i : Math.round(n + i) - Math.round(n)), e.clipExtent && _.restore();
			}
		});
	}
	let x = dt(o);
	return s.getTriangles().forEach(function(e, t, n) {
		let r = e.source, i = e.target, o = r[0][0], s = r[0][1], l = r[1][0], u = r[1][1], f = r[2][0], p = r[2][1], v = h((i[0][0] - x[0]) / a), y = h(-(i[0][1] - x[1]) / a), S = h((i[1][0] - x[0]) / a), C = h(-(i[1][1] - x[1]) / a), w = h((i[2][0] - x[0]) / a), T = h(-(i[2][1] - x[1]) / a), E = o, ee = s;
		o = 0, s = 0, l -= E, u -= ee, f -= E, p -= ee;
		let D = Tt([
			[
				l,
				u,
				0,
				0,
				S - v
			],
			[
				f,
				p,
				0,
				0,
				w - v
			],
			[
				0,
				0,
				l,
				u,
				C - y
			],
			[
				0,
				0,
				f,
				p,
				T - y
			]
		]);
		if (!D) return;
		if (m.save(), m.beginPath(), cu() || !d) {
			m.moveTo(S, C);
			let e = v - S, t = y - C;
			for (let n = 0; n < 4; n++) m.lineTo(S + h((n + 1) * e / 4), C + h(n * t / 3)), n != 3 && m.lineTo(S + h((n + 1) * e / 4), C + h((n + 1) * t / 3));
			m.lineTo(w, T);
		} else m.moveTo(S, C), m.lineTo(v, y), m.lineTo(w, T);
		m.clip(), m.transform(D[0], D[2], D[1], D[3], v, y), m.translate(g[0] - E, g[3] - ee);
		let O;
		if (_) O = _.canvas, m.scale(b, -b);
		else {
			let e = c[0], t = e.extent;
			O = e.image, m.scale(I(t) / O.width, -ct(t) / O.height);
		}
		m.drawImage(O, 0, 0), m.restore();
	}), _ && (S(_), au.push(_.canvas)), u && (m.save(), m.globalCompositeOperation = "source-over", m.strokeStyle = "black", m.lineWidth = 1, s.getTriangles().forEach(function(e, t, n) {
		let r = e.target, i = (r[0][0] - x[0]) / a, o = -(r[0][1] - x[1]) / a, s = (r[1][0] - x[0]) / a, c = -(r[1][1] - x[1]) / a, l = (r[2][0] - x[0]) / a, u = -(r[2][1] - x[1]) / a;
		m.beginPath(), m.moveTo(s, c), m.lineTo(i, o), m.lineTo(l, u), m.closePath(), m.stroke();
	}), m.restore()), m.canvas;
}
//#endregion
//#region node_modules/ol/reproj/Triangulation.js
var fu = 10, pu = .25, mu = class {
	constructor(e, t, n, r, i, a, o) {
		this.sourceProj_ = e, this.targetProj_ = t;
		let s = {}, c = o ? sr((e) => z(o, fr(e, this.targetProj_, this.sourceProj_))) : dr(this.targetProj_, this.sourceProj_);
		this.transformInv_ = function(e) {
			let t = e[0] + "/" + e[1];
			return s[t] || (s[t] = c(e)), s[t];
		}, this.maxSourceExtent_ = r, this.errorThresholdSquared_ = i * i, this.triangles_ = [], this.wrapsXInSource_ = !1, this.canWrapXInSource_ = this.sourceProj_.canWrapX() && !!r && !!this.sourceProj_.getExtent() && I(r) >= I(this.sourceProj_.getExtent()), this.sourceWorldWidth_ = this.sourceProj_.getExtent() ? I(this.sourceProj_.getExtent()) : null, this.targetWorldWidth_ = this.targetProj_.getExtent() ? I(this.targetProj_.getExtent()) : null;
		let l = dt(n), u = ft(n), d = rt(n), f = nt(n), p = this.transformInv_(l), m = this.transformInv_(u), h = this.transformInv_(d), g = this.transformInv_(f), _ = fu + (a ? Math.max(0, Math.ceil(Math.log2(tt(n) / (a * a * 256 * 256)))) : 0);
		if (this.addQuad_(l, u, d, f, p, m, h, g, _), this.wrapsXInSource_) {
			let e = Infinity;
			this.triangles_.forEach(function(t, n, r) {
				e = Math.min(e, t.source[0][0], t.source[1][0], t.source[2][0]);
			}), this.triangles_.forEach((t) => {
				if (Math.max(t.source[0][0], t.source[1][0], t.source[2][0]) - e > this.sourceWorldWidth_ / 2) {
					let n = [
						[t.source[0][0], t.source[0][1]],
						[t.source[1][0], t.source[1][1]],
						[t.source[2][0], t.source[2][1]]
					];
					n[0][0] - e > this.sourceWorldWidth_ / 2 && (n[0][0] -= this.sourceWorldWidth_), n[1][0] - e > this.sourceWorldWidth_ / 2 && (n[1][0] -= this.sourceWorldWidth_), n[2][0] - e > this.sourceWorldWidth_ / 2 && (n[2][0] -= this.sourceWorldWidth_);
					let r = Math.min(n[0][0], n[1][0], n[2][0]);
					Math.max(n[0][0], n[1][0], n[2][0]) - r < this.sourceWorldWidth_ / 2 && (t.source = n);
				}
			});
		}
		s = {};
	}
	addTriangle_(e, t, n, r, i, a) {
		this.triangles_.push({
			source: [
				r,
				i,
				a
			],
			target: [
				e,
				t,
				n
			]
		});
	}
	addQuad_(e, t, n, r, i, a, o, s, c) {
		let l = Fe([
			i,
			a,
			o,
			s
		]), u = this.sourceWorldWidth_ ? I(l) / this.sourceWorldWidth_ : null, d = this.sourceWorldWidth_, f = this.sourceProj_.canWrapX() && u > .5 && u < 1, p = !1;
		if (c > 0 && (this.targetProj_.isGlobal() && this.targetWorldWidth_ && (p = I(Fe([
			e,
			t,
			n,
			r
		])) / this.targetWorldWidth_ > pu || p), !f && this.sourceProj_.isGlobal() && u && (p = u > pu || p)), !p && this.maxSourceExtent_ && isFinite(l[0]) && isFinite(l[1]) && isFinite(l[2]) && isFinite(l[3]) && !pt(l, this.maxSourceExtent_)) return;
		let m = 0;
		if (!p && (!isFinite(i[0]) || !isFinite(i[1]) || !isFinite(a[0]) || !isFinite(a[1]) || !isFinite(o[0]) || !isFinite(o[1]) || !isFinite(s[0]) || !isFinite(s[1]))) {
			if (c > 0) p = !0;
			else if (m = (!isFinite(i[0]) || !isFinite(i[1]) ? 8 : 0) + (!isFinite(a[0]) || !isFinite(a[1]) ? 4 : 0) + (!isFinite(o[0]) || !isFinite(o[1]) ? 2 : 0) + +(!isFinite(s[0]) || !isFinite(s[1])), m != 1 && m != 2 && m != 4 && m != 8) return;
		}
		if (c > 0) {
			if (!p) {
				let t = [(e[0] + n[0]) / 2, (e[1] + n[1]) / 2], r = this.transformInv_(t), a;
				a = f ? (Ot(i[0], d) + Ot(o[0], d)) / 2 - Ot(r[0], d) : (i[0] + o[0]) / 2 - r[0];
				let s = (i[1] + o[1]) / 2 - r[1];
				p = a * a + s * s > this.errorThresholdSquared_;
			}
			if (p) {
				if (Math.abs(e[0] - n[0]) <= Math.abs(e[1] - n[1])) {
					let l = [(t[0] + n[0]) / 2, (t[1] + n[1]) / 2], u = this.transformInv_(l), d = [(r[0] + e[0]) / 2, (r[1] + e[1]) / 2], f = this.transformInv_(d);
					this.addQuad_(e, t, l, d, i, a, u, f, c - 1), this.addQuad_(d, l, n, r, f, u, o, s, c - 1);
				} else {
					let l = [(e[0] + t[0]) / 2, (e[1] + t[1]) / 2], u = this.transformInv_(l), d = [(n[0] + r[0]) / 2, (n[1] + r[1]) / 2], f = this.transformInv_(d);
					this.addQuad_(e, l, d, r, i, u, f, s, c - 1), this.addQuad_(l, t, n, d, u, a, o, f, c - 1);
				}
				return;
			}
		}
		if (f) {
			if (!this.canWrapXInSource_) return;
			this.wrapsXInSource_ = !0;
		}
		m & 11 || this.addTriangle_(e, n, r, i, o, s), m & 14 || this.addTriangle_(e, n, t, i, o, a), m && (m & 13 || this.addTriangle_(t, r, e, a, s, i), m & 7 || this.addTriangle_(t, r, n, a, s, o));
	}
	calculateSourceExtent() {
		let e = We();
		return this.triangles_.forEach(function(t, n, r) {
			let i = t.source;
			Ze(e, i[0]), Ze(e, i[1]), Ze(e, i[2]);
		}), e;
	}
	getTriangles() {
		return this.triangles_;
	}
}, hu = .5, gu = class extends Yl {
	constructor(e, t, n, r, i, a, o, s, c, l, u, d) {
		super(i, L.IDLE, d), this.renderEdges_ = u !== void 0 && u, this.pixelRatio_ = o, this.gutter_ = s, this.canvas_ = null, this.sourceTileGrid_ = t, this.targetTileGrid_ = r, this.wrappedTileCoord_ = a || i, this.sourceTiles_ = [], this.sourcesListenerKeys_ = null, this.sourceZ_ = 0, this.clipExtent_ = e.canWrapX() ? e.getExtent() : void 0;
		let f = r.getTileCoordExtent(this.wrappedTileCoord_), p = this.targetTileGrid_.getExtent(), m = this.sourceTileGrid_.getExtent(), h = p ? lt(f, p) : f;
		if (tt(h) === 0) {
			this.state = L.EMPTY;
			return;
		}
		let g = e.getExtent();
		g && (m = m ? lt(m, g) : g);
		let _ = r.getResolution(this.wrappedTileCoord_[0]), v = uu(e, n, h, _);
		if (!isFinite(v) || v <= 0) {
			this.state = L.EMPTY;
			return;
		}
		let y = l === void 0 ? hu : l;
		if (this.triangulation_ = new mu(e, n, h, m, v * y, _), this.triangulation_.getTriangles().length === 0) {
			this.state = L.EMPTY;
			return;
		}
		this.sourceZ_ = t.getZForResolution(v);
		let b = this.triangulation_.calculateSourceExtent();
		if (m && (e.canWrapX() ? (b[1] = St(b[1], m[1], m[3]), b[3] = St(b[3], m[1], m[3])) : b = lt(b, m)), !tt(b)) this.state = L.EMPTY;
		else {
			let n = 0, r = 0;
			e.canWrapX() && (n = I(g), r = Math.floor((b[0] - g[0]) / n)), bt(b.slice(), e, !0).forEach((e) => {
				let i = t.getTileRangeForExtentAndZ(e, this.sourceZ_);
				for (let e = i.minX; e <= i.maxX; e++) for (let t = i.minY; t <= i.maxY; t++) {
					let i = r * n;
					this.sourceTiles_.push({
						getTile: () => c(this.sourceZ_, e, t, o),
						offset: i
					});
				}
				++r;
			}), this.sourceTiles_.length === 0 && (this.state = L.EMPTY);
		}
	}
	getImage() {
		return this.canvas_;
	}
	reproject_() {
		let e = [];
		if (this.sourceTiles_.forEach((t) => {
			let n = t.tile;
			if (n && n.getState() == L.LOADED) {
				let r = this.sourceTileGrid_.getTileCoordExtent(n.tileCoord);
				r[0] += t.offset, r[2] += t.offset;
				let i = this.clipExtent_?.slice();
				i && (i[0] += t.offset, i[2] += t.offset), e.push({
					extent: r,
					clipExtent: i,
					image: n.getImage()
				});
			}
		}), this.sourceTiles_.length = 0, e.length === 0) this.state = L.ERROR;
		else {
			let t = this.wrappedTileCoord_[0], n = this.targetTileGrid_.getTileSize(t), r = typeof n == "number" ? n : n[0], i = typeof n == "number" ? n : n[1], a = this.targetTileGrid_.getResolution(t), o = this.sourceTileGrid_.getResolution(this.sourceZ_), s = this.targetTileGrid_.getTileCoordExtent(this.wrappedTileCoord_);
			this.canvas_ = du(r, i, this.pixelRatio_, o, this.sourceTileGrid_.getExtent(), a, s, this.triangulation_, e, this.gutter_, this.renderEdges_, this.interpolate), this.state = L.LOADED;
		}
		this.changed();
	}
	load() {
		for (let e of this.sourceTiles_) e.tile = e.getTile();
		if (this.state == L.IDLE) {
			this.state = L.LOADING, this.changed();
			let e = 0;
			this.sourcesListenerKeys_ = [], this.sourceTiles_.forEach(({ tile: t }) => {
				let n = t.getState();
				if (n == L.IDLE || n == L.LOADING) {
					e++;
					let n = A(t, M.CHANGE, (r) => {
						let i = t.getState();
						(i == L.LOADED || i == L.ERROR || i == L.EMPTY) && (j(n), e--, e === 0 && (this.unlistenSources_(), this.reproject_()));
					});
					this.sourcesListenerKeys_.push(n);
				}
			}), e === 0 ? setTimeout(this.reproject_.bind(this), 0) : this.sourceTiles_.forEach(function({ tile: e }, t, n) {
				e.getState() == L.IDLE && e.load();
			});
		}
	}
	unlistenSources_() {
		this.sourcesListenerKeys_.forEach(j), this.sourcesListenerKeys_ = null;
	}
	release() {
		this.canvas_ &&= (S(this.canvas_.getContext("2d")), au.push(this.canvas_), null), this.sourceTiles_.length = 0, super.release();
	}
}, _u = class {
	constructor(e) {
		this.highWaterMark = e === void 0 ? 2048 : e, this.count_ = 0, this.entries_ = {}, this.oldest_ = null, this.newest_ = null;
	}
	deleteOldest() {
		let e = this.pop();
		e instanceof ie && e.dispose();
	}
	canExpireCache() {
		return this.highWaterMark > 0 && this.getCount() > this.highWaterMark;
	}
	expireCache(e) {
		for (; this.canExpireCache();) this.deleteOldest();
	}
	clear() {
		for (; this.oldest_;) this.deleteOldest();
	}
	containsKey(e) {
		return this.entries_.hasOwnProperty(e);
	}
	forEach(e) {
		let t = this.oldest_;
		for (; t;) e(t.value_, t.key_, this), t = t.newer;
	}
	get(e, t) {
		let n = this.entries_[e];
		return R(n !== void 0, "Tried to get a value for a key that does not exist in the cache"), n === this.newest_ ? n.value_ : (n === this.oldest_ ? (this.oldest_ = this.oldest_.newer, this.oldest_.older = null) : (n.newer.older = n.older, n.older.newer = n.newer), n.newer = null, n.older = this.newest_, this.newest_.newer = n, this.newest_ = n, n.value_);
	}
	remove(e) {
		let t = this.entries_[e];
		return R(t !== void 0, "Tried to get a value for a key that does not exist in the cache"), t === this.newest_ ? (this.newest_ = t.older, this.newest_ && (this.newest_.newer = null)) : t === this.oldest_ ? (this.oldest_ = t.newer, this.oldest_ && (this.oldest_.older = null)) : (t.newer.older = t.older, t.older.newer = t.newer), delete this.entries_[e], --this.count_, t.value_;
	}
	getCount() {
		return this.count_;
	}
	getKeys() {
		let e = Array(this.count_), t = 0, n;
		for (n = this.newest_; n; n = n.older) e[t++] = n.key_;
		return e;
	}
	getValues() {
		let e = Array(this.count_), t = 0, n;
		for (n = this.newest_; n; n = n.older) e[t++] = n.value_;
		return e;
	}
	peekLast() {
		return this.oldest_.value_;
	}
	peekLastKey() {
		return this.oldest_.key_;
	}
	peekFirstKey() {
		return this.newest_.key_;
	}
	peek(e) {
		return this.entries_[e]?.value_;
	}
	pop() {
		let e = this.oldest_;
		return delete this.entries_[e.key_], e.newer && (e.newer.older = null), this.oldest_ = e.newer, this.oldest_ || (this.newest_ = null), --this.count_, e.value_;
	}
	replace(e, t) {
		this.get(e), this.entries_[e].value_ = t;
	}
	set(e, t) {
		R(!(e in this.entries_), "Tried to set a value for a key that is used already");
		let n = {
			key_: e,
			newer: null,
			older: this.newest_,
			value_: t
		};
		this.newest_ ? this.newest_.newer = n : this.oldest_ = n, this.newest_ = n, this.entries_[e] = n, ++this.count_;
	}
	setSize(e) {
		this.highWaterMark = e;
	}
};
//#endregion
//#region node_modules/ol/tilecoord.js
function vu(e, t, n, r) {
	return r === void 0 ? [
		e,
		t,
		n
	] : (r[0] = e, r[1] = t, r[2] = n, r);
}
function yu(e, t, n) {
	return e + "/" + t + "/" + n;
}
function bu(e, t, n, r, i) {
	return `${F(e)},${t},${yu(n, r, i)}`;
}
function xu(e) {
	return Su(e[0], e[1], e[2]);
}
function Su(e, t, n) {
	return (t << e) + n;
}
function Cu(e, t) {
	let n = e[0], r = e[1], i = e[2];
	if (t.getMinZoom() > n || n > t.getMaxZoom()) return !1;
	let a = t.getFullTileRange(n);
	return !a || a.containsXY(r, i);
}
//#endregion
//#region node_modules/ol/render/canvas/ZIndexContext.js
var wu = class {
	constructor() {
		this.instructions_ = [], this.zIndex = 0, this.offset_ = 0, this.pendingMethod_, this.context_ = new Proxy(x(), {
			get: (e, t) => {
				if (typeof e[t] == "function") return this.pendingMethod_ = t, this.pushMethodArgs_;
			},
			set: (e, t, n) => (this.push_(t, n), !0)
		});
	}
	push_(...e) {
		let t = this.instructions_, n = this.zIndex + this.offset_;
		t[n] || (t[n] = []), t[n].push(...e);
	}
	pushMethodArgs_ = (...e) => {
		this.push_(this.pendingMethod_, e);
	};
	pushFunction(e) {
		this.push_(e);
	}
	getContext() {
		return this.context_;
	}
	draw(e) {
		this.instructions_.forEach((t) => {
			for (let n = 0, r = t.length; n < r; ++n) {
				let r = t[n];
				if (typeof r == "function") {
					r(e);
					continue;
				}
				let i = t[++n];
				typeof e[r] == "function" ? e[r](...i) : e[r] = typeof i == "function" ? i(e) : i;
			}
		});
	}
	clear() {
		this.instructions_.length = 0, this.zIndex = 0, this.offset_ = 0;
	}
	offset() {
		this.offset_ = this.instructions_.length, this.zIndex = 0;
	}
}, Tu = 5, Eu = class extends be {
	constructor(e) {
		super(), this.ready = !0, this.boundHandleImageChange_ = this.handleImageChange_.bind(this), this.layer_ = e, this.staleKeys_ = [], this.maxStaleKeys = Tu, this.renderedSourceKey_;
	}
	getStaleKeys() {
		return this.staleKeys_;
	}
	prependStaleKey(e) {
		this.staleKeys_.unshift(e), this.staleKeys_.length > this.maxStaleKeys && (this.staleKeys_.length = this.maxStaleKeys);
	}
	updateStaleKeys(e) {
		this.renderedSourceKey_ ? this.renderedSourceKey_ !== e && (this.prependStaleKey(this.renderedSourceKey_), this.renderedSourceKey_ = e) : this.renderedSourceKey_ = e;
	}
	getFeatures(e) {
		return P();
	}
	getData(e) {
		return null;
	}
	prepareFrame(e) {
		return P();
	}
	renderFrame(e, t) {
		return P();
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {}
	getLayer() {
		return this.layer_;
	}
	handleFontsChanged() {}
	handleImageChange_(e) {
		let t = e.target;
		(t.getState() === J.LOADED || t.getState() === J.ERROR) && this.renderIfReadyAndVisible();
	}
	loadImage(e) {
		let t = e.getState();
		return t != J.LOADED && t != J.ERROR && e.addEventListener(M.CHANGE, this.boundHandleImageChange_), t == J.IDLE && (e.load(), t = e.getState()), t == J.LOADED;
	}
	renderIfReadyAndVisible() {
		let e = this.getLayer();
		e && e.getVisible() && e.getSourceState() === "ready" && e.changed();
	}
	renderDeferred(e) {}
	disposeInternal() {
		delete this.layer_, super.disposeInternal();
	}
}, Du = [], Ou = null;
function ku() {
	Ou = y(1, 1, void 0, { willReadFrequently: !0 });
}
var Au = class extends Eu {
	constructor(e) {
		super(e), this.container = null, this.renderedResolution, this.tempTransform = qr(), this.pixelTransform = qr(), this.inversePixelTransform = qr(), this.context = null, this.deferredContext_ = null, this.containerReused = !1, this.frameState = null;
	}
	getImageData(e, t, n) {
		Ou || ku(), Ou.clearRect(0, 0, 1, 1);
		let r;
		try {
			Ou.drawImage(e, t, n, 1, 1, 0, 0, 1, 1), r = Ou.getImageData(0, 0, 1, 1).data;
		} catch {
			return Ou = null, null;
		}
		return r;
	}
	getBackground(e) {
		let t = this.getLayer().getBackground();
		return typeof t == "function" && (t = t(e.viewState.resolution)), t || void 0;
	}
	useContainer(e, t, n, r, i) {
		if (O(e) && this.pixelTransform[1] === 0 && this.pixelTransform[2] === 0 && this.pixelTransform[4] === 0 && this.pixelTransform[5] === 0 && e.width === r && e.height === i) {
			let t = e, r = t.getContext("2d");
			if (r) {
				this.container = e, this.context = r, this.containerReused = !0, n && (r.fillStyle = n, r.fillRect(0, 0, t.width, t.height));
				return;
			}
		}
		let a = this.getLayer().getClassName(), o, s;
		if (e && e.className === a && (!n || e && e.style.backgroundColor && de($o(e.style.backgroundColor), $o(n)))) {
			let t = e.firstElementChild;
			O(t) && (s = t.getContext("2d"));
		}
		if (s && oi(s.canvas.style.transform, t) ? (this.container = e, this.context = s, this.containerReused = !0) : this.containerReused ? (this.container = null, this.context = null, this.containerReused = !1) : this.container && (this.container.style.backgroundColor = null), !this.container) {
			o = g ? D() : document.createElement("div"), o.className = a;
			let e = o.style;
			e.position = "absolute", e.width = "100%", e.height = "100%", s = y();
			let t = s.canvas;
			o.appendChild(t), e = t.style, e.position = "absolute", e.left = "0", e.transformOrigin = "top left", this.container = o, this.context = s;
		}
		!this.containerReused && n && !this.container.style.backgroundColor && (this.container.style.backgroundColor = n);
	}
	clipUnrotated(e, t, n) {
		let r = dt(n), i = ft(n), a = rt(n), o = nt(n);
		z(t.coordinateToPixelTransform, r), z(t.coordinateToPixelTransform, i), z(t.coordinateToPixelTransform, a), z(t.coordinateToPixelTransform, o);
		let s = this.inversePixelTransform;
		z(s, r), z(s, i), z(s, a), z(s, o), e.save(), e.beginPath(), e.moveTo(Math.round(r[0]), Math.round(r[1])), e.lineTo(Math.round(i[0]), Math.round(i[1])), e.lineTo(Math.round(a[0]), Math.round(a[1])), e.lineTo(Math.round(o[0]), Math.round(o[1])), e.clip();
	}
	prepareContainer(e, t) {
		let n = e.extent, r = e.viewState.resolution, i = e.viewState.rotation, a = e.pixelRatio, o = Math.round(I(n) / r * a), s = Math.round(ct(n) / r * a);
		ei(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / a, 1 / a, i, -o / 2, -s / 2), ti(this.inversePixelTransform, this.pixelTransform);
		let c = ii(this.pixelTransform), l = this.getBackground(e);
		if (this.useContainer(t, c, l, o, s), !this.containerReused) {
			let e = this.context.canvas;
			e.width != o || e.height != s ? (e.width = o, e.height = s) : this.context.clearRect(0, 0, o, s), c !== e.style.transform && (e.style.transform = c);
		}
	}
	dispatchRenderEvent_(e, t, n) {
		let r = this.getLayer();
		if (r.hasListener(e)) {
			let i = new Ml(e, this.inversePixelTransform, n, t);
			r.dispatchEvent(i);
		}
	}
	preRender(e, t) {
		this.frameState = t, !t.declutter && this.dispatchRenderEvent_(lo.PRERENDER, e, t);
	}
	postRender(e, t) {
		t.declutter || this.dispatchRenderEvent_(lo.POSTRENDER, e, t);
	}
	renderDeferredInternal(e) {}
	getRenderContext(e) {
		return e.declutter && !this.deferredContext_ && (this.deferredContext_ = new wu()), e.declutter ? this.deferredContext_.getContext() : this.context;
	}
	renderDeferred(e) {
		e.declutter && (this.dispatchRenderEvent_(lo.PRERENDER, this.context, e), e.declutter && this.deferredContext_ && (this.deferredContext_.draw(this.context), this.deferredContext_.clear()), this.renderDeferredInternal(e), this.dispatchRenderEvent_(lo.POSTRENDER, this.context, e));
	}
	getRenderTransform(e, t, n, r, i, a, o) {
		let s = i / 2, c = a / 2, l = r / t, u = -l, d = -e[0] + o, f = -e[1];
		return ei(this.tempTransform, s, c, l, u, -n, d, f);
	}
	disposeInternal() {
		delete this.frameState, super.disposeInternal();
	}
};
//#endregion
//#region node_modules/ol/renderer/canvas/TileLayer.js
function ju(e, t, n) {
	if (!(n in e)) return e[n] = /* @__PURE__ */ new Set([t]), !0;
	let r = e[n], i = r.has(t);
	return i || r.add(t), !i;
}
function Mu(e, t, n) {
	let r = e[n];
	return r ? r.delete(t) : !1;
}
function Nu(e, t) {
	let n = e.layerStatesArray[e.layerIndex];
	n.extent && (t = lt(t, yr(n.extent, e.viewState.projection)));
	let r = n.layer.getRenderSource();
	if (!r.getWrapX()) {
		let n = r.getTileGridForProjection(e.viewState.projection).getExtent();
		n && (t = lt(t, n));
	}
	return t;
}
var Pu = class extends Au {
	constructor(e, t) {
		super(e), t ||= {}, this.extentChanged = !0, this.renderComplete = !1, this.renderedExtent_ = null, this.renderedPixelRatio, this.renderedProjection = null, this.renderedTiles = [], this.renderedSourceRevision_, this.tempExtent = We(), this.tempTileRange_ = new nu(0, 0, 0, 0), this.tempTileCoord_ = vu(0, 0, 0);
		let n = t.cacheSize === void 0 ? 512 : t.cacheSize;
		this.tileCache_ = new _u(n), this.sourceTileCache_ = null, this.layerExtent = null, this.maxStaleKeys = n * .5;
	}
	getTileCache() {
		return this.tileCache_;
	}
	getSourceTileCache() {
		return this.sourceTileCache_ ||= new _u(512), this.sourceTileCache_;
	}
	getOrCreateTile(e, t, n, r) {
		let i = this.tileCache_, a = this.getLayer().getSource(), o = bu(a, a.getKey(), e, t, n), s;
		if (i.containsKey(o)) s = i.get(o);
		else {
			let c = r.viewState.projection, l = a.getProjection();
			if (s = a.getTile(e, t, n, r.pixelRatio, c, !l || cr(l, c) ? void 0 : this.getSourceTileCache()), !s) return null;
			i.set(o, s);
		}
		return s;
	}
	getTile(e, t, n, r) {
		return this.getOrCreateTile(e, t, n, r) || null;
	}
	getData(e) {
		let t = this.frameState;
		if (!t) return null;
		let n = this.getLayer(), r = z(t.pixelToCoordinateTransform, e.slice()), i = n.getExtent();
		if (i && !Be(i, r)) return null;
		let a = t.viewState, o = n.getRenderSource(), s = o.getTileGridForProjection(a.projection), c = o.getTilePixelRatio(t.pixelRatio);
		for (let e = s.getZForResolution(a.resolution); e >= s.getMinZoom(); --e) {
			let n = s.getTileCoordForCoordAndZ(r, e), i = this.getTile(e, n[1], n[2], t);
			if (!i || i.getState() !== L.LOADED) continue;
			let l = s.getOrigin(e), u = rs(s.getTileSize(e)), d = s.getResolution(e), f;
			if (i instanceof eu || i instanceof gu) f = i.getImage();
			else if (i instanceof $l) {
				if (f = Xl(i.getData()), !f) continue;
			} else continue;
			let p = Math.floor(c * ((r[0] - l[0]) / d - n[1] * u[0])), m = Math.floor(c * ((l[1] - r[1]) / d - n[2] * u[1])), h = Math.round(c * o.getGutterForProjection(a.projection));
			return this.getImageData(f, p + h, m + h);
		}
		return null;
	}
	prepareFrame(e) {
		this.renderedProjection ? e.viewState.projection !== this.renderedProjection && (this.tileCache_.clear(), this.renderedProjection = e.viewState.projection) : this.renderedProjection = e.viewState.projection;
		let t = this.getLayer().getSource();
		if (!t) return !1;
		let n = t.getRevision();
		return this.renderedSourceRevision_ ? this.renderedSourceRevision_ !== n && (this.renderedSourceRevision_ = n, this.renderedSourceKey_ === t.getKey() && (this.tileCache_.clear(), this.sourceTileCache_?.clear())) : this.renderedSourceRevision_ = n, !0;
	}
	enqueueTilesForNextExtent() {
		return !0;
	}
	enqueueTiles(e, t, n, r, i) {
		let a = e.viewState, o = this.getLayer(), s = o.getRenderSource(), c = s.getTileGridForProjection(a.projection), l = F(s);
		l in e.wantedTiles || (e.wantedTiles[l] = {});
		let u = e.wantedTiles[l], d = o.getMapInternal(), f = Math.max(n - i, c.getMinZoom(), c.getZForResolution(Math.min(o.getMaxResolution(), d ? d.getView().getResolutionForZoom(Math.max(o.getMinZoom(), 0)) : c.getResolution(0)), s.zDirection)), p = a.rotation, m = p ? st(a.center, a.resolution, p, e.size) : void 0;
		for (let i = n; i >= f; --i) {
			let n = c.getTileRangeForExtentAndZ(t, i, this.tempTileRange_), a = c.getResolution(i);
			for (let t = n.minX; t <= n.maxX; ++t) for (let o = n.minY; o <= n.maxY; ++o) {
				if (p && !c.tileCoordIntersectsViewport([
					i,
					t,
					o
				], m)) continue;
				let n = this.getTile(i, t, o, e);
				if (!n || !ju(r, n, i)) continue;
				let s = n.getKey();
				if (u[s] = !0, n.getState() === L.IDLE && !e.tileQueue.isKeyQueued(s)) {
					let r = vu(i, t, o, this.tempTileCoord_);
					e.tileQueue.enqueue([
						n,
						l,
						c.getTileCoordCenter(r),
						a
					]);
				}
			}
		}
	}
	findStaleTile_(e, t) {
		let n = this.tileCache_, r = e[0], i = e[1], a = e[2], o = this.getStaleKeys();
		for (let e = 0; e < o.length; ++e) {
			let s = bu(this.getLayer().getSource(), o[e], r, i, a);
			if (n.containsKey(s)) {
				let e = n.peek(s);
				if (e.getState() === L.LOADED) return e.endTransition(F(this)), ju(t, e, r), !0;
			}
		}
		return !1;
	}
	findAltTiles_(e, t, n, r) {
		let i = e.getTileRangeForTileCoordAndZ(t, n, this.tempTileRange_);
		if (!i) return !1;
		let a = !0, o = this.tileCache_, s = this.getLayer().getRenderSource(), c = s.getKey();
		for (let e = i.minX; e <= i.maxX; ++e) for (let t = i.minY; t <= i.maxY; ++t) {
			let i = bu(s, c, n, e, t), l = !1;
			if (o.containsKey(i)) {
				let e = o.peek(i);
				e.getState() === L.LOADED && (ju(r, e, n), l = !0);
			}
			l || (a = !1);
		}
		return a;
	}
	renderFrame(e, t) {
		this.renderComplete = !0;
		let n = e.layerStatesArray[e.layerIndex], r = e.viewState, i = r.projection, a = r.resolution, o = r.center, s = e.pixelRatio, c = this.getLayer(), l = c.getSource(), u = l.getTileGridForProjection(i), d = u.getZForResolution(a, l.zDirection), f = u.getResolution(d);
		this.updateStaleKeys(l.getKey());
		let p = e.extent, m = l.getTilePixelRatio(s);
		this.prepareContainer(e, t);
		let h = this.context.canvas.width, g = this.context.canvas.height;
		this.layerExtent = n.extent ? yr(n.extent, i) : null, this.layerExtent && (p = lt(p, this.layerExtent));
		let _ = f * h / 2 / m, v = f * g / 2 / m, y = [
			o[0] - _,
			o[1] - v,
			o[0] + _,
			o[1] + v
		], b = {};
		this.renderedTiles.length = 0;
		let x = c.getPreload();
		if (e.nextExtent && this.enqueueTilesForNextExtent()) {
			let t = u.getZForResolution(r.nextResolution, l.zDirection), n = Nu(e, e.nextExtent);
			this.enqueueTiles(e, n, t, b, x);
		}
		let S = Nu(e, p);
		if (this.enqueueTiles(e, S, d, b, 0), x > 0 && setTimeout(() => {
			this.enqueueTiles(e, S, d - 1, b, x - 1);
		}, 0), !(d in b)) return this.container;
		let C = F(this), w = e.time;
		for (let t of b[d]) {
			let n = t.getState();
			if (n === L.EMPTY) continue;
			let r = t.tileCoord;
			if (n === L.LOADED && t.getAlpha(C, w) === 1) {
				t.endTransition(C);
				continue;
			}
			if (n !== L.ERROR && (this.renderComplete = !1), this.findStaleTile_(r, b)) {
				Mu(b, t, d), e.animate = !0;
				continue;
			}
			if (this.findAltTiles_(u, r, d + 1, b)) continue;
			let i = u.getMinZoom();
			for (let e = d - 1; e >= i && !this.findAltTiles_(u, r, e, b); --e);
		}
		let T = f / a * s / m, E = this.getRenderContext(e);
		ei(this.tempTransform, h / 2, g / 2, T, T, 0, -h / 2, -g / 2), this.layerExtent && this.clipUnrotated(E, e, this.layerExtent), l.getInterpolate() || (E.imageSmoothingEnabled = !1), this.preRender(E, e);
		let ee = Object.keys(b).map(Number);
		ee.sort(oe);
		let D = [], O = [], k = [];
		for (let t = ee.length - 1; t >= 0; --t) {
			let n = ee[t], r = l.getTilePixelSize(n, s, i), a = u.getResolution(n) / f, o = r[0] * a * T, c = r[1] * a * T, p = u.getTileCoordForCoordAndZ(dt(y), n), h = u.getTileCoordExtent(p), g = z(this.tempTransform, [m * (h[0] - y[0]) / f, m * (y[3] - h[3]) / f]), _ = m * l.getGutterForProjection(i);
			for (let t of b[n]) {
				if (t.getState() !== L.LOADED) continue;
				let r = t.tileCoord, i = p[1] - r[1], a = Math.round(g[0] - (i - 1) * o), s = p[2] - r[2], u = Math.round(g[1] - (s - 1) * c), f = Math.round(g[0] - i * o), m = Math.round(g[1] - s * c), h = a - f, v = u - m, y = n === d;
				if (y && t.inTransition(C)) {
					k.push({
						tile: t,
						x: f,
						y: m,
						w: h,
						h: v,
						gutter: _
					}), this.renderedTiles.unshift(t), this.updateUsedTiles(e.usedTiles, l, t);
					continue;
				}
				let b = [
					f,
					m,
					f + h,
					m + v
				], x = [];
				for (let e = 0, t = D.length; e < t; ++e) n < O[e] && pt(b, D[e]) && x.push(D[e]);
				let S;
				x.length > 0 && (S = xt(b, x)), D.push(b), O.push(n), this.drawTile(t, e, f, m, h, v, _, y, S), this.renderedTiles.unshift(t), this.updateUsedTiles(e.usedTiles, l, t);
			}
		}
		for (let t = 0, n = k.length; t < n; ++t) {
			let { tile: n, x: r, y: i, w: a, h: o, gutter: s } = k[t];
			this.drawTile(n, e, r, i, a, o, s, !0, void 0);
		}
		return this.renderedResolution = f, this.extentChanged = !this.renderedExtent_ || !Ye(this.renderedExtent_, y), this.renderedExtent_ = y, this.renderedPixelRatio = s, this.postRender(this.context, e), this.layerExtent && E.restore(), E.imageSmoothingEnabled = !0, this.renderComplete && e.postRenderFunctions.push((e, t) => {
			let n = F(l), r = t.wantedTiles[n], i = r ? Object.keys(r).length : 0;
			this.updateCacheSize(i), this.tileCache_.expireCache(), this.sourceTileCache_?.expireCache();
		}), this.container;
	}
	updateCacheSize(e) {
		this.tileCache_.highWaterMark = Math.max(this.tileCache_.highWaterMark, e * 2);
	}
	drawTile(e, t, n, r, i, a, o, s, c) {
		let l;
		if (e instanceof $l) {
			if (l = Xl(e.getData()), !l) throw Error("Rendering array data is not yet supported");
		} else l = this.getTileImage(e);
		if (!l) return;
		let u = this.getRenderContext(t), d = F(this), f = t.layerStatesArray[t.layerIndex], p = f.opacity * (s ? e.getAlpha(d, t.time) : 1), m = p !== u.globalAlpha;
		m && (u.save(), u.globalAlpha = p);
		let h = l.width - 2 * o, g = l.height - 2 * o;
		if (c) {
			let e = h / i, t = g / a;
			for (let i = 0, a = c.length; i < a; ++i) {
				let a = c[i], s = a[0], d = a[1], f = a[2] - a[0], p = a[3] - a[1];
				u.drawImage(l, o + (s - n) * e, o + (d - r) * t, f * e, p * t, s, d, f, p);
			}
		} else u.drawImage(l, o, o, h, g, n, r, i, a);
		m && u.restore(), p === f.opacity ? s && e.endTransition(d) : t.animate = !0;
	}
	getImage() {
		let e = this.context;
		return e ? e.canvas : null;
	}
	getTileImage(e) {
		return e.getImage();
	}
	updateUsedTiles(e, t, n) {
		let r = F(t);
		r in e || (e[r] = {}), e[r][n.getKey()] = !0;
	}
}, Fu = {
	PRELOAD: "preload",
	USE_INTERIM_TILES_ON_ERROR: "useInterimTilesOnError"
}, Iu = class extends uo {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e), n = e.cacheSize;
		delete e.cacheSize, delete t.preload, delete t.useInterimTilesOnError, super(t), this.on, this.once, this.un, this.cacheSize_ = n, this.setPreload(e.preload === void 0 ? 0 : e.preload), this.setUseInterimTilesOnError(e.useInterimTilesOnError === void 0 || e.useInterimTilesOnError);
	}
	getCacheSize() {
		return this.cacheSize_;
	}
	getPreload() {
		return this.get(Fu.PRELOAD);
	}
	setPreload(e) {
		this.set(Fu.PRELOAD, e);
	}
	getUseInterimTilesOnError() {
		return this.get(Fu.USE_INTERIM_TILES_ON_ERROR);
	}
	setUseInterimTilesOnError(e) {
		this.set(Fu.USE_INTERIM_TILES_ON_ERROR, e);
	}
	getData(e) {
		return super.getData(e);
	}
}, Lu = class extends Iu {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new Pu(this, { cacheSize: this.getCacheSize() });
	}
}, Ru = [
	0,
	0,
	0
], zu = 5, Bu = class {
	constructor(e) {
		let t = e.minZoom, n = e.resolutions;
		t === void 0 && n && (t = n.findIndex((e) => e !== void 0)), this.minZoom = t === void 0 ? 0 : t, this.resolutions_ = n, R(fe(this.resolutions_, (e, t) => t - e, !0), "`resolutions` must be sorted in descending order");
		let r;
		if (!e.origins) {
			for (let e = 0, t = this.resolutions_.length - 1; e < t; ++e) if (!r) r = this.resolutions_[e] / this.resolutions_[e + 1];
			else if (this.resolutions_[e] / this.resolutions_[e + 1] !== r) {
				r = void 0;
				break;
			}
		}
		this.zoomFactor_ = r, this.maxZoom = this.resolutions_.length - 1, this.origin_ = e.origin === void 0 ? null : e.origin, this.origins_ = null, e.origins !== void 0 && (this.origins_ = e.origins, R(this.origins_.length == this.resolutions_.length, "Number of `origins` and `resolutions` must be equal"));
		let i = e.extent;
		i !== void 0 && !this.origin_ && !this.origins_ && (this.origin_ = dt(i)), R(!this.origin_ && this.origins_ || this.origin_ && !this.origins_, "Either `origin` or `origins` must be configured, never both"), this.tileSizes_ = null, e.tileSizes !== void 0 && (this.tileSizes_ = e.tileSizes, R(this.tileSizes_.length == this.resolutions_.length, "Number of `tileSizes` and `resolutions` must be equal")), this.tileSize_ = e.tileSize === void 0 ? this.tileSizes_ ? null : 256 : e.tileSize, R(!this.tileSize_ && this.tileSizes_ || this.tileSize_ && !this.tileSizes_, "Either `tileSize` or `tileSizes` must be configured, never both"), this.extent_ = i === void 0 ? null : i, this.fullTileRanges_ = null, this.tmpSize_ = [0, 0], this.tmpExtent_ = [
			0,
			0,
			0,
			0
		], e.tileRanges === void 0 ? e.sizes === void 0 ? i && this.calculateTileRanges_(i) : this.fullTileRanges_ = e.sizes.map((e, t) => {
			let n = new nu(Math.min(0, e[0]), Math.max(e[0] - 1, -1), Math.min(0, e[1]), Math.max(e[1] - 1, -1));
			if (i) {
				let e = this.getTileRangeForExtentAndZ(i, t);
				n.minX = Math.max(e.minX, n.minX), n.maxX = Math.min(e.maxX, n.maxX), n.minY = Math.max(e.minY, n.minY), n.maxY = Math.min(e.maxY, n.maxY);
			}
			return n;
		}) : this.fullTileRanges_ = e.tileRanges;
	}
	forEachTileCoord(e, t, n) {
		let r = this.getTileRangeForExtentAndZ(e, t);
		for (let e = r.minX, i = r.maxX; e <= i; ++e) for (let i = r.minY, a = r.maxY; i <= a; ++i) n([
			t,
			e,
			i
		]);
	}
	forEachTileCoordParentTileRange(e, t, n, r) {
		let i, a, o, s = null, c = e[0] - 1;
		for (this.zoomFactor_ === 2 ? (a = e[1], o = e[2]) : s = this.getTileCoordExtent(e, r); c >= this.minZoom;) {
			if (a !== void 0 && o !== void 0 ? (a = Math.floor(a / 2), o = Math.floor(o / 2), i = ru(a, a, o, o, n)) : i = this.getTileRangeForExtentAndZ(s, c, n), t(c, i)) return !0;
			--c;
		}
		return !1;
	}
	getExtent() {
		return this.extent_;
	}
	getMaxZoom() {
		return this.maxZoom;
	}
	getMinZoom() {
		return this.minZoom;
	}
	getOrigin(e) {
		return this.origin_ ? this.origin_ : this.origins_[e];
	}
	getOrigins() {
		return this.origins_;
	}
	getResolution(e) {
		return this.resolutions_[e];
	}
	getResolutions() {
		return this.resolutions_;
	}
	getTileCoordChildTileRange(e, t, n) {
		if (e[0] < this.maxZoom) {
			if (this.zoomFactor_ === 2) {
				let n = e[1] * 2, r = e[2] * 2;
				return ru(n, n + 1, r, r + 1, t);
			}
			let r = this.getTileCoordExtent(e, n || this.tmpExtent_);
			return this.getTileRangeForExtentAndZ(r, e[0] + 1, t);
		}
		return null;
	}
	getTileRangeForTileCoordAndZ(e, t, n) {
		if (t > this.maxZoom || t < this.minZoom) return null;
		let r = e[0], i = e[1], a = e[2];
		if (t === r) return ru(i, a, i, a, n);
		if (this.zoomFactor_) {
			let e = this.zoomFactor_ ** +(t - r), o = Math.floor(i * e), s = Math.floor(a * e);
			return t < r ? ru(o, o, s, s, n) : ru(o, Math.floor(e * (i + 1)) - 1, s, Math.floor(e * (a + 1)) - 1, n);
		}
		let o = this.getTileCoordExtent(e, this.tmpExtent_);
		return this.getTileRangeForExtentAndZ(o, t, n);
	}
	getTileRangeForExtentAndZ(e, t, n) {
		this.getTileCoordForXYAndZ_(e[0], e[3], t, !1, Ru);
		let r = Ru[1], i = Ru[2];
		this.getTileCoordForXYAndZ_(e[2], e[1], t, !0, Ru);
		let a = Ru[1], o = Ru[2];
		return ru(r, a, i, o, n);
	}
	getTileCoordCenter(e) {
		let t = this.getOrigin(e[0]), n = this.getResolution(e[0]), r = rs(this.getTileSize(e[0]), this.tmpSize_);
		return [t[0] + (e[1] + .5) * r[0] * n, t[1] - (e[2] + .5) * r[1] * n];
	}
	getTileCoordExtent(e, t) {
		let n = this.getOrigin(e[0]), r = this.getResolution(e[0]), i = rs(this.getTileSize(e[0]), this.tmpSize_), a = n[0] + e[1] * i[0] * r, o = n[1] - (e[2] + 1) * i[1] * r;
		return Ge(a, o, a + i[0] * r, o + i[1] * r, t);
	}
	getTileCoordForCoordAndResolution(e, t, n) {
		return this.getTileCoordForXYAndResolution_(e[0], e[1], t, !1, n);
	}
	getTileCoordForXYAndResolution_(e, t, n, r, i) {
		let a = this.getZForResolution(n), o = n / this.getResolution(a), s = this.getOrigin(a), c = rs(this.getTileSize(a), this.tmpSize_), l = o * (e - s[0]) / n / c[0], u = o * (s[1] - t) / n / c[1];
		return r ? (l = Mt(l, zu) - 1, u = Mt(u, zu) - 1) : (l = jt(l, zu), u = jt(u, zu)), vu(a, l, u, i);
	}
	getTileCoordForXYAndZ_(e, t, n, r, i) {
		let a = this.getOrigin(n), o = this.getResolution(n), s = rs(this.getTileSize(n), this.tmpSize_), c = (e - a[0]) / o / s[0], l = (a[1] - t) / o / s[1];
		return r ? (c = Mt(c, zu) - 1, l = Mt(l, zu) - 1) : (c = jt(c, zu), l = jt(l, zu)), vu(n, c, l, i);
	}
	getTileCoordForCoordAndZ(e, t, n) {
		return this.getTileCoordForXYAndZ_(e[0], e[1], t, !1, n);
	}
	getTileCoordResolution(e) {
		return this.resolutions_[e[0]];
	}
	getTileSize(e) {
		return this.tileSize_ ? this.tileSize_ : this.tileSizes_[e];
	}
	getFullTileRange(e) {
		return this.fullTileRanges_ ? this.fullTileRanges_[e] : this.extent_ ? this.getTileRangeForExtentAndZ(this.extent_, e) : null;
	}
	getZForResolution(e, t) {
		return St(ce(this.resolutions_, e, t || 0), this.minZoom, this.maxZoom);
	}
	tileCoordIntersectsViewport(e, t) {
		return Ii(t, 0, t.length, 2, this.getTileCoordExtent(e));
	}
	calculateTileRanges_(e) {
		let t = this.resolutions_.length, n = Array(t);
		for (let r = this.minZoom; r < t; ++r) n[r] = this.getTileRangeForExtentAndZ(e, r);
		this.fullTileRanges_ = n;
	}
}, Vu = /\{z\}/g, Hu = /\{x\}/g, Uu = /\{y\}/g, Wu = /\{-y\}/g;
function Gu(e, t, n, r, i) {
	return e.replace(Vu, t.toString()).replace(Hu, n.toString()).replace(Uu, r.toString()).replace(Wu, function() {
		if (i === void 0) throw Error("If the URL template has a {-y} placeholder, the grid extent must be known");
		return (i - r).toString();
	});
}
function Ku(e) {
	let t = [], n = /\{([a-z])-([a-z])\}/.exec(e);
	if (n) {
		let r = n[1].charCodeAt(0), i = n[2].charCodeAt(0), a;
		for (a = r; a <= i; ++a) t.push(e.replace(n[0], String.fromCharCode(a)));
		return t;
	}
	if (n = /\{(\d+)-(\d+)\}/.exec(e), n) {
		let r = parseInt(n[2], 10);
		for (let i = parseInt(n[1], 10); i <= r; i++) t.push(e.replace(n[0], i.toString()));
		return t;
	}
	return t.push(e), t;
}
//#endregion
//#region node_modules/ol/tileurlfunction.js
function qu(e, t) {
	return (function(n, r, i) {
		if (!n) return;
		let a, o = n[0];
		if (t) {
			let e = t.getFullTileRange(o);
			e && (a = e.getHeight() - 1);
		}
		return Gu(e, o, n[1], n[2], a);
	});
}
function Ju(e, t) {
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = qu(e[i], t);
	return Yu(r);
}
function Yu(e) {
	return e.length === 1 ? e[0] : (function(t, n, r) {
		return t ? e[Ot(xu(t), e.length)](t, n, r) : void 0;
	});
}
//#endregion
//#region node_modules/ol/tilegrid.js
function Xu(e) {
	let t = e.getDefaultTileGrid();
	return t || (t = ed(e), e.setDefaultTileGrid(t)), t;
}
function Zu(e, t, n) {
	let r = t[0], i = e.getTileCoordCenter(t), a = td(n);
	if (!Be(a, i)) {
		let t = I(a), n = Math.ceil((a[0] - i[0]) / t);
		return i[0] += t * n, e.getTileCoordForCoordAndZ(i, r);
	}
	return t;
}
function Qu(e, t, n, r) {
	r = r === void 0 ? "top-left" : r;
	let i = $u(e, t, n);
	return new Bu({
		extent: e,
		origin: at(e, r),
		resolutions: i,
		tileSize: n
	});
}
function $u(e, t, n, r) {
	t = t === void 0 ? 42 : t, n = rs(n === void 0 ? 256 : n);
	let i = ct(e), a = I(e);
	r = r > 0 ? r : Math.max(a / n[0], i / n[1]);
	let o = t + 1, s = Array(o);
	for (let e = 0; e < o; ++e) s[e] = r / 2 ** e;
	return s;
}
function ed(e, t, n, r) {
	return Qu(td(e), t, n, r);
}
function td(e) {
	e = nr(e);
	let t = e.getExtent();
	if (!t) {
		let n = 180 * Kt.degrees / e.getMetersPerUnit();
		t = Ge(-n, -n, n, n);
	}
	return t;
}
//#endregion
//#region node_modules/ol/source/Source.js
var nd = class extends we {
	constructor(e) {
		super(), this.projection = nr(e.projection), this.attributions_ = rd(e.attributions), this.attributionsCollapsible_ = e.attributionsCollapsible ?? !0, this.loading = !1, this.state_ = e.state === void 0 ? "ready" : e.state, this.wrapX_ = e.wrapX !== void 0 && e.wrapX, this.interpolate_ = !!e.interpolate, this.viewResolver = null, this.viewRejector = null;
		let t = this;
		this.viewPromise_ = new Promise(function(e, n) {
			t.viewResolver = e, t.viewRejector = n;
		});
	}
	getAttributions() {
		return this.attributions_;
	}
	getAttributionsCollapsible() {
		return this.attributionsCollapsible_;
	}
	getProjection() {
		return this.projection;
	}
	getResolutions(e) {
		return null;
	}
	getView() {
		return this.viewPromise_;
	}
	ready() {
		let e = this.getState();
		return e === "ready" ? Promise.resolve() : e === "error" ? Promise.reject(/* @__PURE__ */ Error("Source failed to load")) : new Promise((e, t) => {
			let n = () => {
				let r = this.getState();
				r === "ready" ? (this.un("change", n), e()) : r === "error" && (this.un("change", n), t(/* @__PURE__ */ Error("Source failed to load")));
			};
			this.on("change", n);
		});
	}
	getState() {
		return this.state_;
	}
	getWrapX() {
		return this.wrapX_;
	}
	getInterpolate() {
		return this.interpolate_;
	}
	refresh() {
		this.changed();
	}
	setAttributions(e) {
		this.attributions_ = rd(e), this.changed();
	}
	setState(e) {
		this.state_ = e, this.changed();
	}
};
function rd(e) {
	return e ? typeof e == "function" ? e : (Array.isArray(e) || (e = [e]), (t) => e) : null;
}
//#endregion
//#region node_modules/ol/source/Tile.js
var id = class extends nd {
	constructor(e) {
		super({
			attributions: e.attributions,
			attributionsCollapsible: e.attributionsCollapsible,
			projection: e.projection,
			state: e.state,
			wrapX: e.wrapX,
			interpolate: e.interpolate
		}), this.on, this.once, this.un, this.tilePixelRatio_ = e.tilePixelRatio === void 0 ? 1 : e.tilePixelRatio, this.tileGrid = e.tileGrid === void 0 ? null : e.tileGrid, this.tileGrid && rs(this.tileGrid.getTileSize(this.tileGrid.getMinZoom()), [256, 256]), this.tmpSize = [0, 0], this.key_ = e.key || F(this), this.tileOptions = {
			transition: e.transition,
			interpolate: e.interpolate
		}, this.zDirection = e.zDirection ? e.zDirection : 0;
	}
	getGutterForProjection(e) {
		return 0;
	}
	getKey() {
		return this.key_;
	}
	setKey(e) {
		this.key_ !== e && (this.key_ = e, this.changed());
	}
	getResolutions(e) {
		let t = e ? this.getTileGridForProjection(e) : this.tileGrid;
		return t ? t.getResolutions() : null;
	}
	getTile(e, t, n, r, i, a) {
		return P();
	}
	getTileGrid() {
		return this.tileGrid;
	}
	getTileGridForProjection(e) {
		return this.tileGrid ? this.tileGrid : Xu(e);
	}
	getTilePixelRatio(e) {
		return this.tilePixelRatio_;
	}
	getTilePixelSize(e, t, n) {
		let r = this.getTileGridForProjection(n), i = this.getTilePixelRatio(t), a = rs(r.getTileSize(e), this.tmpSize);
		return i == 1 ? a : ns(a, i, this.tmpSize);
	}
	getTileCoordForTileUrlFunction(e, t) {
		let n = t === void 0 ? this.getProjection() : t, r = t === void 0 && this.tileGrid || this.getTileGridForProjection(n);
		return this.getWrapX() && n.isGlobal() && (e = Zu(r, e, n)), Cu(e, r) ? e : null;
	}
	clear() {}
	refresh() {
		this.clear(), super.refresh();
	}
}, ad = class extends ve {
	constructor(e, t) {
		super(e), this.tile = t;
	}
}, od = {
	TILELOADSTART: "tileloadstart",
	TILELOADEND: "tileloadend",
	TILELOADERROR: "tileloaderror"
}, sd = class e extends id {
	constructor(t) {
		super({
			attributions: t.attributions,
			cacheSize: t.cacheSize,
			projection: t.projection,
			state: t.state,
			tileGrid: t.tileGrid,
			tilePixelRatio: t.tilePixelRatio,
			wrapX: t.wrapX,
			transition: t.transition,
			interpolate: t.interpolate,
			key: t.key,
			attributionsCollapsible: t.attributionsCollapsible,
			zDirection: t.zDirection
		}), this.generateTileUrlFunction_ = this.tileUrlFunction === e.prototype.tileUrlFunction, this.tileLoadFunction = t.tileLoadFunction, t.tileUrlFunction && (this.tileUrlFunction = t.tileUrlFunction), this.urls = null, t.urls ? this.setUrls(t.urls) : t.url && this.setUrl(t.url), this.tileLoadingKeys_ = {};
	}
	getTileLoadFunction() {
		return this.tileLoadFunction;
	}
	getTileUrlFunction() {
		return Object.getPrototypeOf(this).tileUrlFunction === this.tileUrlFunction ? this.tileUrlFunction.bind(this) : this.tileUrlFunction;
	}
	getUrls() {
		return this.urls;
	}
	handleTileChange(e) {
		let t = e.target, n = F(t), r = t.getState(), i;
		r == L.LOADING ? (this.tileLoadingKeys_[n] = !0, i = od.TILELOADSTART) : n in this.tileLoadingKeys_ && (delete this.tileLoadingKeys_[n], i = r == L.ERROR ? od.TILELOADERROR : r == L.LOADED ? od.TILELOADEND : void 0), i != null && this.dispatchEvent(new ad(i, t));
	}
	setTileLoadFunction(e) {
		this.tileLoadFunction = e, this.changed();
	}
	setTileUrlFunction(e, t) {
		this.tileUrlFunction = e, t === void 0 ? this.changed() : this.setKey(t);
	}
	setUrl(e) {
		let t = Ku(e);
		this.urls = t, this.setUrls(t);
	}
	setUrls(e) {
		this.urls = e;
		let t = e.join("\n");
		this.generateTileUrlFunction_ ? this.setTileUrlFunction(Ju(e, this.tileGrid), t) : this.setKey(t);
	}
	tileUrlFunction(e, t, n) {}
}, cd = class extends sd {
	constructor(e) {
		super({
			attributions: e.attributions,
			cacheSize: e.cacheSize,
			projection: e.projection,
			state: e.state,
			tileGrid: e.tileGrid,
			tileLoadFunction: e.tileLoadFunction ? e.tileLoadFunction : ld,
			tilePixelRatio: e.tilePixelRatio,
			tileUrlFunction: e.tileUrlFunction,
			url: e.url,
			urls: e.urls,
			wrapX: e.wrapX,
			transition: e.transition,
			interpolate: e.interpolate === void 0 || e.interpolate,
			key: e.key,
			attributionsCollapsible: e.attributionsCollapsible,
			zDirection: e.zDirection
		}), this.crossOrigin = e.crossOrigin === void 0 ? null : e.crossOrigin, this.referrerPolicy = e.referrerPolicy, this.tileClass = e.tileClass === void 0 ? eu : e.tileClass, this.tileGridForProjection = {}, this.reprojectionErrorThreshold_ = e.reprojectionErrorThreshold, this.renderReprojectionEdges_ = !1;
	}
	getGutterForProjection(e) {
		return this.getProjection() && e && !cr(this.getProjection(), e) ? 0 : this.getGutter();
	}
	getGutter() {
		return 0;
	}
	getKey() {
		let e = super.getKey();
		return this.getInterpolate() || (e += ":disable-interpolation"), e;
	}
	getTileGridForProjection(e) {
		let t = this.getProjection();
		if (this.tileGrid && (!t || cr(t, e))) return this.tileGrid;
		let n = F(e);
		return n in this.tileGridForProjection || (this.tileGridForProjection[n] = Xu(e)), this.tileGridForProjection[n];
	}
	createTile_(e, t, n, r, i, a) {
		let o = [
			e,
			t,
			n
		], s = this.getTileCoordForTileUrlFunction(o, i), c = s ? this.tileUrlFunction(s, r, i) : void 0, l = new this.tileClass(o, c === void 0 ? L.EMPTY : L.IDLE, c === void 0 ? "" : c, {
			crossOrigin: this.crossOrigin,
			referrerPolicy: this.referrerPolicy
		}, this.tileLoadFunction, this.tileOptions);
		return l.key = a, l.addEventListener(M.CHANGE, this.handleTileChange.bind(this)), l;
	}
	getTile(e, t, n, r, i, a) {
		let o = this.getProjection();
		if (!o || !i || cr(o, i)) return this.getTileInternal(e, t, n, r, o || i);
		let s = [
			e,
			t,
			n
		], c = this.getKey(), l = new gu(o, this.getTileGridForProjection(o), i, this.getTileGridForProjection(i), s, this.getTileCoordForTileUrlFunction(s, i), this.getTilePixelRatio(r), this.getGutter(), (e, t, n, r) => this.getTileInternal(e, t, n, r, o, a), this.reprojectionErrorThreshold_, this.renderReprojectionEdges_, this.tileOptions);
		return l.key = c, l;
	}
	getTileInternal(e, t, n, r, i, a) {
		let o = this.getKey(), s = bu(this, o, e, t, n);
		if (a && a.containsKey(s)) return a.get(s);
		let c = this.createTile_(e, t, n, r, i, o);
		return a?.set(s, c), c;
	}
	setRenderReprojectionEdges(e) {
		this.renderReprojectionEdges_ != e && (this.renderReprojectionEdges_ = e, this.changed());
	}
	setTileGridForProjection(e, t) {
		let n = nr(e);
		if (n) {
			let e = F(n);
			e in this.tileGridForProjection || (this.tileGridForProjection[e] = t);
		}
	}
};
function ld(e, t) {
	if (g) {
		let n = e.getCrossOrigin(), r = "same-origin", i = "same-origin";
		n === "anonymous" || n === "" ? (r = "cors", i = "omit") : n === "use-credentials" && (r = "cors", i = "include");
		let a = {
			mode: r,
			credentials: i,
			referrerPolicy: e.getReferrerPolicy()
		};
		fetch(t, a).then((e) => {
			if (!e.ok) throw Error(`HTTP ${e.status}`);
			return e.blob();
		}).then((e) => createImageBitmap(e)).then((t) => {
			let n = e.getImage();
			n.width = t.width, n.height = t.height, n.getContext("2d").drawImage(t, 0, 0), t.close?.(), n.dispatchEvent(new Event("load"));
		}).catch(() => {
			e.getImage().dispatchEvent(new Event("error"));
		});
		return;
	}
	e.getImage().src = t;
}
//#endregion
//#region node_modules/ol/source/Zoomify.js
var ud = class extends eu {
	constructor(e, t, n, r, i, a, o) {
		super(t, n, r, i, a, o), this.zoomifyImage_ = null, this.tileSize_ = e;
	}
	getImage() {
		if (this.zoomifyImage_) return this.zoomifyImage_;
		let e = super.getImage();
		if (this.state == L.LOADED) {
			let t = this.tileSize_;
			if (e.width == t[0] && e.height == t[1]) return this.zoomifyImage_ = e, e;
			let n = y(t[0], t[1]);
			return n.drawImage(e, 0, 0), this.zoomifyImage_ = n.canvas, n.canvas;
		}
		return e;
	}
}, dd = class extends cd {
	constructor(e) {
		let t = e.size, n = e.tierSizeCalculation === void 0 ? "default" : e.tierSizeCalculation, r = e.tilePixelRatio || 1, i = t[0], a = t[1], o = [], s = e.tileSize || 256, c = s * r;
		switch (n) {
			case "default":
				for (; i > c || a > c;) o.push([Math.ceil(i / c), Math.ceil(a / c)]), c += c;
				break;
			case "truncated":
				let e = i, t = a;
				for (; e > c || t > c;) o.push([Math.ceil(e / c), Math.ceil(t / c)]), e >>= 1, t >>= 1;
				break;
			default: throw Error("Unknown `tierSizeCalculation` configured");
		}
		o.push([1, 1]), o.reverse();
		let l = [r], u = [0];
		for (let e = 1, t = o.length; e < t; e++) l.push(r << e), u.push(o[e - 1][0] * o[e - 1][1] + u[e - 1]);
		l.reverse();
		let d = new Bu({
			tileSize: s,
			extent: e.extent || [
				0,
				-a,
				i,
				0
			],
			resolutions: l
		}), f = e.url;
		f && !f.includes("{TileGroup}") && !f.includes("{tileIndex}") && (f += "{TileGroup}/{z}-{x}-{y}.jpg");
		let p = Ku(f), m = s * r;
		function h(e) {
			return (function(t, n, r) {
				if (!t) return;
				let i = t[0], a = t[1], s = t[2], c = a + s * o[i][0], l = {
					z: i,
					x: a,
					y: s,
					tileIndex: c,
					TileGroup: "TileGroup" + ((c + u[i]) / m | 0)
				};
				return e.replace(/\{(\w+?)\}/g, function(e, t) {
					return l[t];
				});
			});
		}
		let g = Yu(p.map(h)), _ = ud.bind(null, rs(s * r));
		super({
			attributions: e.attributions,
			cacheSize: e.cacheSize,
			crossOrigin: e.crossOrigin,
			referrerPolicy: e.referrerPolicy,
			interpolate: e.interpolate,
			projection: e.projection,
			tilePixelRatio: r,
			reprojectionErrorThreshold: e.reprojectionErrorThreshold,
			tileClass: _,
			tileGrid: d,
			tileUrlFunction: g,
			transition: e.transition
		}), this.zDirection = e.zDirection;
		let v = g(d.getTileCoordForCoordAndResolution(it(d.getExtent()), l[l.length - 1]), 1, null), y = new Image();
		y.addEventListener("error", () => {
			m = s, this.changed();
		}), y.src = v;
	}
}, fd = function(e) {
	if (!e) return null;
	var t = e.getViewport().getElementsByClassName("ol-fixedoverlay")[0];
	return t || (e.getViewport().querySelector(".ol-layers") ? (t = document.createElement("canvas"), t.className = "ol-fixedoverlay", e.getViewport().querySelector(".ol-layers").after(t), e.on("precompose", function(n) {
		t.width = e.getSize()[0] * n.frameState.pixelRatio, t.height = e.getSize()[1] * n.frameState.pixelRatio;
	})) : t = e.getViewport().querySelector("canvas")), t;
}, pd = class extends Te {
	constructor(e) {
		e ||= {}, super(e), this.setStyle(e.style);
	}
	setMap(e) {
		this.getCanvas(e);
		var t = this.getMap();
		if (this._listener &&= (xe(this._listener), null), super.setMap(e), t) try {
			t.renderSync();
		} catch {}
		e && (this._listener = e.on("postcompose", this._draw.bind(this)));
	}
	getCanvas(e) {
		return fd(e);
	}
	getContext(e) {
		var t = e.context;
		if (!t && this.getMap()) {
			var n = this.getMap().getViewport().getElementsByClassName("ol-fixedoverlay")[0];
			t = n ? n.getContext("2d") : null;
		}
		return t;
	}
	setStyle(e) {
		this._style = e || new Bc({});
	}
	getStyle() {
		return this._style;
	}
	getStroke() {
		return this._style.getStroke() || this._style.setStroke(new zc({
			color: "#000",
			width: 1.25
		})), this._style.getStroke();
	}
	getFill() {
		return this._style.getFill() || this._style.setFill(new Ic({ color: "#fff" })), this._style.getFill();
	}
	getTextStroke() {
		var e = this._style.getText();
		return e ||= new Kc({}), e.getStroke() || e.setStroke(new zc({
			color: "#fff",
			width: 3
		})), e.getStroke();
	}
	getTextFill() {
		var e = this._style.getText();
		return e ||= new Kc({}), e.getFill() || e.setFill(new Ic({ color: "#fff" })), e.getFill();
	}
	getTextFont() {
		var e = this._style.getText();
		return e ||= new Kc({}), e.getFont() || e.setFont("12px sans-serif"), e.getFont();
	}
	_draw() {
		console.warn("[CanvasBase] draw function not implemented.");
	}
}, md = class extends pd {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = "ol-graticule ol-unselectable ol-hidden", super({ element: t }), this.set("projection", e.projection || "EPSG:4326");
		var n = new qt({ code: this.get("projection") }).getMetersPerUnit();
		for (this.fac = 1; n / this.fac > 10;) this.fac *= 10;
		this.fac = 1e4 / this.fac, this.set("maxResolution", e.maxResolution || Infinity), this.set("step", e.step || .1), this.set("stepCoord", e.stepCoord || 1), this.set("spacing", e.spacing || 40), this.set("intervals", e.intervals), this.set("precision", e.precision), this.set("margin", e.margin || 0), this.set("borderWidth", e.borderWidth || 5), this.set("stroke", e.stroke !== !1), this.formatCoord = e.formatCoord || function(e) {
			return e;
		}, e.style instanceof Bc ? this.setStyle(e.style) : this.setStyle(new Bc({
			stroke: new zc({
				color: "#000",
				width: 1
			}),
			fill: new Ic({ color: "#fff" }),
			text: new Kc({
				stroke: new zc({
					color: "#fff",
					width: 2
				}),
				fill: new Ic({ color: "#000" })
			})
		}));
	}
	setStyle(e) {
		this._style = e;
	}
	_draw(e) {
		if (!(this.get("maxResolution") < e.frameState.viewState.resolution)) {
			for (var t = this.getContext(e), n = t.canvas, r = e.frameState.pixelRatio, i = n.width / r, a = n.height / r, o = this.get("projection"), s = this.getMap(), c = [
				s.getCoordinateFromPixel([0, 0]),
				s.getCoordinateFromPixel([i, 0]),
				s.getCoordinateFromPixel([i, a]),
				s.getCoordinateFromPixel([0, a])
			], l = -Infinity, u = Infinity, d = -Infinity, f = Infinity, p = 0, m; m = c[p]; p++) c[p] = fr(m, s.getView().getProjection(), o), l = Math.max(l, c[p][0]), u = Math.min(u, c[p][0]), d = Math.max(d, c[p][1]), f = Math.min(f, c[p][1]);
			var h = this.get("spacing"), g = this.get("step"), _ = this.get("stepCoord"), v = this.get("borderWidth"), y = this.get("margin");
			if ((l - u) / g * h > i) {
				var b = Math.round((l - u) / i * h / g);
				g *= b, g > this.fac && (g = Math.round(g / this.fac) * this.fac);
			}
			var x = this.get("intervals");
			if (Array.isArray(x)) {
				var S = x[0];
				for (let e = 0, t = x.length; e < t && !(g >= x[e]); ++e) S = x[e];
				g = S;
			}
			var C = this.get("precision"), w = g;
			C > 0 && g > C && (w = g / Math.ceil(g / C)), u = Math.floor(u / g) * g - g, f = Math.floor(f / g) * g - g, l = Math.floor(l / g) * g + 2 * g, d = Math.floor(d / g) * g + 2 * g;
			var T = nr(o).getExtent();
			T && (u < T[0] && (u = T[0]), f < T[1] && (f = T[1]), l > T[2] && (l = T[2] + g), d > T[3] && (d = T[3] + g));
			var E = this.getStyle().getStroke() && this.get("stroke"), ee = this.getStyle().getText(), D = this.getStyle().getFill();
			t.save(), t.scale(r, r), t.beginPath(), t.rect(y, y, i - 2 * y, a - 2 * y), t.clip(), t.beginPath();
			var O = {
				top: [],
				left: [],
				bottom: [],
				right: []
			}, k, te, A, ne, j;
			for (k = u; k < l; k += g) for (ne = fr([k, f], o, s.getView().getProjection()), ne = s.getPixelFromCoordinate(ne), E && t.moveTo(ne[0], ne[1]), A = ne, te = f + w; te <= d; te += w) j = fr([k, te], o, s.getView().getProjection()), j = s.getPixelFromCoordinate(j), E && t.lineTo(j[0], j[1]), A[1] > 0 && j[1] < 0 && O.top.push([k, A]), A[1] > a && j[1] < a && O.bottom.push([k, A]), A = j;
			for (te = f; te < d; te += g) for (ne = fr([u, te], o, s.getView().getProjection()), ne = s.getPixelFromCoordinate(ne), E && t.moveTo(ne[0], ne[1]), A = ne, k = u + w; k <= l; k += w) j = fr([k, te], o, s.getView().getProjection()), j = s.getPixelFromCoordinate(j), E && t.lineTo(j[0], j[1]), A[0] < 0 && j[0] > 0 && O.left.push([te, A]), A[0] < i && j[0] > i && O.right.push([te, A]), A = j;
			if (E && (t.strokeStyle = this.getStyle().getStroke().getColor(), t.lineWidth = this.getStyle().getStroke().getWidth(), t.stroke()), ee) {
				t.fillStyle = this.getStyle().getText().getFill().getColor(), t.strokeStyle = this.getStyle().getText().getStroke().getColor(), t.lineWidth = this.getStyle().getText().getStroke().getWidth(), t.font = this.getStyle().getText().getFont(), t.textAlign = "center", t.textBaseline = "hanging";
				var M, N, re = (D ? v : 0) + y + 2;
				for (p = 0; M = O.top[p]; p++) Math.round(M[0] / this.get("step")) % _ || (N = this.formatCoord(M[0], "top"), t.strokeText(N, M[1][0], re), t.fillText(N, M[1][0], re));
				for (t.textBaseline = "alphabetic", p = 0; M = O.bottom[p]; p++) Math.round(M[0] / this.get("step")) % _ || (N = this.formatCoord(M[0], "bottom"), t.strokeText(N, M[1][0], a - re), t.fillText(N, M[1][0], a - re));
				for (t.textBaseline = "middle", t.textAlign = "left", p = 0; M = O.left[p]; p++) Math.round(M[0] / this.get("step")) % _ || (N = this.formatCoord(M[0], "left"), t.strokeText(N, re, M[1][1]), t.fillText(N, re, M[1][1]));
				for (t.textAlign = "right", p = 0; M = O.right[p]; p++) Math.round(M[0] / this.get("step")) % _ || (N = this.formatCoord(M[0], "right"), t.strokeText(N, i - re, M[1][1]), t.fillText(N, i - re, M[1][1]));
			}
			if (D) {
				var ie = this.getStyle().getFill().getColor(), ae, oe;
				for ((oe = this.getStyle().getStroke()) ? ae = this.getStyle().getStroke().getColor() : (ae = ie, ie = "#fff"), t.strokeStyle = ae, t.lineWidth = oe ? oe.getWidth() : 1, p = 1; p < O.top.length; p++) t.beginPath(), t.rect(O.top[p - 1][1][0], y, O.top[p][1][0] - O.top[p - 1][1][0], v), t.fillStyle = Math.round(O.top[p][0] / g) % 2 ? ae : ie, t.fill(), t.stroke();
				for (p = 1; p < O.bottom.length; p++) t.beginPath(), t.rect(O.bottom[p - 1][1][0], a - v - y, O.bottom[p][1][0] - O.bottom[p - 1][1][0], v), t.fillStyle = Math.round(O.bottom[p][0] / g) % 2 ? ae : ie, t.fill(), t.stroke();
				for (p = 1; p < O.left.length; p++) t.beginPath(), t.rect(y, O.left[p - 1][1][1], v, O.left[p][1][1] - O.left[p - 1][1][1]), t.fillStyle = Math.round(O.left[p][0] / g) % 2 ? ae : ie, t.fill(), t.stroke();
				for (p = 1; p < O.right.length; p++) t.beginPath(), t.rect(i - v - y, O.right[p - 1][1][1], v, O.right[p][1][1] - O.right[p - 1][1][1]), t.fillStyle = Math.round(O.right[p][0] / g) % 2 ? ae : ie, t.fill(), t.stroke();
				t.beginPath(), t.fillStyle = ae, t.rect(y, y, v, v), t.rect(y, a - v - y, v, v), t.rect(i - v - y, y, v, v), t.rect(i - v - y, a - v - y, v, v), t.fill();
			}
			t.restore();
		}
	}
}, hd = class {
	drawCustom(e, t, n, r, i) {}
	drawGeometry(e) {}
	setStyle(e) {}
	drawCircle(e, t, n) {}
	drawFeature(e, t, n) {}
	drawGeometryCollection(e, t, n) {}
	drawLineString(e, t, n) {}
	drawMultiLineString(e, t, n) {}
	drawMultiPoint(e, t, n) {}
	drawMultiPolygon(e, t, n) {}
	drawPoint(e, t, n) {}
	drawPolygon(e, t, n) {}
	drawText(e, t, n) {}
	setFillStrokeStyle(e, t) {}
	setImageStyle(e, t) {}
	setTextStyle(e, t) {}
}, Y = {
	BEGIN_GEOMETRY: 0,
	BEGIN_PATH: 1,
	CIRCLE: 2,
	CLOSE_PATH: 3,
	CUSTOM: 4,
	DRAW_CHARS: 5,
	DRAW_IMAGE: 6,
	END_GEOMETRY: 7,
	FILL: 8,
	MOVE_TO_LINE_TO: 9,
	SET_FILL_STYLE: 10,
	SET_STROKE_STYLE: 11,
	STROKE: 12
}, gd = [Y.FILL], _d = [Y.STROKE], vd = [Y.BEGIN_PATH], yd = [Y.CLOSE_PATH], bd = class extends hd {
	constructor(e, t, n, r) {
		super(), this.tolerance = e, this.maxExtent = t, this.pixelRatio = r, this.maxLineWidth = 0, this.resolution = n, this.beginGeometryInstruction1_ = null, this.beginGeometryInstruction2_ = null, this.bufferedMaxExtent_ = null, this.instructions = [], this.coordinates = [], this.tmpCoordinate_ = [], this.hitDetectionInstructions = [], this.state = {};
	}
	applyPixelRatio(e) {
		let t = this.pixelRatio;
		return t == 1 ? e : e.map(function(e) {
			return e * t;
		});
	}
	appendFlatPointCoordinates(e, t) {
		let n = this.getBufferedMaxExtent(), r = this.tmpCoordinate_, i = this.coordinates, a = i.length;
		for (let o = 0, s = e.length; o < s; o += t) r[0] = e[o], r[1] = e[o + 1], Be(n, r) && (i[a++] = r[0], i[a++] = r[1]);
		return a;
	}
	appendFlatLineCoordinates(e, t, n, r, i, a) {
		let o = this.coordinates, s = o.length, c = this.getBufferedMaxExtent();
		a && (t += r);
		let l = e[t], u = e[t + 1], d = this.tmpCoordinate_, f = !0, p, m, h;
		for (p = t + r; p < n; p += r) d[0] = e[p], d[1] = e[p + 1], h = Ue(c, d), h === m ? h === Pe.INTERSECTING ? (o[s++] = d[0], o[s++] = d[1], f = !1) : f = !0 : (f &&= (o[s++] = l, o[s++] = u, !1), o[s++] = d[0], o[s++] = d[1]), l = d[0], u = d[1], m = h;
		return (i && f || p === t + r) && (o[s++] = l, o[s++] = u), s;
	}
	drawCustomCoordinates_(e, t, n, r, i) {
		for (let a = 0, o = n.length; a < o; ++a) {
			let o = n[a], s = this.appendFlatLineCoordinates(e, t, o, r, !1, !1);
			i.push(s), t = o;
		}
		return t;
	}
	drawCustom(e, t, n, r, i) {
		this.beginGeometry(e, t, i);
		let a = e.getType(), o = e.getStride(), s = this.coordinates.length, c, l, u, d, f;
		switch (a) {
			case "MultiPolygon":
				c = e.getOrientedFlatCoordinates(), d = [];
				let t = e.getEndss();
				f = 0;
				for (let e = 0, n = t.length; e < n; ++e) {
					let n = [];
					f = this.drawCustomCoordinates_(c, f, t[e], o, n), d.push(n);
				}
				this.instructions.push([
					Y.CUSTOM,
					s,
					d,
					e,
					n,
					Ai,
					i
				]), this.hitDetectionInstructions.push([
					Y.CUSTOM,
					s,
					d,
					e,
					r || n,
					Ai,
					i
				]);
				break;
			case "Polygon":
			case "MultiLineString":
				u = [], c = a == "Polygon" ? e.getOrientedFlatCoordinates() : e.getFlatCoordinates(), f = this.drawCustomCoordinates_(c, 0, e.getEnds(), o, u), this.instructions.push([
					Y.CUSTOM,
					s,
					u,
					e,
					n,
					ki,
					i
				]), this.hitDetectionInstructions.push([
					Y.CUSTOM,
					s,
					u,
					e,
					r || n,
					ki,
					i
				]);
				break;
			case "LineString":
			case "Circle":
				c = e.getFlatCoordinates(), l = this.appendFlatLineCoordinates(c, 0, c.length, o, !1, !1), this.instructions.push([
					Y.CUSTOM,
					s,
					l,
					e,
					n,
					Oi,
					i
				]), this.hitDetectionInstructions.push([
					Y.CUSTOM,
					s,
					l,
					e,
					r || n,
					Oi,
					i
				]);
				break;
			case "MultiPoint":
				c = e.getFlatCoordinates(), l = this.appendFlatPointCoordinates(c, o), l > s && (this.instructions.push([
					Y.CUSTOM,
					s,
					l,
					e,
					n,
					Oi,
					i
				]), this.hitDetectionInstructions.push([
					Y.CUSTOM,
					s,
					l,
					e,
					r || n,
					Oi,
					i
				]));
				break;
			case "Point": c = e.getFlatCoordinates(), this.coordinates.push(c[0], c[1]), l = this.coordinates.length, this.instructions.push([
				Y.CUSTOM,
				s,
				l,
				e,
				n,
				void 0,
				i
			]), this.hitDetectionInstructions.push([
				Y.CUSTOM,
				s,
				l,
				e,
				r || n,
				void 0,
				i
			]);
		}
		this.endGeometry(t);
	}
	beginGeometry(e, t, n) {
		this.beginGeometryInstruction1_ = [
			Y.BEGIN_GEOMETRY,
			t,
			0,
			e,
			n
		], this.instructions.push(this.beginGeometryInstruction1_), this.beginGeometryInstruction2_ = [
			Y.BEGIN_GEOMETRY,
			t,
			0,
			e,
			n
		], this.hitDetectionInstructions.push(this.beginGeometryInstruction2_);
	}
	finish() {
		return {
			instructions: this.instructions,
			hitDetectionInstructions: this.hitDetectionInstructions,
			coordinates: this.coordinates
		};
	}
	reverseHitDetectionInstructions() {
		let e = this.hitDetectionInstructions;
		e.reverse();
		let t, n = e.length, r, i, a = -1;
		for (t = 0; t < n; ++t) r = e[t], i = r[0], i == Y.END_GEOMETRY ? a = t : i == Y.BEGIN_GEOMETRY && (r[2] = t, le(this.hitDetectionInstructions, a, t), a = -1);
	}
	fillStyleToState(e, t = {}) {
		if (e) {
			let n = e.getColor();
			t.fillPatternScale = n && typeof n == "object" && "src" in n ? this.pixelRatio : 1, t.fillStyle = cc(n || "#000") ?? void 0;
		} else t.fillStyle = void 0;
		return t;
	}
	strokeStyleToState(e, t = {}) {
		if (e) {
			t.strokeStyle = cc(e.getColor() || hc);
			let n = e.getLineCap();
			t.lineCap = n === void 0 ? fc : n;
			let r = e.getLineDash();
			t.lineDash = r ? r.slice() : pc, t.lineDashOffset = e.getLineDashOffset() || 0;
			let i = e.getLineJoin();
			t.lineJoin = i === void 0 ? mc : i;
			let a = e.getWidth();
			t.lineWidth = a === void 0 ? 1 : a;
			let o = e.getMiterLimit();
			t.miterLimit = o === void 0 ? 10 : o, t.strokeOffset = e.getOffset() ?? 0, t.lineWidth > this.maxLineWidth && (this.maxLineWidth = t.lineWidth, this.bufferedMaxExtent_ = null);
		} else t.strokeStyle = void 0, t.lineCap = void 0, t.lineDash = null, t.lineDashOffset = void 0, t.lineJoin = void 0, t.lineWidth = void 0, t.miterLimit = void 0, t.strokeOffset = void 0;
		return t;
	}
	setFillStrokeStyle(e, t) {
		let n = this.state;
		this.fillStyleToState(e, n), this.strokeStyleToState(t, n);
	}
	createFill(e) {
		let t = e.fillStyle, n = [Y.SET_FILL_STYLE, t];
		return typeof t != "string" && n.push(e.fillPatternScale), n;
	}
	applyStroke(e) {
		this.instructions.push(this.createStroke(e));
	}
	createStroke(e) {
		return [
			Y.SET_STROKE_STYLE,
			e.strokeStyle,
			e.lineWidth * this.pixelRatio,
			e.lineCap,
			e.lineJoin,
			e.miterLimit,
			e.lineDash ? this.applyPixelRatio(e.lineDash) : null,
			e.lineDashOffset * this.pixelRatio
		];
	}
	updateFillStyle(e, t) {
		let n = e.fillStyle;
		(n !== void 0 && typeof n != "string" || e.currentFillStyle != n) && (this.instructions.push(t.call(this, e)), e.currentFillStyle = n);
	}
	updateStrokeStyle(e, t) {
		let n = e.strokeStyle, r = e.lineCap, i = e.lineDash, a = e.lineDashOffset, o = e.lineJoin, s = e.lineWidth, c = e.miterLimit, l = e.strokeOffset;
		(e.currentStrokeStyle != n || e.currentLineCap != r || i != e.currentLineDash && !de(e.currentLineDash, i) || e.currentLineDashOffset != a || e.currentLineJoin != o || e.currentLineWidth != s || e.currentMiterLimit != c || e.currentStrokeOffset != l) && (t.call(this, e), e.currentStrokeStyle = n, e.currentLineCap = r, e.currentLineDash = i, e.currentLineDashOffset = a, e.currentLineJoin = o, e.currentLineWidth = s, e.currentMiterLimit = c, e.currentStrokeOffset = l);
	}
	endGeometry(e) {
		this.beginGeometryInstruction1_[2] = this.instructions.length, this.beginGeometryInstruction1_ = null, this.beginGeometryInstruction2_[2] = this.hitDetectionInstructions.length, this.beginGeometryInstruction2_ = null;
		let t = [Y.END_GEOMETRY, e];
		this.instructions.push(t), this.hitDetectionInstructions.push(t);
	}
	getBufferedMaxExtent() {
		if (!this.bufferedMaxExtent_ && (this.bufferedMaxExtent_ = Re(this.maxExtent), this.maxLineWidth > 0)) {
			let e = this.resolution * (this.maxLineWidth + 1) / 2;
			Le(this.bufferedMaxExtent_, e, this.bufferedMaxExtent_);
		}
		return this.bufferedMaxExtent_;
	}
}, xd = class extends bd {
	constructor(e, t, n, r) {
		super(e, t, n, r), this.hitDetectionImage_ = null, this.image_ = null, this.imagePixelRatio_ = void 0, this.anchorX_ = void 0, this.anchorY_ = void 0, this.height_ = void 0, this.opacity_ = void 0, this.originX_ = void 0, this.originY_ = void 0, this.rotateWithView_ = void 0, this.rotation_ = void 0, this.scale_ = void 0, this.width_ = void 0, this.declutterMode_ = void 0, this.declutterImageWithText_ = void 0;
	}
	drawPoint(e, t, n) {
		if (!this.image_ || this.maxExtent && !Be(this.maxExtent, e.getFlatCoordinates())) return;
		this.beginGeometry(e, t, n);
		let r = e.getFlatCoordinates(), i = e.getStride(), a = this.coordinates.length, o = this.appendFlatPointCoordinates(r, i);
		this.instructions.push([
			Y.DRAW_IMAGE,
			a,
			o,
			this.image_,
			this.anchorX_ * this.imagePixelRatio_,
			this.anchorY_ * this.imagePixelRatio_,
			Math.ceil(this.height_ * this.imagePixelRatio_),
			this.opacity_,
			this.originX_ * this.imagePixelRatio_,
			this.originY_ * this.imagePixelRatio_,
			this.rotateWithView_,
			this.rotation_,
			[this.scale_[0] * this.pixelRatio / this.imagePixelRatio_, this.scale_[1] * this.pixelRatio / this.imagePixelRatio_],
			Math.ceil(this.width_ * this.imagePixelRatio_),
			this.declutterMode_,
			this.declutterImageWithText_
		]), this.hitDetectionInstructions.push([
			Y.DRAW_IMAGE,
			a,
			o,
			this.hitDetectionImage_,
			this.anchorX_,
			this.anchorY_,
			this.height_,
			1,
			this.originX_,
			this.originY_,
			this.rotateWithView_,
			this.rotation_,
			this.scale_,
			this.width_,
			this.declutterMode_,
			this.declutterImageWithText_
		]), this.endGeometry(t);
	}
	drawMultiPoint(e, t, n) {
		if (!this.image_) return;
		this.beginGeometry(e, t, n);
		let r = e.getFlatCoordinates(), i = [];
		for (let t = 0, n = r.length; t < n; t += e.getStride()) (!this.maxExtent || Be(this.maxExtent, r.slice(t, t + 2))) && i.push(r[t], r[t + 1]);
		let a = this.coordinates.length, o = this.appendFlatPointCoordinates(i, 2);
		this.instructions.push([
			Y.DRAW_IMAGE,
			a,
			o,
			this.image_,
			this.anchorX_ * this.imagePixelRatio_,
			this.anchorY_ * this.imagePixelRatio_,
			Math.ceil(this.height_ * this.imagePixelRatio_),
			this.opacity_,
			this.originX_ * this.imagePixelRatio_,
			this.originY_ * this.imagePixelRatio_,
			this.rotateWithView_,
			this.rotation_,
			[this.scale_[0] * this.pixelRatio / this.imagePixelRatio_, this.scale_[1] * this.pixelRatio / this.imagePixelRatio_],
			Math.ceil(this.width_ * this.imagePixelRatio_),
			this.declutterMode_,
			this.declutterImageWithText_
		]), this.hitDetectionInstructions.push([
			Y.DRAW_IMAGE,
			a,
			o,
			this.hitDetectionImage_,
			this.anchorX_,
			this.anchorY_,
			this.height_,
			1,
			this.originX_,
			this.originY_,
			this.rotateWithView_,
			this.rotation_,
			this.scale_,
			this.width_,
			this.declutterMode_,
			this.declutterImageWithText_
		]), this.endGeometry(t);
	}
	finish() {
		return this.reverseHitDetectionInstructions(), this.anchorX_ = void 0, this.anchorY_ = void 0, this.hitDetectionImage_ = null, this.image_ = null, this.imagePixelRatio_ = void 0, this.height_ = void 0, this.scale_ = void 0, this.opacity_ = void 0, this.originX_ = void 0, this.originY_ = void 0, this.rotateWithView_ = void 0, this.rotation_ = void 0, this.width_ = void 0, super.finish();
	}
	setImageStyle(e, t) {
		let n = e.getAnchor(), r = e.getSize(), i = e.getOrigin();
		this.imagePixelRatio_ = e.getPixelRatio(this.pixelRatio), this.anchorX_ = n[0], this.anchorY_ = n[1], this.hitDetectionImage_ = e.getHitDetectionImage(), this.image_ = e.getImage(this.pixelRatio), this.height_ = r[1], this.opacity_ = e.getOpacity(), this.originX_ = i[0], this.originY_ = i[1], this.rotateWithView_ = e.getRotateWithView(), this.rotation_ = e.getRotation(), this.scale_ = e.getScaleArray(), this.width_ = r[0], this.declutterMode_ = e.getDeclutterMode(), this.declutterImageWithText_ = t;
	}
}, Sd = class extends bd {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	drawFlatCoordinates_(e, t, n, r, i) {
		let a = this.coordinates.length, o = this.appendFlatLineCoordinates(e, t, n, r, !1, !1);
		return this.instructions.push([
			Y.MOVE_TO_LINE_TO,
			a,
			o,
			i * this.pixelRatio
		]), this.hitDetectionInstructions.push([
			Y.MOVE_TO_LINE_TO,
			a,
			o,
			i
		]), n;
	}
	drawLineString(e, t, n) {
		let r = this.state, i = r.strokeStyle, a = r.lineWidth, o = r.strokeOffset;
		if (i === void 0 || a === void 0) return;
		this.updateStrokeStyle(r, this.applyStroke), this.beginGeometry(e, t, n), this.hitDetectionInstructions.push([
			Y.SET_STROKE_STYLE,
			hc,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			pc,
			0
		], vd);
		let s = e.getFlatCoordinates(), c = e.getStride();
		this.drawFlatCoordinates_(s, 0, s.length, c, o), this.hitDetectionInstructions.push(_d), this.endGeometry(t);
	}
	drawMultiLineString(e, t, n) {
		let r = this.state, i = r.strokeStyle, a = r.lineWidth, o = r.strokeOffset;
		if (i === void 0 || a === void 0) return;
		this.updateStrokeStyle(r, this.applyStroke), this.beginGeometry(e, t, n), this.hitDetectionInstructions.push([
			Y.SET_STROKE_STYLE,
			hc,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			pc,
			0
		], vd);
		let s = e.getEnds(), c = e.getFlatCoordinates(), l = e.getStride(), u = 0;
		for (let e = 0, t = s.length; e < t; ++e) u = this.drawFlatCoordinates_(c, u, s[e], l, o);
		this.hitDetectionInstructions.push(_d), this.endGeometry(t);
	}
	finish() {
		let e = this.state;
		return e.lastStroke != null && e.lastStroke != this.coordinates.length && this.instructions.push(_d), this.reverseHitDetectionInstructions(), this.state = null, super.finish();
	}
	applyStroke(e) {
		e.lastStroke != null && e.lastStroke != this.coordinates.length && (this.instructions.push(_d), e.lastStroke = this.coordinates.length), e.lastStroke = 0, super.applyStroke(e), this.instructions.push(vd);
	}
}, Cd = class extends bd {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	drawFlatCoordinatess_(e, t, n, r, i) {
		let a = this.state, o = a.fillStyle !== void 0, s = a.strokeStyle !== void 0, c = n.length;
		this.instructions.push(vd), this.hitDetectionInstructions.push(vd);
		for (let a = 0; a < c; ++a) {
			let o = n[a], c = this.coordinates.length, l = this.appendFlatLineCoordinates(e, t, o, r, !0, !s);
			this.instructions.push([
				Y.MOVE_TO_LINE_TO,
				c,
				l,
				i * this.pixelRatio,
				!0
			]), this.hitDetectionInstructions.push([
				Y.MOVE_TO_LINE_TO,
				c,
				l,
				i,
				!0
			]), s && (this.instructions.push(yd), this.hitDetectionInstructions.push(yd)), t = o;
		}
		return o && (this.instructions.push(gd), this.hitDetectionInstructions.push(gd)), s && (this.instructions.push(_d), this.hitDetectionInstructions.push(_d)), t;
	}
	drawCircle(e, t, n) {
		let r = this.state, i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawCircle(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([Y.SET_FILL_STYLE, dc]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			Y.SET_STROKE_STYLE,
			hc,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			pc,
			0
		]);
		let s = e.getFlatCoordinates(), c = e.getStride(), l = this.coordinates.length;
		this.appendFlatLineCoordinates(s, 0, s.length, c, !1, !1);
		let u = [
			Y.CIRCLE,
			l,
			o
		];
		this.instructions.push(vd, u), this.hitDetectionInstructions.push(vd, u), r.fillStyle !== void 0 && (this.instructions.push(gd), this.hitDetectionInstructions.push(gd)), r.strokeStyle !== void 0 && (this.instructions.push(_d), this.hitDetectionInstructions.push(_d)), this.endGeometry(t);
	}
	drawPolygon(e, t, n) {
		let r = this.state, i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawPolygon(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([Y.SET_FILL_STYLE, dc]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			Y.SET_STROKE_STYLE,
			hc,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			pc,
			0
		]);
		let s = e.getEnds(), c = e.getOrientedFlatCoordinates(), l = e.getStride();
		this.drawFlatCoordinatess_(c, 0, s, l, o), this.endGeometry(t);
	}
	drawMultiPolygon(e, t, n) {
		let r = this.state, i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawMultiPolygon(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([Y.SET_FILL_STYLE, dc]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			Y.SET_STROKE_STYLE,
			hc,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			pc,
			0
		]);
		let s = e.getEndss(), c = e.getOrientedFlatCoordinates(), l = e.getStride(), u = 0;
		for (let e = 0, t = s.length; e < t; ++e) u = this.drawFlatCoordinatess_(c, u, s[e], l, o);
		this.endGeometry(t);
	}
	finish() {
		this.reverseHitDetectionInstructions(), this.state = null;
		let e = this.tolerance;
		if (e !== 0) {
			let t = this.coordinates;
			for (let n = 0, r = t.length; n < r; ++n) t[n] = Bi(t[n], e);
		}
		return super.finish();
	}
	setFillStrokeStyles_() {
		let e = this.state;
		this.updateFillStyle(e, this.createFill), this.updateStrokeStyle(e, this.applyStroke);
	}
	handleStrokeOffset_(e) {
		let t = this.state, n = t.fillStyle, r = t.strokeStyle, i = t.strokeOffset;
		return Math.abs(i) > 0 && n !== void 0 && r !== void 0 && (t.strokeStyle = void 0, t.strokeOffset = 0, e(), t.fillStyle = void 0, t.strokeStyle = r, t.strokeOffset = i, e(), t.fillStyle = n, !0);
	}
}, wd = 0, Td = 1;
function Ed(e, t, n, r, i, a, o, s) {
	let c = o - i, l = s - a, u = 0, d = 1;
	if (c === 0) {
		if (i < e || i > n) return !1;
	} else {
		let t = (e - i) / c, r = (n - i) / c;
		if (t > r) {
			let e = t;
			t = r, r = e;
		}
		if (t > u && (u = t), r < d && (d = r), u > d) return !1;
	}
	if (l === 0) {
		if (a < t || a > r) return !1;
	} else {
		let e = (t - a) / l, n = (r - a) / l;
		if (e > n) {
			let t = e;
			e = n, n = t;
		}
		if (e > u && (u = e), n < d && (d = n), u > d) return !1;
	}
	return wd = u, Td = d, !0;
}
function Dd(e, t, n, r) {
	let i = r[0], a = r[1], o = r[2], s = r[3], c = [], l = [], u = !1, d, f, p = 0;
	for (let r = 0, m = t.length; r < m; ++r) {
		let m = t[r], h = e[p], g = e[p + 1], _ = !1;
		for (let t = p + n; t < m; t += n) {
			let n = e[t], r = e[t + 1];
			if (Ed(i, a, o, s, h, g, n, r)) {
				let e = n - h, t = r - g, i = h + wd * e, a = g + wd * t, o = h + Td * e, s = g + Td * t;
				u && _ && i === d && a === f ? c.push(o, s) : (u && l.push(c.length), c.push(i, a, o, s), u = !0), d = o, f = s, _ = !0;
			}
			h = n, g = r;
		}
		p = m;
	}
	return u && l.push(c.length), {
		flatCoordinates: c,
		ends: l
	};
}
//#endregion
//#region node_modules/ol/geom/flat/linechunk.js
function Od(e, t, n, r, i) {
	let a = [], o = n, s = 0, c = t.slice(n, 2);
	for (; s < e && o + i < r;) {
		let [n, r] = c.slice(-2), l = t[o + i], u = t[o + i + 1], d = Math.sqrt((l - n) * (l - n) + (u - r) * (u - r));
		if (s += d, s >= e) {
			let t = (e - s + d) / d, f = kt(n, l, t), p = kt(r, u, t);
			c.push(f, p), a.push(c), c = [f, p], s == e && (o += i), s = 0;
		} else if (s < e) c.push(t[o + i], t[o + i + 1]), o += i;
		else {
			let e = d - s, t = kt(n, l, e / d), f = kt(r, u, e / d);
			c.push(t, f), a.push(c), c = [t, f], s = 0, o += i;
		}
	}
	return s > 0 && a.push(c), a;
}
//#endregion
//#region node_modules/ol/geom/flat/straightchunk.js
function kd(e, t, n, r, i) {
	let a = n, o = n, s = 0, c = 0, l = n, u, d, f, p, m, h, g, _, v, y;
	for (d = n; d < r; d += i) {
		let n = t[d], r = t[d + 1];
		m !== void 0 && (v = n - m, y = r - h, p = Math.sqrt(v * v + y * y), g !== void 0 && (c += f, u = Math.acos((g * v + _ * y) / (f * p)), u > e && (c > s && (s = c, a = l, o = d), c = 0, l = d - i)), f = p, g = v, _ = y), m = n, h = r;
	}
	return c += p, c > s ? [l, d] : [a, o];
}
//#endregion
//#region node_modules/ol/render/canvas/TextBuilder.js
var Ad = {
	left: 0,
	center: .5,
	right: 1,
	top: 0,
	middle: .5,
	hanging: .2,
	alphabetic: .8,
	ideographic: .8,
	bottom: 1
}, jd = {
	Circle: Cd,
	Default: bd,
	Image: xd,
	LineString: Sd,
	Polygon: Cd,
	Text: class extends bd {
		constructor(e, t, n, r) {
			super(e, t, n, r), this.labels_ = null, this.text_ = "", this.textOffsetX_ = 0, this.textOffsetY_ = 0, this.textRotateWithView_ = void 0, this.textKeepUpright_ = void 0, this.textRotation_ = 0, this.textFillState_ = null, this.fillStates = {}, this.fillStates[dc] = { fillStyle: dc }, this.textStrokeState_ = null, this.strokeStates = {}, this.textState_ = {}, this.textStates = {}, this.textKey_ = "", this.fillKey_ = "", this.strokeKey_ = "", this.declutterMode_ = void 0, this.declutterImageWithText_ = void 0;
		}
		finish() {
			let e = super.finish();
			return e.textStates = this.textStates, e.fillStates = this.fillStates, e.strokeStates = this.strokeStates, e;
		}
		drawText(e, t, n) {
			let r = this.textFillState_, i = this.textStrokeState_, a = this.textState_;
			if (this.text_ === "" || !a || !r && !i) return;
			let o = this.coordinates, s = o.length, c = e.getType(), l = null, u = e.getStride();
			if (a.placement === "line" && (c == "LineString" || c == "MultiLineString" || c == "Polygon" || c == "MultiPolygon")) {
				let r = e.getExtent();
				if (!pt(this.maxExtent, r)) return;
				let i;
				if (l = e.getFlatCoordinates(), c == "LineString") i = [l.length];
				else if (c == "MultiLineString") i = e.getEnds();
				else if (c == "Polygon") i = e.getEnds().slice(0, 1);
				else if (c == "MultiPolygon") {
					let t = e.getEndss();
					i = [];
					for (let e = 0, n = t.length; e < n; ++e) i.push(t[e][0]);
				}
				if ((c == "LineString" || c == "MultiLineString") && !Ve(this.getBufferedMaxExtent(), r)) {
					let e = Dd(l, i, u, this.getBufferedMaxExtent());
					if (l = e.flatCoordinates, i = e.ends, u = 2, i.length === 0) return;
				}
				this.beginGeometry(e, t, n);
				let d = a.repeat, f = d ? void 0 : a.textAlign, p = 0;
				for (let e = 0, t = i.length; e < t; ++e) {
					let t;
					t = d ? Od(d * this.resolution, l, p, i[e], u) : [l.slice(p, i[e])];
					for (let n = 0, r = t.length; n < r; ++n) {
						let r = t[n], c = 0, l = r.length;
						if (f == null) {
							let e = kd(a.maxAngle, r, 0, r.length, 2);
							c = e[0], l = e[1];
						}
						for (let e = c; e < l; e += u) o.push(r[e], r[e + 1]);
						let d = o.length;
						p = i[e], this.drawChars_(s, d), s = d;
					}
				}
				this.endGeometry(t);
			} else {
				let r = a.overflow ? null : [];
				switch (c) {
					case "Point":
					case "MultiPoint":
						l = e.getFlatCoordinates();
						break;
					case "LineString":
						l = e.getFlatMidpoint();
						break;
					case "Circle":
						l = e.getCenter();
						break;
					case "MultiLineString":
						l = e.getFlatMidpoints(), u = 2;
						break;
					case "Polygon":
						l = e.getFlatInteriorPoint(), a.overflow || r.push(l[2] / this.resolution), u = 3;
						break;
					case "MultiPolygon":
						let t = e.getFlatInteriorPoints();
						l = [];
						for (let e = 0, n = t.length; e < n; e += 3) a.overflow || r.push(t[e + 2] / this.resolution), l.push(t[e], t[e + 1]);
						if (l.length === 0) return;
						u = 2;
				}
				let i = this.appendFlatPointCoordinates(l, u);
				if (i === s) return;
				if (r && (i - s) / 2 !== l.length / u) {
					let e = s / 2;
					r = r.filter((t, n) => {
						let r = o[(e + n) * 2] === l[n * u] && o[(e + n) * 2 + 1] === l[n * u + 1];
						return r || --e, r;
					});
				}
				this.saveTextStates_();
				let d = a.backgroundFill ? this.createFill(this.fillStyleToState(a.backgroundFill)) : null, f = a.backgroundStroke ? this.createStroke(this.strokeStyleToState(a.backgroundStroke)) : null;
				this.beginGeometry(e, t, n);
				let p = a.padding;
				if (p != vc && (a.scale[0] < 0 || a.scale[1] < 0)) {
					let e = a.padding[0], t = a.padding[1], n = a.padding[2], r = a.padding[3];
					a.scale[0] < 0 && (t = -t, r = -r), a.scale[1] < 0 && (e = -e, n = -n), p = [
						e,
						t,
						n,
						r
					];
				}
				let m = this.pixelRatio;
				this.instructions.push([
					Y.DRAW_IMAGE,
					s,
					i,
					null,
					NaN,
					NaN,
					NaN,
					1,
					0,
					0,
					this.textRotateWithView_,
					this.textRotation_,
					[1, 1],
					NaN,
					this.declutterMode_,
					this.declutterImageWithText_,
					p == vc ? vc : p.map(function(e) {
						return e * m;
					}),
					d,
					f,
					this.text_,
					this.textKey_,
					this.strokeKey_,
					this.fillKey_,
					this.textOffsetX_,
					this.textOffsetY_,
					r
				]);
				let h = 1 / m, g = d ? d.slice(0) : null;
				g && (g[1] = dc), this.hitDetectionInstructions.push([
					Y.DRAW_IMAGE,
					s,
					i,
					null,
					NaN,
					NaN,
					NaN,
					1,
					0,
					0,
					this.textRotateWithView_,
					this.textRotation_,
					[h, h],
					NaN,
					this.declutterMode_,
					this.declutterImageWithText_,
					p,
					g,
					f,
					this.text_,
					this.textKey_,
					this.strokeKey_,
					this.fillKey_ ? dc : this.fillKey_,
					this.textOffsetX_,
					this.textOffsetY_,
					r
				]), this.endGeometry(t);
			}
		}
		saveTextStates_() {
			let e = this.textStrokeState_, t = this.textState_, n = this.textFillState_, r = this.strokeKey_;
			e && (r in this.strokeStates || (this.strokeStates[r] = {
				strokeStyle: e.strokeStyle,
				lineCap: e.lineCap,
				lineDashOffset: e.lineDashOffset,
				lineWidth: e.lineWidth,
				lineJoin: e.lineJoin,
				miterLimit: e.miterLimit,
				lineDash: e.lineDash
			}));
			let i = this.textKey_;
			i in this.textStates || (this.textStates[i] = {
				font: t.font,
				textAlign: t.textAlign || "center",
				justify: t.justify,
				textBaseline: t.textBaseline || "middle",
				scale: t.scale
			});
			let a = this.fillKey_;
			n && (a in this.fillStates || (this.fillStates[a] = { fillStyle: n.fillStyle }));
		}
		drawChars_(e, t) {
			let n = this.textStrokeState_, r = this.textState_, i = this.strokeKey_, a = this.textKey_, o = this.fillKey_;
			this.saveTextStates_();
			let s = this.pixelRatio, c = Ad[r.textBaseline], l = this.textOffsetX_ * s, u = this.textOffsetY_ * s, d = this.text_, f = n ? n.lineWidth * Math.abs(r.scale[0]) / 2 : 0;
			this.instructions.push([
				Y.DRAW_CHARS,
				e,
				t,
				c,
				r.overflow,
				o,
				r.maxAngle,
				s,
				u,
				i,
				f * s,
				d,
				a,
				1,
				this.declutterMode_,
				this.textKeepUpright_,
				l
			]), this.hitDetectionInstructions.push([
				Y.DRAW_CHARS,
				e,
				t,
				c,
				r.overflow,
				o && dc,
				r.maxAngle,
				s,
				u,
				i,
				f * s,
				d,
				a,
				1 / s,
				this.declutterMode_,
				this.textKeepUpright_,
				l
			]);
		}
		setTextStyle(e, t) {
			let n, r, i;
			if (!e) this.text_ = "";
			else {
				let t = e.getFill();
				t ? (r = this.textFillState_, r || (r = {}, this.textFillState_ = r), r.fillStyle = cc(t.getColor() || "#000")) : (r = null, this.textFillState_ = r);
				let a = e.getStroke();
				if (!a) i = null, this.textStrokeState_ = i;
				else {
					i = this.textStrokeState_, i || (i = {}, this.textStrokeState_ = i);
					let e = a.getLineDash(), t = a.getLineDashOffset(), n = a.getWidth(), r = a.getMiterLimit();
					i.lineCap = a.getLineCap() || "round", i.lineDash = e ? e.slice() : pc, i.lineDashOffset = t === void 0 ? 0 : t, i.lineJoin = a.getLineJoin() || "round", i.lineWidth = n === void 0 ? 1 : n, i.miterLimit = r === void 0 ? 10 : r, i.strokeStyle = cc(a.getColor() || "#000");
				}
				n = this.textState_;
				let o = e.getFont() || "10px sans-serif";
				Tc(o);
				let s = e.getScaleArray();
				n.overflow = e.getOverflow(), n.font = o, n.maxAngle = e.getMaxAngle(), n.placement = e.getPlacement(), n.textAlign = e.getTextAlign(), n.repeat = e.getRepeat(), n.justify = e.getJustify(), n.textBaseline = e.getTextBaseline() || "middle", n.backgroundFill = e.getBackgroundFill(), n.backgroundStroke = e.getBackgroundStroke(), n.padding = e.getPadding() || vc, n.scale = s === void 0 ? [1, 1] : s;
				let c = e.getOffsetX(), l = e.getOffsetY(), u = e.getRotateWithView(), d = e.getKeepUpright(), f = e.getRotation();
				this.text_ = e.getText() || "", this.textOffsetX_ = c === void 0 ? 0 : c, this.textOffsetY_ = l === void 0 ? 0 : l, this.textRotateWithView_ = u !== void 0 && u, this.textKeepUpright_ = d === void 0 || d, this.textRotation_ = f === void 0 ? 0 : f, this.strokeKey_ = i ? (typeof i.strokeStyle == "string" ? i.strokeStyle : F(i.strokeStyle)) + i.lineCap + i.lineDashOffset + "|" + i.lineWidth + i.lineJoin + i.miterLimit + "[" + i.lineDash.join() + "]" : "", this.textKey_ = n.font + n.scale + (n.textAlign || "?") + (n.repeat || "?") + (n.justify || "?") + (n.textBaseline || "?"), this.fillKey_ = r && r.fillStyle ? typeof r.fillStyle == "string" ? r.fillStyle : "|" + F(r.fillStyle) : "";
			}
			this.declutterMode_ = e.getDeclutterMode(), this.declutterImageWithText_ = t;
		}
	}
}, Md = class {
	constructor(e, t, n, r) {
		this.tolerance_ = e, this.maxExtent_ = t, this.pixelRatio_ = r, this.resolution_ = n, this.buildersByZIndex_ = {};
	}
	finish() {
		let e = {};
		for (let t in this.buildersByZIndex_) {
			e[t] = e[t] || {};
			let n = this.buildersByZIndex_[t];
			for (let r in n) {
				let i = n[r].finish();
				e[t][r] = i;
			}
		}
		return e;
	}
	getBuilder(e, t) {
		let n = e === void 0 ? "0" : e.toString(), r = this.buildersByZIndex_[n];
		r === void 0 && (r = {}, this.buildersByZIndex_[n] = r);
		let i = r[t];
		if (i === void 0) {
			let e = jd[t];
			i = new e(this.tolerance_, this.maxExtent_, this.resolution_, this.pixelRatio_), r[t] = i;
		}
		return i;
	}
};
//#endregion
//#region node_modules/ol/geom/flat/length.js
function Nd(e, t, n, r) {
	let i = e[t], a = e[t + 1], o = 0;
	for (let s = t + r; s < n; s += r) {
		let t = e[s], n = e[s + 1];
		o += Math.sqrt((t - i) * (t - i) + (n - a) * (n - a)), i = t, a = n;
	}
	return o;
}
//#endregion
//#region node_modules/ol/geom/flat/lineoffset.js
function Pd(e, t, n, r, i, a, o, s) {
	o ??= [], s ??= r;
	let c = e[t + r], l = e[t + r + 1], u = e[n - 2 * r], d = e[n - 2 * r + 1], f, p, m, h, g, _, v, y, b = 0;
	for (let x = t; x < n; x += r) {
		m = f, h = p, g = void 0, _ = void 0, x + r < n && (g = e[x + r], _ = e[x + r + 1]), a && x === t && (m = u, h = d), a && x === n - r && (g = c, _ = l), f = e[x], p = e[x + 1], [v, y] = Fd(f, p, m, h, g, _, i), o[b++] = v, o[b++] = y;
		for (let t = 2; t < s; t++) o[b++] = e[x + t];
	}
	return o.length != b && (o.length = b), o;
}
function Fd(e, t, n, r, i, a, o) {
	let s, c;
	n !== void 0 && r !== void 0 ? (s = e - n, c = t - r) : i !== void 0 && a !== void 0 ? (s = i - e, c = a - t) : (s = 1, c = 0);
	let l = Math.hypot(s, c), u = s / l, d = c / l;
	if (s = -d, c = u, n === void 0 || r === void 0 || i === void 0 || a === void 0) return [e + s * o, t + c * o];
	let f = Vt([e, t], [n, r], [i, a]);
	if (Math.cos(f) > .998) return [e + u * o, t + d * o];
	let p = Math.cos(f / 2), m = Math.sin(f / 2), h = m * s + p * c, g = -p * s + m * c, _ = 1 / m * h, v = 1 / m * g;
	return [e + _ * o, t + v * o];
}
function Id(e, t, n = !1) {
	for (let r = 0, i = e.length - 2; r < i; r += t) {
		let i = n && r === 0 ? e.length - 3 * t : e.length - 2 * t;
		for (let n = i; n > r + t; n -= t) {
			let i = e[r], a = e[r + 1], o = e[r + t], s = e[r + t + 1], c = e[n], l = e[n + 1], u = e[n + t], d = e[n + t + 1], f = (d - l) * (o - i) - (u - c) * (s - a);
			if (f === 0) continue;
			let p = ((u - c) * (a - l) - (d - l) * (i - c)) / f, m = ((o - i) * (a - l) - (s - a) * (i - c)) / f;
			if (p > 0 && p < 1 && m > 0 && m < 1) {
				let c = i + p * (o - i), l = a + p * (s - a);
				e[r + t] = c, e[r + t + 1] = l, e.splice(r + 2 * t, n - r - t);
				break;
			}
		}
	}
	return e;
}
//#endregion
//#region node_modules/ol/geom/flat/textpath.js
var Ld;
function Rd() {
	return Ld ||= new Intl.Segmenter(void 0, { granularity: "grapheme" }), Ld;
}
function zd(e, t, n, r, i, a, o, s, c, l, u, d, f = !0) {
	let p = e[t], m = e[t + 1], h = 0, g = 0, _ = 0, v = 0;
	function y() {
		h = p, g = m, t += r, p = e[t], m = e[t + 1], v += _, _ = Math.sqrt((p - h) * (p - h) + (m - g) * (m - g));
	}
	do
		y();
	while (t < n - r && v + _ < a);
	let b = _ === 0 ? 0 : (a - v) / _, x = kt(h, p, b), S = kt(g, m, b), C = t - r, w = v, T = a + s * c(l, i, u);
	for (; t < n - r && v + _ < T;) y();
	b = _ === 0 ? 0 : (T - v) / _;
	let E = kt(h, p, b), ee = kt(g, m, b), D = !1;
	if (f) {
		if (d) {
			let e = [
				x,
				S,
				E,
				ee
			];
			ci(e, 0, 4, 2, d, e, e), D = e[0] > e[2];
		} else D = x > E;
	}
	let O = Math.PI, k = [], te = C + r === t;
	t = C, _ = 0, v = w, p = e[t], m = e[t + 1];
	let A;
	if (te) return y(), A = Math.atan2(m - g, p - h), D && (A += A > 0 ? -O : O), k[0] = [
		(E + x) / 2,
		(ee + S) / 2,
		(T - a) / 2,
		A,
		i
	], k;
	i = i.replace(/\n/g, " ");
	let ne = Array.from(Rd().segment(i), (e) => e.segment);
	for (let e = 0, i = ne.length; e < i;) {
		y();
		let d = Math.atan2(m - g, p - h);
		if (D && (d += d > 0 ? -O : O), A !== void 0) {
			let e = d - A;
			if (e += e > O ? -2 * O : e < -O ? 2 * O : 0, Math.abs(e) > o) return null;
		}
		A = d;
		let f = e, x = 0;
		for (; e < i; ++e) {
			let o = s * c(l, ne[D ? i - e - 1 : e], u);
			if (t + r < n && v + _ < a + x + o / 2) break;
			x += o;
		}
		if (e === f) continue;
		let S = (D ? ne.slice(i - e, i - f) : ne.slice(f, e)).join("");
		b = _ === 0 ? 0 : (a + x / 2 - v) / _;
		let C = kt(h, p, b), w = kt(g, m, b);
		k.push([
			C,
			w,
			x / 2,
			d,
			S
		]), a += x;
	}
	return k;
}
//#endregion
//#region node_modules/ol/render/canvas/Executor.js
var Bd = We(), Vd = [], Hd = [], Ud = [], Wd = [];
function Gd(e) {
	return e[3].declutterBox;
}
var Kd = /* @__PURE__ */ RegExp("[֑-ࣿיִ-﷿ﹰ-ﻼࠀ-࿿-]");
function qd(e, t) {
	return t === "start" ? t = Kd.test(e) ? "right" : "left" : t === "end" && (t = Kd.test(e) ? "left" : "right"), Ad[t];
}
function Jd(e, t, n) {
	return n > 0 && e.push("\n", ""), e.push(t, ""), e;
}
function Yd(e, t, n) {
	return n % 2 == 0 && (e += t), e;
}
var Xd = class {
	constructor(e, t, n, r, i) {
		this.overlaps = n, this.pixelRatio = t, this.resolution = e, this.alignAndScaleFill_, this.instructions = r.instructions, this.coordinates = r.coordinates, this.coordinateCache_ = {}, this.renderedTransform_ = qr(), this.hitDetectionInstructions = r.hitDetectionInstructions, this.pixelCoordinates_ = null, this.viewRotation_ = 0, this.fillStates = r.fillStates || {}, this.strokeStates = r.strokeStates || {}, this.textStates = r.textStates || {}, this.widths_ = {}, this.labels_ = {}, this.zIndexContext_ = i ? new wu() : null;
	}
	getZIndexContext() {
		return this.zIndexContext_;
	}
	createLabel(e, t, n, r) {
		let i = e + t + n + r;
		if (this.labels_[i]) return this.labels_[i];
		let a = r ? this.strokeStates[r] : null, o = n ? this.fillStates[n] : null, s = this.textStates[t], c = this.pixelRatio, l = [s.scale[0] * c, s.scale[1] * c], u = s.justify ? Ad[s.justify] : qd(Array.isArray(e) ? e[0] : e, s.textAlign || "center"), d = r && a.lineWidth ? a.lineWidth : 0, f = Array.isArray(e) ? e : String(e).split("\n").reduce(Jd, []), { width: p, height: m, widths: h, heights: g, lineWidths: _ } = Ac(s, f), v = p + d, y = [], b = (v + 2) * l[0], x = (m + d) * l[1], S = {
			width: b < 0 ? Math.floor(b) : Math.ceil(b),
			height: x < 0 ? Math.floor(x) : Math.ceil(x),
			contextInstructions: y
		};
		(l[0] != 1 || l[1] != 1) && y.push("scale", l), r && (y.push("strokeStyle", a.strokeStyle), y.push("lineWidth", d), y.push("lineCap", a.lineCap), y.push("lineJoin", a.lineJoin), y.push("miterLimit", a.miterLimit), y.push("setLineDash", [a.lineDash]), y.push("lineDashOffset", a.lineDashOffset)), n && y.push("fillStyle", o.fillStyle), y.push("textBaseline", "middle"), y.push("textAlign", "center");
		let C = .5 - u, w = u * v + C * d, T = [], E = [], ee = 0, D = 0, O = 0, k = 0, te;
		for (let e = 0, t = f.length; e < t; e += 2) {
			let t = f[e];
			if (t === "\n") {
				D += ee, ee = 0, w = u * v + C * d, ++k;
				continue;
			}
			let i = f[e + 1] || s.font;
			i !== te && (r && T.push("font", i), n && E.push("font", i), te = i), ee = Math.max(ee, g[O]);
			let a = [
				t,
				w + C * h[O] + u * (h[O] - _[k]),
				.5 * (d + ee) + D
			];
			w += h[O], r && T.push("strokeText", a), n && E.push("fillText", a), ++O;
		}
		return Array.prototype.push.apply(y, T), Array.prototype.push.apply(y, E), this.labels_[i] = S, S;
	}
	replayTextBackground_(e, t, n, r, i, a, o) {
		e.beginPath(), e.moveTo.apply(e, t), e.lineTo.apply(e, n), e.lineTo.apply(e, r), e.lineTo.apply(e, i), e.lineTo.apply(e, t), a && (this.alignAndScaleFill_ = a[2], e.fillStyle = a[1], this.fill_(e)), o && (this.setStrokeStyle_(e, o), e.stroke());
	}
	calculateImageOrLabelDimensions_(e, t, n, r, i, a, o, s, c, l, u, d, f, p, m, h) {
		o *= d[0], s *= d[1];
		let g = n - o, _ = r - s, v = i + c > e ? e - c : i, y = a + l > t ? t - l : a, b = p[3] + v * d[0] + p[1], x = p[0] + y * d[1] + p[2], S = g - p[3], C = _ - p[0];
		(m || u !== 0) && (Vd[0] = S, Wd[0] = S, Vd[1] = C, Hd[1] = C, Hd[0] = S + b, Ud[0] = Hd[0], Ud[1] = C + x, Wd[1] = Ud[1]);
		let w;
		return u === 0 ? Ge(Math.min(S, S + b), Math.min(C, C + x), Math.max(S, S + b), Math.max(C, C + x), Bd) : (w = ei(qr(), n, r, 1, 1, u, -n, -r), z(w, Vd), z(w, Hd), z(w, Ud), z(w, Wd), Ge(Math.min(Vd[0], Hd[0], Ud[0], Wd[0]), Math.min(Vd[1], Hd[1], Ud[1], Wd[1]), Math.max(Vd[0], Hd[0], Ud[0], Wd[0]), Math.max(Vd[1], Hd[1], Ud[1], Wd[1]), Bd)), f && (g = Math.round(g), _ = Math.round(_)), {
			drawImageX: g,
			drawImageY: _,
			drawImageW: v,
			drawImageH: y,
			originX: c,
			originY: l,
			declutterBox: {
				minX: Bd[0],
				minY: Bd[1],
				maxX: Bd[2],
				maxY: Bd[3],
				value: h
			},
			canvasTransform: w,
			scale: d
		};
	}
	replayImageOrLabel_(e, t, n, r, i, a, o) {
		let s = !!(a || o), c = r.declutterBox, l = o ? o[2] * r.scale[0] / 2 : 0;
		return c.minX - l <= t[0] && c.maxX + l >= 0 && c.minY - l <= t[1] && c.maxY + l >= 0 && (s && this.replayTextBackground_(e, Vd, Hd, Ud, Wd, a, o), jc(e, r.canvasTransform, i, n, r.originX, r.originY, r.drawImageW, r.drawImageH, r.drawImageX, r.drawImageY, r.scale)), !0;
	}
	fill_(e) {
		let t = this.alignAndScaleFill_;
		if (t) {
			let n = z(this.renderedTransform_, [0, 0]), r = 512 * this.pixelRatio;
			e.save(), e.translate(n[0] % r, n[1] % r), t !== 1 && e.scale(t, t);
		}
		e.fill(), t && e.restore();
	}
	setStrokeStyle_(e, t) {
		e.strokeStyle = t[1], t[1] && (e.lineWidth = t[2], e.lineCap = t[3], e.lineJoin = t[4], e.miterLimit = t[5], e.lineDashOffset = t[7], e.setLineDash(t[6]));
	}
	drawLabelWithPointPlacement_(e, t, n, r) {
		let i = this.textStates[t], a = this.createLabel(e, t, r, n), o = this.strokeStates[n], s = this.pixelRatio, c = qd(Array.isArray(e) ? e[0] : e, i.textAlign || "center"), l = Ad[i.textBaseline || "middle"], u = o && o.lineWidth ? o.lineWidth : 0;
		return {
			label: a,
			anchorX: c * (a.width / s - 2 * i.scale[0]) + 2 * (.5 - c) * u,
			anchorY: l * a.height / s + 2 * (.5 - l) * u
		};
	}
	execute_(e, t, n, r, i, a, o, s) {
		let c = this.zIndexContext_, l;
		this.pixelCoordinates_ && de(n, this.renderedTransform_) ? l = this.pixelCoordinates_ : (this.pixelCoordinates_ ||= [], l = si(this.coordinates, 0, this.coordinates.length, 2, n, this.pixelCoordinates_), Zr(this.renderedTransform_, n));
		let u = 0, d = r.length, f = 0, p, m = [], h, g, _, v, y, b, x, S, C, w, T, E, ee, D = 0, O = 0, k = this.coordinateCache_, te = this.viewRotation_, A = Math.round(Math.atan2(-n[1], n[0]) * 0xe8d4a51000) / 0xe8d4a51000, ne = {
			context: e,
			pixelRatio: this.pixelRatio,
			resolution: this.resolution,
			rotation: te
		}, j = this.instructions != r || this.overlaps ? 0 : 200, M, N, re, ie;
		for (; u < d;) {
			let n = r[u];
			switch (n[0]) {
				case Y.BEGIN_GEOMETRY:
					M = n[1], ie = n[3], M.getGeometry() ? o !== void 0 && !pt(o, ie.getExtent()) ? u = n[2] + 1 : ++u : u = n[2], c && (c.zIndex = n[4]);
					break;
				case Y.BEGIN_PATH:
					D > j && (this.fill_(e), D = 0), O > j && (e.stroke(), O = 0), !D && !O && (e.beginPath(), y = NaN, b = NaN), ++u;
					break;
				case Y.CIRCLE:
					f = n[1], _ = n[2] ?? 0;
					let r = l[f], d = l[f + 1], ae = l[f + 2] - _, oe = l[f + 3] - _, se = ae - r, ce = oe - d, le = Math.sqrt(se * se + ce * ce);
					e.moveTo(r + le, d), e.arc(r, d, le, 0, 2 * Math.PI, !0), ++u;
					break;
				case Y.CLOSE_PATH:
					e.closePath(), ++u;
					break;
				case Y.CUSTOM:
					f = n[1], p = n[2];
					let ue = n[3], de = n[4], fe = n[5];
					ne.geometry = ue, ne.feature = M, u in k || (k[u] = []);
					let pe = k[u];
					fe ? fe(l, f, p, 2, pe) : (pe[0] = l[f], pe[1] = l[f + 1], pe.length = 2), c && (c.zIndex = n[6]), de(pe, ne), ++u;
					break;
				case Y.DRAW_IMAGE:
					f = n[1], p = n[2], C = n[3], h = n[4], g = n[5];
					let me = n[6], he = n[7], ge = n[8], _e = n[9], ve = n[10], ye = n[11], be = n[12], xe = n[13];
					v = n[14] || "declutter";
					let P = n[15];
					if (!C && n.length >= 20) {
						w = n[19], T = n[20], E = n[21], ee = n[22];
						let e = this.drawLabelWithPointPlacement_(w, T, E, ee);
						C = e.label, n[3] = C;
						let t = n[23];
						h = (e.anchorX - t) * this.pixelRatio, n[4] = h;
						let r = n[24];
						g = (e.anchorY - r) * this.pixelRatio, n[5] = g, me = C.height, n[6] = me, xe = C.width, n[13] = xe;
					}
					let Se;
					n.length > 25 && (Se = n[25]);
					let F, Ce, we;
					n.length > 17 ? (F = n[16], Ce = n[17], we = n[18]) : (F = vc, Ce = null, we = null), ve && A ? ye += te : !ve && !A && (ye -= te);
					let Te = 0;
					for (; f < p; f += 2) {
						if (Se && Se[Te++] < xe / this.pixelRatio) continue;
						let n = this.calculateImageOrLabelDimensions_(C.width, C.height, l[f], l[f + 1], xe, me, h, g, ge, _e, ye, be, i, F, !!Ce || !!we, M), r = [
							e,
							t,
							C,
							n,
							he,
							Ce,
							we
						];
						if (s) {
							let e, t, i;
							if (P) {
								let n = p - f;
								if (!P[n]) {
									P[n] = {
										args: r,
										declutterMode: v
									};
									continue;
								}
								let a = P[n];
								e = a.args, t = a.declutterMode, delete P[n], i = Gd(e);
							}
							let a, o;
							if (e && (t !== "declutter" || !s.collides(i)) && (a = !0), (v !== "declutter" || !s.collides(n.declutterBox)) && (o = !0), t === "declutter" && v === "declutter") {
								let e = a && o;
								a = e, o = e;
							}
							a && (t !== "none" && s.insert(i), this.replayImageOrLabel_.apply(this, e)), o && (v !== "none" && s.insert(n.declutterBox), this.replayImageOrLabel_.apply(this, r));
						} else this.replayImageOrLabel_.apply(this, r);
					}
					++u;
					break;
				case Y.DRAW_CHARS:
					let Ee = n[1], De = n[2], Oe = n[3], ke = n[4];
					ee = n[5];
					let Ae = n[6], je = n[7], Me = n[8];
					E = n[9];
					let Ne = n[10];
					w = n[11], Array.isArray(w) && (w = w.reduce(Yd, "")), T = n[12];
					let Pe = [n[13], n[13]];
					v = n[14] || "declutter";
					let Fe = n[15], Ie = n[16], Le = this.textStates[T], Re = Le.font, ze = [Le.scale[0] * je, Le.scale[1] * je], Be;
					Re in this.widths_ ? Be = this.widths_[Re] : (Be = {}, this.widths_[Re] = Be);
					let Ve = Nd(l, Ee, De, 2), He = Math.abs(ze[0]) * kc(Re, w, Be);
					if (ke || He <= Ve) {
						let n = this.textStates[T].textAlign, r = (Ve - He) * qd(w, n), i = zd(l, Ee, De, 2, w, r, Ae, Math.abs(ze[0]), kc, Re, Be, A ? 0 : this.viewRotation_, Fe);
						drawChars: if (i) {
							let n = [], r, a, o, c, l;
							if (E) for (r = 0, a = i.length; r < a; ++r) {
								l = i[r], o = l[4], c = this.createLabel(o, T, "", E), h = l[2] + (ze[0] < 0 ? -Ne : Ne) - Ie, g = Oe * c.height + (.5 - Oe) * 2 * Ne * ze[1] / ze[0] - Me;
								let a = this.calculateImageOrLabelDimensions_(c.width, c.height, l[0], l[1], c.width, c.height, h, g, 0, 0, l[3], Pe, !1, vc, !1, M);
								if (s && v === "declutter" && s.collides(a.declutterBox)) break drawChars;
								n.push([
									e,
									t,
									c,
									a,
									1,
									null,
									null
								]);
							}
							if (ee) for (r = 0, a = i.length; r < a; ++r) {
								l = i[r], o = l[4], c = this.createLabel(o, T, ee, ""), h = l[2] - Ie, g = Oe * c.height - Me;
								let a = this.calculateImageOrLabelDimensions_(c.width, c.height, l[0], l[1], c.width, c.height, h, g, 0, 0, l[3], Pe, !1, vc, !1, M);
								if (s && v === "declutter" && s.collides(a.declutterBox)) break drawChars;
								n.push([
									e,
									t,
									c,
									a,
									1,
									null,
									null
								]);
							}
							s && v !== "none" && s.load(n.map(Gd));
							for (let e = 0, t = n.length; e < t; ++e) this.replayImageOrLabel_.apply(this, n[e]);
						}
					}
					++u;
					break;
				case Y.END_GEOMETRY:
					if (a !== void 0) {
						M = n[1];
						let e = a(M, ie, v);
						if (e) return e;
					}
					++u;
					break;
				case Y.FILL:
					j ? D++ : this.fill_(e), ++u;
					break;
				case Y.MOVE_TO_LINE_TO:
					f = n[1], p = n[2], _ = n[3];
					let Ue, We, Ge;
					if (_) {
						let e = (n[4] ?? !1) || Math.abs(l[f] - l[p - 2]) < 1e-6 && Math.abs(l[f + 1] - l[p - 1]) < 1e-6;
						Pd(l, f, p, 2, _, e, m), Id(m, 2, e), Ue = m, We = 0, Ge = Ue.length;
					} else Ue = l, We = f, Ge = p;
					N = Ue[We], re = Ue[We + 1], e.moveTo(N, re), y = N + .5 | 0, b = re + .5 | 0;
					for (let t = We + 2; t < Ge; t += 2) N = Ue[t], re = Ue[t + 1], x = N + .5 | 0, S = re + .5 | 0, (t == Ge - 2 || x !== y || S !== b) && (e.lineTo(N, re), y = x, b = S);
					++u;
					break;
				case Y.SET_FILL_STYLE:
					this.alignAndScaleFill_ = n[2], D ? (this.fill_(e), D = 0, O &&= (e.stroke(), 0)) : O && n[1] && (e.stroke(), O = 0), e.fillStyle = n[1], ++u;
					break;
				case Y.SET_STROKE_STYLE:
					D && n[1] && (this.fill_(e), D = 0), O &&= (e.stroke(), 0), this.setStrokeStyle_(e, n), ++u;
					break;
				case Y.STROKE:
					j ? O++ : e.stroke(), ++u;
					break;
				default: ++u;
			}
		}
		D && this.fill_(e), O && e.stroke();
	}
	execute(e, t, n, r, i, a) {
		this.viewRotation_ = r, this.execute_(e, t, n, this.instructions, i, void 0, void 0, a);
	}
	executeHitDetection(e, t, n, r, i) {
		return this.viewRotation_ = n, this.execute_(e, [e.canvas.width, e.canvas.height], t, this.hitDetectionInstructions, !0, r, i);
	}
}, Zd = [
	"Polygon",
	"Circle",
	"LineString",
	"Image",
	"Text",
	"Default"
], Qd = ["Image", "Text"], $d = Zd.filter((e) => !Qd.includes(e)), ef = !1, tf = !1;
function nf() {
	let e = 0, t = (t) => {
		let n = y(1, 1, null, { willReadFrequently: t }), r = 0, i = performance.now();
		for (; performance.now() - i < 50; ++r) n.fillStyle = `rgba(255,0,${r % 256},1)`, n.fillRect(0, 0, 1, 1), n.getImageData(0, 0, 1, 1);
		return e = r > e ? r : e, r;
	};
	ef = {
		[t(!0)]: !0,
		[t(!1)]: !1,
		[t(void 0)]: void 0
	}[e], tf = !0;
}
var rf = class {
	constructor(e, t, n, r, i, a, o) {
		this.maxExtent_ = e, this.overlaps_ = r, this.pixelRatio_ = n, this.resolution_ = t, this.renderBuffer_ = a, this.executorsByZIndex_ = {}, this.hitDetectionContext_ = null, this.hitDetectionTransform_ = qr(), this.renderedContext_ = null, this.deferredZIndexContexts_ = {}, this.createExecutors_(i, o);
	}
	clip(e, t) {
		let n = this.getClipCoords(t);
		e.beginPath(), e.moveTo(n[0], n[1]), e.lineTo(n[2], n[3]), e.lineTo(n[4], n[5]), e.lineTo(n[6], n[7]), e.clip();
	}
	createExecutors_(e, t) {
		for (let n in e) {
			let r = this.executorsByZIndex_[n];
			r === void 0 && (r = {}, this.executorsByZIndex_[n] = r);
			let i = e[n];
			for (let e in i) {
				let n = i[e];
				r[e] = new Xd(this.resolution_, this.pixelRatio_, this.overlaps_, n, t);
			}
		}
	}
	hasExecutors(e) {
		for (let t in this.executorsByZIndex_) {
			let n = this.executorsByZIndex_[t];
			for (let t = 0, r = e.length; t < r; ++t) if (e[t] in n) return !0;
		}
		return !1;
	}
	forEachFeatureAtCoordinate(e, t, n, r, i, a) {
		tf === !1 && nf(), r = Math.round(r);
		let o = r * 2 + 1, s = ei(this.hitDetectionTransform_, r + .5, r + .5, 1 / t, -1 / t, -n, -e[0], -e[1]), c = !this.hitDetectionContext_;
		c && (this.hitDetectionContext_ = y(o, o, null, { willReadFrequently: ef }));
		let l = this.hitDetectionContext_;
		l.canvas.width !== o || l.canvas.height !== o ? (l.canvas.width = o, l.canvas.height = o) : c || l.clearRect(0, 0, o, o);
		let u;
		this.renderBuffer_ !== void 0 && (u = We(), Ze(u, e), Le(u, t * (this.renderBuffer_ + r), u));
		let d = of(r), f;
		function p(e, t, n) {
			let s = l.getImageData(0, 0, o, o).data;
			for (let c = 0, u = d.length; c < u; c++) if (s[d[c]] > 0) {
				if (!a || n === "none" || f !== "Image" && f !== "Text" || a.includes(e)) {
					let n = (d[c] - 3) / 4, a = r - n % o, s = r - (n / o | 0), l = i(e, t, a * a + s * s);
					if (l) return l;
				}
				l.clearRect(0, 0, o, o);
				break;
			}
		}
		let m = Object.keys(this.executorsByZIndex_).map(Number);
		m.sort(oe);
		let h, g, _, v, b;
		for (h = m.length - 1; h >= 0; --h) {
			let e = m[h].toString();
			for (_ = this.executorsByZIndex_[e], g = Zd.length - 1; g >= 0; --g) if (f = Zd[g], v = _[f], v !== void 0 && (b = v.executeHitDetection(l, s, n, p, u), b)) return b;
		}
	}
	getClipCoords(e) {
		let t = this.maxExtent_;
		if (!t) return null;
		let n = t[0], r = t[1], i = t[2], a = t[3], o = [
			n,
			r,
			n,
			a,
			i,
			a,
			i,
			r
		];
		return si(o, 0, 8, 2, e, o), o;
	}
	isEmpty() {
		return te(this.executorsByZIndex_);
	}
	execute(e, t, n, r, i, a, o) {
		let s = Object.keys(this.executorsByZIndex_).map(Number);
		s.sort(o ? se : oe), a ||= Zd;
		let c = Zd.length;
		for (let l = 0, u = s.length; l < u; ++l) {
			let u = s[l].toString(), d = this.executorsByZIndex_[u];
			for (let u = 0, f = a.length; u < f; ++u) {
				let f = a[u], p = d[f];
				if (p !== void 0) {
					let a = o === null ? void 0 : p.getZIndexContext(), u = a ? a.getContext() : e, d = this.maxExtent_ && f !== "Image" && f !== "Text";
					if (d && (u.save(), this.clip(u, n)), !a || f === "Text" || f === "Image" ? p.execute(u, t, n, r, i, o) : a.pushFunction((e) => p.execute(e, t, n, r, i, o)), d && u.restore(), a) {
						a.offset();
						let e = s[l] * c + Zd.indexOf(f);
						this.deferredZIndexContexts_[e] || (this.deferredZIndexContexts_[e] = []), this.deferredZIndexContexts_[e].push(a);
					}
				}
			}
		}
		this.renderedContext_ = e;
	}
	getDeferredZIndexContexts() {
		return this.deferredZIndexContexts_;
	}
	getRenderedContext() {
		return this.renderedContext_;
	}
	renderDeferred() {
		let e = this.deferredZIndexContexts_, t = Object.keys(e).map(Number).sort(oe);
		for (let n = 0, r = t.length; n < r; ++n) e[t[n]].forEach((e) => {
			e.draw(this.renderedContext_), e.clear();
		}), e[t[n]].length = 0;
	}
}, af = {};
function of(e) {
	if (af[e] !== void 0) return af[e];
	let t = e * 2 + 1, n = e * e, r = Array(n + 1);
	for (let i = 0; i <= e; ++i) for (let a = 0; a <= e; ++a) {
		let o = i * i + a * a;
		if (o > n) break;
		let s = r[o];
		s || (s = [], r[o] = s), s.push(((e + i) * t + (e + a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e + a)) * 4 + 3), a > 0 && (s.push(((e + i) * t + (e - a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e - a)) * 4 + 3));
	}
	let i = [];
	for (let e = 0, t = r.length; e < t; ++e) r[e] && i.push(...r[e]);
	return af[e] = i, i;
}
//#endregion
//#region node_modules/ol/render/canvas/Immediate.js
var sf = class extends hd {
	constructor(e, t, n, r, i, a, o) {
		super(), this.context_ = e, this.pixelRatio_ = t, this.extent_ = n, this.transform_ = r, this.transformRotation_ = r ? At(Math.atan2(r[1], r[0]), 10) : 0, this.viewRotation_ = i, this.squaredTolerance_ = a, this.userTransform_ = o, this.contextFillState_ = null, this.contextStrokeState_ = null, this.contextTextState_ = null, this.fillState_ = null, this.strokeState_ = null, this.image_ = null, this.imageAnchorX_ = 0, this.imageAnchorY_ = 0, this.imageHeight_ = 0, this.imageOpacity_ = 0, this.imageOriginX_ = 0, this.imageOriginY_ = 0, this.imageRotateWithView_ = !1, this.imageRotation_ = 0, this.imageScale_ = [0, 0], this.imageWidth_ = 0, this.text_ = "", this.textOffsetX_ = 0, this.textOffsetY_ = 0, this.textRotateWithView_ = !1, this.textRotation_ = 0, this.textScale_ = [0, 0], this.textFillState_ = null, this.textStrokeState_ = null, this.textState_ = null, this.pixelCoordinates_ = [], this.tmpLocalTransform_ = qr();
	}
	drawImages_(e, t, n, r) {
		if (!this.image_) return;
		let i = si(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.tmpLocalTransform_, s = a.globalAlpha;
		this.imageOpacity_ != 1 && (a.globalAlpha = s * this.imageOpacity_);
		let c = this.imageRotation_;
		this.transformRotation_ === 0 && (c -= this.viewRotation_), this.imageRotateWithView_ && (c += this.viewRotation_);
		for (let e = 0, t = i.length; e < t; e += 2) {
			let t = i[e] - this.imageAnchorX_, n = i[e + 1] - this.imageAnchorY_;
			if (c !== 0 || this.imageScale_[0] != 1 || this.imageScale_[1] != 1) {
				let e = t + this.imageAnchorX_, r = n + this.imageAnchorY_;
				ei(o, e, r, 1, 1, c, -e, -r), a.save(), a.transform.apply(a, o), a.translate(e, r), a.scale(this.imageScale_[0], this.imageScale_[1]), a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, -this.imageAnchorX_, -this.imageAnchorY_, this.imageWidth_, this.imageHeight_), a.restore();
			} else a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, t, n, this.imageWidth_, this.imageHeight_);
		}
		this.imageOpacity_ != 1 && (a.globalAlpha = s);
	}
	drawText_(e, t, n, r) {
		if (!this.textState_ || this.text_ === "") return;
		this.textFillState_ && this.setContextFillState_(this.textFillState_), this.textStrokeState_ && this.setContextStrokeState_(this.textStrokeState_), this.setContextTextState_(this.textState_);
		let i = si(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.textRotation_;
		for (this.transformRotation_ === 0 && (o -= this.viewRotation_), this.textRotateWithView_ && (o += this.viewRotation_); t < n; t += r) {
			let e = i[t] + this.textOffsetX_, n = i[t + 1] + this.textOffsetY_;
			o !== 0 || this.textScale_[0] != 1 || this.textScale_[1] != 1 ? (a.save(), a.translate(e - this.textOffsetX_, n - this.textOffsetY_), a.rotate(o), a.translate(this.textOffsetX_, this.textOffsetY_), a.scale(this.textScale_[0], this.textScale_[1]), this.textStrokeState_ && a.strokeText(this.text_, 0, 0), this.textFillState_ && a.fillText(this.text_, 0, 0), a.restore()) : (this.textStrokeState_ && a.strokeText(this.text_, e, n), this.textFillState_ && a.fillText(this.text_, e, n));
		}
	}
	moveToLineTo_(e, t, n, r, i, a) {
		let o = this.context_, s = si(e, t, n, r, this.transform_, this.pixelCoordinates_);
		if (Math.abs(a) > 0) {
			let e = s.length, t = i || Math.abs(s[0] - s[e - 2]) < 1e-6 && Math.abs(s[1] - s[e - 1]) < 1e-6;
			s = Pd(s, 0, e, 2, a, t, s), Id(s, 2, t);
		}
		o.moveTo(s[0], s[1]);
		let c = s.length;
		i && (c -= 2);
		for (let e = 2; e < c; e += 2) o.lineTo(s[e], s[e + 1]);
		return i && o.closePath(), n;
	}
	drawRings_(e, t, n, r, i) {
		for (let a = 0, o = n.length; a < o; ++a) t = this.moveToLineTo_(e, t, n[a], r, !0, i);
		return t;
	}
	drawCircle(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), pt(this.extent_, e.getExtent())) {
			if (this.fillState_ || this.strokeState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				let t = _i(e, this.transform_, this.pixelCoordinates_), n = t[2] - t[0], r = t[3] - t[1], i = Math.sqrt(n * n + r * r), a = this.context_;
				a.beginPath(), a.arc(t[0], t[1], i, 0, 2 * Math.PI), this.fillState_ && a.fill(), this.strokeState_ && a.stroke();
			}
			this.text_ !== "" && this.drawText_(e.getCenter(), 0, 2, 2);
		}
	}
	setStyle(e) {
		this.setFillStrokeStyle(e.getFill(), e.getStroke()), this.setImageStyle(e.getImage()), this.setTextStyle(e.getText());
	}
	setTransform(e) {
		this.transform_ = e;
	}
	drawGeometry(e) {
		switch (e.getType()) {
			case "Point":
				this.drawPoint(e);
				break;
			case "LineString":
				this.drawLineString(e);
				break;
			case "Polygon":
				this.drawPolygon(e);
				break;
			case "MultiPoint":
				this.drawMultiPoint(e);
				break;
			case "MultiLineString":
				this.drawMultiLineString(e);
				break;
			case "MultiPolygon":
				this.drawMultiPolygon(e);
				break;
			case "GeometryCollection":
				this.drawGeometryCollection(e);
				break;
			case "Circle": this.drawCircle(e);
		}
	}
	drawFeature(e, t) {
		let n = t.getGeometryFunction()(e);
		n && (this.setStyle(t), this.drawGeometry(n));
	}
	drawGeometryCollection(e) {
		let t = e.getGeometriesArray();
		for (let e = 0, n = t.length; e < n; ++e) this.drawGeometry(t[e]);
	}
	drawPoint(e) {
		this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_));
		let t = e.getFlatCoordinates(), n = e.getStride();
		this.image_ && this.drawImages_(t, 0, t.length, n), this.text_ !== "" && this.drawText_(t, 0, t.length, n);
	}
	drawMultiPoint(e) {
		this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_));
		let t = e.getFlatCoordinates(), n = e.getStride();
		this.image_ && this.drawImages_(t, 0, t.length, n), this.text_ !== "" && this.drawText_(t, 0, t.length, n);
	}
	drawLineString(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), pt(this.extent_, e.getExtent())) {
			if (this.strokeState_) {
				this.setContextStrokeState_(this.strokeState_);
				let t = this.context_, n = e.getFlatCoordinates();
				t.beginPath(), this.moveToLineTo_(n, 0, n.length, e.getStride(), !1, this.strokeState_.strokeOffset), t.stroke();
			}
			if (this.text_ !== "") {
				let t = e.getFlatMidpoint();
				this.drawText_(t, 0, 2, 2);
			}
		}
	}
	drawMultiLineString(e) {
		this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_));
		let t = e.getExtent();
		if (pt(this.extent_, t)) {
			if (this.strokeState_) {
				this.setContextStrokeState_(this.strokeState_);
				let t = this.context_, n = e.getFlatCoordinates(), r = 0, i = e.getEnds(), a = e.getStride();
				t.beginPath();
				for (let e = 0, t = i.length; e < t; ++e) r = this.moveToLineTo_(n, r, i[e], a, !1, this.strokeState_.strokeOffset);
				t.stroke();
			}
			if (this.text_ !== "") {
				let t = e.getFlatMidpoints();
				this.drawText_(t, 0, t.length, 2);
			}
		}
	}
	drawPolygon(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), pt(this.extent_, e.getExtent())) {
			if (this.strokeState_ || this.fillState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				let t = this.context_;
				t.beginPath(), this.drawRings_(e.getOrientedFlatCoordinates(), 0, e.getEnds(), e.getStride(), this.strokeState_?.strokeOffset), this.fillState_ && t.fill(), this.strokeState_ && t.stroke();
			}
			if (this.text_ !== "") {
				let t = e.getFlatInteriorPoint();
				this.drawText_(t, 0, 2, 2);
			}
		}
	}
	drawMultiPolygon(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), pt(this.extent_, e.getExtent())) {
			if (this.strokeState_ || this.fillState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				let t = this.context_, n = e.getOrientedFlatCoordinates(), r = 0, i = e.getEndss(), a = e.getStride();
				t.beginPath();
				for (let e = 0, t = i.length; e < t; ++e) {
					let t = i[e];
					r = this.drawRings_(n, r, t, a, this.strokeState_?.strokeOffset);
				}
				this.fillState_ && t.fill(), this.strokeState_ && t.stroke();
			}
			if (this.text_ !== "") {
				let t = e.getFlatInteriorPoints();
				this.drawText_(t, 0, t.length, 2);
			}
		}
	}
	setContextFillState_(e) {
		let t = this.context_, n = this.contextFillState_;
		n ? n.fillStyle != e.fillStyle && (n.fillStyle = e.fillStyle, t.fillStyle = e.fillStyle) : (t.fillStyle = e.fillStyle, this.contextFillState_ = { fillStyle: e.fillStyle });
	}
	setContextStrokeState_(e) {
		let t = this.context_, n = this.contextStrokeState_;
		n ? (n.lineCap != e.lineCap && (n.lineCap = e.lineCap, t.lineCap = e.lineCap), de(n.lineDash, e.lineDash) || t.setLineDash(n.lineDash = e.lineDash), n.lineDashOffset != e.lineDashOffset && (n.lineDashOffset = e.lineDashOffset, t.lineDashOffset = e.lineDashOffset), n.lineJoin != e.lineJoin && (n.lineJoin = e.lineJoin, t.lineJoin = e.lineJoin), n.lineWidth != e.lineWidth && (n.lineWidth = e.lineWidth, t.lineWidth = e.lineWidth), n.miterLimit != e.miterLimit && (n.miterLimit = e.miterLimit, t.miterLimit = e.miterLimit), n.strokeStyle != e.strokeStyle && (n.strokeStyle = e.strokeStyle, t.strokeStyle = e.strokeStyle)) : (t.lineCap = e.lineCap, t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset, t.lineJoin = e.lineJoin, t.lineWidth = e.lineWidth, t.miterLimit = e.miterLimit, t.strokeStyle = e.strokeStyle, this.contextStrokeState_ = {
			lineCap: e.lineCap,
			lineDash: e.lineDash,
			lineDashOffset: e.lineDashOffset,
			lineJoin: e.lineJoin,
			lineWidth: e.lineWidth,
			miterLimit: e.miterLimit,
			strokeStyle: e.strokeStyle
		});
	}
	setContextTextState_(e) {
		let t = this.context_, n = this.contextTextState_, r = e.textAlign ? e.textAlign : gc;
		n ? (n.font != e.font && (n.font = e.font, t.font = e.font), n.textAlign != r && (n.textAlign = r, t.textAlign = r), n.textBaseline != e.textBaseline && (n.textBaseline = e.textBaseline, t.textBaseline = e.textBaseline)) : (t.font = e.font, t.textAlign = r, t.textBaseline = e.textBaseline, this.contextTextState_ = {
			font: e.font,
			textAlign: r,
			textBaseline: e.textBaseline
		});
	}
	setFillStrokeStyle(e, t) {
		if (!e) this.fillState_ = null;
		else {
			let t = e.getColor();
			this.fillState_ = { fillStyle: cc(t || dc) };
		}
		if (!t) this.strokeState_ = null;
		else {
			let e = t.getColor(), n = t.getLineCap(), r = t.getLineDash(), i = t.getLineDashOffset(), a = t.getLineJoin(), o = t.getWidth(), s = t.getMiterLimit(), c = r || pc, l = t.getOffset();
			this.strokeState_ = {
				lineCap: n === void 0 ? fc : n,
				lineDash: this.pixelRatio_ === 1 ? c : c.map((e) => e * this.pixelRatio_),
				lineDashOffset: (i || 0) * this.pixelRatio_,
				lineJoin: a === void 0 ? mc : a,
				lineWidth: (o === void 0 ? 1 : o) * this.pixelRatio_,
				miterLimit: s === void 0 ? 10 : s,
				strokeStyle: cc(e || hc),
				strokeOffset: (l ?? 0) * this.pixelRatio_
			};
		}
	}
	setImageStyle(e) {
		let t;
		if (!e || !(t = e.getSize())) {
			this.image_ = null;
			return;
		}
		let n = e.getPixelRatio(this.pixelRatio_), r = e.getAnchor(), i = e.getOrigin();
		this.image_ = e.getImage(this.pixelRatio_), this.imageAnchorX_ = r[0] * n, this.imageAnchorY_ = r[1] * n, this.imageHeight_ = t[1] * n, this.imageOpacity_ = e.getOpacity(), this.imageOriginX_ = i[0], this.imageOriginY_ = i[1], this.imageRotateWithView_ = e.getRotateWithView(), this.imageRotation_ = e.getRotation();
		let a = e.getScaleArray();
		this.imageScale_ = [a[0] * this.pixelRatio_ / n, a[1] * this.pixelRatio_ / n], this.imageWidth_ = t[0] * n;
	}
	setTextStyle(e) {
		if (!e) this.text_ = "";
		else {
			let t = e.getFill();
			if (!t) this.textFillState_ = null;
			else {
				let e = t.getColor();
				this.textFillState_ = { fillStyle: cc(e || dc) };
			}
			let n = e.getStroke();
			if (!n) this.textStrokeState_ = null;
			else {
				let e = n.getColor(), t = n.getLineCap(), r = n.getLineDash(), i = n.getLineDashOffset(), a = n.getLineJoin(), o = n.getWidth(), s = n.getMiterLimit();
				this.textStrokeState_ = {
					lineCap: t === void 0 ? fc : t,
					lineDash: r || pc,
					lineDashOffset: i || 0,
					lineJoin: a === void 0 ? mc : a,
					lineWidth: o === void 0 ? 1 : o,
					miterLimit: s === void 0 ? 10 : s,
					strokeStyle: cc(e || hc)
				};
			}
			let r = e.getFont(), i = e.getOffsetX(), a = e.getOffsetY(), o = e.getRotateWithView(), s = e.getRotation(), c = e.getScaleArray(), l = e.getText(), u = e.getTextAlign(), d = e.getTextBaseline();
			this.textState_ = {
				font: r === void 0 ? uc : r,
				textAlign: u === void 0 ? gc : u,
				textBaseline: d === void 0 ? _c : d
			}, this.text_ = l === void 0 ? "" : Array.isArray(l) ? l.reduce((e, t, n) => e += n % 2 ? " " : t, "") : l, this.textOffsetX_ = i === void 0 ? 0 : this.pixelRatio_ * i, this.textOffsetY_ = a === void 0 ? 0 : this.pixelRatio_ * a, this.textRotateWithView_ = o !== void 0 && o, this.textRotation_ = s === void 0 ? 0 : s, this.textScale_ = [this.pixelRatio_ * c[0], this.pixelRatio_ * c[1]];
		}
	}
}, cf = .5;
function lf(e, t, n, r, i, a, o, s, c) {
	let l = c ? vr(i, c) : i, u = y(e[0] * cf, e[1] * cf);
	u.imageSmoothingEnabled = !1;
	let d = u.canvas, f = new sf(u, cf, i, null, o, s, c ? lr(hr(), c) : null), p = n.length, m = Math.floor(16777215 / p), h = {};
	for (let e = 1; e <= p; ++e) {
		let t = n[e - 1], i = t.getStyleFunction() || r;
		if (!i) continue;
		let o = i(t, a);
		if (!o) continue;
		Array.isArray(o) || (o = [o]);
		let s = (e * m).toString(16).padStart(7, "#00000");
		for (let e = 0, n = o.length; e < n; ++e) {
			let n = o[e], r = n.getGeometryFunction()(t);
			if (!r || !pt(l, r.getExtent())) continue;
			let i = n.clone(), a = i.getFill();
			a && a.setColor(s);
			let c = i.getStroke();
			c && (c.setColor(s), c.setLineDash(null)), i.setText(void 0);
			let u = n.getImage();
			if (u) {
				let e = u.getImageSize();
				if (!e) continue;
				let t = y(e[0], e[1], void 0, { alpha: !1 }), n = t.canvas;
				t.fillStyle = s, t.fillRect(0, 0, n.width, n.height), i.setImage(new Rc({
					img: n,
					anchor: u.getAnchor(),
					anchorXUnits: "pixels",
					anchorYUnits: "pixels",
					offset: u.getOrigin(),
					opacity: 1,
					size: u.getSize(),
					scale: u.getScale(),
					rotation: u.getRotation(),
					rotateWithView: u.getRotateWithView()
				}));
			}
			let d = i.getZIndex() || 0, f = h[d];
			f || (f = {}, h[d] = f, f.Polygon = [], f.Circle = [], f.LineString = [], f.Point = []);
			let p = r.getType();
			if (p === "GeometryCollection") {
				let e = r.getGeometriesArrayRecursive();
				for (let t = 0, n = e.length; t < n; ++t) {
					let n = e[t];
					f[n.getType().replace("Multi", "")].push(n, i);
				}
			} else f[p.replace("Multi", "")].push(r, i);
		}
	}
	let g = Object.keys(h).map(Number).sort(oe);
	for (let e = 0, n = g.length; e < n; ++e) {
		let n = h[g[e]];
		for (let e in n) {
			let r = n[e];
			for (let e = 0, n = r.length; e < n; e += 2) {
				f.setStyle(r[e + 1]);
				for (let n = 0, i = t.length; n < i; ++n) f.setTransform(t[n]), f.drawGeometry(r[e]);
			}
		}
	}
	return u.getImageData(0, 0, d.width, d.height);
}
function uf(e, t, n) {
	let r = [];
	if (n) {
		let i = Math.floor(Math.round(e[0]) * cf), a = Math.floor(Math.round(e[1]) * cf), o = (St(i, 0, n.width - 1) + St(a, 0, n.height - 1) * n.width) * 4, s = n.data[o], c = n.data[o + 1], l = n.data[o + 2] + 256 * (c + 256 * s), u = Math.floor(16777215 / t.length);
		l && l % u === 0 && r.push(t[l / u - 1]);
	}
	return r;
}
//#endregion
//#region node_modules/ol/renderer/vector.js
var df = .5, ff = {
	Point: wf,
	LineString: xf,
	Polygon: Ef,
	MultiPoint: Tf,
	MultiLineString: Sf,
	MultiPolygon: Cf,
	GeometryCollection: bf,
	Circle: gf
};
function pf(e, t) {
	return parseInt(F(e), 10) - parseInt(F(t), 10);
}
function mf(e, t) {
	let n = hf(e, t);
	return n * n;
}
function hf(e, t) {
	return df * e / t;
}
function gf(e, t, n, r, i) {
	let a = n.getFill(), o = n.getStroke();
	if (a || o) {
		let s = e.getBuilder(n.getZIndex(), "Circle");
		s.setFillStrokeStyle(a, o), s.drawCircle(t, r, i);
	}
	let s = n.getText();
	if (s && s.getText()) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(s), a.drawText(t, r, i);
	}
}
function _f(e, t, n, r, i, a, o, s) {
	let c = [], l = n.getImage();
	if (l) {
		let e = !0, t = l.getImageState();
		t == J.LOADED || t == J.ERROR ? e = !1 : t == J.IDLE && l.load(), e && c.push(l.ready());
	}
	let u = n.getFill();
	u && u.loading() && c.push(u.ready());
	let d = c.length > 0;
	return d && Promise.all(c).then(() => i(null)), vf(e, t, n, r, a, o, s), d;
}
function vf(e, t, n, r, i, a, o) {
	let s = n.getGeometryFunction()(t);
	if (!s) return;
	let c = s.simplifyTransformed(r, i);
	if (n.getRenderer()) yf(e, c, n, t, o);
	else {
		let r = ff[c.getType()];
		r(e, c, n, t, o, a);
	}
}
function yf(e, t, n, r, i) {
	if (t.getType() == "GeometryCollection") {
		let a = t.getGeometries();
		for (let t = 0, o = a.length; t < o; ++t) yf(e, a[t], n, r, i);
		return;
	}
	e.getBuilder(n.getZIndex(), "Default").drawCustom(t, r, n.getRenderer(), n.getHitDetectionRenderer(), i);
}
function bf(e, t, n, r, i, a) {
	let o = t.getGeometriesArray(), s, c;
	for (s = 0, c = o.length; s < c; ++s) {
		let t = ff[o[s].getType()];
		t(e, o[s], n, r, i, a);
	}
}
function xf(e, t, n, r, i) {
	let a = n.getStroke();
	if (a) {
		let o = e.getBuilder(n.getZIndex(), "LineString");
		o.setFillStrokeStyle(null, a), o.drawLineString(t, r, i);
	}
	let o = n.getText();
	if (o && o.getText()) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(o), a.drawText(t, r, i);
	}
}
function Sf(e, t, n, r, i) {
	let a = n.getStroke();
	if (a) {
		let o = e.getBuilder(n.getZIndex(), "LineString");
		o.setFillStrokeStyle(null, a), o.drawMultiLineString(t, r, i);
	}
	let o = n.getText();
	if (o && o.getText()) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(o), a.drawText(t, r, i);
	}
}
function Cf(e, t, n, r, i) {
	let a = n.getFill(), o = n.getStroke();
	if (o || a) {
		let s = e.getBuilder(n.getZIndex(), "Polygon");
		s.setFillStrokeStyle(a, o), s.drawMultiPolygon(t, r, i);
	}
	let s = n.getText();
	if (s && s.getText()) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(s), a.drawText(t, r, i);
	}
}
function wf(e, t, n, r, i, a) {
	let o = n.getImage(), s = n.getText(), c = s && s.getText(), l = a && o && c ? {} : void 0;
	if (o) {
		if (o.getImageState() != J.LOADED) return;
		let a = e.getBuilder(n.getZIndex(), "Image");
		a.setImageStyle(o, l), a.drawPoint(t, r, i);
	}
	if (c) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(s, l), a.drawText(t, r, i);
	}
}
function Tf(e, t, n, r, i, a) {
	let o = n.getImage(), s = o && o.getOpacity() !== 0, c = n.getText(), l = c && c.getText(), u = a && s && l ? {} : void 0;
	if (s) {
		if (o.getImageState() != J.LOADED) return;
		let a = e.getBuilder(n.getZIndex(), "Image");
		a.setImageStyle(o, u), a.drawMultiPoint(t, r, i);
	}
	if (l) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(c, u), a.drawText(t, r, i);
	}
}
function Ef(e, t, n, r, i) {
	let a = n.getFill(), o = n.getStroke();
	if (a || o) {
		let s = e.getBuilder(n.getZIndex(), "Polygon");
		s.setFillStrokeStyle(a, o), s.drawPolygon(t, r, i);
	}
	let s = n.getText();
	if (s && s.getText()) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(s), a.drawText(t, r, i);
	}
}
//#endregion
//#region node_modules/ol/renderer/canvas/VectorLayer.js
var Df = class extends Au {
	constructor(e) {
		super(e), this.boundHandleStyleImageChange_ = this.handleStyleImageChange_.bind(this), this.animatingOrInteracting_, this.hitDetectionImageData_ = null, this.clipExtent_ = null, this.extendX_ = !1, this.renderedFeatures_ = null, this.renderedRevision_ = -1, this.renderedResolution_ = NaN, this.renderedExtent_ = We(), this.wrappedRenderedExtent_ = We(), this.renderedRotation_, this.renderedCenter_ = null, this.renderedProjection_ = null, this.renderedPixelRatio_ = 1, this.renderedRenderOrder_ = null, this.renderedFrameDeclutter_, this.replayGroup_ = null, this.replayGroupChanged = !0, this.clipping = !0, this.targetContext_ = null, this.opacity_ = 1;
	}
	renderWorlds(e, t, n) {
		let r = t.extent, i = t.viewState, a = i.center, o = i.resolution, s = i.projection, c = i.rotation, l = s.getExtent(), u = this.getLayer().getSource(), d = this.getLayer().getDeclutter(), f = t.pixelRatio, p = t.viewHints, m = !(p[Lr.ANIMATING] || p[Lr.INTERACTING]), h = this.context, g = Math.round(I(r) / o * f), _ = Math.round(ct(r) / o * f), v = u.getWrapX() && s.canWrapX(), y = v ? I(l) : null, b = v ? Math.ceil((r[2] - l[2]) / y) + (this.extendX_ ? 2 : 1) : 1, x = v ? Math.floor((r[0] - l[0]) / y) - +!!this.extendX_ : 0;
		do {
			let r = this.getRenderTransform(a, o, 0, f, g, _, x * y);
			t.declutter && (r = r.slice(0)), e.execute(h, [h.canvas.width, h.canvas.height], r, c, m, n === void 0 ? Zd : n ? Qd : $d, n ? d && t.declutter[d] : void 0);
		} while (++x < b);
	}
	setDrawContext_() {
		this.opacity_ !== 1 && (this.targetContext_ = this.context, this.context = y(this.context.canvas.width, this.context.canvas.height, Du));
	}
	resetDrawContext_() {
		if (this.opacity_ !== 1 && this.targetContext_) {
			let e = this.targetContext_.globalAlpha;
			this.targetContext_.globalAlpha = this.opacity_, this.targetContext_.drawImage(this.context.canvas, 0, 0), this.targetContext_.globalAlpha = e, S(this.context), Du.push(this.context.canvas), this.context = this.targetContext_, this.targetContext_ = null;
		}
	}
	renderDeclutter(e) {
		!this.replayGroup_ || !this.getLayer().getDeclutter() || this.renderWorlds(this.replayGroup_, e, !0);
	}
	renderDeferredInternal(e) {
		this.replayGroup_ && (this.clipExtent_ && this.clipUnrotated(this.context, e, this.clipExtent_), this.replayGroup_.renderDeferred(), this.clipExtent_ &&= (this.context.restore(), null), this.resetDrawContext_());
	}
	renderFrame(e, t) {
		let n = e.layerStatesArray[e.layerIndex];
		this.opacity_ = n.opacity;
		let r = e.viewState;
		this.prepareContainer(e, t);
		let i = this.context, a = this.replayGroup_, o = a && !a.isEmpty();
		if (!o && !(this.getLayer().hasListener(lo.PRERENDER) || this.getLayer().hasListener(lo.POSTRENDER))) return this.container;
		this.setDrawContext_(), this.preRender(i, e);
		let s = r.projection;
		this.clipExtent_ = null;
		let c = !1;
		if (o && n.extent && this.clipping) {
			let t = yr(n.extent, s);
			o = pt(t, e.extent), o && !Ve(t, e.extent) && (e.declutter ? this.clipExtent_ = t : (this.clipUnrotated(i, e, t), c = !0));
		}
		return o && this.renderWorlds(a, e, !this.getLayer().getDeclutter() && void 0), c && i.restore(), this.postRender(i, e), this.renderedRotation_ !== r.rotation && (this.renderedRotation_ = r.rotation, this.hitDetectionImageData_ = null), e.declutter || this.resetDrawContext_(), this.container;
	}
	getFeatures(e) {
		return new Promise((t) => {
			if (this.frameState && !this.hitDetectionImageData_ && !this.animatingOrInteracting_) {
				let e = this.frameState.size.slice(), t = this.renderedCenter_, n = this.renderedResolution_, r = this.renderedRotation_, i = this.renderedProjection_, a = this.wrappedRenderedExtent_, o = this.getLayer(), s = [], c = e[0] * cf, l = e[1] * cf;
				s.push(this.getRenderTransform(t, n, r, cf, c, l, 0).slice());
				let u = o.getSource(), d = i.getExtent();
				if (u.getWrapX() && i.canWrapX() && !Ve(d, a)) {
					let e = a[0], i = I(d), o = 0, u;
					for (; e < d[0];) --o, u = i * o, s.push(this.getRenderTransform(t, n, r, cf, c, l, u).slice()), e += i;
					for (o = 0, e = a[2]; e > d[2];) ++o, u = i * o, s.push(this.getRenderTransform(t, n, r, cf, c, l, u).slice()), e -= i;
				}
				let f = hr();
				this.hitDetectionImageData_ = lf(e, s, this.renderedFeatures_, o.getStyleFunction(), a, n, r, mf(n, this.renderedPixelRatio_), f ? i : null);
			}
			t(uf(e, this.renderedFeatures_, this.hitDetectionImageData_));
		});
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		if (!this.replayGroup_) return;
		let a = t.viewState.resolution, o = t.viewState.rotation, s = this.getLayer(), c = {}, l = function(e, t, n) {
			let a = F(e), o = c[a];
			if (!o) {
				if (n === 0) return c[a] = !0, r(e, s, t);
				i.push(c[a] = {
					feature: e,
					layer: s,
					geometry: t,
					distanceSq: n,
					callback: r
				});
			} else if (o !== !0 && n < o.distanceSq) {
				if (n === 0) return c[a] = !0, i.splice(i.lastIndexOf(o), 1), r(e, s, t);
				o.geometry = t, o.distanceSq = n;
			}
		}, u = this.getLayer().getDeclutter();
		return this.replayGroup_.forEachFeatureAtCoordinate(e, a, o, n, l, u ? t.declutter?.[u]?.all().map((e) => e.value) : null);
	}
	handleFontsChanged() {
		let e = this.getLayer();
		e.getVisible() && this.replayGroup_ && e.changed();
	}
	handleStyleImageChange_(e) {
		this.renderIfReadyAndVisible();
	}
	prepareFrame(e) {
		let t = this.getLayer(), n = t.getSource();
		if (!n) return !1;
		let r = e.viewHints[Lr.ANIMATING], i = e.viewHints[Lr.INTERACTING], a = t.getUpdateWhileAnimating(), o = t.getUpdateWhileInteracting();
		if (this.ready && !a && r || !o && i) return this.animatingOrInteracting_ = !0, !0;
		this.animatingOrInteracting_ = !1;
		let s = e.extent, c = e.viewState, l = c.projection, u = c.resolution, d = e.pixelRatio, f = t.getRevision(), p = t.getRenderBuffer(), m = t.getRenderOrder();
		m === void 0 && (m = pf);
		let h = c.center.slice(), g = Le(s, p * u), _ = g.slice(), v = [g.slice()], y = l.getExtent(), b = n.getWrapX() && l.canWrapX();
		if (this.extendX_ = !1, b) {
			let e = n.getExtent();
			e && !mt(e) && (this.extendX_ = e[0] < y[0] || e[2] > y[2]);
		}
		if (b && (!Ve(y, e.extent) || this.extendX_)) {
			let e = I(y), t = Math.max(I(g) / 2, e), n = y[0], r = y[2];
			this.extendX_ && (n -= e, r += e), g[0] = n - t, g[2] = r + t, zt(h, l);
			let i = yt(v[0], l);
			i[0] < y[0] && i[2] < y[2] ? v.push([
				i[0] + e,
				i[1],
				i[2] + e,
				i[3]
			]) : i[0] > y[0] && i[2] > y[2] && v.push([
				i[0] - e,
				i[1],
				i[2] - e,
				i[3]
			]);
		}
		if (this.ready && this.renderedResolution_ == u && this.renderedPixelRatio_ === d && this.renderedRevision_ == f && this.renderedRenderOrder_ == m && this.renderedFrameDeclutter_ === !!e.declutter && Ve(this.wrappedRenderedExtent_, g)) return de(this.renderedExtent_, _) || (this.hitDetectionImageData_ = null, this.renderedExtent_ = _), this.renderedCenter_ = h, this.replayGroupChanged = !1, !0;
		this.replayGroup_ = null;
		let x = new Md(hf(u, d), g, u, d), S = hr(), C;
		if (S) {
			for (let e = 0, t = v.length; e < t; ++e) {
				let t = v[e], r = vr(t, l);
				n.loadFeatures(r, br(u, l), S);
			}
			C = lr(S, l);
		} else for (let e = 0, t = v.length; e < t; ++e) n.loadFeatures(v[e], u, l);
		let w = mf(u, d), T = !0, E = (e, n) => {
			let r, i = e.getStyleFunction() || t.getStyleFunction();
			if (i && (r = i(e, u)), r) {
				let t = this.renderFeature(e, w, r, x, C, this.getLayer().getDeclutter(), n);
				T &&= !t;
			}
		}, ee = vr(g, l), D = n.getFeaturesInExtent(ee);
		m && D.sort(m);
		for (let e = 0, t = D.length; e < t; ++e) E(D[e], e);
		this.renderedFeatures_ = D, this.ready = T;
		let O = x.finish(), k = new rf(g, u, d, n.getOverlaps(), O, t.getRenderBuffer(), !!e.declutter);
		return this.renderedResolution_ = u, this.renderedRevision_ = f, this.renderedRenderOrder_ = m, this.renderedFrameDeclutter_ = !!e.declutter, this.renderedExtent_ = _, this.wrappedRenderedExtent_ = g, this.renderedCenter_ = h, this.renderedProjection_ = l, this.renderedPixelRatio_ = d, this.replayGroup_ = k, this.hitDetectionImageData_ = null, this.replayGroupChanged = !0, !0;
	}
	renderFeature(e, t, n, r, i, a, o) {
		if (!n) return !1;
		let s = !1;
		if (Array.isArray(n)) for (let c = 0, l = n.length; c < l; ++c) s = _f(r, e, n[c], t, this.boundHandleStyleImageChange_, i, a, o) || s;
		else s = _f(r, e, n, t, this.boundHandleStyleImageChange_, i, a, o);
		return s;
	}
}, Of = class extends Al {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new Df(this);
	}
}, kf = {
	image: [
		"Polygon",
		"Circle",
		"LineString",
		"Image",
		"Text"
	],
	hybrid: ["Polygon", "LineString"],
	vector: []
}, Af = {
	hybrid: [
		"Image",
		"Text",
		"Default"
	],
	vector: [
		"Polygon",
		"Circle",
		"LineString",
		"Image",
		"Text",
		"Default"
	]
}, jf = class extends Pu {
	constructor(e, t) {
		super(e, t), this.boundHandleStyleImageChange_ = this.handleStyleImageChange_.bind(this), this.renderedLayerRevision_, this.renderedPixelToCoordinateTransform_ = null, this.renderedRotation_, this.renderedOpacity_ = 1, this.tmpTransform_ = qr(), this.tileClipContexts_ = null;
	}
	enqueueTilesForNextExtent() {
		return this.getLayer().getRenderMode() !== "vector";
	}
	drawTile(e, t, n, r, i, a, o, s, c) {
		this.updateExecutorGroup_(e, t.pixelRatio, t.viewState.projection), this.tileImageNeedsRender_(e) && this.renderTileImage_(e, t), super.drawTile(e, t, n, r, i, a, o, s, c);
	}
	getTile(e, t, n, r) {
		let i = this.getOrCreateTile(e, t, n, r);
		if (!i) return null;
		let a = r.viewState, o = a.resolution, s = r.viewHints, c = this.getLayer().getSource(), l = c.getTileGridForProjection(a.projection), u = !(s[Lr.ANIMATING] || s[Lr.INTERACTING]), d = l.getZForResolution(o, c.zDirection) === e;
		return u && d ? i.wantedResolution = o : i.wantedResolution ||= l.getResolution(e), i;
	}
	prepareFrame(e) {
		let t = this.getLayer().getRevision();
		return this.renderedLayerRevision_ !== t && (this.renderedLayerRevision_ = t, this.renderedTiles.length = 0), super.prepareFrame(e);
	}
	updateExecutorGroup_(e, t, n) {
		let r = this.getLayer(), i = r.getRevision(), a = r.getRenderOrder() || null, o = e.wantedResolution, s = e.getReplayState(r);
		if (!s.dirty && s.renderedResolution === o && s.renderedRevision == i && s.renderedPixelRatio === t && s.renderedRenderOrder == a) return;
		let c = r.getSource(), l = !!r.getDeclutter(), u = c.getTileGrid(), d = c.getTileGridForProjection(n).getTileCoordExtent(e.wrappedTileCoord), f = c.getSourceTiles(t, n, e), p = F(r);
		delete e.hitDetectionImageData[p], e.executorGroups[p] = [], s.dirty = !1;
		for (let i = 0, m = f.length; i < m; ++i) {
			let m = f[i];
			if (m.getState() != L.LOADED) continue;
			let h = c.getProjection(), g = m.tileCoord, _ = u.getTileCoordExtent(g);
			n && h && !cr(n, h) && (_ = pr(_, h, n, 32));
			let v = lt(d, _), y = Le(v, r.getRenderBuffer() * o, this.tempExtent), b = Ye(_, v) ? null : y, x = new Md(0, v, o, t), S = mf(o, t), C = function(e, t) {
				let n, i = e.getStyleFunction() || r.getStyleFunction();
				if (i && (n = i(e, o)), n) {
					let r = this.renderFeature(e, S, n, x, l, t);
					s.dirty = s.dirty || r;
				}
			}, w = m.getFeatures();
			a && a !== s.renderedRenderOrder && w.sort(a);
			for (let e = 0, t = w.length; e < t; ++e) {
				let t = w[e];
				n && m.projection && !cr(n, m.projection) && (t = t.clone(), t.getGeometry().applyTransform(dr(m.projection, n))), (!b || pt(b, t.getGeometry().getExtent())) && C.call(this, t, e);
			}
			let T = x.finish(), E = new rf(r.getRenderMode() !== "vector" && l && f.length === 1 ? null : v, o, t, c.getOverlaps(), T, r.getRenderBuffer(), !0);
			e.executorGroups[p].push(E);
		}
		s.renderedRevision = i, s.renderedPixelRatio = t, s.renderedRenderOrder = a, s.renderedResolution = o;
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		let a = t.viewState.resolution, o = t.viewState.rotation;
		n ??= 0;
		let s = this.getLayer(), c = s.getSource().getTileGridForProjection(t.viewState.projection), l = s.getRenderBuffer(), u = Fe([e]);
		Le(u, a * (l + n), u);
		let d = {}, f = function(e, t, n) {
			let a = e.getId();
			a === void 0 && (a = F(e));
			let o = d[a];
			if (!o) {
				if (n === 0) return d[a] = !0, r(e, s, t);
				i.push(d[a] = {
					feature: e,
					layer: s,
					geometry: t,
					distanceSq: n,
					callback: r
				});
			} else if (o !== !0 && n < o.distanceSq) {
				if (n === 0) return d[a] = !0, i.splice(i.lastIndexOf(o), 1), r(e, s, t);
				o.geometry = t, o.distanceSq = n;
			}
		}, p = this.renderedTiles, m = F(s), h = s.getDeclutter(), g = h ? t.declutter?.[h]?.all().map((e) => e.value) : null, _;
		foundFeature: for (let t = p.length - 1; t >= 0; --t) {
			let r = p[t];
			if (!pt(c.getTileCoordExtent(r.wrappedTileCoord), u)) continue;
			let i = r.executorGroups[m];
			for (let t = 0, r = i.length; t < r; ++t) if (_ = i[t].forEachFeatureAtCoordinate(e, a, o, n, f, g), _) break foundFeature;
		}
		return _;
	}
	getFeatures(e) {
		return this.renderedTiles.length === 0 ? Promise.resolve([]) : new Promise((t, n) => {
			let r = this.getLayer(), i = r.getSource(), a = this.renderedProjection, o = a.getExtent(), s = this.renderedResolution, c = i.getTileGridForProjection(a), l = z(this.renderedPixelToCoordinateTransform_, e.slice()), u = c.getTileCoordForCoordAndResolution(l, s).toString(), d = this.renderedTiles.find((e) => e.tileCoord.toString() === u && e.getState() === L.LOADED);
			if (!d || d.loadingSourceTiles > 0) {
				t([]);
				return;
			}
			i.getWrapX() && a.canWrapX() && !Ve(o, c.getTileCoordExtent(d.tileCoord)) && zt(l, a);
			let f = F(r), p = dt(c.getTileCoordExtent(d.wrappedTileCoord)), m = [(l[0] - p[0]) / s, (p[1] - l[1]) / s], h = d.getSourceTiles().reduce((e, t) => e.concat(t.getFeatures()), []), g = d.hitDetectionImageData[f];
			if (!g) {
				let e = rs(c.getTileSize(c.getZForResolution(s, i.zDirection))), t = this.renderedRotation_;
				g = lf(e, [this.getRenderTransform(c.getTileCoordCenter(d.wrappedTileCoord), s, 0, cf, e[0] * cf, e[1] * cf, 0)], h, r.getStyleFunction(), c.getTileCoordExtent(d.wrappedTileCoord), d.getReplayState(r).renderedResolution, t), d.hitDetectionImageData[f] = g;
			}
			t(uf(m, h, g));
		});
	}
	getFeaturesInExtent(e) {
		let t = [], n = this.getTileCache();
		if (n.getCount() === 0) return t;
		let r = this.getLayer().getSource().getTileGridForProjection(this.frameState.viewState.projection), i = r.getZForResolution(this.renderedResolution), a = {};
		return n.forEach((n) => {
			if (n.tileCoord[0] !== i || n.getState() !== L.LOADED) return;
			let o = n.getSourceTiles();
			for (let n = 0, i = o.length; n < i; ++n) {
				let i = o[n], s = i.getKey();
				if (s in a) continue;
				a[s] = !0;
				let c = i.tileCoord;
				if (pt(e, r.getTileCoordExtent(c))) {
					let n = i.getFeatures();
					if (n) for (let r = 0, i = n.length; r < i; ++r) {
						let i = n[r];
						pt(e, i.getGeometry().getExtent()) && t.push(i);
					}
				}
			}
		}), t;
	}
	handleFontsChanged() {
		let e = this.getLayer();
		e.getVisible() && this.renderedLayerRevision_ !== void 0 && e.changed();
	}
	handleStyleImageChange_(e) {
		this.renderIfReadyAndVisible();
	}
	renderDeclutter(e, t) {
		let n = this.context, r = n.globalAlpha;
		n.globalAlpha = t.opacity;
		let i = e.viewHints, a = !(i[Lr.ANIMATING] || i[Lr.INTERACTING]), o = [this.context.canvas.width, this.context.canvas.height], s = this.getLayer().getDeclutter(), c = s ? e.declutter?.[s] : void 0, l = F(this.getLayer()), u = this.renderedTiles;
		for (let t = 0, n = u.length; t < n; ++t) {
			let n = u[t], r = n.executorGroups[l];
			if (r) for (let t = r.length - 1; t >= 0; --t) r[t].execute(this.context, o, this.getTileRenderTransform(n, e), e.viewState.rotation, a, Qd, c);
		}
		n.globalAlpha = r;
	}
	renderDeferredInternal(e) {
		let t = this.renderedTiles, n = F(this.getLayer()), r = t.reduce((e, t, r) => (t.executorGroups[n].forEach((t) => e.push({
			executorGroup: t,
			index: r
		})), e), []), i = r.map(({ executorGroup: e }) => e.getDeferredZIndexContexts()), a = {};
		for (let e = 0, t = r.length; e < t; ++e) {
			let t = r[e].executorGroup.getDeferredZIndexContexts();
			for (let e in t) a[e] = !0;
		}
		let o = Object.keys(a).map(Number).sort(oe);
		this.layerExtent && this.clipUnrotated(this.context, e, this.layerExtent), o.forEach((e) => {
			i.forEach((t, n) => {
				t[e] && (t[e].forEach((e) => {
					let { executorGroup: t, index: i } = r[n], a = t.getRenderedContext(), o = a.globalAlpha;
					a.globalAlpha = this.renderedOpacity_;
					let s = this.tileClipContexts_[i];
					s && s.draw(a), e.draw(a), s && a.restore(), a.globalAlpha = o, e.clear();
				}), t[e].length = 0);
			});
		}), this.layerExtent && this.context.restore();
	}
	getTileRenderTransform(e, t) {
		let n = t.pixelRatio, r = t.viewState, i = r.center, a = r.resolution, o = r.rotation, s = t.size, c = Math.round(s[0] * n), l = Math.round(s[1] * n), u = this.getLayer().getSource().getTileGridForProjection(t.viewState.projection), d = e.tileCoord, f = u.getTileCoordExtent(e.wrappedTileCoord), p = u.getTileCoordExtent(d, this.tempExtent)[0] - f[0];
		return Yr(Qr(this.inversePixelTransform.slice(), 1 / n, 1 / n), this.getRenderTransform(i, a, o, n, c, l, p));
	}
	clipTileContext_(e, t, n, r, i, a) {
		let o = [];
		for (let e = 0, a = n.length; e < a; ++e) i < r[e] && pt(t, n[e]) && o.push(n[e]);
		if (o.length === 0) return !1;
		let s = xt(t, o);
		e.save(), e.beginPath();
		for (let t = 0, n = s.length; t < n; ++t) {
			let n = s[t], r = z(a, [n[0], n[1]]), i = z(a, [n[0], n[3]]), o = z(a, [n[2], n[3]]), c = z(a, [n[2], n[1]]);
			e.moveTo(r[0], r[1]), e.lineTo(i[0], i[1]), e.lineTo(o[0], o[1]), e.lineTo(c[0], c[1]), e.closePath();
		}
		return e.clip(), !0;
	}
	postRender(e, t) {
		let n = t.viewHints, r = !(n[Lr.ANIMATING] || n[Lr.INTERACTING]);
		this.renderedPixelToCoordinateTransform_ = t.pixelToCoordinateTransform.slice(), this.renderedRotation_ = t.viewState.rotation, this.renderedOpacity_ = t.layerStatesArray[t.layerIndex].opacity;
		let i = this.getLayer(), a = i.getRenderMode(), o = e.globalAlpha;
		e.globalAlpha = this.renderedOpacity_;
		let s = i.getDeclutter(), c = s ? Af[a].filter((e) => !Qd.includes(e)) : Af[a], l = t.viewState, u = l.rotation;
		this.layerExtent && this.clipUnrotated(e, t, this.layerExtent);
		let d = i.getSource(), f = d.getTileGridForProjection(l.projection).getZForResolution(l.resolution, d.zDirection), p = this.renderedTiles, m = [], h = [], g = [], _ = F(i), v = !0;
		for (let n = p.length - 1; n >= 0; --n) {
			let a = p[n];
			v &&= !a.getReplayState(i).dirty;
			let o = a.executorGroups[_].filter((e) => e.hasExecutors(c));
			if (o.length === 0) continue;
			let l = this.getTileRenderTransform(a, t), d = a.tileCoord[0], y = !1, b = o[0].getClipCoords(Gr), x = e, S;
			if (b) {
				let e = [
					b[0],
					b[1],
					b[4],
					b[5]
				];
				S = new wu(), x = S.getContext(), f !== d && (y = this.clipTileContext_(x, e, m, h, d, l)), m.push(e), h.push(d);
			}
			for (let n = 0, i = o.length; n < i; ++n) o[n].execute(e, [e.canvas.width, e.canvas.height], l, u, r, c, t.declutter?.[s]);
			y && (x === e ? x.restore() : g[n] = S);
		}
		this.layerExtent && e.restore(), e.globalAlpha = o, this.ready = v, this.tileClipContexts_ = g, t.declutter || this.renderDeferredInternal(t), super.postRender(e, t);
	}
	renderFeature(e, t, n, r, i, a) {
		if (!n) return !1;
		let o = !1;
		if (Array.isArray(n)) for (let s = 0, c = n.length; s < c; ++s) o = _f(r, e, n[s], t, this.boundHandleStyleImageChange_, void 0, i, a) || o;
		else o = _f(r, e, n, t, this.boundHandleStyleImageChange_, void 0, i, a);
		return o;
	}
	tileImageNeedsRender_(e) {
		let t = this.getLayer();
		if (t.getRenderMode() === "vector") return !1;
		let n = e.getReplayState(t), r = t.getRevision(), i = e.wantedResolution;
		return n.renderedTileResolution !== i || n.renderedTileRevision !== r;
	}
	renderTileImage_(e, t) {
		let n = this.getLayer(), r = e.getReplayState(n), i = n.getRevision(), a = e.executorGroups[F(n)];
		r.renderedTileRevision = i;
		let o = e.wrappedTileCoord, s = o[0], c = n.getSource(), l = t.pixelRatio, u = t.viewState.projection, d = c.getTileGridForProjection(u), f = d.getResolution(e.tileCoord[0]), p = t.pixelRatio / e.wantedResolution * f, m = d.getResolution(s), h = e.getContext();
		l = Math.round(Math.max(l, p / l));
		let g = c.getTilePixelSize(s, l, u);
		h.canvas.width = g[0], h.canvas.height = g[1];
		let _ = l / p;
		if (_ !== 1) {
			let e = Jr(this.tmpTransform_);
			Qr(e, _, _), h.setTransform.apply(h, e);
		}
		let v = d.getTileCoordExtent(o, this.tempExtent), y = p / m, b = Jr(this.tmpTransform_);
		Qr(b, y, -y), $r(b, -v[0], -v[3]);
		for (let e = 0, t = a.length; e < t; ++e) a[e].execute(h, [h.canvas.width * _, h.canvas.height * _], b, 0, !0, kf[n.getRenderMode()], null);
		r.renderedTileResolution = e.wantedResolution;
	}
}, Mf = class extends Al {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.preload;
		let n = e.cacheSize === void 0 ? 0 : e.cacheSize;
		delete e.cacheSize, delete t.useInterimTilesOnError, super(t), this.on, this.once, this.un, this.cacheSize_ = n;
		let r = e.renderMode || "hybrid";
		R(r == "hybrid" || r == "vector", "`renderMode` must be `'hybrid'` or `'vector'`"), this.renderMode_ = r, this.setPreload(e.preload ? e.preload : 0), this.setUseInterimTilesOnError(e.useInterimTilesOnError === void 0 || e.useInterimTilesOnError), this.getBackground, this.setBackground;
	}
	createRenderer() {
		return new jf(this, { cacheSize: this.cacheSize_ });
	}
	getFeatures(e) {
		return super.getFeatures(e);
	}
	getFeaturesInExtent(e) {
		return this.getRenderer().getFeaturesInExtent(e);
	}
	getRenderMode() {
		return this.renderMode_;
	}
	getPreload() {
		return this.get(Fu.PRELOAD);
	}
	getUseInterimTilesOnError() {
		return this.get(Fu.USE_INTERIM_TILES_ON_ERROR);
	}
	setPreload(e) {
		this.set(Fu.PRELOAD, e);
	}
	setUseInterimTilesOnError(e) {
		this.set(Fu.USE_INTERIM_TILES_ON_ERROR, e);
	}
}, Nf = class extends Au {
	constructor(e) {
		super(e), this.image = null, this.renderedSourceRevision_ = 0;
	}
	getImage() {
		return this.image ? this.image.getImage() : null;
	}
	prepareFrame(e) {
		let t = e.layerStatesArray[e.layerIndex], n = e.pixelRatio, r = e.viewState, i = r.resolution, a = this.getLayer().getSource(), o = e.viewHints, s = e.extent;
		if (t.extent !== void 0 && (s = lt(s, yr(t.extent, r.projection))), !o[Lr.ANIMATING] && !o[Lr.INTERACTING] && !mt(s)) {
			if (a) {
				!this.getLayer().rendered && this.renderedSourceRevision_ !== a.getRevision() && (this.image = null), this.renderedSourceRevision_ = a.getRevision();
				let e = r.projection, t = a.getImage(s, i, n, e);
				t && (this.loadImage(t) ? this.image = t : t.getState() === J.EMPTY && (this.image = null));
			} else this.image = null;
		}
		return !!this.image;
	}
	getData(e) {
		let t = this.frameState;
		if (!t) return null;
		let n = this.getLayer(), r = z(t.pixelToCoordinateTransform, e.slice()), i = n.getExtent();
		if (i && !Be(i, r)) return null;
		let a = this.image.getExtent(), o = this.image.getImage(), s = I(a), c = Math.floor(o.width * ((r[0] - a[0]) / s));
		if (c < 0 || c >= o.width) return null;
		let l = ct(a), u = Math.floor(o.height * ((a[3] - r[1]) / l));
		return u < 0 || u >= o.height ? null : this.getImageData(o, c, u);
	}
	renderFrame(e, t) {
		let n = this.image, r = n.getExtent(), i = n.getResolution(), [a, o] = Array.isArray(i) ? i : [i, i], s = n.getPixelRatio(), c = e.layerStatesArray[e.layerIndex], l = e.pixelRatio, u = e.viewState, d = u.center, f = u.resolution, p = l * a / (f * s), m = l * o / (f * s);
		this.prepareContainer(e, t);
		let h = this.context.canvas.width, g = this.context.canvas.height, _ = this.getRenderContext(e), v = !1, y = !0;
		if (c.extent) {
			let t = yr(c.extent, u.projection);
			y = pt(t, e.extent), v = y && !Ve(t, e.extent), v && this.clipUnrotated(_, e, t);
		}
		let b = n.getImage(), x = ei(this.tempTransform, h / 2, g / 2, p, m, 0, s * (r[0] - d[0]) / a, s * (d[1] - r[3]) / o);
		this.renderedResolution = o * l / s;
		let S = b.width * x[0], C = b.height * x[3];
		if (this.getLayer().getSource().getInterpolate() || (_.imageSmoothingEnabled = !1), this.preRender(_, e), y && S >= .5 && C >= .5) {
			let e = x[4], t = x[5], n = c.opacity;
			n !== 1 && (_.save(), _.globalAlpha = n), _.drawImage(b, 0, 0, +b.width, +b.height, e, t, S, C), n !== 1 && _.restore();
		}
		return this.postRender(this.context, e), v && _.restore(), _.imageSmoothingEnabled = !0, this.container;
	}
}, Pf = class extends uo {
	constructor(e) {
		e ||= {}, super(e);
	}
}, Ff = class extends Pf {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new Nf(this);
	}
	getData(e) {
		return super.getData(e);
	}
};
//#endregion
//#region node_modules/ol/vec/mat4.js
function If() {
	return [
		1,
		0,
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		0,
		1
	];
}
function Lf(e) {
	return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = 1, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = 1, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, e;
}
function Rf(e, t) {
	return e[0] = t[0], e[1] = t[1], e[4] = t[2], e[5] = t[3], e[12] = t[4], e[13] = t[5], e;
}
function zf(e, t, n, r, i) {
	return i ??= If(), i[0] = e[0] * t, i[1] = e[1] * t, i[2] = e[2] * t, i[3] = e[3] * t, i[4] = e[4] * n, i[5] = e[5] * n, i[6] = e[6] * n, i[7] = e[7] * n, i[8] = e[8] * r, i[9] = e[9] * r, i[10] = e[10] * r, i[11] = e[11] * r, i[12] = e[12], i[13] = e[13], i[14] = e[14], i[15] = e[15], i;
}
function Bf(e, t, n, r, i) {
	i ??= If();
	let a, o, s, c, l, u, d, f, p, m, h, g;
	return e === i ? (i[12] = e[0] * t + e[4] * n + e[8] * r + e[12], i[13] = e[1] * t + e[5] * n + e[9] * r + e[13], i[14] = e[2] * t + e[6] * n + e[10] * r + e[14], i[15] = e[3] * t + e[7] * n + e[11] * r + e[15]) : (a = e[0], o = e[1], s = e[2], c = e[3], l = e[4], u = e[5], d = e[6], f = e[7], p = e[8], m = e[9], h = e[10], g = e[11], i[0] = a, i[1] = o, i[2] = s, i[3] = c, i[4] = l, i[5] = u, i[6] = d, i[7] = f, i[8] = p, i[9] = m, i[10] = h, i[11] = g, i[12] = a * t + l * n + p * r + e[12], i[13] = o * t + u * n + m * r + e[13], i[14] = s * t + d * n + h * r + e[14], i[15] = c * t + f * n + g * r + e[15]), i;
}
function Vf(e, t, n) {
	n ??= If();
	let r = Math.cos(t), i = Math.sin(t), a = r, o = -i, s = i, c = r, l = e[0], u = e[1], d = e[2], f = e[3], p = e[4], m = e[5], h = e[6], g = e[7];
	return n[0] = a * l + o * p, n[1] = a * u + o * m, n[2] = a * d + o * h, n[3] = a * f + o * g, n[4] = s * l + c * p, n[5] = s * u + c * m, n[6] = s * d + c * h, n[7] = s * f + c * g, n !== e && (n[8] = e[8], n[9] = e[9], n[10] = e[10], n[11] = e[11], n[12] = e[12], n[13] = e[13], n[14] = e[14], n[15] = e[15]), n;
}
//#endregion
//#region node_modules/ol/webgl.js
var Hf = 34962, Uf = 34963, Wf = 35040, Gf = 35044, Kf = 35048, qf = 5121, Jf = 5123, Yf = 5125, Xf = 5126, Zf = [
	"experimental-webgl",
	"webgl",
	"webkit-3d",
	"moz-webgl"
];
function Qf(e, t) {
	t = Object.assign({
		preserveDrawingBuffer: !0,
		antialias: !f
	}, t);
	let n = Zf.length;
	for (let r = 0; r < n; ++r) try {
		let n = e.getContext(Zf[r], t);
		if (n) return n;
	} catch {}
	return null;
}
//#endregion
//#region node_modules/ol/webgl/Buffer.js
var $f = {
	STATIC_DRAW: Gf,
	STREAM_DRAW: Wf,
	DYNAMIC_DRAW: Kf
}, ep = class {
	constructor(e, t) {
		this.array_ = null, this.type_ = e, R(e === 34962 || e === 34963, "A `WebGLArrayBuffer` must either be of type `ELEMENT_ARRAY_BUFFER` or `ARRAY_BUFFER`"), this.usage_ = t === void 0 ? $f.STATIC_DRAW : t;
	}
	ofSize(e) {
		return this.array_ = new (tp(this.type_))(e), this;
	}
	fromArray(e) {
		return this.array_ = tp(this.type_).from(e), this;
	}
	fromArrayBuffer(e) {
		return this.array_ = new (tp(this.type_))(e), this;
	}
	getType() {
		return this.type_;
	}
	getArray() {
		return this.array_;
	}
	setArray(e) {
		let t = tp(this.type_);
		if (!(e instanceof t)) throw Error(`Expected ${t}`);
		this.array_ = e;
	}
	getUsage() {
		return this.usage_;
	}
	getSize() {
		return this.array_ ? this.array_.length : 0;
	}
};
function tp(e) {
	switch (e) {
		case Hf: return Float32Array;
		case Uf: return Uint32Array;
		default: return Float32Array;
	}
}
//#endregion
//#region node_modules/ol/webgl/ContextEventType.js
var np = {
	LOST: "webglcontextlost",
	RESTORED: "webglcontextrestored"
}, rp = "\n  precision mediump float;\n\n  attribute vec2 a_position;\n  varying vec2 v_texCoord;\n  varying vec2 v_screenCoord;\n\n  uniform vec2 u_screenSize;\n\n  void main() {\n    v_texCoord = a_position * 0.5 + 0.5;\n    v_screenCoord = v_texCoord * u_screenSize;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n  }\n", ip = "\n  precision mediump float;\n\n  uniform sampler2D u_image;\n  uniform float u_opacity;\n\n  varying vec2 v_texCoord;\n\n  void main() {\n    gl_FragColor = texture2D(u_image, v_texCoord) * u_opacity;\n  }\n", ap = class {
	constructor(e) {
		this.gl_ = e.webGlContext;
		let t = this.gl_;
		this.scaleRatio_ = e.scaleRatio || 1, this.renderTargetTexture_ = t.createTexture(), this.renderTargetTextureSize_ = null, this.frameBuffer_ = t.createFramebuffer(), this.depthBuffer_ = t.createRenderbuffer();
		let n = t.createShader(t.VERTEX_SHADER);
		t.shaderSource(n, e.vertexShader || rp), t.compileShader(n);
		let r = t.createShader(t.FRAGMENT_SHADER);
		if (t.shaderSource(r, e.fragmentShader || ip), t.compileShader(r), !t.getShaderParameter(r, t.COMPILE_STATUS)) {
			let e = `Fragment shader compilation failed: ${t.getShaderInfoLog(r)}`;
			throw Error(e);
		}
		this.renderTargetProgram_ = t.createProgram(), t.attachShader(this.renderTargetProgram_, n), t.attachShader(this.renderTargetProgram_, r), t.linkProgram(this.renderTargetProgram_), this.renderTargetVerticesBuffer_ = t.createBuffer(), t.bindBuffer(t.ARRAY_BUFFER, this.renderTargetVerticesBuffer_), t.bufferData(t.ARRAY_BUFFER, new Float32Array([
			-1,
			-1,
			1,
			-1,
			-1,
			1,
			1,
			-1,
			1,
			1,
			-1,
			1
		]), t.STATIC_DRAW), this.renderTargetAttribLocation_ = t.getAttribLocation(this.renderTargetProgram_, "a_position"), this.renderTargetUniformLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_screenSize"), this.renderTargetOpacityLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_opacity"), this.renderTargetTextureLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_image"), this.uniforms_ = [], e.uniforms && Object.keys(e.uniforms).forEach((n) => {
			this.uniforms_.push({
				value: e.uniforms[n],
				location: t.getUniformLocation(this.renderTargetProgram_, n)
			});
		});
	}
	getRenderTargetTexture() {
		return this.renderTargetTexture_;
	}
	getGL() {
		return this.gl_;
	}
	init(e) {
		let t = this.getGL(), n = [t.drawingBufferWidth * this.scaleRatio_, t.drawingBufferHeight * this.scaleRatio_];
		if (t.bindFramebuffer(t.FRAMEBUFFER, this.getFrameBuffer()), t.bindRenderbuffer(t.RENDERBUFFER, this.getDepthBuffer()), t.viewport(0, 0, n[0], n[1]), !this.renderTargetTextureSize_ || this.renderTargetTextureSize_[0] !== n[0] || this.renderTargetTextureSize_[1] !== n[1]) {
			this.renderTargetTextureSize_ = n;
			let e = t.RGBA, r = t.RGBA, i = t.UNSIGNED_BYTE;
			t.bindTexture(t.TEXTURE_2D, this.renderTargetTexture_), t.texImage2D(t.TEXTURE_2D, 0, e, n[0], n[1], 0, r, i, null), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, this.renderTargetTexture_, 0), t.renderbufferStorage(t.RENDERBUFFER, t.DEPTH_COMPONENT16, n[0], n[1]), t.framebufferRenderbuffer(t.FRAMEBUFFER, t.DEPTH_ATTACHMENT, t.RENDERBUFFER, this.depthBuffer_);
		}
	}
	apply(e, t, n, r) {
		let i = this.getGL(), a = e.size;
		if (i.bindFramebuffer(i.FRAMEBUFFER, t ? t.getFrameBuffer() : null), i.activeTexture(i.TEXTURE0), i.bindTexture(i.TEXTURE_2D, this.renderTargetTexture_), !t) {
			let t = F(i.canvas);
			if (!e.renderTargets[t]) {
				let n = i.getContextAttributes();
				n && n.preserveDrawingBuffer && (i.clearColor(0, 0, 0, 0), i.clearDepth(1), i.clear(i.COLOR_BUFFER_BIT | i.DEPTH_BUFFER_BIT)), e.renderTargets[t] = !0;
			}
		}
		i.disable(i.DEPTH_TEST), i.enable(i.BLEND), i.blendFunc(i.ONE, i.ONE_MINUS_SRC_ALPHA), i.viewport(0, 0, i.drawingBufferWidth, i.drawingBufferHeight), i.bindBuffer(i.ARRAY_BUFFER, this.renderTargetVerticesBuffer_), i.useProgram(this.renderTargetProgram_), i.enableVertexAttribArray(this.renderTargetAttribLocation_), i.vertexAttribPointer(this.renderTargetAttribLocation_, 2, i.FLOAT, !1, 0, 0), i.uniform2f(this.renderTargetUniformLocation_, a[0], a[1]), i.uniform1i(this.renderTargetTextureLocation_, 0);
		let o = e.layerStatesArray[e.layerIndex].opacity;
		i.uniform1f(this.renderTargetOpacityLocation_, o), this.applyUniforms(e), n && n(i, e), i.drawArrays(i.TRIANGLES, 0, 6), r && r(i, e);
	}
	getFrameBuffer() {
		return this.frameBuffer_;
	}
	getDepthBuffer() {
		return this.depthBuffer_;
	}
	applyUniforms(e) {
		let t = this.getGL(), n, r = 1;
		this.uniforms_.forEach(function(i) {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof ImageData) i.texture ||= t.createTexture(), t.activeTexture(t[`TEXTURE${r}`]), t.bindTexture(t.TEXTURE_2D, i.texture), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), n instanceof ImageData ? t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, n.width, n.height, 0, t.UNSIGNED_BYTE, new Uint8Array(n.data)) : t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n), t.uniform1i(i.location, r++);
			else if (Array.isArray(n)) switch (n.length) {
				case 2:
					t.uniform2f(i.location, n[0], n[1]);
					return;
				case 3:
					t.uniform3f(i.location, n[0], n[1], n[2]);
					return;
				case 4:
					t.uniform4f(i.location, n[0], n[1], n[2], n[3]);
					return;
				case 16:
					t.uniformMatrix4fv(i.location, !1, n);
					return;
				default: return;
			}
			else typeof n == "number" && t.uniform1f(i.location, n);
		});
	}
}, op = {
	PROJECTION_MATRIX: "u_projectionMatrix",
	INVERT_PROJECTION_MATRIX: "u_invertProjectionMatrix",
	TIME: "u_time",
	ZOOM: "u_zoom",
	RESOLUTION: "u_resolution",
	ROTATION: "u_rotation",
	VIEWPORT_SIZE_PX: "u_viewportSizePx",
	PIXEL_RATIO: "u_pixelRatio",
	HIT_DETECTION: "u_hitDetection"
}, sp = {
	UNSIGNED_BYTE: qf,
	UNSIGNED_SHORT: Jf,
	UNSIGNED_INT: Yf,
	FLOAT: Xf
}, cp = {};
function lp(e) {
	return "shared/" + e;
}
var up = 0;
function dp() {
	let e = "unique/" + up;
	return up += 1, e;
}
function fp(e) {
	let t = cp[e];
	if (!t) {
		let n = document.createElement("canvas");
		n.width = 1, n.height = 1, n.style.position = "absolute", n.style.left = "0", t = {
			users: 0,
			context: Qf(n)
		}, cp[e] = t;
	}
	return t.users += 1, t.context;
}
function pp(e) {
	let t = cp[e];
	if (!t || (--t.users, t.users > 0)) return;
	let n = t.context, r = n.getExtension("WEBGL_lose_context");
	r && r.loseContext();
	let i = n.canvas;
	i.width = 1, i.height = 1, delete cp[e];
}
var mp = class extends ie {
	constructor(e) {
		super(), e ||= {}, this.boundHandleWebGLContextLost_ = this.handleWebGLContextLost.bind(this), this.boundHandleWebGLContextRestored_ = this.handleWebGLContextRestored.bind(this), this.canvasCacheKey_ = e.canvasCacheKey ? lp(e.canvasCacheKey) : dp(), this.gl_ = fp(this.canvasCacheKey_), this.bufferCache_ = {}, this.extensionCache_ = {}, this.currentProgram_ = null, this.needsToBeRecreated_ = !1;
		let t = this.gl_.canvas;
		t.addEventListener(np.LOST, this.boundHandleWebGLContextLost_), t.addEventListener(np.RESTORED, this.boundHandleWebGLContextRestored_), this.tmpMat4_ = If(), this.uniformLocationsByProgram_ = {}, this.attribLocationsByProgram_ = {}, this.uniforms_ = [], e.uniforms && this.setUniforms(e.uniforms), this.postProcessPasses_ = e.postProcesses?.length ? e.postProcesses.map((e) => new ap({
			webGlContext: this.gl_,
			scaleRatio: e.scaleRatio,
			vertexShader: e.vertexShader,
			fragmentShader: e.fragmentShader,
			uniforms: e.uniforms
		})) : [new ap({ webGlContext: this.gl_ })], this.shaderCompileErrors_ = null, this.startTime_ = Date.now(), this.maxAttributeCount_ = this.gl_.getParameter(this.gl_.MAX_VERTEX_ATTRIBS);
	}
	setUniforms(e) {
		this.uniforms_ = [], this.addUniforms(e);
	}
	addUniforms(e) {
		for (let t in e) this.uniforms_.push({
			name: t,
			value: e[t]
		});
	}
	canvasCacheKeyMatches(e) {
		return this.canvasCacheKey_ === lp(e);
	}
	getExtension(e) {
		if (e in this.extensionCache_) return this.extensionCache_[e];
		let t = this.gl_.getExtension(e);
		return this.extensionCache_[e] = t, t;
	}
	getInstancedRenderingExtension_() {
		let e = this.getExtension("ANGLE_instanced_arrays");
		return R(!!e, "WebGL extension 'ANGLE_instanced_arrays' is required for vector rendering"), e;
	}
	bindBuffer(e) {
		let t = this.gl_, n = F(e), r = this.bufferCache_[n];
		r || (r = {
			buffer: e,
			webGlBuffer: t.createBuffer()
		}, this.bufferCache_[n] = r), t.bindBuffer(e.getType(), r.webGlBuffer);
	}
	flushBufferData(e) {
		let t = this.gl_;
		this.bindBuffer(e), t.bufferData(e.getType(), e.getArray(), e.getUsage());
	}
	deleteBuffer(e) {
		let t = F(e);
		delete this.bufferCache_[t];
	}
	disposeInternal() {
		let e = this.gl_.canvas;
		e.removeEventListener(np.LOST, this.boundHandleWebGLContextLost_), e.removeEventListener(np.RESTORED, this.boundHandleWebGLContextRestored_), pp(this.canvasCacheKey_), delete this.gl_;
	}
	prepareDraw(e, t, n) {
		let r = this.gl_, i = this.getCanvas(), a = e.size, o = e.pixelRatio;
		(i.width !== a[0] * o || i.height !== a[1] * o) && (i.width = a[0] * o, i.height = a[1] * o, i.style.width = a[0] + "px", i.style.height = a[1] + "px");
		for (let t = this.postProcessPasses_.length - 1; t >= 0; t--) this.postProcessPasses_[t].init(e);
		r.bindTexture(r.TEXTURE_2D, null), r.clearColor(0, 0, 0, 0), r.depthRange(0, 1), r.clearDepth(1), r.clear(r.COLOR_BUFFER_BIT | r.DEPTH_BUFFER_BIT), r.enable(r.BLEND), r.blendFunc(r.ONE, t ? r.ZERO : r.ONE_MINUS_SRC_ALPHA), n ? (r.enable(r.DEPTH_TEST), r.depthFunc(r.LEQUAL)) : r.disable(r.DEPTH_TEST);
	}
	bindFrameBuffer(e, t) {
		let n = this.getGL();
		n.bindFramebuffer(n.FRAMEBUFFER, e), t && n.framebufferTexture2D(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0, n.TEXTURE_2D, t, 0);
	}
	bindInitialFrameBuffer() {
		let e = this.getGL(), t = this.postProcessPasses_[0].getFrameBuffer();
		e.bindFramebuffer(e.FRAMEBUFFER, t);
		let n = this.postProcessPasses_[0].getRenderTargetTexture();
		e.framebufferTexture2D(e.FRAMEBUFFER, e.COLOR_ATTACHMENT0, e.TEXTURE_2D, n, 0);
	}
	bindTexture(e, t, n) {
		let r = this.gl_;
		r.activeTexture(r.TEXTURE0 + t), r.bindTexture(r.TEXTURE_2D, e), r.uniform1i(this.getUniformLocation(n), t);
	}
	bindAttribute(e, t, n) {
		let r = this.getGL();
		this.bindBuffer(e);
		let i = this.getAttributeLocation(t);
		r.enableVertexAttribArray(i), r.vertexAttribPointer(i, n, r.FLOAT, !1, 0, 0);
	}
	prepareDrawToRenderTarget(e, t, n, r) {
		let i = this.gl_, a = t.getSize();
		i.bindFramebuffer(i.FRAMEBUFFER, t.getFramebuffer()), i.bindRenderbuffer(i.RENDERBUFFER, t.getDepthbuffer()), i.viewport(0, 0, a[0], a[1]), i.bindTexture(i.TEXTURE_2D, t.getTexture()), i.clearColor(0, 0, 0, 0), i.depthRange(0, 1), i.clearDepth(1), i.clear(i.COLOR_BUFFER_BIT | i.DEPTH_BUFFER_BIT), i.enable(i.BLEND), i.blendFunc(i.ONE, n ? i.ZERO : i.ONE_MINUS_SRC_ALPHA), r ? (i.enable(i.DEPTH_TEST), i.depthFunc(i.LEQUAL)) : i.disable(i.DEPTH_TEST);
	}
	drawElements(e, t) {
		let n = this.gl_;
		this.getExtension("OES_element_index_uint");
		let r = n.UNSIGNED_INT, i = t - e, a = e * 4;
		n.drawElements(n.TRIANGLES, i, r, a);
	}
	drawElementsInstanced(e, t, n) {
		let r = this.gl_;
		this.getExtension("OES_element_index_uint");
		let i = this.getInstancedRenderingExtension_(), a = r.UNSIGNED_INT, o = t - e, s = e * 4;
		i.drawElementsInstancedANGLE(r.TRIANGLES, o, a, s, n);
		for (let e = 0; e < this.maxAttributeCount_; e++) i.vertexAttribDivisorANGLE(e, 0);
	}
	finalizeDraw(e, t, n) {
		for (let r = 0, i = this.postProcessPasses_.length; r < i; r++) r === i - 1 ? this.postProcessPasses_[r].apply(e, null, t, n) : this.postProcessPasses_[r].apply(e, this.postProcessPasses_[r + 1]);
	}
	getCanvas() {
		return this.gl_.canvas;
	}
	getGL() {
		return this.gl_;
	}
	applyFrameState(e) {
		let t = e.size, n = e.viewState.rotation, r = e.pixelRatio;
		this.setUniformFloatValue(op.TIME, (Date.now() - this.startTime_) * .001), this.setUniformFloatValue(op.ZOOM, e.viewState.zoom), this.setUniformFloatValue(op.RESOLUTION, e.viewState.resolution), this.setUniformFloatValue(op.PIXEL_RATIO, r), this.setUniformFloatVec2(op.VIEWPORT_SIZE_PX, [t[0], t[1]]), this.setUniformFloatValue(op.ROTATION, n);
	}
	applyHitDetectionUniform(e) {
		let t = this.getUniformLocation(op.HIT_DETECTION);
		this.getGL().uniform1i(t, +!!e), e && this.setUniformFloatValue(op.PIXEL_RATIO, .5);
	}
	applyUniforms(e) {
		let t = this.gl_, n, r = 0;
		this.uniforms_.forEach((i) => {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof HTMLImageElement || n instanceof ImageData || n instanceof WebGLTexture) {
				n instanceof WebGLTexture && !i.texture ? (i.prevValue = void 0, i.texture = n) : i.texture ||= (i.prevValue = void 0, t.createTexture()), this.bindTexture(i.texture, r, i.name), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE);
				let e = !(n instanceof HTMLImageElement) || n.complete;
				!(n instanceof WebGLTexture) && e && i.prevValue !== n && (i.prevValue = n, t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n)), r++;
			} else if (Array.isArray(n) && n.length === 6) this.setUniformMatrixValue(i.name, Rf(this.tmpMat4_, n));
			else if (Array.isArray(n) && n.length <= 4) switch (n.length) {
				case 2:
					t.uniform2f(this.getUniformLocation(i.name), n[0], n[1]);
					return;
				case 3:
					t.uniform3f(this.getUniformLocation(i.name), n[0], n[1], n[2]);
					return;
				case 4:
					t.uniform4f(this.getUniformLocation(i.name), n[0], n[1], n[2], n[3]);
					return;
				default: return;
			}
			else typeof n == "number" && t.uniform1f(this.getUniformLocation(i.name), n);
		});
	}
	useProgram(e, t) {
		this.disableAllAttributes_(), this.gl_.useProgram(e), this.currentProgram_ = e, t && (this.applyFrameState(t), this.applyUniforms(t));
	}
	compileShader(e, t) {
		let n = this.gl_, r = n.createShader(t);
		return n.shaderSource(r, e), n.compileShader(r), r;
	}
	getProgram(e, t) {
		let n = this.gl_, r = this.compileShader(e, n.FRAGMENT_SHADER), i = this.compileShader(t, n.VERTEX_SHADER), a = n.createProgram();
		if (n.attachShader(a, r), n.attachShader(a, i), n.linkProgram(a), !n.getShaderParameter(r, n.COMPILE_STATUS)) {
			let e = `Fragment shader compilation failed: ${n.getShaderInfoLog(r)}`;
			throw Error(e);
		}
		if (n.deleteShader(r), !n.getShaderParameter(i, n.COMPILE_STATUS)) {
			let e = `Vertex shader compilation failed: ${n.getShaderInfoLog(i)}`;
			throw Error(e);
		}
		if (n.deleteShader(i), !n.getProgramParameter(a, n.LINK_STATUS)) {
			let e = `GL program linking failed: ${n.getProgramInfoLog(a)}`;
			throw Error(e);
		}
		return a;
	}
	getUniformLocation(e) {
		let t = F(this.currentProgram_);
		return this.uniformLocationsByProgram_[t] === void 0 && (this.uniformLocationsByProgram_[t] = {}), this.uniformLocationsByProgram_[t][e] === void 0 && (this.uniformLocationsByProgram_[t][e] = this.gl_.getUniformLocation(this.currentProgram_, e)), this.uniformLocationsByProgram_[t][e];
	}
	getAttributeLocation(e) {
		let t = F(this.currentProgram_);
		return this.attribLocationsByProgram_[t] === void 0 && (this.attribLocationsByProgram_[t] = {}), this.attribLocationsByProgram_[t][e] === void 0 && (this.attribLocationsByProgram_[t][e] = this.gl_.getAttribLocation(this.currentProgram_, e)), this.attribLocationsByProgram_[t][e];
	}
	makeProjectionTransform(e, t, n) {
		let r = e.size, i = n ? 0 : e.viewState.rotation, a = e.viewState.resolution, o = e.viewState.center;
		return ei(t, 0, 0, 2 / (a * r[0]), 2 / (a * r[1]), -i, -o[0], -o[1]), t;
	}
	setUniformFloatValue(e, t) {
		this.gl_.uniform1f(this.getUniformLocation(e), t);
	}
	setUniformFloatVec2(e, t) {
		this.gl_.uniform2fv(this.getUniformLocation(e), t);
	}
	setUniformFloatVec4(e, t) {
		this.gl_.uniform4fv(this.getUniformLocation(e), t);
	}
	setUniformMatrixValue(e, t) {
		this.gl_.uniformMatrix4fv(this.getUniformLocation(e), !1, t);
	}
	disableAllAttributes_() {
		for (let e = 0; e < this.maxAttributeCount_; e++) this.gl_.disableVertexAttribArray(e);
	}
	enableAttributeArray_(e, t, n, r, i, a) {
		let o = this.getAttributeLocation(e);
		o < 0 || (this.gl_.enableVertexAttribArray(o), this.gl_.vertexAttribPointer(o, t, n, !1, r, i), a && this.getInstancedRenderingExtension_().vertexAttribDivisorANGLE(o, 1));
	}
	enableAttributes_(e, t) {
		let n = hp(e), r = 0;
		for (let i = 0; i < e.length; i++) {
			let a = e[i];
			a.name && this.enableAttributeArray_(a.name, a.size, a.type || 5126, n, r, t), r += a.size * gp(a.type);
		}
	}
	enableAttributes(e) {
		this.enableAttributes_(e, !1);
	}
	enableAttributesInstanced(e) {
		this.enableAttributes_(e, !0);
	}
	handleWebGLContextLost(e) {
		k(this.bufferCache_), this.currentProgram_ = null, e.preventDefault();
	}
	handleWebGLContextRestored() {
		this.needsToBeRecreated_ = !0;
	}
	needsToBeRecreated() {
		return this.needsToBeRecreated_;
	}
	createTexture(e, t, n, r) {
		let i = this.gl_;
		n ||= i.createTexture();
		let a = r ? i.NEAREST : i.LINEAR;
		i.bindTexture(i.TEXTURE_2D, n), i.texParameteri(i.TEXTURE_2D, i.TEXTURE_MIN_FILTER, a), i.texParameteri(i.TEXTURE_2D, i.TEXTURE_MAG_FILTER, a), i.texParameteri(i.TEXTURE_2D, i.TEXTURE_WRAP_S, i.CLAMP_TO_EDGE), i.texParameteri(i.TEXTURE_2D, i.TEXTURE_WRAP_T, i.CLAMP_TO_EDGE);
		let o = i.RGBA, s = i.RGBA, c = i.UNSIGNED_BYTE;
		return t instanceof Uint8Array ? i.texImage2D(i.TEXTURE_2D, 0, o, e[0], e[1], 0, s, c, t) : t ? i.texImage2D(i.TEXTURE_2D, 0, o, s, c, t) : i.texImage2D(i.TEXTURE_2D, 0, o, e[0], e[1], 0, s, c, null), n;
	}
};
function hp(e) {
	let t = 0;
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		t += r.size * gp(r.type);
	}
	return t;
}
function gp(e) {
	switch (e) {
		case sp.UNSIGNED_BYTE: return Uint8Array.BYTES_PER_ELEMENT;
		case sp.UNSIGNED_SHORT: return Uint16Array.BYTES_PER_ELEMENT;
		case sp.UNSIGNED_INT: return Uint32Array.BYTES_PER_ELEMENT;
		case sp.FLOAT:
		default: return Float32Array.BYTES_PER_ELEMENT;
	}
}
//#endregion
//#region node_modules/ol/renderer/webgl/Layer.js
var _p = class e extends Eu {
	constructor(e, t) {
		super(e), t ||= {}, this.inversePixelTransform_ = qr(), this.postProcesses_ = t.postProcesses, this.uniforms_ = t.uniforms, this.helper, this.onMapChanged_ = () => {
			this.clearCache(), this.removeHelper();
		}, e.addChangeListener(B.MAP, this.onMapChanged_), this.dispatchPreComposeEvent = this.dispatchPreComposeEvent.bind(this), this.dispatchPostComposeEvent = this.dispatchPostComposeEvent.bind(this);
	}
	dispatchPreComposeEvent(e, t) {
		let n = this.getLayer();
		if (n.hasListener(lo.PRECOMPOSE)) {
			let r = new Ml(lo.PRECOMPOSE, void 0, t, e);
			n.dispatchEvent(r);
		}
	}
	dispatchPostComposeEvent(e, t) {
		let n = this.getLayer();
		if (n.hasListener(lo.POSTCOMPOSE)) {
			let r = new Ml(lo.POSTCOMPOSE, void 0, t, e);
			n.dispatchEvent(r);
		}
	}
	reset(e) {
		this.uniforms_ = e.uniforms, this.helper && this.helper.setUniforms(this.uniforms_);
	}
	removeHelper() {
		this.helper && (this.helper.dispose(), delete this.helper);
	}
	prepareFrame(t) {
		if (this.getLayer().getRenderSource()) {
			let n = !0, r = -1, i;
			for (let a = 0, o = t.layerStatesArray.length; a < o; a++) {
				let o = t.layerStatesArray[a].layer, s = o.getRenderer();
				if (!(s instanceof e)) {
					n = !0;
					continue;
				}
				let c = o.getClassName();
				if ((n || c !== i) && (r += 1, n = !1), i = c, s === this) break;
			}
			let a = "map/" + t.mapId + "/group/" + r;
			(!this.helper || !this.helper.canvasCacheKeyMatches(a) || this.helper.needsToBeRecreated()) && (this.removeHelper(), this.helper = new mp({
				postProcesses: this.postProcesses_,
				uniforms: this.uniforms_,
				canvasCacheKey: a
			}), i && (this.helper.getCanvas().className = i), this.afterHelperCreated());
		}
		return this.prepareFrameInternal(t);
	}
	afterHelperCreated() {}
	prepareFrameInternal(e) {
		return !0;
	}
	clearCache() {}
	setPostProcesses(e) {
		this.postProcesses_ = e, this.removeHelper();
	}
	getPostProcesses() {
		return this.postProcesses_;
	}
	disposeInternal() {
		this.clearCache(), this.removeHelper(), this.getLayer()?.removeChangeListener(B.MAP, this.onMapChanged_), super.disposeInternal();
	}
	dispatchRenderEvent_(e, t, n) {
		let r = this.getLayer();
		if (r.hasListener(e)) {
			ei(this.inversePixelTransform_, 0, 0, n.pixelRatio, -n.pixelRatio, 0, 0, -n.size[1]);
			let i = new Ml(e, this.inversePixelTransform_, n, t);
			r.dispatchEvent(i);
		}
	}
	preRender(e, t) {
		this.dispatchRenderEvent_(lo.PRERENDER, e, t);
	}
	postRender(e, t) {
		this.dispatchRenderEvent_(lo.POSTRENDER, e, t);
	}
}, vp = {
	...op,
	TILE_TRANSFORM: "u_tileTransform",
	TRANSITION_ALPHA: "u_transitionAlpha",
	DEPTH: "u_depth",
	RENDER_EXTENT: "u_renderExtent",
	GLOBAL_ALPHA: "u_globalAlpha",
	TILE_TEXTURE_ARRAY: "u_tileTextures",
	TEXTURE_PIXEL_WIDTH: "u_texturePixelWidth",
	TEXTURE_PIXEL_HEIGHT: "u_texturePixelHeight",
	TEXTURE_RESOLUTION: "u_textureResolution"
};
({ TEXTURE_COORD: "a_textureCoord" }).TEXTURE_COORD, sp.FLOAT;
//#endregion
//#region node_modules/ol/webgl/PaletteTexture.js
var yp = class {
	constructor(e, t) {
		this.name = e, this.data = t, this.texture_ = null;
	}
	getTexture(e) {
		if (!this.texture_) {
			let t = e.createTexture();
			e.bindTexture(e.TEXTURE_2D, t), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, this.data.length / 4, 1, 0, e.RGBA, e.UNSIGNED_BYTE, this.data), this.texture_ = t;
		}
		return this.texture_;
	}
	delete(e) {
		this.texture_ && e.deleteTexture(this.texture_), this.texture_ = null;
	}
};
//#endregion
//#region node_modules/ol/expr/gpu.js
function bp(e, t) {
	return `operator_${e}_${Object.keys(t.functions).length}`;
}
function xp(e) {
	let t = e.toString();
	return t.includes(".") ? t : t + ".0";
}
function Sp(e) {
	if (e.length < 2 || e.length > 4) throw Error("`formatArray` can only output `vec2`, `vec3` or `vec4` arrays.");
	return `vec${e.length}(${e.map(xp).join(", ")})`;
}
function Cp(e) {
	let t = $o(e), n = t.length > 3 ? t[3] : 1;
	return Sp([
		t[0] / 255,
		t[1] / 255,
		t[2] / 255,
		n
	]);
}
function wp(e) {
	return Sp(rs(e));
}
var Tp = {}, Ep = 0;
function Dp(e) {
	return e in Tp || (Tp[e] = Ep++), Tp[e];
}
function Op(e) {
	return xp(Dp(e));
}
function kp(e) {
	return "u_var_" + e;
}
function Ap(e) {
	return {
		variables: /* @__PURE__ */ new Map(),
		properties: /* @__PURE__ */ new Map(),
		functions: {},
		bandCount: 0,
		featureId: !1,
		geometryType: !1,
		inputVariables: e
	};
}
var jp = "getBandValue", Mp = "u_paletteTextures", Np = "featureId", Pp = "geometryType", Fp = -9999999;
function Ip(e, t, n, r) {
	let i = ys(e, t, n);
	return r.properties = new Map([...r.properties, ...n.properties]), r.variables = new Map([...r.variables, ...n.variables]), Rp(i, t, r);
}
function X(e) {
	return (t, n, r) => {
		let i = n.args.length, a = Array(i);
		for (let e = 0; e < i; ++e) a[e] = Rp(n.args[e], r, t);
		return e(a, t);
	};
}
var Lp = {
	[W.Get]: (e, t) => {
		let n = "a_prop_" + t.args[0].value;
		return hs(t.type, as) && (n = `(${n} > 0.0)`), n;
	},
	[W.Id]: (e) => (e.featureId = !0, "a_featureId"),
	[W.GeometryType]: (e) => (e.geometryType = !0, "a_geometryType"),
	[W.LineMetric]: () => "currentLineMetric",
	[W.Var]: (e, t) => {
		let n = t.args[0].value, r = kp(n);
		return hs(t.type, as) && (r = `(${r} > 0.0)`), r;
	},
	[W.Has]: (e, t) => `(a_prop_${t.args[0].value} != ${xp(Fp)})`,
	[W.Resolution]: () => "u_resolution",
	[W.Zoom]: () => "u_zoom",
	[W.Time]: () => "u_time",
	[W.Any]: X((e) => `(${e.join(" || ")})`),
	[W.All]: X((e) => `(${e.join(" && ")})`),
	[W.Not]: X(([e]) => `(!${e})`),
	[W.Equal]: X(([e, t]) => `(${e} == ${t})`),
	[W.NotEqual]: X(([e, t]) => `(${e} != ${t})`),
	[W.GreaterThan]: X(([e, t]) => `(${e} > ${t})`),
	[W.GreaterThanOrEqualTo]: X(([e, t]) => `(${e} >= ${t})`),
	[W.LessThan]: X(([e, t]) => `(${e} < ${t})`),
	[W.LessThanOrEqualTo]: X(([e, t]) => `(${e} <= ${t})`),
	[W.Multiply]: X((e) => `(${e.join(" * ")})`),
	[W.Divide]: X(([e, t]) => `(${e} / ${t})`),
	[W.Add]: X((e) => `(${e.join(" + ")})`),
	[W.Subtract]: X(([e, t]) => `(${e} - ${t})`),
	[W.Clamp]: X(([e, t, n]) => `clamp(${e}, ${t}, ${n})`),
	[W.Mod]: X(([e, t]) => `mod(${e}, ${t})`),
	[W.Pow]: X(([e, t]) => `pow(${e}, ${t})`),
	[W.Abs]: X(([e]) => `abs(${e})`),
	[W.Floor]: X(([e]) => `floor(${e})`),
	[W.Ceil]: X(([e]) => `ceil(${e})`),
	[W.Round]: X(([e]) => `floor(${e} + 0.5)`),
	[W.Sin]: X(([e]) => `sin(${e})`),
	[W.Cos]: X(([e]) => `cos(${e})`),
	[W.Atan]: X(([e, t]) => t === void 0 ? `atan(${e})` : `atan(${e}, ${t})`),
	[W.Sqrt]: X(([e]) => `sqrt(${e})`),
	[W.Match]: X((e) => {
		let t = e[0], n = e[e.length - 1], r = null;
		for (let i = e.length - 3; i >= 1; i -= 2) r = `(${t} == ${e[i]} ? ${e[i + 1]} : ${r || n})`;
		return r;
	}),
	[W.Between]: X(([e, t, n]) => `(${e} >= ${t} && ${e} <= ${n})`),
	[W.Interpolate]: X(([e, t, ...n]) => {
		let r = "";
		for (let i = 0; i < n.length - 2; i += 2) {
			let a = n[i], o = r || n[i + 1], s = n[i + 2], c = n[i + 3], l;
			l = e === xp(1) ? `(${t} - ${a}) / (${s} - ${a})` : `(pow(${e}, (${t} - ${a})) - 1.0) / (pow(${e}, (${s} - ${a})) - 1.0)`, r = `mix(${o}, ${c}, clamp(${l}, 0.0, 1.0))`;
		}
		return r;
	}),
	[W.Case]: X((e) => {
		let t = e[e.length - 1], n = null;
		for (let r = e.length - 3; r >= 0; r -= 2) n = `(${e[r]} ? ${e[r + 1]} : ${n || t})`;
		return n;
	}),
	[W.In]: X(([e, ...t], n) => {
		let r = bp("in", n), i = [];
		for (let e = 0; e < t.length; e += 1) i.push(`  if (inputValue == ${t[e]}) { return true; }`);
		return n.functions[r] = `bool ${r}(float inputValue) {
${i.join("\n")}
  return false;
}`, `${r}(${e})`;
	}),
	[W.Array]: X((e) => `vec${e.length}(${e.join(", ")})`),
	[W.Color]: X((e) => {
		if (e.length === 1) return `vec4(vec3(${e[0]} / 255.0), 1.0)`;
		if (e.length === 2) return `vec4(vec3(${e[0]} / 255.0), ${e[1]})`;
		let t = e.slice(0, 3).map((e) => `${e} / 255.0`);
		if (e.length === 3) return `vec4(${t.join(", ")}, 1.0)`;
		let n = e[3];
		return `vec4(${t.join(", ")}, ${n})`;
	}),
	[W.Band]: X(([e, t, n], r) => {
		if (!(jp in r.functions)) {
			let e = "", t = r.bandCount || 1;
			for (let n = 0; n < t; n++) {
				let r = Math.floor(n / 4), i = n % 4;
				n === t - 1 && i === 1 && (i = 3);
				let a = `${vp.TILE_TEXTURE_ARRAY}[${r}]`;
				e += `  if (band == ${n + 1}.0) {
    return texture2D(${a}, v_textureCoord + vec2(dx, dy))[${i}];
  }
`;
			}
			r.functions[jp] = `float getBandValue(float band, float xOffset, float yOffset) {
  float dx = xOffset / ${vp.TEXTURE_PIXEL_WIDTH};
  float dy = yOffset / ${vp.TEXTURE_PIXEL_HEIGHT};
${e}
}`;
		}
		return `${jp}(${e}, ${t ?? "0.0"}, ${n ?? "0.0"})`;
	}),
	[W.Palette]: (e, t) => {
		let [n, ...r] = t.args, i = r.length, a = new Uint8Array(i * 4);
		for (let e = 0; e < r.length; e++) {
			let t = r[e].value, n = $o(t), i = e * 4;
			a[i] = n[0], a[i + 1] = n[1], a[i + 2] = n[2], a[i + 3] = n[3] * 255;
		}
		e.paletteTextures ||= [];
		let o = `${Mp}[${e.paletteTextures.length}]`, s = new yp(o, a);
		return e.paletteTextures.push(s), `texture2D(${o}, vec2((${Rp(n, V, e)} + 0.5) / ${i}.0, 0.5))`;
	}
};
function Rp(e, t, n) {
	if (e instanceof _s) {
		let r = Lp[e.operator];
		if (r === void 0) throw Error(`No compiler defined for this operator: ${JSON.stringify(e.operator)}`);
		return r(n, e, t);
	}
	if ((e.type & V) > 0) return xp(e.value);
	if ((e.type & as) > 0) return e.value.toString();
	if ((e.type & H) > 0) return Op(e.value.toString());
	if ((e.type & U) > 0) return Cp(e.value);
	if ((e.type & os) > 0) return Sp(e.value);
	if ((e.type & ss) > 0) return wp(e.value);
	throw Error(`Unexpected expression ${e.value} (expected type ${fs(t)})`);
}
//#endregion
//#region node_modules/ol/style/flat.js
function zp() {
	return {
		"fill-color": "rgba(255,255,255,0.4)",
		"stroke-color": "#3399CC",
		"stroke-width": 1.25,
		"circle-radius": 5,
		"circle-fill-color": "rgba(255,255,255,0.4)",
		"circle-stroke-width": 1.25,
		"circle-stroke-color": "#3399CC"
	};
}
//#endregion
//#region node_modules/ol/render/webgl/bufferUtil.js
var Bp = .985;
//#endregion
//#region node_modules/ol/render/webgl/compileUtil.js
function Z(e, t, n, r) {
	return Ip(t, n, r ?? vs(e.inputVariables), e);
}
function Vp(e) {
	let t = $o(e), n = t[0] * 256, r = t[1], i = t[2] * 256, a = Math.round(t[3] * 255);
	return [n + r, i + a];
}
var Hp = "vec4 unpackColor(vec2 packedColor) {\n  return vec4(\n    min(floor(packedColor[0] / 256.0) / 255.0, 1.0),\n    min(mod(packedColor[0], 256.0) / 255.0, 1.0),\n    min(floor(packedColor[1] / 256.0) / 255.0, 1.0),\n    min(mod(packedColor[1], 256.0) / 255.0, 1.0)\n  );\n}";
function Up(e) {
	return e === U || e === ss ? 2 : e === os ? 4 : e === H ? 3 : 1;
}
function Wp(e) {
	if (e === H) return "float";
	let t = Up(e);
	return t > 1 ? `vec${t}` : "float";
}
function Gp(e, t) {
	for (let n of t.variables.entries()) {
		let [t, r] = n, i = kp(t), a = Wp(r);
		r === U && (a = "vec4"), e.addUniform(i, a);
	}
	for (let n of t.properties.entries()) {
		let [t, r] = n, i = Wp(r), a = `a_prop_${t}`;
		r === U ? e.addAttribute(a, i, `unpackColor(${a})`, "vec4") : e.addAttribute(a, i);
	}
	for (let n in t.functions) e.addVertexShaderFunction(t.functions[n]), e.addFragmentShaderFunction(t.functions[n]);
}
function Kp(e, t) {
	let n = {};
	for (let r of e.variables.entries()) {
		let [e, i] = r, a = kp(e);
		n[a] = () => {
			let n = t[e];
			if (i === as) return +!!n;
			if (i === U) {
				let e = [...$o(n || "#eee")];
				return e[0] /= 255, e[1] /= 255, e[2] /= 255, e[3] ??= 1, e;
			}
			return i === H ? Dp(n) : n;
		};
	}
	return n;
}
function qp(e) {
	let t = {};
	for (let n of e.properties.entries()) {
		let [e, r] = n, i = (t) => {
			let n = t.get(e);
			return r === U ? Vp([...$o(n || "#eee")]) : r === as ? +!!n : n;
		};
		t[`prop_${e}`] = {
			size: Up(r),
			callback: i
		};
	}
	return t;
}
//#endregion
//#region node_modules/ol/render/webgl/float64Util.js
function Jp(e) {
	return e - Yp(e);
}
function Yp(e) {
	return Math.fround(e);
}
//#endregion
//#region node_modules/ol/render/webgl/ShaderBuilder.js
var Xp = `#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform float u_one;
uniform mat4 u_projectionMatrix;
uniform mat4 u_invertProjectionMatrix;
uniform vec2 u_viewportSizePx;
uniform float u_pixelRatio;
uniform float u_globalAlpha;
uniform float u_time;
uniform float u_zoom;
uniform float u_resolution;
uniform float u_rotation;
uniform vec4 u_renderExtent;
uniform float u_depth;
uniform mediump int u_hitDetection;

// these 64-bits floats are split into high/low
uniform vec2 u_df_patternOriginX;
uniform vec2 u_df_patternOriginY;
uniform vec2 u_df_patternScaleRatio;

const float PI = 3.141592653589793238;
const float TWO_PI = 2.0 * PI;
float currentLineMetric = 0.; // an actual value will be used in the stroke shaders

vec2 pxToWorld(vec2 pxPos) {
  vec2 screenPos = 2.0 * pxPos / u_viewportSizePx - 1.0;
  return (u_invertProjectionMatrix * vec4(screenPos, 0.0, 1.0)).xy;
}

vec2 worldToPx(vec2 worldPos) {
  vec4 screenPos = u_projectionMatrix * vec4(worldPos, 0.0, 1.0);
  return (0.5 * screenPos.xy + 0.5) * u_viewportSizePx;
}
${Hp}

vec2 df_from(float value) {
  return vec2(value, 0.);
}

float df_float(vec2 df) {
  return df.x;
}

vec2 df_add(vec2 dfa, vec2 dfb) {
  vec2 dfc;
  float t1, t2, e;

  t1 = dfa.x * u_one + dfb.x * u_one;
  e = t1 * u_one - dfa.x * u_one;
  t2 = ((dfb.x - e) + (dfa.x - (t1 - e))) * u_one + dfa.y + dfb.y * u_one;

  dfc.x = t1 * u_one + t2 * u_one;
  dfc.y = t2 - (dfc.x - t1) * u_one;
  return dfc;
}

vec2 df_sub(vec2 dfa, vec2 dfb) {
  vec2 dfc;
  float e, t1, t2;

  t1 = dfa.x - dfb.x;
  e = t1 - dfa.x;
  t2 = ((-dfb.x - e) + (dfa.x - (t1 - e))) + dfa.y - dfb.y;

  dfc.x = t1 + t2;
  dfc.y = t2 - (dfc.x - t1);
  return dfc;
}

vec2 df_mul(vec2 dfa, vec2 dfb) {
  vec2 dfc;
  float c11, c21, c2, e, t1, t2;
  float a1, a2, b1, b2, cona, conb, split = 4097.;

  cona = dfa.x * split * u_one;
  conb = dfb.x * split * u_one;
  a1 = cona * u_one - (cona - dfa.x);
  b1 = conb * u_one - (conb - dfb.x);
  a2 = dfa.x * u_one - a1;
  b2 = dfb.x * u_one - b1 * u_one;

  c11 = dfa.x * u_one * dfb.x * u_one;
  c21 = a2 * b2 * u_one + (a2 * b1 + (a1 * b2 + (a1 * b1 - c11))) * u_one;

  c2 = dfa.x * dfb.y * u_one + dfa.y * dfb.x * u_one;

  t1 = c11 + c2 * u_one;
  e = t1 - c11 * u_one;
  t2 = dfa.y * dfb.y * u_one + ((c2 - e) + (c11 - (t1 - e))) + c21 * u_one;

  dfc.x = t1 * u_one + t2 * u_one;
  dfc.y = t2 - (dfc.x - t1) * u_one;

  return dfc;
}

vec2 df_div(vec2 dfa, vec2 dfb) {
  vec2 dfc;
  float c11, c21, c2, e, t1, t2, t11, t12, t21, t22;
  float a1, a2, b1, b2, cona, conb, split = 4097.;
  float s1, s2;

  s1 = dfa.x / dfb.x * u_one;
  cona = s1 * split * u_one;
  conb = dfb.x * split * u_one;
  a1 = cona - (cona - s1) * u_one;
  b1 = conb - (conb - dfb.x) * u_one;
  a2 = s1 - a1 * u_one;
  b2 = dfb.x - b1 * u_one;

  c11 = s1 * dfb.x * u_one;
  c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2 * u_one;

  c2 = s1 * dfb.y * u_one;

  t1 = c11 + c2 * u_one;
  e  = t1 - c11 * u_one;
  t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 * u_one;

  t12 = t1 + t2 * u_one;
  t22 = t2 - (t12 - t1) * u_one;

  t11 = dfa.x - t12 * u_one;
  e   = t11 - dfa.x * u_one;
  t21 = ((-t12 - e) + (dfa.x - (t11 - e))) + dfa.y - t22 * u_one;

  s2 = (t11 + t21) / dfb.x * u_one;

  dfc.x = s1 + s2 * u_one;
  dfc.y = s2 - (dfc.x - s1) * u_one;

  return dfc;
}

float df_mod(vec2 df, vec2 m) {
  vec2 q = df_div(df, m) * u_one;
  float qf = floor(q.x);
  float frac = q.x - qf + q.y * u_one;
  if (frac < 0.0) qf -= 1.0;
  if (frac >= 1.0) qf += 1.0;
  vec2 prod = df_mul(df_from(qf), m);
  vec2 rem = df_add(df_from(df.x), df_from(-prod.x)) * u_one;
  rem.y += df.y - prod.y;
  return rem.x + rem.y * u_one;
}

`, Zp = zp(), Qp = class {
	constructor() {
		this.uniforms_ = [], this.attributes_ = [], this.hasSymbol_ = !1, this.symbolSizeExpression_ = `vec2(${xp(Zp["circle-radius"])} + ${xp(Zp["circle-stroke-width"] * .5)})`, this.symbolRotationExpression_ = "0.0", this.symbolOffsetExpression_ = "vec2(0.0)", this.symbolColorExpression_ = Cp(Zp["circle-fill-color"]), this.texCoordExpression_ = "vec4(0.0, 0.0, 1.0, 1.0)", this.fragmentDiscardExpression_ = null, this.shapeDiscardExpression_ = null, this.symbolRotateWithView_ = !1, this.hasStroke_ = !1, this.strokeWidthExpression_ = xp(Zp["stroke-width"]), this.strokeColorExpression_ = Cp(Zp["stroke-color"]), this.strokeOffsetExpression_ = "0.", this.strokeCapExpression_ = Op("round"), this.strokeJoinExpression_ = Op("round"), this.strokeMiterLimitExpression_ = "10.", this.strokeDistanceFieldExpression_ = "-1000.", this.strokePatternLengthExpression_ = null, this.hasFill_ = !1, this.fillColorExpression_ = Cp(Zp["fill-color"]), this.fillPatternSizeExpression_ = null, this.vertexShaderFunctions_ = [], this.fragmentShaderFunctions_ = [];
	}
	addUniform(e, t) {
		return this.uniforms_.push({
			name: e,
			type: t
		}), this;
	}
	addAttribute(e, t, n, r) {
		return this.attributes_.push({
			name: e,
			type: t,
			varyingName: e.replace(/^a_/, "v_"),
			varyingType: r ?? t,
			varyingExpression: n ?? e
		}), this;
	}
	setSymbolSizeExpression(e) {
		return this.hasSymbol_ = !0, this.symbolSizeExpression_ = e, this;
	}
	getSymbolSizeExpression() {
		return this.symbolSizeExpression_;
	}
	setSymbolRotationExpression(e) {
		return this.symbolRotationExpression_ = e, this;
	}
	setSymbolOffsetExpression(e) {
		return this.symbolOffsetExpression_ = e, this;
	}
	getSymbolOffsetExpression() {
		return this.symbolOffsetExpression_;
	}
	setSymbolColorExpression(e) {
		return this.hasSymbol_ = !0, this.symbolColorExpression_ = e, this;
	}
	getSymbolColorExpression() {
		return this.symbolColorExpression_;
	}
	setTextureCoordinateExpression(e) {
		return this.texCoordExpression_ = e, this;
	}
	setFragmentDiscardExpression(e) {
		return this.fragmentDiscardExpression_ = e, this;
	}
	getFragmentDiscardExpression() {
		return this.fragmentDiscardExpression_;
	}
	setShapeDiscardExpression(e) {
		return this.shapeDiscardExpression_ = e, this;
	}
	getShapeDiscardExpression() {
		return this.shapeDiscardExpression_;
	}
	setSymbolRotateWithView(e) {
		return this.symbolRotateWithView_ = e, this;
	}
	setStrokeWidthExpression(e) {
		return this.hasStroke_ = !0, this.strokeWidthExpression_ = e, this;
	}
	setStrokeColorExpression(e) {
		return this.hasStroke_ = !0, this.strokeColorExpression_ = e, this;
	}
	getStrokeColorExpression() {
		return this.strokeColorExpression_;
	}
	setStrokeOffsetExpression(e) {
		return this.strokeOffsetExpression_ = e, this;
	}
	setStrokeCapExpression(e) {
		return this.strokeCapExpression_ = e, this;
	}
	setStrokeJoinExpression(e) {
		return this.strokeJoinExpression_ = e, this;
	}
	setStrokeMiterLimitExpression(e) {
		return this.strokeMiterLimitExpression_ = e, this;
	}
	setStrokeDistanceFieldExpression(e) {
		return this.strokeDistanceFieldExpression_ = e, this;
	}
	setStrokePatternLengthExpression(e) {
		return this.strokePatternLengthExpression_ = e, this;
	}
	getStrokePatternLengthExpression() {
		return this.strokePatternLengthExpression_;
	}
	setFillColorExpression(e) {
		return this.hasFill_ = !0, this.fillColorExpression_ = e, this;
	}
	getFillColorExpression() {
		return this.fillColorExpression_;
	}
	setFillPatternSizeExpression(e) {
		return this.fillPatternSizeExpression_ = e, this;
	}
	getFillPatternSizeExpression() {
		return this.fillPatternSizeExpression_;
	}
	addVertexShaderFunction(e) {
		return this.vertexShaderFunctions_.includes(e) || this.vertexShaderFunctions_.push(e), this;
	}
	addFragmentShaderFunction(e) {
		return this.fragmentShaderFunctions_.includes(e) || this.fragmentShaderFunctions_.push(e), this;
	}
	getSymbolVertexShader() {
		return this.hasSymbol_ ? `${Xp}
${this.uniforms_.map((e) => `uniform ${e.type} ${e.name};`).join("\n")}
attribute vec2 a_position;
attribute vec2 a_localPosition;
attribute vec2 a_hitColor;

varying vec2 v_texCoord;
varying vec2 v_quadCoord;
varying vec4 v_hitColor;
varying vec2 v_centerPx;
varying float v_angle;
varying vec2 v_quadSizePx;

${this.attributes_.map((e) => `attribute ${e.type} ${e.name};
varying ${e.varyingType} ${e.varyingName};`).join("\n")}
${this.vertexShaderFunctions_.join("\n")}
vec2 pxToScreen(vec2 coordPx) {
  vec2 scaled = coordPx / u_viewportSizePx / 0.5;
  return scaled;
}

vec2 screenToPx(vec2 coordScreen) {
  return (coordScreen * 0.5 + 0.5) * u_viewportSizePx;
}

void main(void) {
  v_quadSizePx = ${this.symbolSizeExpression_};
  vec2 halfSizePx = v_quadSizePx * 0.5;
  vec2 centerOffsetPx = ${this.symbolOffsetExpression_};
  vec2 offsetPx = centerOffsetPx + a_localPosition * halfSizePx * vec2(1., -1.);
  float angle = ${this.symbolRotationExpression_}${this.symbolRotateWithView_ ? " + u_rotation" : ""};
  float c = cos(-angle);
  float s = sin(-angle);
  offsetPx = vec2(c * offsetPx.x - s * offsetPx.y, s * offsetPx.x + c * offsetPx.y);
  vec4 center = u_projectionMatrix * vec4(a_position, 0.0, 1.0);
  gl_Position = center + vec4(pxToScreen(offsetPx), u_depth, 0.);
  vec4 texCoord = ${this.texCoordExpression_};
  float u = mix(texCoord.s, texCoord.p, a_localPosition.x * 0.5 + 0.5);
  float v = mix(texCoord.t, texCoord.q, a_localPosition.y * 0.5 + 0.5);
  v_texCoord = vec2(u, v);
  v_hitColor = unpackColor(a_hitColor);
  v_angle = angle;
  c = cos(-v_angle);
  s = sin(-v_angle);
  centerOffsetPx = vec2(c * centerOffsetPx.x - s * centerOffsetPx.y, s * centerOffsetPx.x + c * centerOffsetPx.y);
  v_centerPx = screenToPx(center.xy) + centerOffsetPx;
${this.attributes_.map((e) => `  ${e.varyingName} = ${e.varyingExpression};`).join("\n")}
${this.shapeDiscardExpression_ ? `  if (${this.shapeDiscardExpression_}) { gl_Position = vec4(2.0, 2.0, 0.0, 0.0); }` : ""}
}` : null;
	}
	getSymbolFragmentShader() {
		return this.hasSymbol_ ? `${Xp}
${this.uniforms_.map((e) => `uniform ${e.type} ${e.name};`).join("\n")}
varying vec2 v_texCoord;
varying vec4 v_hitColor;
varying vec2 v_centerPx;
varying float v_angle;
varying vec2 v_quadSizePx;
${this.attributes_.map((e) => `varying ${e.varyingType} ${e.varyingName};`).join("\n")}
${this.fragmentShaderFunctions_.join("\n")}

void main(void) {
${this.attributes_.map((e) => `  ${e.varyingType} ${e.name} = ${e.varyingName}; // assign to original attribute name`).join("\n")}
${this.fragmentDiscardExpression_ ? `  if (${this.fragmentDiscardExpression_}) { discard; }` : ""}
  vec2 coordsPx = gl_FragCoord.xy / u_pixelRatio - v_centerPx; // relative to center
  float c = cos(v_angle);
  float s = sin(v_angle);
  coordsPx = vec2(c * coordsPx.x - s * coordsPx.y, s * coordsPx.x + c * coordsPx.y);
  gl_FragColor = ${this.symbolColorExpression_};
  gl_FragColor.rgb *= gl_FragColor.a;
  if (u_hitDetection > 0) {
    if (gl_FragColor.a < 0.05) { discard; };
    gl_FragColor = v_hitColor;
  }
}` : null;
	}
	getStrokeVertexShader() {
		return this.hasStroke_ ? `${Xp}
${this.uniforms_.map((e) => `uniform ${e.type} ${e.name};`).join("\n")}
attribute vec2 a_segmentStart;
attribute vec2 a_segmentEnd;
attribute vec2 a_localPosition;
attribute float a_measureStart;
attribute float a_measureEnd;
attribute float a_angleTangentSum;
attribute float a_distanceLow;
attribute float a_distanceHigh;
attribute vec2 a_joinAngles;
attribute vec2 a_hitColor;

varying vec2 v_segmentStartPx;
varying vec2 v_segmentEndPx;
varying float v_angleStart;
varying float v_angleEnd;
varying float v_width;
varying vec4 v_hitColor;
varying float v_distancePx;
varying float v_measureStart;
varying float v_measureEnd;

${this.attributes_.map((e) => `attribute ${e.type} ${e.name};
varying ${e.varyingType} ${e.varyingName};`).join("\n")}
${this.vertexShaderFunctions_.join("\n")}

vec4 pxToScreen(vec2 pxPos) {
  vec2 screenPos = 2.0 * pxPos / u_viewportSizePx - 1.0;
  return vec4(screenPos, u_depth, 1.0);
}

bool isCap(float joinAngle) {
  return joinAngle < -0.1;
}

vec2 getJoinOffsetDirection(vec2 normalPx, float joinAngle) {
  float halfAngle = joinAngle / 2.0;
  float c = cos(halfAngle);
  float s = sin(halfAngle);
  vec2 angleBisectorNormal = vec2(s * normalPx.x + c * normalPx.y, -c * normalPx.x + s * normalPx.y);
  float length = 1.0 / s;
  return angleBisectorNormal * length;
}

vec2 getOffsetPoint(vec2 point, vec2 normal, float joinAngle, float offsetPx) {
  // if on a cap or the join angle is too high, offset the line along the segment normal
  if (cos(joinAngle) > 0.998 || isCap(joinAngle)) {
    return point - normal * offsetPx;
  }
  // offset is applied along the inverted normal (positive offset goes "right" relative to line direction)
  return point - getJoinOffsetDirection(normal, joinAngle) * offsetPx;
}

void main(void) {
  v_angleStart = a_joinAngles.x;
  v_angleEnd = a_joinAngles.y;
  float startEndRatio = a_localPosition.x * 0.5 + 0.5;
  currentLineMetric = mix(a_measureStart, a_measureEnd, startEndRatio);
  // we're reading the fractional part while keeping the sign (so -4.12 gives -0.12, 3.45 gives 0.45)

  float lineWidth = ${this.strokeWidthExpression_};
  float lineOffsetPx = ${this.strokeOffsetExpression_};

  // compute segment start/end in px with offset
  vec2 segmentStartPx = worldToPx(a_segmentStart);
  vec2 segmentEndPx = worldToPx(a_segmentEnd);
  vec2 tangentPx = normalize(segmentEndPx - segmentStartPx);
  vec2 normalPx = vec2(-tangentPx.y, tangentPx.x);
  segmentStartPx = getOffsetPoint(segmentStartPx, normalPx, v_angleStart, lineOffsetPx),
  segmentEndPx = getOffsetPoint(segmentEndPx, normalPx, v_angleEnd, lineOffsetPx);

  // compute current vertex position
  float normalDir = -1. * a_localPosition.y;
  float tangentDir = -1. * a_localPosition.x;
  float angle = mix(v_angleStart, v_angleEnd, startEndRatio);
  vec2 joinDirection;
  vec2 positionPx = mix(segmentStartPx, segmentEndPx, startEndRatio);
  // if angle is too high, do not make a proper join
  if (cos(angle) > ${Bp} || isCap(angle)) {
    joinDirection = normalPx * normalDir - tangentPx * tangentDir;
  } else {
    joinDirection = getJoinOffsetDirection(normalPx * normalDir, angle);
  }
  positionPx = positionPx + joinDirection * (lineWidth * 0.5 + 1.); // adding 1 pixel for antialiasing
  gl_Position = pxToScreen(positionPx);

  v_segmentStartPx = segmentStartPx;
  v_segmentEndPx = segmentEndPx;
  v_width = lineWidth;
  v_hitColor = unpackColor(a_hitColor);

  v_distancePx = a_distanceLow / u_resolution - (lineOffsetPx * a_angleTangentSum);
  float distanceHighPx = a_distanceHigh / u_resolution;
  ${this.strokePatternLengthExpression_ === null ? "" : `v_distancePx = mod(v_distancePx, ${this.strokePatternLengthExpression_});
  distanceHighPx = mod(distanceHighPx, ${this.strokePatternLengthExpression_});
  `}v_distancePx += distanceHighPx;

  v_measureStart = a_measureStart;
  v_measureEnd = a_measureEnd;
${this.attributes_.map((e) => `  ${e.varyingName} = ${e.varyingExpression};`).join("\n")}
${this.shapeDiscardExpression_ ? `  if (${this.shapeDiscardExpression_}) { gl_Position = vec4(2.0, 2.0, 0.0, 0.0); }` : ""}
}` : null;
	}
	getStrokeFragmentShader() {
		return this.hasStroke_ ? `${Xp}
${this.uniforms_.map((e) => `uniform ${e.type} ${e.name};`).join("\n")}
varying vec2 v_segmentStartPx;
varying vec2 v_segmentEndPx;
varying float v_angleStart;
varying float v_angleEnd;
varying float v_width;
varying vec4 v_hitColor;
varying float v_distancePx;
varying float v_measureStart;
varying float v_measureEnd;
${this.attributes_.map((e) => `varying ${e.varyingType} ${e.varyingName};`).join("\n")}
${this.fragmentShaderFunctions_.join("\n")}

bool isCap(float joinAngle) {
  return joinAngle < -0.1;
}

float segmentDistanceField(vec2 point, vec2 start, vec2 end, float width) {
  vec2 tangent = normalize(end - start);
  vec2 normal = vec2(-tangent.y, tangent.x);
  vec2 startToPoint = point - start;
  return abs(dot(startToPoint, normal)) - width * 0.5;
}

float buttCapDistanceField(vec2 point, vec2 start, vec2 end) {
  vec2 startToPoint = point - start;
  vec2 tangent = normalize(end - start);
  return dot(startToPoint, -tangent);
}

float squareCapDistanceField(vec2 point, vec2 start, vec2 end, float width) {
  return buttCapDistanceField(point, start, end) - width * 0.5;
}

float roundCapDistanceField(vec2 point, vec2 start, vec2 end, float width) {
  float onSegment = max(0., 1000. * dot(point - start, end - start)); // this is very high when inside the segment
  return length(point - start) - width * 0.5 - onSegment;
}

float roundJoinDistanceField(vec2 point, vec2 start, vec2 end, float width) {
  return roundCapDistanceField(point, start, end, width);
}

float bevelJoinField(vec2 point, vec2 start, vec2 end, float width, float joinAngle) {
  vec2 startToPoint = point - start;
  vec2 tangent = normalize(end - start);
  float c = cos(joinAngle * 0.5);
  float s = sin(joinAngle * 0.5);
  float direction = -sign(sin(joinAngle));
  vec2 bisector = vec2(c * tangent.x - s * tangent.y, s * tangent.x + c * tangent.y);
  float radius = width * 0.5 * s;
  return dot(startToPoint, bisector * direction) - radius;
}

float miterJoinDistanceField(vec2 point, vec2 start, vec2 end, float width, float joinAngle) {
  if (cos(joinAngle) > ${Bp}) { // avoid risking a division by zero
    return bevelJoinField(point, start, end, width, joinAngle);
  }
  float miterLength = 1. / sin(joinAngle * 0.5);
  float miterLimit = ${this.strokeMiterLimitExpression_};
  if (miterLength > miterLimit) {
    return bevelJoinField(point, start, end, width, joinAngle);
  }
  return -1000.;
}

float capDistanceField(vec2 point, vec2 start, vec2 end, float width, float capType) {
   if (capType == ${Op("butt")}) {
    return buttCapDistanceField(point, start, end);
  } else if (capType == ${Op("square")}) {
    return squareCapDistanceField(point, start, end, width);
  }
  return roundCapDistanceField(point, start, end, width);
}

float joinDistanceField(vec2 point, vec2 start, vec2 end, float width, float joinAngle, float joinType) {
  if (joinType == ${Op("bevel")}) {
    return bevelJoinField(point, start, end, width, joinAngle);
  } else if (joinType == ${Op("miter")}) {
    return miterJoinDistanceField(point, start, end, width, joinAngle);
  }
  return roundJoinDistanceField(point, start, end, width);
}

float computeSegmentPointDistance(vec2 point, vec2 start, vec2 end, float width, float joinAngle, float capType, float joinType) {
  if (isCap(joinAngle)) {
    return capDistanceField(point, start, end, width, capType);
  }
  return joinDistanceField(point, start, end, width, joinAngle, joinType);
}

float distanceFromSegment(vec2 point, vec2 start, vec2 end) {
  vec2 tangent = end - start;
  vec2 startToPoint = point - start;
  // inspire by capsule fn in https://iquilezles.org/articles/distfunctions/
  float h = clamp(dot(startToPoint, tangent) / dot(tangent, tangent), 0.0, 1.0);
  return length(startToPoint - tangent * h);
}

void main(void) {
${this.attributes_.map((e) => `  ${e.varyingType} ${e.name} = ${e.varyingName}; // assign to original attribute name`).join("\n")}

  vec2 currentPointPx = gl_FragCoord.xy / u_pixelRatio;
  vec2 worldPos = pxToWorld(currentPointPx);
  if (
    abs(u_renderExtent[0] - u_renderExtent[2]) > 0.0 && (
      worldPos[0] < u_renderExtent[0] ||
      worldPos[1] < u_renderExtent[1] ||
      worldPos[0] > u_renderExtent[2] ||
      worldPos[1] > u_renderExtent[3]
    )
  ) {
    discard;
  }

  float segmentLengthPx = length(v_segmentEndPx - v_segmentStartPx);
  segmentLengthPx = max(segmentLengthPx, 1.17549429e-38); // avoid divide by zero
  vec2 segmentTangent = (v_segmentEndPx - v_segmentStartPx) / segmentLengthPx;
  vec2 segmentNormal = vec2(-segmentTangent.y, segmentTangent.x);
  vec2 startToPointPx = currentPointPx - v_segmentStartPx;
  float lengthToPointPx = max(0., min(dot(segmentTangent, startToPointPx), segmentLengthPx));
  float currentLengthPx = lengthToPointPx + v_distancePx;
  float currentRadiusPx = distanceFromSegment(currentPointPx, v_segmentStartPx, v_segmentEndPx);
  float currentRadiusRatio = dot(segmentNormal, startToPointPx) * 2. / v_width;
  currentLineMetric = mix(v_measureStart, v_measureEnd, lengthToPointPx / segmentLengthPx);

${this.fragmentDiscardExpression_ ? `  if (${this.fragmentDiscardExpression_}) { discard; }` : ""}

  float capType = ${this.strokeCapExpression_};
  float joinType = ${this.strokeJoinExpression_};
  float segmentStartDistance = computeSegmentPointDistance(currentPointPx, v_segmentStartPx, v_segmentEndPx, v_width, v_angleStart, capType, joinType);
  float segmentEndDistance = computeSegmentPointDistance(currentPointPx, v_segmentEndPx, v_segmentStartPx, v_width, v_angleEnd, capType, joinType);
  float distanceField = max(
    segmentDistanceField(currentPointPx, v_segmentStartPx, v_segmentEndPx, v_width),
    max(segmentStartDistance, segmentEndDistance)
  );
  distanceField = max(distanceField, ${this.strokeDistanceFieldExpression_});

  vec4 color = ${this.strokeColorExpression_};
  color.a *= smoothstep(0.5, -0.5, distanceField);
  gl_FragColor = color;
  gl_FragColor.a *= u_globalAlpha;
  gl_FragColor.rgb *= gl_FragColor.a;
  if (u_hitDetection > 0) {
    if (gl_FragColor.a < 0.1) { discard; };
    gl_FragColor = v_hitColor;
  }
}` : null;
	}
	getFillVertexShader() {
		return this.hasFill_ ? `${Xp}
${this.uniforms_.map((e) => `uniform ${e.type} ${e.name};`).join("\n")}
attribute vec2 a_position;
attribute vec2 a_hitColor;

varying vec4 v_hitColor;
varying vec2 v_patternOriginPx;
varying vec2 v_patternSizePx;

${this.attributes_.map((e) => `attribute ${e.type} ${e.name};
varying ${e.varyingType} ${e.varyingName};`).join("\n")}
${this.vertexShaderFunctions_.join("\n")}
void main(void) {
  gl_Position = u_projectionMatrix * vec4(a_position, u_depth, 1.0);
  v_hitColor = unpackColor(a_hitColor);
${this.fillPatternSizeExpression_ === null ? "  v_patternOriginPx = vec2(0.);" : `
  // this computes the pattern offset in screenspace using double-float arithmetics
  v_patternSizePx = ${this.fillPatternSizeExpression_};
  vec2 patternSizeScaledX = df_mul(df_from(v_patternSizePx.x), u_df_patternScaleRatio);
  vec2 patternSizeScaledY = df_mul(df_from(v_patternSizePx.y), u_df_patternScaleRatio);
  v_patternOriginPx = vec2(
    df_mod(u_df_patternOriginX, patternSizeScaledX),
    df_mod(u_df_patternOriginY, patternSizeScaledY)
  );

  // reapply rotation to the pattern origin
  v_patternOriginPx -= u_viewportSizePx / 2.; // translate to viewport center
  v_patternOriginPx = vec2(
    cos(-u_rotation) * v_patternOriginPx.x - sin(-u_rotation) * v_patternOriginPx.y,
    sin(-u_rotation) * v_patternOriginPx.x + cos(-u_rotation) * v_patternOriginPx.y
  );
  v_patternOriginPx += u_viewportSizePx / 2.; // translate back
`}
${this.attributes_.map((e) => `  ${e.varyingName} = ${e.varyingExpression};`).join("\n")}
${this.shapeDiscardExpression_ ? `  if (${this.shapeDiscardExpression_}) { gl_Position = vec4(2.0, 2.0, 0.0, 0.0); }` : ""}
}` : null;
	}
	getFillFragmentShader() {
		return this.hasFill_ ? `${Xp}
${this.uniforms_.map((e) => `uniform ${e.type} ${e.name};`).join("\n")}
varying vec4 v_hitColor;
varying vec2 v_patternOriginPx;
varying vec2 v_patternSizePx;
${this.attributes_.map((e) => `varying ${e.varyingType} ${e.varyingName};`).join("\n")}
${this.fragmentShaderFunctions_.join("\n")}

void main(void) {
${this.attributes_.map((e) => `  ${e.varyingType} ${e.name} = ${e.varyingName}; // assign to original attribute name`).join("\n")}
  vec2 pxPos = gl_FragCoord.xy / u_pixelRatio;
  vec2 worldPos = pxToWorld(pxPos);
  if (
    abs(u_renderExtent[0] - u_renderExtent[2]) > 0.0 && (
      worldPos[0] < u_renderExtent[0] ||
      worldPos[1] < u_renderExtent[1] ||
      worldPos[0] > u_renderExtent[2] ||
      worldPos[1] > u_renderExtent[3]
    )
  ) {
    discard;
  }
${this.fragmentDiscardExpression_ ? `  if (${this.fragmentDiscardExpression_}) { discard; }` : ""}
  gl_FragColor = ${this.fillColorExpression_};
  gl_FragColor.a *= u_globalAlpha;
  gl_FragColor.rgb *= gl_FragColor.a;
  if (u_hitDetection > 0) {
    if (gl_FragColor.a < 0.1) { discard; };
    gl_FragColor = v_hitColor;
  }
}` : null;
	}
};
//#endregion
//#region node_modules/ol/geom/flat/interpolate.js
function $p(e, t, n, r, i, a, o) {
	let s, c, l = (n - t) / r;
	if (l === 1) s = t;
	else if (l === 2) s = t, c = i;
	else if (l !== 0) {
		let a = e[t], o = e[t + 1], l = 0, u = [0];
		for (let i = t + r; i < n; i += r) {
			let t = e[i], n = e[i + 1];
			l += Math.sqrt((t - a) * (t - a) + (n - o) * (n - o)), u.push(l), a = t, o = n;
		}
		let d = i * l, f = ae(u, d);
		f < 0 ? (c = (d - u[-f - 2]) / (u[-f - 1] - u[-f - 2]), s = t + (-f - 2) * r) : s = t + f * r;
	}
	o = o > 1 ? o : 2, a ||= Array(o);
	for (let t = 0; t < o; ++t) a[t] = s === void 0 ? NaN : c === void 0 ? e[s + t] : kt(e[s + t], e[s + r + t], c);
	return a;
}
//#endregion
//#region node_modules/ol/geom/flat/center.js
function em(e, t, n, r) {
	let i = [], a = We();
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		a = Je(e, t, s[0], r), i.push((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), t = s[s.length - 1];
	}
	return i;
}
//#endregion
//#region node_modules/ol/render/Feature.js
var tm = qr(), nm = class e {
	constructor(e, t, n, r, i, a) {
		this.styleFunction, this.extent_, this.id_ = a, this.type_ = e, this.flatCoordinates_ = t, this.flatInteriorPoints_ = null, this.flatMidpoints_ = null, this.ends_ = n || null, this.properties_ = i, this.squaredTolerance_, this.stride_ = r, this.simplifiedGeometry_;
	}
	get(e) {
		return this.properties_[e];
	}
	getExtent() {
		return this.extent_ ||= this.type_ === "Point" ? qe(this.flatCoordinates_) : Je(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_), this.extent_;
	}
	getFlatInteriorPoint() {
		if (!this.flatInteriorPoints_) {
			let e = it(this.getExtent());
			this.flatInteriorPoints_ = Gi(this.flatCoordinates_, 0, this.ends_, this.stride_, e, 0);
		}
		return this.flatInteriorPoints_;
	}
	getFlatInteriorPoints() {
		if (!this.flatInteriorPoints_) {
			let e = Zi(this.flatCoordinates_, this.ends_), t = em(this.flatCoordinates_, 0, e, this.stride_);
			this.flatInteriorPoints_ = Ki(this.flatCoordinates_, 0, e, this.stride_, t);
		}
		return this.flatInteriorPoints_;
	}
	getFlatMidpoint() {
		return this.flatMidpoints_ ||= $p(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_, .5), this.flatMidpoints_;
	}
	getFlatMidpoints() {
		if (!this.flatMidpoints_) {
			this.flatMidpoints_ = [];
			let e = this.flatCoordinates_, t = 0, n = this.ends_;
			for (let r = 0, i = n.length; r < i; ++r) {
				let i = n[r], a = $p(e, t, i, this.stride_, .5);
				ue(this.flatMidpoints_, a), t = i;
			}
		}
		return this.flatMidpoints_;
	}
	getId() {
		return this.id_;
	}
	getOrientedFlatCoordinates() {
		return this.flatCoordinates_;
	}
	getGeometry() {
		return this;
	}
	getSimplifiedGeometry(e) {
		return this;
	}
	simplifyTransformed(e, t) {
		return this;
	}
	getProperties() {
		return this.properties_;
	}
	getPropertiesInternal() {
		return this.properties_;
	}
	getStride() {
		return this.stride_;
	}
	getStyleFunction() {
		return this.styleFunction;
	}
	getType() {
		return this.type_;
	}
	transform(e) {
		e = nr(e);
		let t = e.getExtent(), n = e.getWorldExtent();
		if (t && n) {
			let e = ct(n) / ct(t);
			ei(tm, n[0], n[3], e, -e, 0, 0, 0), si(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_, tm, this.flatCoordinates_);
		}
	}
	applyTransform(e) {
		e(this.flatCoordinates_, this.flatCoordinates_, this.stride_);
	}
	clone() {
		return new e(this.type_, this.flatCoordinates_.slice(), this.ends_?.slice(), this.stride_, Object.assign({}, this.properties_), this.id_);
	}
	getEnds() {
		return this.ends_;
	}
	enableSimplifyTransformed() {
		return this.simplifyTransformed = ge((t, n) => {
			if (t === this.squaredTolerance_) return this.simplifiedGeometry_;
			this.simplifiedGeometry_ = this.clone(), n && this.simplifiedGeometry_.applyTransform(n);
			let r = this.simplifiedGeometry_.getFlatCoordinates(), i;
			switch (this.type_) {
				case "LineString":
					r.length = Ri(r, 0, this.simplifiedGeometry_.flatCoordinates_.length, this.simplifiedGeometry_.stride_, t, r, 0), i = [r.length];
					break;
				case "MultiLineString":
					i = [], r.length = zi(r, 0, this.simplifiedGeometry_.ends_, this.simplifiedGeometry_.stride_, t, r, 0, i);
					break;
				case "Polygon": i = [], r.length = Hi(r, 0, this.simplifiedGeometry_.ends_, this.simplifiedGeometry_.stride_, Math.sqrt(t), r, 0, i);
			}
			return i && (this.simplifiedGeometry_ = new e(this.type_, r, i, this.stride_, this.properties_, this.id_)), this.squaredTolerance_ = t, this.simplifiedGeometry_;
		}), this;
	}
};
nm.prototype.getFlatCoordinates = nm.prototype.getOrientedFlatCoordinates;
//#endregion
//#region node_modules/ol/render/webgl/MixedGeometryBatch.js
var rm = class e {
	constructor() {
		this.globalCounter_ = 0, this.refToFeature_ = /* @__PURE__ */ new Map(), this.uidToRef_ = /* @__PURE__ */ new Map(), this.freeGlobalRef_ = [], this.polygonBatch = {
			entries: {},
			geometriesCount: 0,
			verticesCount: 0,
			ringsCount: 0
		}, this.pointBatch = {
			entries: {},
			geometriesCount: 0
		}, this.lineStringBatch = {
			entries: {},
			geometriesCount: 0,
			verticesCount: 0
		};
	}
	addFeatures(e, t) {
		for (let n = 0; n < e.length; n++) this.addFeature(e[n], t);
	}
	addFeature(e, t) {
		let n = e.getGeometry();
		n && (t && (n = n.clone(), n.applyTransform(t)), this.addGeometry_(n, e));
	}
	clearFeatureEntryInPointBatch_(e) {
		let t = F(e), n = this.pointBatch.entries[t];
		if (n) return this.pointBatch.geometriesCount -= n.flatCoordss.length, delete this.pointBatch.entries[t], n;
	}
	clearFeatureEntryInLineStringBatch_(e) {
		let t = F(e), n = this.lineStringBatch.entries[t];
		if (n) return this.lineStringBatch.verticesCount -= n.verticesCount, this.lineStringBatch.geometriesCount -= n.flatCoordss.length, delete this.lineStringBatch.entries[t], n;
	}
	clearFeatureEntryInPolygonBatch_(e) {
		let t = F(e), n = this.polygonBatch.entries[t];
		if (n) return this.polygonBatch.verticesCount -= n.verticesCount, this.polygonBatch.ringsCount -= n.ringsCount, this.polygonBatch.geometriesCount -= n.flatCoordss.length, delete this.polygonBatch.entries[t], n;
	}
	addGeometry_(e, t) {
		let n = e.getType();
		switch (n) {
			case "GeometryCollection": {
				let n = e.getGeometriesArray();
				for (let e of n) this.addGeometry_(e, t);
				break;
			}
			case "MultiPolygon": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEndss(), t, F(t), r.getStride());
				break;
			}
			case "MultiLineString": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEnds(), t, F(t), r.getStride());
				break;
			}
			case "MultiPoint": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, F(t), r.getStride());
				break;
			}
			case "Polygon": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEnds(), t, F(t), r.getStride());
				break;
			}
			case "Point": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, F(t), r.getStride());
				break;
			}
			case "LineString":
			case "LinearRing": {
				let r = e, i = r.getStride();
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, F(t), i, r.getLayout?.());
				break;
			}
		}
	}
	addCoordinates_(e, t, n, r, i, a, o) {
		let s;
		switch (e) {
			case "MultiPolygon": {
				let e = n;
				for (let n = 0, s = e.length; n < s; n++) {
					let s = e[n], c = n > 0 ? e[n - 1] : null, l = c ? c[c.length - 1] : 0, u = s[s.length - 1];
					s = l > 0 ? s.map((e) => e - l) : s, this.addCoordinates_("Polygon", t.slice(l, u), s, r, i, a, o);
				}
				break;
			}
			case "MultiLineString": {
				let e = n;
				for (let n = 0, s = e.length; n < s; n++) {
					let s = n > 0 ? e[n - 1] : 0;
					this.addCoordinates_("LineString", t.slice(s, e[n]), null, r, i, a, o);
				}
				break;
			}
			case "MultiPoint":
				for (let e = 0, n = t.length; e < n; e += a) this.addCoordinates_("Point", t.slice(e, e + 2), null, r, i, null, null);
				break;
			case "Polygon": {
				let e = n;
				if (r instanceof nm) {
					let n = Zi(t, e);
					if (n.length > 1) {
						this.addCoordinates_("MultiPolygon", t, n, r, i, a, o);
						return;
					}
				}
				this.polygonBatch.entries[i] || (this.polygonBatch.entries[i] = this.addRefToEntry_(i, {
					feature: r,
					flatCoordss: [],
					verticesCount: 0,
					ringsCount: 0,
					ringsVerticesCounts: []
				})), s = t.length / a;
				let c = n.length, l = n.map((e, t, n) => t > 0 ? (e - n[t - 1]) / a : e / a);
				this.polygonBatch.verticesCount += s, this.polygonBatch.ringsCount += c, this.polygonBatch.geometriesCount++, this.polygonBatch.entries[i].flatCoordss.push(im(t, a)), this.polygonBatch.entries[i].ringsVerticesCounts.push(l), this.polygonBatch.entries[i].verticesCount += s, this.polygonBatch.entries[i].ringsCount += c;
				for (let n = 0, s = e.length; n < s; n++) {
					let s = n > 0 ? e[n - 1] : 0;
					this.addCoordinates_("LinearRing", t.slice(s, e[n]), null, r, i, a, o);
				}
				break;
			}
			case "Point":
				this.pointBatch.entries[i] || (this.pointBatch.entries[i] = this.addRefToEntry_(i, {
					feature: r,
					flatCoordss: []
				})), this.pointBatch.geometriesCount++, this.pointBatch.entries[i].flatCoordss.push(t);
				break;
			case "LineString":
			case "LinearRing": this.lineStringBatch.entries[i] || (this.lineStringBatch.entries[i] = this.addRefToEntry_(i, {
				feature: r,
				flatCoordss: [],
				verticesCount: 0
			})), s = t.length / a, this.lineStringBatch.verticesCount += s, this.lineStringBatch.geometriesCount++, this.lineStringBatch.entries[i].flatCoordss.push(am(t, a, o)), this.lineStringBatch.entries[i].verticesCount += s;
		}
	}
	addRefToEntry_(e, t) {
		let n = this.uidToRef_.get(e), r = n || this.freeGlobalRef_.pop() || ++this.globalCounter_;
		return t.ref = r, n || (this.refToFeature_.set(r, t.feature), this.uidToRef_.set(e, r)), t;
	}
	removeRef_(e, t) {
		if (!e) throw Error("This feature has no ref: " + t);
		this.refToFeature_.delete(e), this.uidToRef_.delete(t), this.freeGlobalRef_.push(e);
	}
	changeFeature(e, t) {
		if (!this.uidToRef_.get(F(e))) return;
		this.removeFeature(e);
		let n = e.getGeometry();
		n && (t && (n = n.clone(), n.applyTransform(t)), this.addGeometry_(n, e));
	}
	removeFeature(e) {
		let t = this.clearFeatureEntryInPointBatch_(e);
		t = this.clearFeatureEntryInPolygonBatch_(e) || t, t = this.clearFeatureEntryInLineStringBatch_(e) || t, t && this.removeRef_(t.ref, F(t.feature));
	}
	clear() {
		this.polygonBatch.entries = {}, this.polygonBatch.geometriesCount = 0, this.polygonBatch.verticesCount = 0, this.polygonBatch.ringsCount = 0, this.lineStringBatch.entries = {}, this.lineStringBatch.geometriesCount = 0, this.lineStringBatch.verticesCount = 0, this.pointBatch.entries = {}, this.pointBatch.geometriesCount = 0, this.globalCounter_ = 0, this.freeGlobalRef_ = [], this.refToFeature_.clear(), this.uidToRef_.clear();
	}
	getFeatureFromRef(e) {
		return this.refToFeature_.get(e);
	}
	isEmpty() {
		return this.globalCounter_ === 0;
	}
	filter(t) {
		let n = new e();
		n.globalCounter_ = this.globalCounter_, n.uidToRef_ = this.uidToRef_, n.refToFeature_ = this.refToFeature_;
		let r = !0;
		for (let e of this.refToFeature_.values()) t(e) && (n.addFeature(e), r = !1);
		return r ? new e() : n;
	}
};
function im(e, t) {
	return t === 2 ? e : e.filter((e, n) => n % t < 2);
}
function am(e, t, n) {
	return t === 3 && n === "XYM" ? e : t === 4 ? e.filter((e, n) => n % t !== 2) : t === 3 ? e.map((e, n) => n % t === 2 ? 0 : e) : Array(e.length * 1.5).fill(0).map((t, n) => n % 3 == 2 ? 0 : e[Math.round(n / 1.5)]);
}
//#endregion
//#region node_modules/ol/webgl/LabelsArray.js
var om = new TextEncoder(), sm = 1e5, cm = class {
	constructor() {
		this.array_ = new Uint8Array(sm), this.actualSize_ = 0, this.labelPositionMap_ = /* @__PURE__ */ new Map();
	}
	push(e) {
		if (e === "") return [0, 0];
		if (this.labelPositionMap_.has(e)) return this.labelPositionMap_.get(e);
		let t = om.encode(e);
		if (this.actualSize_ + t.length > this.array_.length) {
			let e = new Uint8Array(this.array_.length + sm);
			e.set(this.array_), this.array_ = e;
		}
		let n = this.actualSize_;
		this.array_.set(t, n), this.actualSize_ += t.length;
		let r = [n, t.length];
		return this.labelPositionMap_.set(e, r), r;
	}
	getArray() {
		return this.array_;
	}
};
//#endregion
//#region node_modules/ol/worker/textOverlay.js
function lm() {
	let e = "function t(t,e){return t>e?1:t<e?-1:0}function e(t,e,i){for(;e<i;){const n=t[e];t[e]=t[i],t[i]=n,++e,--i}}function i(t,e){const i=Array.isArray(e)?e:[e],n=i.length;for(let e=0;e<n;e++)t[t.length]=i[e]}function n(t,e){const i=t.length;if(i!==e.length)return!1;for(let n=0;n<i;n++)if(t[n]!==e[n])return!1;return!0}const r=\"undefined\"!=typeof navigator&&void 0!==navigator.userAgent?navigator.userAgent.toLowerCase():\"\";r.includes(\"safari\")&&!r.includes(\"chrom\")&&(r.includes(\"version/15.4\")||/cpu (os|iphone os) 15_4 like mac os x/.test(r)),r.includes(\"webkit\")&&r.includes(\"edge\"),r.includes(\"macintosh\");const s=\"undefined\"!=typeof WorkerGlobalScope&&\"undefined\"!=typeof OffscreenCanvas&&self instanceof WorkerGlobalScope,o=\"undefined\"!=typeof Image&&Image.prototype.decode;function a(t,e,i,n){let r;return r=s?new class extends OffscreenCanvas{style={}}(t??300,e??150):document.createElement(\"canvas\"),t&&(r.width=t),e&&(r.height=e),r.getContext(\"2d\",n)}let l;function h(){return l||(l=a(1,1)),l}function c(t,e,i){return Math.min(Math.max(t,e),i)}function u(t,e,i,n,r,s){const o=r-i,a=s-n;if(0!==o||0!==a){const l=((t-i)*o+(e-n)*a)/(o*o+a*a);l>1?(i=r,n=s):l>0&&(i+=o*l,n+=a*l)}return f(t,e,i,n)}function f(t,e,i,n){const r=i-t,s=n-e;return r*r+s*s}function d(t){return 180*t/Math.PI}function g(t){return t*Math.PI/180}function p(t,e,i){return t+i*(e-t)}function _(t,e,i){if(t>=e&&t<i)return t;const n=i-e;return((t-e)%n+n)%n+e}!function(){let t=!1;try{const e=Object.defineProperty({},\"passive\",{get:function(){t=!0}});window.addEventListener(\"_\",null,e),window.removeEventListener(\"_\",null,e)}catch{}}();const m=[NaN,NaN,NaN,0];let y;const w=/^rgba?\\(\\s*(\\d+%?)\\s+(\\d+%?)\\s+(\\d+%?)(?:\\s*\\/\\s*(\\d+%|\\d*\\.\\d+|[01]))?\\s*\\)$/i,x=/^rgba?\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)(?:\\s*,\\s*(\\d+%|\\d*\\.\\d+|[01]))?\\s*\\)$/i,v=/^rgba?\\(\\s*(\\d+%)\\s*,\\s*(\\d+%)\\s*,\\s*(\\d+%)(?:\\s*,\\s*(\\d+%|\\d*\\.\\d+|[01]))?\\s*\\)$/i,S=/^#([\\da-f]{3,4}|[\\da-f]{6}|[\\da-f]{8})$/i;function C(t,e){return t.endsWith(\"%\")?Number(t.substring(0,t.length-1))/e:Number(t)}function b(t){throw new Error('failed to parse \"'+t+'\" as color')}function M(t){if(t.toLowerCase().startsWith(\"rgb\")){const e=t.match(x)||t.match(w)||t.match(v);if(e){const t=e[4],i=100/255;return[c(C(e[1],i)+.5|0,0,255),c(C(e[2],i)+.5|0,0,255),c(C(e[3],i)+.5|0,0,255),void 0!==t?c(C(t,100),0,1):1]}b(t)}if(t.startsWith(\"#\")){if(S.test(t)){const e=t.substring(1),i=e.length<=4?1:2,n=[0,0,0,255];for(let t=0,r=e.length;t<r;t+=i){let r=parseInt(e.substring(t,t+i),16);1===i&&(r+=r<<4),n[t/i]=r}return n[3]=n[3]/255,n}b(t)}const e=(y||(y=a(1,1,0,{willReadFrequently:!0,desynchronized:!0})),y);e.fillStyle=\"#abcdef\";let i=e.fillStyle;e.fillStyle=t,e.fillStyle===i&&(e.fillStyle=\"#fedcba\",i=e.fillStyle,e.fillStyle=t,e.fillStyle===i&&b(t));const n=e.fillStyle;if(n.startsWith(\"#\")||n.startsWith(\"rgba\"))return M(n);e.clearRect(0,0,1,1),e.fillRect(0,0,1,1);const r=Array.from(e.getImageData(0,0,1,1).data);return r[3]=function(t,e){const i=Math.pow(10,e);return Math.round(t*i)/i}(r[3]/255,3),r}const I={};let E=0;function k(t){if(4===t.length)return t;const e=t.slice();return e[3]=1,e}function A(t){return t>.0031308?269.025*Math.pow(t,1/2.4)-14.025:3294.6*t}function P(t){return t>.2068965?Math.pow(t,3):108/841*(t-4/29)}function O(t){return t>10.314724?Math.pow((t+14.025)/269.025,2.4):t/3294.6}function R(t){return t>.0088564?Math.pow(t,1/3):t/(108/841)+4/29}function L(t){const e=O(t[0]),i=O(t[1]),n=O(t[2]),r=R(.222488403*e+.716873169*i+.06060791*n),s=500*(R(.452247074*e+.399439023*i+.148375274*n)-r),o=200*(r-R(.016863605*e+.117638439*i+.865350722*n)),a=Math.atan2(o,s)*(180/Math.PI);return[116*r-16,Math.sqrt(s*s+o*o),a<0?a+360:a,t[3]]}function D(t){if(\"none\"===t)return m;if(I.hasOwnProperty(t))return I[t];if(E>=1024){let t=0;for(const e in I)3&t++||(delete I[e],--E)}const e=M(t);4!==e.length&&b(t);for(const i of e)isNaN(i)&&b(t);return I[t]=e,++E,e}function F(t){return Array.isArray(t)?t:D(t)}function T(t){let e=t[0];e!=(0|e)&&(e=e+.5|0);let i=t[1];i!=(0|i)&&(i=i+.5|0);let n=t[2];n!=(0|n)&&(n=n+.5|0);return\"rgba(\"+e+\",\"+i+\",\"+n+\",\"+(void 0===t[3]?1:Math.round(1e3*t[3])/1e3)+\")\"}function z(t,e){return Array.isArray(t)?t:(void 0===e?e=[t,t]:(e[0]=t,e[1]=t),e)}let $=0;const W=1<<$++,G=1<<$++,N=1<<$++,X=1<<$++,Y=1<<$++,B=1<<$++,U=Math.pow(2,6)-1,j={[W]:\"boolean\",[G]:\"number\",[N]:\"string\",[X]:\"color\",[Y]:\"number[]\",[B]:\"size\"},V=Object.keys(j).map(Number).sort(t);function q(t){const e=[];for(const i of V)J(t,i)&&e.push(j[i]);return 0===e.length?\"untyped\":e.length<3?e.join(\" or \"):e.slice(0,-1).join(\", \")+\", or \"+e[e.length-1]}function J(t,e){return(t&e)===e}function K(t,e){return!!(t&e)}function H(t,e){return t===e}class Z{constructor(t,e){if(!function(t){return t in j}(t))throw new Error(`literal expressions must have a specific type, got ${q(t)}`);this.type=t,this.value=e}}class Q{constructor(t,e,...i){this.type=t,this.operator=e,this.args=i}}function tt(t){return{variables:new Map,properties:new Map,featureId:!1,geometryType:!1,mCoordinate:!1,mapState:!1,inputVariables:t}}function et(t,e,i){switch(typeof t){case\"boolean\":if(H(e,N))return new Z(N,t?\"true\":\"false\");if(!J(e,W))throw new Error(`got a boolean, but expected ${q(e)}`);return new Z(W,t);case\"number\":if(H(e,B))return new Z(B,z(t));if(H(e,W))return new Z(W,!!t);if(H(e,N))return new Z(N,t.toString());if(!J(e,G))throw new Error(`got a number, but expected ${q(e)}`);return new Z(G,t);case\"string\":if(H(e,X))return new Z(X,D(t));if(H(e,W))return new Z(W,!!t);if(!J(e,N))throw new Error(`got a string, but expected ${q(e)}`);return new Z(N,t)}if(!Array.isArray(t))throw new Error(\"expression must be an array or a primitive value\");if(0===t.length)throw new Error(\"empty expression\");if(\"string\"==typeof t[0])return function(t,e,i){const n=t[0],r=Jt[n];if(!r)throw new Error(`unknown operator: ${n}`);return r(t,e,i)}(t,e,i);for(const e of t)if(\"number\"!=typeof e)throw new Error(\"expected an array of numbers\");if(H(e,B)){if(2!==t.length)throw new Error(`expected an array of two values for a size, got ${t.length}`);return new Z(B,t)}if(H(e,X)){if(3===t.length)return new Z(X,[...t,1]);if(4===t.length)return new Z(X,t);throw new Error(`expected an array of 3 or 4 values for a color, got ${t.length}`)}if(!J(e,Y))throw new Error(`got an array of numbers, but expected ${q(e)}`);return new Z(Y,t)}const it=\"get\",nt=\"var\",rt=\"concat\",st=\"geometry-type\",ot=\"line-metric\",at=\"any\",lt=\"all\",ht=\"!\",ct=\"resolution\",ut=\"zoom\",ft=\"time\",dt=\"==\",gt=\"!=\",pt=\">\",_t=\">=\",mt=\"<\",yt=\"<=\",wt=\"*\",xt=\"/\",vt=\"+\",St=\"-\",Ct=\"clamp\",bt=\"%\",Mt=\"^\",It=\"abs\",Et=\"floor\",kt=\"ceil\",At=\"round\",Pt=\"sin\",Ot=\"cos\",Rt=\"atan\",Lt=\"sqrt\",Dt=\"match\",Ft=\"between\",Tt=\"interpolate\",zt=\"coalesce\",$t=\"case\",Wt=\"in\",Gt=\"number\",Nt=\"string\",Xt=\"array\",Yt=\"color\",Bt=\"id\",Ut=\"band\",jt=\"palette\",Vt=\"to-string\",qt=\"has\",Jt={[it]:re(Qt(1,1/0),Kt),[nt]:function(t,e,i){const n=t[1];if(\"string\"!=typeof n)throw new Error(\"expected a string argument for var operation\");let r=e;const s=i.inputVariables?.[n];if(void 0!==s){const t=et(s,U,i);if(!(t instanceof Z))throw new Error(`style variables should only be literal values (no expressions!), variable name: ${n}`);let o=t.type;if(\"string\"==typeof s&&K(r,X)&&!K(r,N)?o=X:Array.isArray(s)&&2===s.length&&K(r,B)&&!K(r,Y)&&(o=B),r&=o,0===r)throw new Error(`the type expected from the var operator (${q(e)}) did not have any overlap with the type of the corresponding style variables (${q(o)}), variable name: ${n}`)}if(i.variables.has(n)){const t=i.variables.get(n);if(r&=t,0===r)throw new Error(`a new type expected from the var operator (${q(e)}) did not have any overlap with the previous type expected for it (${q(t)}), variable name: ${n}`)}return i.variables.set(n,r),new Q(r,\"var\",new Z(N,n))},[qt]:re(Qt(1,1/0),Kt),[Bt]:re(function(t,e,i){i.featureId=!0},Zt),[rt]:re(Qt(2,1/0),ee(N)),[st]:re(function(t,e,i){i.geometryType=!0},Zt),[ot]:re(function(t,e,i){i.mCoordinate=!0},Zt),[ct]:re(Ht,Zt),[ut]:re(Ht,Zt),[ft]:re(Ht,Zt),[at]:re(Qt(2,1/0),ee(W)),[lt]:re(Qt(2,1/0),ee(W)),[ht]:re(Qt(1,1),ee(W)),[dt]:re(Qt(2,2),ie()),[gt]:re(Qt(2,2),ie()),[pt]:re(Qt(2,2),ee(G)),[_t]:re(Qt(2,2),ee(G)),[mt]:re(Qt(2,2),ee(G)),[yt]:re(Qt(2,2),ee(G)),[wt]:re(Qt(2,1/0),te),[zt]:re(Qt(2,1/0),te),[xt]:re(Qt(2,2),ee(G)),[vt]:re(Qt(2,1/0),ee(G)),[St]:re(Qt(2,2),ee(G)),[Ct]:re(Qt(3,3),ee(G)),[bt]:re(Qt(2,2),ee(G)),[Mt]:re(Qt(2,2),ee(G)),[It]:re(Qt(1,1),ee(G)),[Et]:re(Qt(1,1),ee(G)),[kt]:re(Qt(1,1),ee(G)),[At]:re(Qt(1,1),ee(G)),[Pt]:re(Qt(1,1),ee(G)),[Ot]:re(Qt(1,1),ee(G)),[Rt]:re(Qt(1,2),ee(G)),[Lt]:re(Qt(1,1),ee(G)),[Dt]:re(Qt(4,1/0),ne,function(t,e,i){const n=t.length-1,r=et(t[t.length-1],e,i);let s=N|G|W;const o=new Array(n-2);for(let e=0;e<n-2;e+=2){try{s&=et(t[e+2],s,i).type}catch(t){throw new Error(`failed to parse argument ${e+1} of match expression: ${t.message}`)}if(0===s)throw new Error(\"no common type was found among the arguments of match expression\")}for(let e=0;e<n-2;e+=2){try{const n=et(t[e+2],s,i);o[e]=n}catch(t){throw new Error(`failed to parse argument ${e+1} of match expression: ${t.message}`)}try{const n=et(t[e+3],r.type,i);o[e+1]=n}catch(t){throw new Error(`failed to parse argument ${e+2} of match expression: ${t.message}`)}}const a=et(t[1],s,i);return[a,...o,r]}),[Ft]:re(Qt(3,3),ee(G)),[Tt]:re(Qt(6,1/0),ne,function(t,e,i){const n=t[1];let r;switch(n[0]){case\"linear\":r=1;break;case\"exponential\":const t=n[1];if(\"number\"!=typeof t||t<=0)throw new Error(`expected a number base for exponential interpolation, got ${JSON.stringify(t)} instead`);r=t;break;default:throw new Error(`invalid interpolation type: ${JSON.stringify(n)}`)}const s=new Z(G,r);let o;try{o=et(t[2],G,i)}catch(t){throw new Error(`failed to parse argument 1 in interpolate expression: ${t.message}`)}const a=new Array(t.length-3);for(let n=0;n<a.length;n+=2){try{const e=et(t[n+3],G,i);a[n]=e}catch(t){throw new Error(`failed to parse argument ${n+2} for interpolate expression: ${t.message}`)}try{const r=et(t[n+4],e,i);a[n+1]=r}catch(t){throw new Error(`failed to parse argument ${n+3} for interpolate expression: ${t.message}`)}}return[s,o,...a]}),[$t]:re(Qt(3,1/0),function(t,e,i){const n=t[0],r=t.length-1;if(r%2==0)throw new Error(`expected an odd number of arguments for ${n}, got ${r} instead`)},function(t,e,i){const n=et(t[t.length-1],e,i),r=new Array(t.length-1);for(let e=0;e<r.length-1;e+=2){try{const n=et(t[e+1],W,i);r[e]=n}catch(t){throw new Error(`failed to parse argument ${e} of case expression: ${t.message}`)}try{const s=et(t[e+2],n.type,i);r[e+1]=s}catch(t){throw new Error(`failed to parse argument ${e+1} of case expression: ${t.message}`)}}return r[r.length-1]=n,r}),[Wt]:re(Qt(2,2),function(t,e,i){let n,r=t[2];if(!Array.isArray(r))throw new Error('the second argument for the \"in\" operator must be an array');if(\"literal\"===r[0]){if(r=r[1],!Array.isArray(r))throw new Error('failed to parse \"in\" expression: the literal operator must be followed by an array')}else if(\"string\"==typeof r[0])throw new Error('for the \"in\" operator, a string array should be wrapped in a \"literal\" operator to disambiguate from expressions');n=\"string\"==typeof r[0]?N:G;const s=new Array(r.length);for(let t=0;t<s.length;t++)try{const e=et(r[t],n,i);s[t]=e}catch(e){throw new Error(`failed to parse haystack item ${t} for \"in\" expression: ${e.message}`)}const o=et(t[1],n,i);return[o,...s]}),[Gt]:re(Qt(1,1/0),ee(U)),[Nt]:re(Qt(1,1/0),ee(U)),[Xt]:re(Qt(1,1/0),ee(G)),[Yt]:re(Qt(1,4),ee(G)),[Ut]:re(Qt(1,3),ee(G)),[jt]:re(Qt(2,2),function(t,e,i){let n;try{n=et(t[1],G,i)}catch(t){throw new Error(`failed to parse first argument in palette expression: ${t.message}`)}const r=t[2];if(!Array.isArray(r))throw new Error(\"the second argument of palette must be an array\");const s=new Array(r.length);for(let t=0;t<s.length;t++){let e;try{e=et(r[t],X,i)}catch(e){throw new Error(`failed to parse color at index ${t} in palette expression: ${e.message}`)}if(!(e instanceof Z))throw new Error(`the palette color at index ${t} must be a literal value`);s[t]=e}return[n,...s]}),[Vt]:re(Qt(1,1),ee(W|G|N|X))};function Kt(t,e,i){const n=t.length-1,r=new Array(n);for(let s=0;s<n;++s){const n=t[s+1];switch(typeof n){case\"number\":r[s]=new Z(G,n);break;case\"string\":r[s]=new Z(N,n);break;default:throw new Error(`expected a string key or numeric array index for a get operation, got ${n}`)}0===s&&i.properties.set(String(n),e)}return r}function Ht(t,e,i){i.mapState=!0}function Zt(t,e,i){const n=t[0];if(1!==t.length)throw new Error(`expected no arguments for ${n} operation`);return[]}function Qt(t,e){return function(i,n,r){const s=i[0],o=i.length-1;if(t===e){if(o!==t){throw new Error(`expected ${t} argument${1===t?\"\":\"s\"} for ${s}, got ${o}`)}}else if(o<t||o>e){throw new Error(`expected ${e===1/0?`${t} or more`:`${t} to ${e}`} arguments for ${s}, got ${o}`)}}}function te(t,e,i){const n=t.length-1,r=new Array(n);for(let s=0;s<n;++s){const n=et(t[s+1],e,i);r[s]=n}return r}function ee(t){return function(e,i,n){const r=e.length-1,s=new Array(r);for(let i=0;i<r;++i){const r=et(e[i+1],t,n);s[i]=r}return s}}function ie(){return function(t,e,i){const n=t[0],r=t.length-1,s=new Array(r);let o=U;for(let e=0;e<r;++e){o&=et(t[e+1],o,i).type}if(0===o)throw new Error(`no common type was found among the arguments of ${n}`);for(let e=0;e<r;++e){const n=et(t[e+1],o,i);s[e]=n}return s}}function ne(t,e,i){const n=t[0],r=t.length-1;if(r%2==1)throw new Error(`expected an even number of arguments for operation ${n}, got ${r} instead`)}function re(...t){return function(e,i,n){const r=e[0];let s;for(let r=0;r<t.length;r++){const o=t[r](e,i,n);if(r==t.length-1){if(!o)throw new Error(\"expected last argument validator to return the parsed args\");s=o}}return new Q(i,r,...s)}}function se(t){if(!t)return\"\";const e=t.getType();switch(e){case\"Point\":case\"LineString\":case\"Polygon\":return e;case\"MultiPoint\":case\"MultiLineString\":case\"MultiPolygon\":return e.substring(5);case\"Circle\":return\"Polygon\";case\"GeometryCollection\":return se(t.getGeometries()[0]);default:return\"\"}}var oe=0,ae=1,le=2,he=4,ce=8,ue=16;function fe(t,e,i){let n,r;return n=e<t[0]?t[0]-e:t[2]<e?e-t[2]:0,r=i<t[1]?t[1]-i:t[3]<i?i-t[3]:0,n*n+r*r}function de(t,e){return ge(t,e[0],e[1])}function ge(t,e,i){return t[0]<=e&&e<=t[2]&&t[1]<=i&&i<=t[3]}function pe(t,e){const i=t[0],n=t[1],r=t[2],s=t[3],o=e[0],a=e[1];let l=oe;return o<i?l|=ue:o>r&&(l|=he),a<n?l|=ce:a>s&&(l|=le),l===oe&&(l=ae),l}function _e(t,e,i,n,r){return r?(r[0]=t,r[1]=e,r[2]=i,r[3]=n,r):[t,e,i,n]}function me(t){return _e(1/0,1/0,-1/0,-1/0,t)}function ye(t,e){const i=t[0],n=t[1];return _e(i,n,i,n,e)}function we(t,e,i,n,r){return xe(me(r),t,e,i,n)}function xe(t,e,i,n,r){for(;i<n;i+=r)ve(t,e[i],e[i+1]);return t}function ve(t,e,i){t[0]=Math.min(t[0],e),t[1]=Math.min(t[1],i),t[2]=Math.max(t[2],e),t[3]=Math.max(t[3],i)}function Se(t){return[(t[0]+t[2])/2,(t[1]+t[3])/2]}function Ce(t){return t[3]-t[1]}function be(t,e){return t[0]<=e[2]&&t[2]>=e[0]&&t[1]<=e[3]&&t[3]>=e[1]}function Me(t,e,i,n){let r=t[e],s=t[e+1],o=0;for(let a=e+n;a<i;a+=n){const e=t[a],i=t[a+1];o+=Math.sqrt((e-r)*(e-r)+(i-s)*(i-s)),r=e,s=i}return o}function Ie(t,e,i,n,r,s,o,a){o=o??[],a=a??n;const l=t[e+n],h=t[e+n+1],c=t[i-2*n],u=t[i-2*n+1];let f,d,g,p,_,m,y,w,x=0;for(let v=e;v<i;v+=n){g=f,p=d,_=void 0,m=void 0,v+n<i&&(_=t[v+n],m=t[v+n+1]),s&&v===e&&(g=c,p=u),s&&v===i-n&&(_=l,m=h),f=t[v],d=t[v+1],[y,w]=Ee(f,d,g,p,_,m,r),o[x++]=y,o[x++]=w;for(let e=2;e<a;e++)o[x++]=t[v+e]}return o.length!=x&&(o.length=x),o}function Ee(t,e,i,n,r,s,o){let a,l;void 0!==i&&void 0!==n?(a=t-i,l=e-n):void 0!==r&&void 0!==s?(a=r-t,l=s-e):(a=1,l=0);const h=Math.hypot(a,l),u=a/h,f=l/h;if(a=-f,l=u,void 0===i||void 0===n)return[t+a*o,e+l*o];if(void 0===r||void 0===s)return[t+a*o,e+l*o];const d=function(t,e,i){const n=Math.sqrt((e[0]-t[0])*(e[0]-t[0])+(e[1]-t[1])*(e[1]-t[1])),r=[(e[0]-t[0])/n,(e[1]-t[1])/n],s=[-r[1],r[0]],o=Math.sqrt((i[0]-t[0])*(i[0]-t[0])+(i[1]-t[1])*(i[1]-t[1])),a=[(i[0]-t[0])/o,(i[1]-t[1])/o];let l=0===n||0===o?0:Math.acos(c(a[0]*r[0]+a[1]*r[1],-1,1));return l=Math.max(l,1e-5),a[0]*s[0]+a[1]*s[1]>0?l:2*Math.PI-l}([t,e],[i,n],[r,s]);if(Math.cos(d)>.998)return[t+u*o,e+f*o];const g=Math.cos(d/2),p=Math.sin(d/2);return[t+(p*a+g*l)*(1/p)*o,e+(-g*a+p*l)*(1/p)*o]}function ke(t,e,i=!1){for(let n=0,r=t.length-2;n<r;n+=e){for(let r=i&&0===n?t.length-3*e:t.length-2*e;r>n+e;r-=e){const i=t[n],s=t[n+1],o=t[n+e],a=t[n+e+1],l=t[r],h=t[r+1],c=t[r+e],u=t[r+e+1],f=(u-h)*(o-i)-(c-l)*(a-s);if(0===f)continue;const d=((c-l)*(s-h)-(u-h)*(i-l))/f,g=((o-i)*(s-h)-(a-s)*(i-l))/f;if(d>0&&d<1&&g>0&&g<1){const l=i+d*(o-i),h=s+d*(a-s);t[n+e]=l,t[n+e+1]=h,t.splice(n+2*e,r-n-e);break}}}return t}function Ae(t,e,i,n,r,s,o){s=s||[],o=o||2;let a=0;for(let l=e;l<i;l+=n){const e=t[l],i=t[l+1];s[a++]=r[0]*e+r[2]*i+r[4],s[a++]=r[1]*e+r[3]*i+r[5];for(let e=2;e<o;e++)s[a++]=t[l+e]}return s&&s.length!=a&&(s.length=a),s}function Pe(t,e,i,n,r,s,o){o=o||[];const a=Math.cos(r),l=Math.sin(r),h=s[0],c=s[1];let u=0;for(let r=e;r<i;r+=n){const e=t[r]-h,i=t[r+1]-c;o[u++]=h+e*a-i*l,o[u++]=c+e*l+i*a;for(let e=r+2;e<r+n;++e)o[u++]=t[e]}return o&&o.length!=u&&(o.length=u),o}let Oe;function Re(t,e,i,n,r,s,o,a,l,h,c,u,f=!0){let d=t[e],g=t[e+1],_=0,m=0,y=0,w=0;function x(){_=d,m=g,d=t[e+=n],g=t[e+1],w+=y,y=Math.sqrt((d-_)*(d-_)+(g-m)*(g-m))}do{x()}while(e<i-n&&w+y<s);let v=0===y?0:(s-w)/y;const S=p(_,d,v),C=p(m,g,v),b=e-n,M=w,I=s+a*l(h,r,c);for(;e<i-n&&w+y<I;)x();v=0===y?0:(I-w)/y;const E=p(_,d,v),k=p(m,g,v);let A=!1;if(f)if(u){const t=[S,C,E,k];Pe(t,0,4,2,u,t,t),A=t[0]>t[2]}else A=S>E;const P=Math.PI,O=[],R=b+n===e;let L;if(y=0,w=M,d=t[e=b],g=t[e+1],R){x(),L=Math.atan2(g-m,d-_),A&&(L+=L>0?-P:P);const t=(E+S)/2,e=(k+C)/2;return O[0]=[t,e,(I-s)/2,L,r],O}r=r.replace(/\\n/g,\" \");const D=Array.from((Oe||(Oe=new Intl.Segmenter(void 0,{granularity:\"grapheme\"})),Oe).segment(r),t=>t.segment);for(let t=0,r=D.length;t<r;){x();let u=Math.atan2(g-m,d-_);if(A&&(u+=u>0?-P:P),void 0!==L){let t=u-L;if(t+=t>P?-2*P:t<-P?2*P:0,Math.abs(t)>o)return null}L=u;const f=t;let S=0;for(;t<r;++t){const o=a*l(h,D[A?r-t-1:t],c);if(e+n<i&&w+y<s+S+o/2)break;S+=o}if(t===f)continue;const C=(A?D.slice(r-t,r-f):D.slice(f,t)).join(\"\");v=0===y?0:(s+S/2-w)/y;const b=p(_,d,v),M=p(m,g,v);O.push([b,M,S/2,u,C]),s+=S}return O}function Le(t,e){if(!t)throw new Error(e)}const De=[1,0,0,1,0,0];function Fe(){return De.slice(0)}function Te(t,e){const i=t[0],n=t[1],r=t[2],s=t[3],o=t[4],a=t[5],l=e[0],h=e[1],c=e[2],u=e[3],f=e[4],d=e[5];return t[0]=i*l+r*h,t[1]=n*l+s*h,t[2]=i*c+r*u,t[3]=n*c+s*u,t[4]=i*f+r*d+o,t[5]=n*f+s*d+a,t}function ze(t,e){const i=e[0],n=e[1];return e[0]=t[0]*i+t[2]*n+t[4],e[1]=t[1]*i+t[3]*n+t[5],e}function $e(t,e,i,n,r,s,o,a){const l=Math.sin(s),h=Math.cos(s);return t[0]=n*h,t[1]=r*l,t[2]=-n*l,t[3]=r*h,t[4]=o*n*h-a*n*l+e,t[5]=o*r*l+a*r*h+i,t}function We(t){return function(t,e){const i=(n=e,n[0]*n[3]-n[1]*n[2]);var n;Le(0!==i,\"Transformation matrix cannot be inverted\");const r=e[0],s=e[1],o=e[2],a=e[3],l=e[4],h=e[5];return t[0]=a/i,t[1]=-s/i,t[2]=-o/i,t[3]=r/i,t[4]=(o*h-a*l)/i,t[5]=-(r*h-s*l)/i,t}(t,t)}new Array(6);var Ge=\"propertychange\";function Ne(t){for(const e in t)delete t[e]}function Xe(t){let e;for(e in t)return!1;return!e}function Ye(t,e,i,n,r){if(r){const n=i;i=function(r){return t.removeEventListener(e,i),n.call(this,r)}}const s={target:t,type:e,listener:i};return t.addEventListener(e,i),s}function Be(t,e,i,n){return Ye(t,e,i,0,!0)}function Ue(t){t&&t.target&&(t.target.removeEventListener(t.type,t.listener),Ne(t))}var je=\"change\";class Ve{constructor(){this.disposed=!1}dispose(){this.disposed||(this.disposed=!0,this.disposeInternal())}disposeInternal(){}}function qe(){}function Je(t){let e,i,r;return function(){const s=Array.prototype.slice.call(arguments);return i&&this===r&&n(s,i)||(r=this,i=s,e=t.apply(this,arguments)),e}}class Ke{constructor(t){this.propagationStopped,this.defaultPrevented,this.type=t,this.target=null}preventDefault(){this.defaultPrevented=!0}stopPropagation(){this.propagationStopped=!0}}class He extends Ve{constructor(t){super(),this.eventTarget_=t,this.pendingRemovals_=null,this.dispatching_=null,this.listeners_=null}addEventListener(t,e){if(!t||!e)return;const i=this.listeners_||(this.listeners_={}),n=i[t]||(i[t]=[]);n.includes(e)||n.push(e)}dispatchEvent(t){const e=\"string\"==typeof t,i=e?t:t.type,n=this.listeners_&&this.listeners_[i];if(!n)return;const r=e?new Ke(t):t;r.target||(r.target=this.eventTarget_||this);const s=this.dispatching_||(this.dispatching_={}),o=this.pendingRemovals_||(this.pendingRemovals_={});let a;i in s||(s[i]=0,o[i]=0),++s[i];for(let t=0,e=n.length;t<e;++t)if(a=\"handleEvent\"in n[t]?n[t].handleEvent(r):n[t].call(this,r),!1===a||r.propagationStopped){a=!1;break}if(0===--s[i]){let t=o[i];for(delete o[i];t--;)this.removeEventListener(i,qe);delete s[i]}return a}disposeInternal(){this.listeners_&&Ne(this.listeners_)}getListeners(t){return this.listeners_&&this.listeners_[t]||void 0}hasListener(t){return!!this.listeners_&&(t?t in this.listeners_:Object.keys(this.listeners_).length>0)}removeEventListener(t,e){if(!this.listeners_)return;const i=this.listeners_[t];if(!i)return;const n=i.indexOf(e);-1!==n&&(this.pendingRemovals_&&t in this.pendingRemovals_?(i[n]=qe,++this.pendingRemovals_[t]):(i.splice(n,1),0===i.length&&delete this.listeners_[t]))}}class Ze extends He{constructor(){super(),this.on=this.onInternal,this.once=this.onceInternal,this.un=this.unInternal,this.revision_=0}changed(){++this.revision_,this.dispatchEvent(je)}getRevision(){return this.revision_}onInternal(t,e){if(Array.isArray(t)){const i=t.length,n=new Array(i);for(let r=0;r<i;++r)n[r]=Ye(this,t[r],e);return n}return Ye(this,t,e)}onceInternal(t,e){let i;if(Array.isArray(t)){const n=t.length;i=new Array(n);for(let r=0;r<n;++r)i[r]=Be(this,t[r],e)}else i=Be(this,t,e);return e.ol_key=i,i}unInternal(t,e){const i=e.ol_key;if(i)!function(t){if(Array.isArray(t))for(let e=0,i=t.length;e<i;++e)Ue(t[e]);else Ue(t)}(i);else if(Array.isArray(t))for(let i=0,n=t.length;i<n;++i)this.removeEventListener(t[i],e);else this.removeEventListener(t,e)}}function Qe(){throw new Error(\"Unimplemented abstract method.\")}let ti=0;function ei(t){return t.ol_uid||(t.ol_uid=String(++ti))}class ii extends Ke{constructor(t,e,i){super(t),this.key=e,this.oldValue=i}}class ni extends Ze{constructor(t){super(),this.on,this.once,this.un,ei(this),this.values_=null,void 0!==t&&this.setProperties(t)}get(t){let e;return this.values_&&this.values_.hasOwnProperty(t)&&(e=this.values_[t]),e}getKeys(){return this.values_&&Object.keys(this.values_)||[]}getProperties(){return this.values_&&Object.assign({},this.values_)||{}}getPropertiesInternal(){return this.values_}hasProperties(){return!!this.values_}notify(t,e){let i;i=`change:${t}`,this.hasListener(i)&&this.dispatchEvent(new ii(i,t,e)),i=Ge,this.hasListener(i)&&this.dispatchEvent(new ii(i,t,e))}addChangeListener(t,e){this.addEventListener(`change:${t}`,e)}removeChangeListener(t,e){this.removeEventListener(`change:${t}`,e)}set(t,e,i){const n=this.values_||(this.values_={});if(i)n[t]=e;else{const i=n[t];n[t]=e,i!==e&&this.notify(t,i)}}setProperties(t,e){for(const i in t)this.set(i,t[i],e)}applyProperties(t){t.values_&&Object.assign(this.values_||(this.values_={}),t.values_)}unset(t,e){if(this.values_&&t in this.values_){const i=this.values_[t];delete this.values_[t],Xe(this.values_)&&(this.values_=null),e||this.notify(t,i)}}}const ri=new RegExp([\"^\\\\s*(?=(?:(?:[-a-z]+\\\\s*){0,2}(italic|oblique))?)\",\"(?=(?:(?:[-a-z]+\\\\s*){0,2}(small-caps))?)\",\"(?=(?:(?:[-a-z]+\\\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)\",\"(?:(?:normal|\\\\1|\\\\2|\\\\3)\\\\s*){0,3}((?:xx?-)?\",\"(?:small|large)|medium|smaller|larger|[\\\\.\\\\d]+(?:\\\\%|in|[cem]m|ex|p[ctx]))\",\"(?:\\\\s*\\\\/\\\\s*(normal|[\\\\.\\\\d]+(?:\\\\%|in|[cem]m|ex|p[ctx])?))\",\"?\\\\s*([-,\\\\\\\"\\\\'\\\\sa-z0-9]+?)\\\\s*$\"].join(\"\"),\"i\"),si=[\"style\",\"variant\",\"weight\",\"size\",\"lineHeight\",\"family\"],oi={normal:400,bold:700},ai=function(t){const e=t.match(ri);if(!e)return null;const i={lineHeight:\"normal\",size:\"1.2em\",style:\"normal\",weight:\"400\",variant:\"normal\"};for(let t=0,n=si.length;t<n;++t){const n=e[t+1];void 0!==n&&(i[si[t]]=\"string\"==typeof n?n.trim():n)}return isNaN(Number(i.weight))&&i.weight in oi&&(i.weight=oi[i.weight]),i.families=i.family.split(/,\\s?/).map(t=>t.trim().replace(/^['\"]|['\"]$/g,\"\")),i},li=\"#000\",hi=\"round\",ci=[],ui=\"round\",fi=\"#000\",di=\"center\",gi=\"middle\",pi=[0,0,0,0],_i=new ni;let mi,yi=null;const wi={},xi=new Set([\"serif\",\"sans-serif\",\"monospace\",\"cursive\",\"fantasy\",\"system-ui\",\"ui-serif\",\"ui-sans-serif\",\"ui-monospace\",\"ui-rounded\",\"emoji\",\"math\",\"fangsong\"]);function vi(t,e,i){return`${t} ${e} 16px \"${i}\"`}const Si=function(){const t=100;let e,i;async function n(t){await i.ready;const e=ai(t),n=e.families[0].toLowerCase(),r=e.weight,s=[];if(i.forEach(t=>{const i=t.family.replace(/^['\"]|['\"]$/g,\"\").toLowerCase(),o=oi[t.weight]||t.weight;i===n&&t.style===e.style&&o==r&&s.push(t)}),0===s.length)return!1;return(await Promise.all(s.map(t=>t.load().then(()=>!0,()=>!1)))).some(t=>t)}async function r(){await i.ready;let s=!0;const o=_i.getProperties(),a=Object.keys(o).filter(e=>o[e]<t);for(let e=a.length-1;e>=0;--e){const i=a[e];let r=o[i];r<t&&(await n(i)?(Ne(wi),_i.set(i,t)):(r+=10,_i.set(i,r,!0),r<t&&(s=!1)))}e=void 0,s||(e=setTimeout(r,100))}return async function(t){i||(i=s?self.fonts:document.fonts);const n=ai(t);if(!n)return;const o=n.families;let a=!1;for(const t of o){if(xi.has(t))continue;const e=vi(n.style,n.weight,t);void 0===_i.get(e)&&(_i.set(e,0,!0),a=!0)}a&&(clearTimeout(e),e=setTimeout(r,100))}}(),Ci=function(){let t;return function(e){let i=wi[e];if(null==i){if(s){const t=ai(e),n=bi(e,\"Žg\");i=(isNaN(Number(t.lineHeight))?1.2:Number(t.lineHeight))*(n.actualBoundingBoxAscent+n.actualBoundingBoxDescent)}else t||(t=document.createElement(\"div\"),t.innerHTML=\"M\",t.style.minHeight=\"0\",t.style.maxHeight=\"none\",t.style.height=\"auto\",t.style.padding=\"0\",t.style.border=\"none\",t.style.position=\"absolute\",t.style.display=\"block\",t.style.left=\"-99999px\"),t.style.font=e,document.body.appendChild(t),i=t.offsetHeight,document.body.removeChild(t);wi[e]=i}return i}}();function bi(t,e){return yi||(yi=a(1,1)),t!=mi&&(yi.font=t,mi=yi.font),yi.measureText(e)}function Mi(t,e){return bi(t,e).width}function Ii(t,e,i){if(e in i)return i[e];const n=e.split(\"\\n\").reduce((e,i)=>Math.max(e,Mi(t,i)),0);return i[e]=n,n}function Ei(t,e,i,n,r,s,o,a,l,h,c){t.save(),1!==i&&(void 0===t.globalAlpha?t.globalAlpha=t=>t.globalAlpha*=i:t.globalAlpha*=i),e&&t.transform.apply(t,e),n.contextInstructions?(t.translate(l,h),t.scale(c[0],c[1]),function(t,e){const i=t.contextInstructions;for(let t=0,n=i.length;t<n;t+=2)Array.isArray(i[t+1])?e[i[t]].apply(e,i[t+1]):e[i[t]]=i[t+1]}(n,t)):c[0]<0||c[1]<0?(t.translate(l,h),t.scale(c[0],c[1]),t.drawImage(n,r,s,o,a,0,0,o,a)):t.drawImage(n,r,s,o,a,l,h,o*c[0],a*c[1]),t.restore()}class ki{constructor(){this.instructions_=[],this.zIndex=0,this.offset_=0,this.pendingMethod_,this.context_=new Proxy(h(),{get:(t,e)=>{if(\"function\"==typeof t[e])return this.pendingMethod_=e,this.pushMethodArgs_},set:(t,e,i)=>(this.push_(e,i),!0)})}push_(...t){const e=this.instructions_,i=this.zIndex+this.offset_;e[i]||(e[i]=[]),e[i].push(...t)}pushMethodArgs_=(...t)=>{this.push_(this.pendingMethod_,t)};pushFunction(t){this.push_(t)}getContext(){return this.context_}draw(t){this.instructions_.forEach(e=>{for(let i=0,n=e.length;i<n;++i){const n=e[i];if(\"function\"==typeof n){n(t);continue}const r=e[++i];\"function\"==typeof t[n]?t[n](...r):t[n]=\"function\"==typeof r?r(t):r}})}clear(){this.instructions_.length=0,this.zIndex=0,this.offset_=0}offset(){this.offset_=this.instructions_.length,this.zIndex=0}}const Ai=0,Pi=1,Oi=2,Ri=3,Li=4,Di=5,Fi=6,Ti=7,zi=8,$i=9,Wi=10,Gi=11,Ni=12;var Xi=0,Yi=1,Bi=2,Ui=3;function ji(t,e){return e&&(t.src=e),t.src&&o?new Promise((e,i)=>t.decode().then(()=>e(t)).catch(n=>t.complete&&t.width?e(t):i(n))):function(t){return new Promise((e,i)=>{function n(){s(),e(t)}function r(){s(),i(new Error(\"Image load error\"))}function s(){t.removeEventListener(\"load\",n),t.removeEventListener(\"error\",r)}t.addEventListener(\"load\",n),t.addEventListener(\"error\",r)})}(t)}function Vi(t,e){return t+\":\"+(e?F(e):\"null\")}const qi=new class{constructor(){this.cache_={},this.patternCache_={},this.cacheSize_=0,this.maxCacheSize_=1024}clear(){this.cache_={},this.patternCache_={},this.cacheSize_=0}canExpireCache(){return this.cacheSize_>this.maxCacheSize_}expire(){if(this.canExpireCache()){let t=0;for(const e in this.cache_){const i=this.cache_[e];3&t++||i.hasListener()||(delete this.cache_[e],delete this.patternCache_[e],--this.cacheSize_)}}}get(t,e){const i=Vi(t,e);return i in this.cache_?this.cache_[i]:null}getPattern(t,e){const i=Vi(t,e);return i in this.patternCache_?this.patternCache_[i]:null}set(t,e,i,n){const r=Vi(t,e),s=r in this.cache_;this.cache_[r]=i,n&&(i.getImageState()===Xi&&i.load(),i.getImageState()===Yi?i.ready().then(()=>{this.patternCache_[r]=h().createPattern(i.getImage(1),\"repeat\")}):this.patternCache_[r]=h().createPattern(i.getImage(1),\"repeat\")),s||++this.cacheSize_}setSize(t){this.maxCacheSize_=t,this.expire()}};let Ji=null;class Ki extends He{constructor(t,e,i,n,r){super(),this.hitDetectionImage_=null,this.image_=t,this.crossOrigin_=i?.crossOrigin,this.referrerPolicy_=i?.referrerPolicy,this.canvas_={},this.color_=r,this.imageState_=void 0===n?Xi:n,this.size_=t&&t.width&&t.height?[t.width,t.height]:null,this.src_=e,this.tainted_,this.ready_=null}initializeImage_(){this.image_=new Image,null!==this.crossOrigin_&&(this.image_.crossOrigin=this.crossOrigin_),void 0!==this.referrerPolicy_&&(this.image_.referrerPolicy=this.referrerPolicy_)}isTainted_(){if(void 0===this.tainted_&&this.imageState_===Bi){Ji||(Ji=a(1,1,0,{willReadFrequently:!0})),Ji.drawImage(this.image_,0,0);try{Ji.getImageData(0,0,1,1),this.tainted_=!1}catch{Ji=null,this.tainted_=!0}}return!0===this.tainted_}dispatchChangeEvent_(){this.dispatchEvent(je)}handleImageError_(){this.imageState_=Ui,this.dispatchChangeEvent_()}handleImageLoad_(){this.imageState_=Bi,this.size_=[this.image_.width,this.image_.height],this.dispatchChangeEvent_()}getImage(t){return this.image_||this.initializeImage_(),this.replaceColor_(t),this.canvas_[t]?this.canvas_[t]:this.image_}setImage(t){this.image_=t}getPixelRatio(t){return this.replaceColor_(t),this.canvas_[t]?t:1}getImageState(){return this.imageState_}getHitDetectionImage(){if(this.image_||this.initializeImage_(),!this.hitDetectionImage_)if(this.isTainted_()){const t=this.size_[0],e=this.size_[1],i=a(t,e);i.fillRect(0,0,t,e),this.hitDetectionImage_=i.canvas}else this.hitDetectionImage_=this.image_;return this.hitDetectionImage_}getSize(){return this.size_}getSrc(){return this.src_}load(){if(this.imageState_===Xi){this.image_||this.initializeImage_(),this.imageState_=Yi;try{void 0!==this.src_&&(this.image_.src=this.src_)}catch{this.handleImageError_()}this.image_ instanceof HTMLImageElement&&ji(this.image_,this.src_).then(t=>{this.image_=t,this.handleImageLoad_()}).catch(this.handleImageError_.bind(this))}}replaceColor_(t){if(!this.color_||this.canvas_[t]||this.imageState_!==Bi)return;const e=this.image_,i=a(Math.ceil(e.width*t),Math.ceil(e.height*t)),n=i.canvas;var r;i.scale(t,t),i.drawImage(e,0,0),i.globalCompositeOperation=\"multiply\",i.fillStyle=\"string\"==typeof(r=this.color_)?r:T(r),i.fillRect(0,0,n.width/t,n.height/t),i.globalCompositeOperation=\"destination-in\",i.drawImage(e,0,0),this.canvas_[t]=n}ready(){return this.ready_||(this.ready_=new Promise(t=>{if(this.imageState_===Bi||this.imageState_===Ui)t();else{const e=()=>{this.imageState_!==Bi&&this.imageState_!==Ui||(this.removeEventListener(je,e),t())};this.addEventListener(je,e)}})),this.ready_}}function Hi(t,e,i,n,r,s){let o=void 0===e?void 0:qi.get(e,r);return o||(o=new Ki(t,t&&\"src\"in t?t.src||void 0:e,i,n,r),qi.set(e,r,o,s)),s&&o&&!qi.getPattern(e,r)&&qi.set(e,r,o,s),o}function Zi(t){return t?Array.isArray(t)?T(t):\"object\"==typeof t&&\"src\"in t?function(t){if(!t.offset||!t.size)return qi.getPattern(t.src,t.color);const e=t.src+\":\"+t.offset,i=qi.getPattern(e,t.color);if(i)return i;const n=qi.get(t.src,null);if(n.getImageState()!==Bi)return null;const r=a(t.size[0],t.size[1]);return r.drawImage(n.getImage(1),t.offset[0],t.offset[1],t.size[0],t.size[1],0,0,t.size[0],t.size[1]),Hi(r.canvas,e,void 0,Bi,t.color,!0),qi.getPattern(e,t.color)}(t):t:null}let Qi=0,tn=1;function en(t,e,i,n,r,s,o,a){const l=o-r,h=a-s;let c=0,u=1;if(0===l){if(r<t||r>i)return!1}else{let e=(t-r)/l,n=(i-r)/l;if(e>n){const t=e;e=n,n=t}if(e>c&&(c=e),n<u&&(u=n),c>u)return!1}if(0===h){if(s<e||s>n)return!1}else{let t=(e-s)/h,i=(n-s)/h;if(t>i){const e=t;t=i,i=e}if(t>c&&(c=t),i<u&&(u=i),c>u)return!1}return Qi=c,tn=u,!0}function nn(t,e,i,n,r){const s=[];let o=i,a=0,l=e.slice(i,2);for(;a<t&&o+r<n;){const[i,n]=l.slice(-2),h=e[o+r],c=e[o+r+1],u=Math.sqrt((h-i)*(h-i)+(c-n)*(c-n));if(a+=u,a>=t){const e=(t-a+u)/u,f=p(i,h,e),d=p(n,c,e);l.push(f,d),s.push(l),l=[f,d],a==t&&(o+=r),a=0}else if(a<t)l.push(e[o+r],e[o+r+1]),o+=r;else{const t=u-a,e=p(i,h,t/u),f=p(n,c,t/u);l.push(e,f),s.push(l),l=[e,f],a=0,o+=r}}return a>0&&s.push(l),s}function rn(t,e,i,n,r){let s,o,a,l,h,c,u,f,d,g,p=i,_=i,m=0,y=0,w=i;for(o=i;o<n;o+=r){const i=e[o],n=e[o+1];void 0!==h&&(d=i-h,g=n-c,l=Math.sqrt(d*d+g*g),void 0!==u&&(y+=a,s=Math.acos((u*d+f*g)/(a*l)),s>t&&(y>m&&(m=y,p=w,_=o),y=0,w=o-r)),a=l,u=d,f=g),h=i,c=n}return y+=l,y>m?[w,o]:[p,_]}function sn(t,e,i,n,r){r=void 0!==r?r:[];let s=0;for(let o=e;o<i;o+=n)r[s++]=t.slice(o,o+n);return r.length=s,r}function on(t,e,i,n,r){r=void 0!==r?r:[];let s=0;for(let o=0,a=i.length;o<a;++o){const a=i[o];r[s++]=sn(t,e,a,n,r[s]),e=a}return r.length=s,r}function an(t,e,i,n,r){r=void 0!==r?r:[];let s=0;for(let o=0,a=i.length;o<a;++o){const a=i[o];r[s++]=1===a.length&&a[0]===e?[]:on(t,e,a,n,r[s]),e=a[a.length-1]}return r.length=s,r}class ln{drawCustom(t,e,i,n,r){}drawGeometry(t){}setStyle(t){}drawCircle(t,e,i){}drawFeature(t,e,i){}drawGeometryCollection(t,e,i){}drawLineString(t,e,i){}drawMultiLineString(t,e,i){}drawMultiPoint(t,e,i){}drawMultiPolygon(t,e,i){}drawPoint(t,e,i){}drawPolygon(t,e,i){}drawText(t,e,i){}setFillStrokeStyle(t,e){}setImageStyle(t,e){}setTextStyle(t,e){}}class hn extends ln{constructor(t,e,i,n){super(),this.tolerance=t,this.maxExtent=e,this.pixelRatio=n,this.maxLineWidth=0,this.resolution=i,this.beginGeometryInstruction1_=null,this.beginGeometryInstruction2_=null,this.bufferedMaxExtent_=null,this.instructions=[],this.coordinates=[],this.tmpCoordinate_=[],this.hitDetectionInstructions=[],this.state={}}applyPixelRatio(t){const e=this.pixelRatio;return 1==e?t:t.map(function(t){return t*e})}appendFlatPointCoordinates(t,e){const i=this.getBufferedMaxExtent(),n=this.tmpCoordinate_,r=this.coordinates;let s=r.length;for(let o=0,a=t.length;o<a;o+=e)n[0]=t[o],n[1]=t[o+1],de(i,n)&&(r[s++]=n[0],r[s++]=n[1]);return s}appendFlatLineCoordinates(t,e,i,n,r,s){const o=this.coordinates;let a=o.length;const l=this.getBufferedMaxExtent();s&&(e+=n);let h=t[e],c=t[e+1];const u=this.tmpCoordinate_;let f,d,g,p=!0;for(f=e+n;f<i;f+=n)u[0]=t[f],u[1]=t[f+1],g=pe(l,u),g!==d?(p&&(o[a++]=h,o[a++]=c,p=!1),o[a++]=u[0],o[a++]=u[1]):g===ae?(o[a++]=u[0],o[a++]=u[1],p=!1):p=!0,h=u[0],c=u[1],d=g;return(r&&p||f===e+n)&&(o[a++]=h,o[a++]=c),a}drawCustomCoordinates_(t,e,i,n,r){for(let s=0,o=i.length;s<o;++s){const o=i[s],a=this.appendFlatLineCoordinates(t,e,o,n,!1,!1);r.push(a),e=o}return e}drawCustom(t,e,i,n,r){this.beginGeometry(t,e,r);const s=t.getType(),o=t.getStride(),a=this.coordinates.length;let l,h,c,u,f;switch(s){case\"MultiPolygon\":l=t.getOrientedFlatCoordinates(),u=[];const e=t.getEndss();f=0;for(let t=0,i=e.length;t<i;++t){const i=[];f=this.drawCustomCoordinates_(l,f,e[t],o,i),u.push(i)}this.instructions.push([Li,a,u,t,i,an,r]),this.hitDetectionInstructions.push([Li,a,u,t,n||i,an,r]);break;case\"Polygon\":case\"MultiLineString\":c=[],l=\"Polygon\"==s?t.getOrientedFlatCoordinates():t.getFlatCoordinates(),f=this.drawCustomCoordinates_(l,0,t.getEnds(),o,c),this.instructions.push([Li,a,c,t,i,on,r]),this.hitDetectionInstructions.push([Li,a,c,t,n||i,on,r]);break;case\"LineString\":case\"Circle\":l=t.getFlatCoordinates(),h=this.appendFlatLineCoordinates(l,0,l.length,o,!1,!1),this.instructions.push([Li,a,h,t,i,sn,r]),this.hitDetectionInstructions.push([Li,a,h,t,n||i,sn,r]);break;case\"MultiPoint\":l=t.getFlatCoordinates(),h=this.appendFlatPointCoordinates(l,o),h>a&&(this.instructions.push([Li,a,h,t,i,sn,r]),this.hitDetectionInstructions.push([Li,a,h,t,n||i,sn,r]));break;case\"Point\":l=t.getFlatCoordinates(),this.coordinates.push(l[0],l[1]),h=this.coordinates.length,this.instructions.push([Li,a,h,t,i,void 0,r]),this.hitDetectionInstructions.push([Li,a,h,t,n||i,void 0,r])}this.endGeometry(e)}beginGeometry(t,e,i){this.beginGeometryInstruction1_=[Ai,e,0,t,i],this.instructions.push(this.beginGeometryInstruction1_),this.beginGeometryInstruction2_=[Ai,e,0,t,i],this.hitDetectionInstructions.push(this.beginGeometryInstruction2_)}finish(){return{instructions:this.instructions,hitDetectionInstructions:this.hitDetectionInstructions,coordinates:this.coordinates}}reverseHitDetectionInstructions(){const t=this.hitDetectionInstructions;let i;t.reverse();const n=t.length;let r,s,o=-1;for(i=0;i<n;++i)r=t[i],s=r[0],s==Ti?o=i:s==Ai&&(r[2]=i,e(this.hitDetectionInstructions,o,i),o=-1)}fillStyleToState(t,e={}){if(t){const i=t.getColor();e.fillPatternScale=i&&\"object\"==typeof i&&\"src\"in i?this.pixelRatio:1,e.fillStyle=Zi(i||li)??void 0}else e.fillStyle=void 0;return e}strokeStyleToState(t,e={}){if(t){const i=t.getColor();e.strokeStyle=Zi(i||fi);const n=t.getLineCap();e.lineCap=void 0!==n?n:hi;const r=t.getLineDash();e.lineDash=r?r.slice():ci;const s=t.getLineDashOffset();e.lineDashOffset=s||0;const o=t.getLineJoin();e.lineJoin=void 0!==o?o:ui;const a=t.getWidth();e.lineWidth=void 0!==a?a:1;const l=t.getMiterLimit();e.miterLimit=void 0!==l?l:10;const h=t.getOffset();e.strokeOffset=h??0,e.lineWidth>this.maxLineWidth&&(this.maxLineWidth=e.lineWidth,this.bufferedMaxExtent_=null)}else e.strokeStyle=void 0,e.lineCap=void 0,e.lineDash=null,e.lineDashOffset=void 0,e.lineJoin=void 0,e.lineWidth=void 0,e.miterLimit=void 0,e.strokeOffset=void 0;return e}setFillStrokeStyle(t,e){const i=this.state;this.fillStyleToState(t,i),this.strokeStyleToState(e,i)}createFill(t){const e=t.fillStyle,i=[Wi,e];return\"string\"!=typeof e&&i.push(t.fillPatternScale),i}applyStroke(t){this.instructions.push(this.createStroke(t))}createStroke(t){return[Gi,t.strokeStyle,t.lineWidth*this.pixelRatio,t.lineCap,t.lineJoin,t.miterLimit,t.lineDash?this.applyPixelRatio(t.lineDash):null,t.lineDashOffset*this.pixelRatio]}updateFillStyle(t,e){const i=t.fillStyle;(void 0!==i&&\"string\"!=typeof i||t.currentFillStyle!=i)&&(this.instructions.push(e.call(this,t)),t.currentFillStyle=i)}updateStrokeStyle(t,e){const i=t.strokeStyle,r=t.lineCap,s=t.lineDash,o=t.lineDashOffset,a=t.lineJoin,l=t.lineWidth,h=t.miterLimit,c=t.strokeOffset;(t.currentStrokeStyle!=i||t.currentLineCap!=r||s!=t.currentLineDash&&!n(t.currentLineDash,s)||t.currentLineDashOffset!=o||t.currentLineJoin!=a||t.currentLineWidth!=l||t.currentMiterLimit!=h||t.currentStrokeOffset!=c)&&(e.call(this,t),t.currentStrokeStyle=i,t.currentLineCap=r,t.currentLineDash=s,t.currentLineDashOffset=o,t.currentLineJoin=a,t.currentLineWidth=l,t.currentMiterLimit=h,t.currentStrokeOffset=c)}endGeometry(t){this.beginGeometryInstruction1_[2]=this.instructions.length,this.beginGeometryInstruction1_=null,this.beginGeometryInstruction2_[2]=this.hitDetectionInstructions.length,this.beginGeometryInstruction2_=null;const e=[Ti,t];this.instructions.push(e),this.hitDetectionInstructions.push(e)}getBufferedMaxExtent(){if(!this.bufferedMaxExtent_&&(this.bufferedMaxExtent_=this.maxExtent.slice(),this.maxLineWidth>0)){const t=this.resolution*(this.maxLineWidth+1)/2;!function(t,e,i){i?(i[0]=t[0]-e,i[1]=t[1]-e,i[2]=t[2]+e,i[3]=t[3]+e):(t[0],t[1],t[2],t[3])}(this.bufferedMaxExtent_,t,this.bufferedMaxExtent_)}return this.bufferedMaxExtent_}}const cn={left:0,center:.5,right:1,top:0,middle:.5,hanging:.2,alphabetic:.8,ideographic:.8,bottom:1};class un extends hn{constructor(t,e,i,n){super(t,e,i,n),this.labels_=null,this.text_=\"\",this.textOffsetX_=0,this.textOffsetY_=0,this.textRotateWithView_=void 0,this.textKeepUpright_=void 0,this.textRotation_=0,this.textFillState_=null,this.fillStates={},this.fillStates[li]={fillStyle:li},this.textStrokeState_=null,this.strokeStates={},this.textState_={},this.textStates={},this.textKey_=\"\",this.fillKey_=\"\",this.strokeKey_=\"\",this.declutterMode_=void 0,this.declutterImageWithText_=void 0}finish(){const t=super.finish();return t.textStates=this.textStates,t.fillStates=this.fillStates,t.strokeStates=this.strokeStates,t}drawText(t,e,i){const n=this.textFillState_,r=this.textStrokeState_,s=this.textState_;if(\"\"===this.text_||!s||!n&&!r)return;const o=this.coordinates;let a=o.length;const l=t.getType();let h=null,c=t.getStride();if(\"line\"!==s.placement||\"LineString\"!=l&&\"MultiLineString\"!=l&&\"Polygon\"!=l&&\"MultiPolygon\"!=l){let n=s.overflow?null:[];switch(l){case\"Point\":case\"MultiPoint\":h=t.getFlatCoordinates();break;case\"LineString\":h=t.getFlatMidpoint();break;case\"Circle\":h=t.getCenter();break;case\"MultiLineString\":h=t.getFlatMidpoints(),c=2;break;case\"Polygon\":h=t.getFlatInteriorPoint(),s.overflow||n.push(h[2]/this.resolution),c=3;break;case\"MultiPolygon\":const e=t.getFlatInteriorPoints();h=[];for(let t=0,i=e.length;t<i;t+=3)s.overflow||n.push(e[t+2]/this.resolution),h.push(e[t],e[t+1]);if(0===h.length)return;c=2}const r=this.appendFlatPointCoordinates(h,c);if(r===a)return;if(n&&(r-a)/2!==h.length/c){let t=a/2;n=n.filter((e,i)=>{const n=o[2*(t+i)]===h[i*c]&&o[2*(t+i)+1]===h[i*c+1];return n||--t,n})}this.saveTextStates_();const u=s.backgroundFill?this.createFill(this.fillStyleToState(s.backgroundFill)):null,f=s.backgroundStroke?this.createStroke(this.strokeStyleToState(s.backgroundStroke)):null;this.beginGeometry(t,e,i);let d=s.padding;if(d!=pi&&(s.scale[0]<0||s.scale[1]<0)){let t=s.padding[0],e=s.padding[1],i=s.padding[2],n=s.padding[3];s.scale[0]<0&&(e=-e,n=-n),s.scale[1]<0&&(t=-t,i=-i),d=[t,e,i,n]}const g=this.pixelRatio;this.instructions.push([Fi,a,r,null,NaN,NaN,NaN,1,0,0,this.textRotateWithView_,this.textRotation_,[1,1],NaN,this.declutterMode_,this.declutterImageWithText_,d==pi?pi:d.map(function(t){return t*g}),u,f,this.text_,this.textKey_,this.strokeKey_,this.fillKey_,this.textOffsetX_,this.textOffsetY_,n]);const p=1/g,_=u?u.slice(0):null;_&&(_[1]=li),this.hitDetectionInstructions.push([Fi,a,r,null,NaN,NaN,NaN,1,0,0,this.textRotateWithView_,this.textRotation_,[p,p],NaN,this.declutterMode_,this.declutterImageWithText_,d,_,f,this.text_,this.textKey_,this.strokeKey_,this.fillKey_?li:this.fillKey_,this.textOffsetX_,this.textOffsetY_,n]),this.endGeometry(e)}else{const n=t.getExtent();if(!be(this.maxExtent,n))return;let r;if(h=t.getFlatCoordinates(),\"LineString\"==l)r=[h.length];else if(\"MultiLineString\"==l)r=t.getEnds();else if(\"Polygon\"==l)r=t.getEnds().slice(0,1);else if(\"MultiPolygon\"==l){const e=t.getEndss();r=[];for(let t=0,i=e.length;t<i;++t)r.push(e[t][0])}if(!(\"LineString\"!=l&&\"MultiLineString\"!=l||(u=this.getBufferedMaxExtent(),f=n,u[0]<=f[0]&&f[2]<=u[2]&&u[1]<=f[1]&&f[3]<=u[3]))){const t=function(t,e,i,n){const r=n[0],s=n[1],o=n[2],a=n[3],l=[],h=[];let c,u,f=!1,d=0;for(let n=0,g=e.length;n<g;++n){const g=e[n];let p=t[d],_=t[d+1],m=!1;for(let e=d+i;e<g;e+=i){const i=t[e],n=t[e+1];if(en(r,s,o,a,p,_,i,n)){const t=i-p,e=n-_,r=p+Qi*t,s=_+Qi*e,o=p+tn*t,a=_+tn*e;f&&m&&r===c&&s===u?l.push(o,a):(f&&h.push(l.length),l.push(r,s,o,a),f=!0),c=o,u=a,m=!0}p=i,_=n}d=g}return f&&h.push(l.length),{flatCoordinates:l,ends:h}}(h,r,c,this.getBufferedMaxExtent());if(h=t.flatCoordinates,r=t.ends,c=2,0===r.length)return}this.beginGeometry(t,e,i);const d=s.repeat,g=d?void 0:s.textAlign;let p=0;for(let t=0,e=r.length;t<e;++t){let e;e=d?nn(d*this.resolution,h,p,r[t],c):[h.slice(p,r[t])];for(let i=0,n=e.length;i<n;++i){const n=e[i];let l=0,h=n.length;if(null==g){const t=rn(s.maxAngle,n,0,n.length,2);l=t[0],h=t[1]}for(let t=l;t<h;t+=c)o.push(n[t],n[t+1]);const u=o.length;p=r[t],this.drawChars_(a,u),a=u}}this.endGeometry(e)}var u,f}saveTextStates_(){const t=this.textStrokeState_,e=this.textState_,i=this.textFillState_,n=this.strokeKey_;t&&(n in this.strokeStates||(this.strokeStates[n]={strokeStyle:t.strokeStyle,lineCap:t.lineCap,lineDashOffset:t.lineDashOffset,lineWidth:t.lineWidth,lineJoin:t.lineJoin,miterLimit:t.miterLimit,lineDash:t.lineDash}));const r=this.textKey_;r in this.textStates||(this.textStates[r]={font:e.font,textAlign:e.textAlign||di,justify:e.justify,textBaseline:e.textBaseline||gi,scale:e.scale});const s=this.fillKey_;i&&(s in this.fillStates||(this.fillStates[s]={fillStyle:i.fillStyle}))}drawChars_(t,e){const i=this.textStrokeState_,n=this.textState_,r=this.strokeKey_,s=this.textKey_,o=this.fillKey_;this.saveTextStates_();const a=this.pixelRatio,l=cn[n.textBaseline],h=this.textOffsetX_*a,c=this.textOffsetY_*a,u=this.text_,f=i?i.lineWidth*Math.abs(n.scale[0])/2:0;this.instructions.push([Di,t,e,l,n.overflow,o,n.maxAngle,a,c,r,f*a,u,s,1,this.declutterMode_,this.textKeepUpright_,h]),this.hitDetectionInstructions.push([Di,t,e,l,n.overflow,o?li:o,n.maxAngle,a,c,r,f*a,u,s,1/a,this.declutterMode_,this.textKeepUpright_,h])}setTextStyle(t,e){let i,n,r;if(t){const e=t.getFill();e?(n=this.textFillState_,n||(n={},this.textFillState_=n),n.fillStyle=Zi(e.getColor()||li)):(n=null,this.textFillState_=n);const s=t.getStroke();if(s){r=this.textStrokeState_,r||(r={},this.textStrokeState_=r);const t=s.getLineDash(),e=s.getLineDashOffset(),i=s.getWidth(),n=s.getMiterLimit();r.lineCap=s.getLineCap()||hi,r.lineDash=t?t.slice():ci,r.lineDashOffset=void 0===e?0:e,r.lineJoin=s.getLineJoin()||ui,r.lineWidth=void 0===i?1:i,r.miterLimit=void 0===n?10:n,r.strokeStyle=Zi(s.getColor()||fi)}else r=null,this.textStrokeState_=r;i=this.textState_;const o=t.getFont()||\"10px sans-serif\";Si(o);const a=t.getScaleArray();i.overflow=t.getOverflow(),i.font=o,i.maxAngle=t.getMaxAngle(),i.placement=t.getPlacement(),i.textAlign=t.getTextAlign(),i.repeat=t.getRepeat(),i.justify=t.getJustify(),i.textBaseline=t.getTextBaseline()||gi,i.backgroundFill=t.getBackgroundFill(),i.backgroundStroke=t.getBackgroundStroke(),i.padding=t.getPadding()||pi,i.scale=void 0===a?[1,1]:a;const l=t.getOffsetX(),h=t.getOffsetY(),c=t.getRotateWithView(),u=t.getKeepUpright(),f=t.getRotation();this.text_=t.getText()||\"\",this.textOffsetX_=void 0===l?0:l,this.textOffsetY_=void 0===h?0:h,this.textRotateWithView_=void 0!==c&&c,this.textKeepUpright_=void 0===u||u,this.textRotation_=void 0===f?0:f,this.strokeKey_=r?(\"string\"==typeof r.strokeStyle?r.strokeStyle:ei(r.strokeStyle))+r.lineCap+r.lineDashOffset+\"|\"+r.lineWidth+r.lineJoin+r.miterLimit+\"[\"+r.lineDash.join()+\"]\":\"\",this.textKey_=i.font+i.scale+(i.textAlign||\"?\")+(i.repeat||\"?\")+(i.justify||\"?\")+(i.textBaseline||\"?\"),this.fillKey_=n&&n.fillStyle?\"string\"==typeof n.fillStyle?n.fillStyle:\"|\"+ei(n.fillStyle):\"\"}else this.text_=\"\";this.declutterMode_=t.getDeclutterMode(),this.declutterImageWithText_=e}}const fn=[1/0,1/0,-1/0,-1/0],dn=[],gn=[],pn=[],_n=[];function mn(t){return t[3].declutterBox}const yn=new RegExp(\"[\"+String.fromCharCode(1425)+\"-\"+String.fromCharCode(2303)+String.fromCharCode(64285)+\"-\"+String.fromCharCode(65023)+String.fromCharCode(65136)+\"-\"+String.fromCharCode(65276)+String.fromCharCode(67584)+\"-\"+String.fromCharCode(69631)+String.fromCharCode(124928)+\"-\"+String.fromCharCode(126975)+\"]\");function wn(t,e){return\"start\"===e?e=yn.test(t)?\"right\":\"left\":\"end\"===e&&(e=yn.test(t)?\"left\":\"right\"),cn[e]}function xn(t,e,i){return i>0&&t.push(\"\\n\",\"\"),t.push(e,\"\"),t}function vn(t,e,i){return i%2==0&&(t+=e),t}class Sn{constructor(t,e,i,n,r){this.overlaps=i,this.pixelRatio=e,this.resolution=t,this.alignAndScaleFill_,this.instructions=n.instructions,this.coordinates=n.coordinates,this.coordinateCache_={},this.renderedTransform_=Fe(),this.hitDetectionInstructions=n.hitDetectionInstructions,this.pixelCoordinates_=null,this.viewRotation_=0,this.fillStates=n.fillStates||{},this.strokeStates=n.strokeStates||{},this.textStates=n.textStates||{},this.widths_={},this.labels_={},this.zIndexContext_=r?new ki:null}getZIndexContext(){return this.zIndexContext_}createLabel(t,e,i,n){const r=t+e+i+n;if(this.labels_[r])return this.labels_[r];const s=n?this.strokeStates[n]:null,o=i?this.fillStates[i]:null,a=this.textStates[e],l=this.pixelRatio,h=[a.scale[0]*l,a.scale[1]*l],c=a.justify?cn[a.justify]:wn(Array.isArray(t)?t[0]:t,a.textAlign||di),u=n&&s.lineWidth?s.lineWidth:0,f=Array.isArray(t)?t:String(t).split(\"\\n\").reduce(xn,[]),{width:d,height:g,widths:p,heights:_,lineWidths:m}=function(t,e){const i=[],n=[],r=[];let s=0,o=0,a=0,l=0;for(let h=0,c=e.length;h<=c;h+=2){const u=e[h];if(\"\\n\"===u||h===c){s=Math.max(s,o),r.push(o),o=0,a+=l,l=0;continue}const f=e[h+1]||t.font,d=Mi(f,u);i.push(d),o+=d;const g=Ci(f);n.push(g),l=Math.max(l,g)}return{width:s,height:a,widths:i,heights:n,lineWidths:r}}(a,f),y=d+u,w=[],x=(y+2)*h[0],v=(g+u)*h[1],S={width:x<0?Math.floor(x):Math.ceil(x),height:v<0?Math.floor(v):Math.ceil(v),contextInstructions:w};1==h[0]&&1==h[1]||w.push(\"scale\",h),n&&(w.push(\"strokeStyle\",s.strokeStyle),w.push(\"lineWidth\",u),w.push(\"lineCap\",s.lineCap),w.push(\"lineJoin\",s.lineJoin),w.push(\"miterLimit\",s.miterLimit),w.push(\"setLineDash\",[s.lineDash]),w.push(\"lineDashOffset\",s.lineDashOffset)),i&&w.push(\"fillStyle\",o.fillStyle),w.push(\"textBaseline\",\"middle\"),w.push(\"textAlign\",\"center\");const C=.5-c;let b=c*y+C*u;const M=[],I=[];let E,k=0,A=0,P=0,O=0;for(let t=0,e=f.length;t<e;t+=2){const e=f[t];if(\"\\n\"===e){A+=k,k=0,b=c*y+C*u,++O;continue}const r=f[t+1]||a.font;r!==E&&(n&&M.push(\"font\",r),i&&I.push(\"font\",r),E=r),k=Math.max(k,_[P]);const s=[e,b+C*p[P]+c*(p[P]-m[O]),.5*(u+k)+A];b+=p[P],n&&M.push(\"strokeText\",s),i&&I.push(\"fillText\",s),++P}return Array.prototype.push.apply(w,M),Array.prototype.push.apply(w,I),this.labels_[r]=S,S}replayTextBackground_(t,e,i,n,r,s,o){t.beginPath(),t.moveTo.apply(t,e),t.lineTo.apply(t,i),t.lineTo.apply(t,n),t.lineTo.apply(t,r),t.lineTo.apply(t,e),s&&(this.alignAndScaleFill_=s[2],t.fillStyle=s[1],this.fill_(t)),o&&(this.setStrokeStyle_(t,o),t.stroke())}calculateImageOrLabelDimensions_(t,e,i,n,r,s,o,a,l,h,c,u,f,d,g,p){let _=i-(o*=u[0]),m=n-(a*=u[1]);const y=r+l>t?t-l:r,w=s+h>e?e-h:s,x=d[3]+y*u[0]+d[1],v=d[0]+w*u[1]+d[2],S=_-d[3],C=m-d[0];let b;return(g||0!==c)&&(dn[0]=S,_n[0]=S,dn[1]=C,gn[1]=C,gn[0]=S+x,pn[0]=gn[0],pn[1]=C+v,_n[1]=pn[1]),0!==c?(b=$e(Fe(),i,n,1,1,c,-i,-n),ze(b,dn),ze(b,gn),ze(b,pn),ze(b,_n),_e(Math.min(dn[0],gn[0],pn[0],_n[0]),Math.min(dn[1],gn[1],pn[1],_n[1]),Math.max(dn[0],gn[0],pn[0],_n[0]),Math.max(dn[1],gn[1],pn[1],_n[1]),fn)):_e(Math.min(S,S+x),Math.min(C,C+v),Math.max(S,S+x),Math.max(C,C+v),fn),f&&(_=Math.round(_),m=Math.round(m)),{drawImageX:_,drawImageY:m,drawImageW:y,drawImageH:w,originX:l,originY:h,declutterBox:{minX:fn[0],minY:fn[1],maxX:fn[2],maxY:fn[3],value:p},canvasTransform:b,scale:u}}replayImageOrLabel_(t,e,i,n,r,s,o){const a=!(!s&&!o),l=n.declutterBox,h=o?o[2]*n.scale[0]/2:0;return l.minX-h<=e[0]&&l.maxX+h>=0&&l.minY-h<=e[1]&&l.maxY+h>=0&&(a&&this.replayTextBackground_(t,dn,gn,pn,_n,s,o),Ei(t,n.canvasTransform,r,i,n.originX,n.originY,n.drawImageW,n.drawImageH,n.drawImageX,n.drawImageY,n.scale)),!0}fill_(t){const e=this.alignAndScaleFill_;if(e){const i=ze(this.renderedTransform_,[0,0]),n=512*this.pixelRatio;t.save(),t.translate(i[0]%n,i[1]%n),1!==e&&t.scale(e,e)}t.fill(),e&&t.restore()}setStrokeStyle_(t,e){t.strokeStyle=e[1],e[1]&&(t.lineWidth=e[2],t.lineCap=e[3],t.lineJoin=e[4],t.miterLimit=e[5],t.lineDashOffset=e[7],t.setLineDash(e[6]))}drawLabelWithPointPlacement_(t,e,i,n){const r=this.textStates[e],s=this.createLabel(t,e,n,i),o=this.strokeStates[i],a=this.pixelRatio,l=wn(Array.isArray(t)?t[0]:t,r.textAlign||di),h=cn[r.textBaseline||gi],c=o&&o.lineWidth?o.lineWidth:0;return{label:s,anchorX:l*(s.width/a-2*r.scale[0])+2*(.5-l)*c,anchorY:h*s.height/a+2*(.5-h)*c}}execute_(t,e,i,r,s,o,a,l){const h=this.zIndexContext_;let c;var u,f;this.pixelCoordinates_&&n(i,this.renderedTransform_)?c=this.pixelCoordinates_:(this.pixelCoordinates_||(this.pixelCoordinates_=[]),c=Ae(this.coordinates,0,this.coordinates.length,2,i,this.pixelCoordinates_),u=this.renderedTransform_,f=i,u[0]=f[0],u[1]=f[1],u[2]=f[2],u[3]=f[3],u[4]=f[4],u[5]=f[5]);let d=0;const g=r.length;let p,_=0;const m=[];let y,w,x,v,S,C,b,M,I,E,k,A,P,O=0,R=0;const L=this.coordinateCache_,D=this.viewRotation_,F=Math.round(1e12*Math.atan2(-i[1],i[0]))/1e12,T={context:t,pixelRatio:this.pixelRatio,resolution:this.resolution,rotation:D},z=this.instructions!=r||this.overlaps?0:200;let $,W,G,N;for(;d<g;){const i=r[d];switch(i[0]){case Ai:$=i[1],N=i[3],$.getGeometry()?void 0===a||be(a,N.getExtent())?++d:d=i[2]+1:d=i[2],h&&(h.zIndex=i[4]);break;case Pi:O>z&&(this.fill_(t),O=0),R>z&&(t.stroke(),R=0),O||R||(t.beginPath(),S=NaN,C=NaN),++d;break;case Oi:_=i[1],x=i[2]??0;const n=c[_],r=c[_+1],u=c[_+2]-x-n,f=c[_+3]-x-r,g=Math.sqrt(u*u+f*f);t.moveTo(n+g,r),t.arc(n,r,g,0,2*Math.PI,!0),++d;break;case Ri:t.closePath(),++d;break;case Li:_=i[1],p=i[2];const X=i[3],Y=i[4],B=i[5];T.geometry=X,T.feature=$,d in L||(L[d]=[]);const U=L[d];B?B(c,_,p,2,U):(U[0]=c[_],U[1]=c[_+1],U.length=2),h&&(h.zIndex=i[6]),Y(U,T),++d;break;case Fi:_=i[1],p=i[2],I=i[3],y=i[4],w=i[5];let j=i[6];const V=i[7],q=i[8],J=i[9],K=i[10];let H=i[11];const Z=i[12];let Q=i[13];v=i[14]||\"declutter\";const tt=i[15];if(!I&&i.length>=20){E=i[19],k=i[20],A=i[21],P=i[22];const t=this.drawLabelWithPointPlacement_(E,k,A,P);I=t.label,i[3]=I;const e=i[23];y=(t.anchorX-e)*this.pixelRatio,i[4]=y;const n=i[24];w=(t.anchorY-n)*this.pixelRatio,i[5]=w,j=I.height,i[6]=j,Q=I.width,i[13]=Q}let et,it,nt,rt;i.length>25&&(et=i[25]),i.length>17?(it=i[16],nt=i[17],rt=i[18]):(it=pi,nt=null,rt=null),K&&F?H+=D:K||F||(H-=D);let st=0;for(;_<p;_+=2){if(et&&et[st++]<Q/this.pixelRatio)continue;const i=this.calculateImageOrLabelDimensions_(I.width,I.height,c[_],c[_+1],Q,j,y,w,q,J,H,Z,s,it,!!nt||!!rt,$),n=[t,e,I,i,V,nt,rt];if(l){let t,e,r,s,o;if(tt){const i=p-_;if(!tt[i]){tt[i]={args:n,declutterMode:v};continue}const s=tt[i];t=s.args,e=s.declutterMode,delete tt[i],r=mn(t)}if(!t||\"declutter\"===e&&l.collides(r)||(s=!0),\"declutter\"===v&&l.collides(i.declutterBox)||(o=!0),\"declutter\"===e&&\"declutter\"===v){const t=s&&o;s=t,o=t}s&&(\"none\"!==e&&l.insert(r),this.replayImageOrLabel_.apply(this,t)),o&&(\"none\"!==v&&l.insert(i.declutterBox),this.replayImageOrLabel_.apply(this,n))}else this.replayImageOrLabel_.apply(this,n)}++d;break;case Di:const ot=i[1],at=i[2],lt=i[3],ht=i[4];P=i[5];const ct=i[6],ut=i[7],ft=i[8];A=i[9];const dt=i[10];E=i[11],Array.isArray(E)&&(E=E.reduce(vn,\"\")),k=i[12];const gt=[i[13],i[13]];v=i[14]||\"declutter\";const pt=i[15],_t=i[16],mt=this.textStates[k],yt=mt.font,wt=[mt.scale[0]*ut,mt.scale[1]*ut];let xt;yt in this.widths_?xt=this.widths_[yt]:(xt={},this.widths_[yt]=xt);const vt=Me(c,ot,at,2),St=Math.abs(wt[0])*Ii(yt,E,xt);if(ht||St<=vt){const i=Re(c,ot,at,2,E,(vt-St)*wn(E,this.textStates[k].textAlign),ct,Math.abs(wt[0]),Ii,yt,xt,F?0:this.viewRotation_,pt);t:if(i){const n=[];let r,s,o,a,h;if(A)for(r=0,s=i.length;r<s;++r){h=i[r],o=h[4],a=this.createLabel(o,k,\"\",A),y=h[2]+(wt[0]<0?-dt:dt)-_t,w=lt*a.height+2*(.5-lt)*dt*wt[1]/wt[0]-ft;const s=this.calculateImageOrLabelDimensions_(a.width,a.height,h[0],h[1],a.width,a.height,y,w,0,0,h[3],gt,!1,pi,!1,$);if(l&&\"declutter\"===v&&l.collides(s.declutterBox))break t;n.push([t,e,a,s,1,null,null])}if(P)for(r=0,s=i.length;r<s;++r){h=i[r],o=h[4],a=this.createLabel(o,k,P,\"\"),y=h[2]-_t,w=lt*a.height-ft;const s=this.calculateImageOrLabelDimensions_(a.width,a.height,h[0],h[1],a.width,a.height,y,w,0,0,h[3],gt,!1,pi,!1,$);if(l&&\"declutter\"===v&&l.collides(s.declutterBox))break t;n.push([t,e,a,s,1,null,null])}l&&\"none\"!==v&&l.load(n.map(mn));for(let t=0,e=n.length;t<e;++t)this.replayImageOrLabel_.apply(this,n[t])}}++d;break;case Ti:if(void 0!==o){$=i[1];const t=o($,N,v);if(t)return t}++d;break;case zi:z?O++:this.fill_(t),++d;break;case $i:let Ct,bt,Mt;if(_=i[1],p=i[2],x=i[3],x){const t=(i[4]??!1)||Math.abs(c[_]-c[p-2])<1e-6&&Math.abs(c[_+1]-c[p-1])<1e-6;Ie(c,_,p,2,x,t,m),ke(m,2,t),Ct=m,bt=0,Mt=Ct.length}else Ct=c,bt=_,Mt=p;W=Ct[bt],G=Ct[bt+1],t.moveTo(W,G),S=W+.5|0,C=G+.5|0;for(let e=bt+2;e<Mt;e+=2)W=Ct[e],G=Ct[e+1],b=W+.5|0,M=G+.5|0,e!=Mt-2&&b===S&&M===C||(t.lineTo(W,G),S=b,C=M);++d;break;case Wi:this.alignAndScaleFill_=i[2],O?(this.fill_(t),O=0,R&&(t.stroke(),R=0)):R&&i[1]&&(t.stroke(),R=0),t.fillStyle=i[1],++d;break;case Gi:O&&i[1]&&(this.fill_(t),O=0),R&&(t.stroke(),R=0),this.setStrokeStyle_(t,i),++d;break;case Ni:z?R++:t.stroke(),++d;break;default:++d}}O&&this.fill_(t),R&&t.stroke()}execute(t,e,i,n,r,s){this.viewRotation_=n,this.execute_(t,e,i,this.instructions,r,void 0,void 0,s)}executeHitDetection(t,e,i,n,r){return this.viewRotation_=i,this.execute_(t,[t.canvas.width,t.canvas.height],e,this.hitDetectionInstructions,!0,n,r)}}function Cn(t,e,i){return bn(et(t,e,i))}function bn(t,e){if(t instanceof Z){if(t.type===X&&\"string\"==typeof t.value){const e=D(t.value);return function(){return e}}return function(){return t.value}}const i=t.operator;switch(i){case Gt:case Nt:case zt:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);switch(e){case zt:return t=>{for(let e=0;e<i;++e){const i=n[e](t);if(null!=i)return i}throw new Error(\"Expected one of the values to be non-null\")};case Gt:case Nt:return t=>{for(let r=0;r<i;++r){const i=n[r](t);if(typeof i===e)return i}throw new Error(`Expected one of the values to be a ${e}`)};default:throw new Error(`Unsupported assertion operator ${e}`)}}(t);case it:case nt:case qt:return function(t){const e=t.args[0],i=e.value;switch(t.operator){case it:return e=>{const n=t.args;let r=e.properties[i];for(let t=1,e=n.length;t<e;++t){r=r[n[t].value]}return r};case nt:return t=>t.variables[i];case qt:return e=>{const n=t.args;if(!(i in e.properties))return!1;let r=e.properties[i];for(let t=1,e=n.length;t<e;++t){const e=n[t].value;if(!r||!Object.hasOwn(r,e))return!1;r=r[e]}return!0};default:throw new Error(`Unsupported accessor operator ${t.operator}`)}}(t);case Bt:return t=>t.featureId;case st:return t=>t.geometryType;case rt:{const e=t.args.map(t=>bn(t));return t=>\"\".concat(...e.map(e=>e(t).toString()))}case ct:return t=>t.resolution;case at:case lt:case Ft:case Wt:case ht:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);switch(e){case at:return t=>{for(let e=0;e<i;++e)if(n[e](t))return!0;return!1};case lt:return t=>{for(let e=0;e<i;++e)if(!n[e](t))return!1;return!0};case Ft:return t=>{const e=n[0](t),i=n[1](t),r=n[2](t);return e>=i&&e<=r};case Wt:return t=>{const e=n[0](t);for(let r=1;r<i;++r)if(e===n[r](t))return!0;return!1};case ht:return t=>!n[0](t);default:throw new Error(`Unsupported logical operator ${e}`)}}(t);case dt:case gt:case mt:case yt:case pt:case _t:return function(t){const e=t.operator,i=bn(t.args[0]),n=bn(t.args[1]);switch(e){case dt:return t=>i(t)===n(t);case gt:return t=>i(t)!==n(t);case mt:return t=>i(t)<n(t);case yt:return t=>i(t)<=n(t);case pt:return t=>i(t)>n(t);case _t:return t=>i(t)>=n(t);default:throw new Error(`Unsupported comparison operator ${e}`)}}(t);case wt:case xt:case vt:case St:case Ct:case bt:case Mt:case It:case Et:case kt:case At:case Pt:case Ot:case Rt:case Lt:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);switch(e){case wt:return t=>{let e=1;for(let r=0;r<i;++r)e*=n[r](t);return e};case xt:return t=>n[0](t)/n[1](t);case vt:return t=>{let e=0;for(let r=0;r<i;++r)e+=n[r](t);return e};case St:return t=>n[0](t)-n[1](t);case Ct:return t=>{const e=n[0](t),i=n[1](t);if(e<i)return i;const r=n[2](t);return e>r?r:e};case bt:return t=>n[0](t)%n[1](t);case Mt:return t=>Math.pow(n[0](t),n[1](t));case It:return t=>Math.abs(n[0](t));case Et:return t=>Math.floor(n[0](t));case kt:return t=>Math.ceil(n[0](t));case At:return t=>Math.round(n[0](t));case Pt:return t=>Math.sin(n[0](t));case Ot:return t=>Math.cos(n[0](t));case Rt:return 2===i?t=>Math.atan2(n[0](t),n[1](t)):t=>Math.atan(n[0](t));case Lt:return t=>Math.sqrt(n[0](t));default:throw new Error(`Unsupported numeric operator ${e}`)}}(t);case $t:return function(t){const e=t.args.length,i=new Array(e);for(let n=0;n<e;++n)i[n]=bn(t.args[n]);return t=>{for(let n=0;n<e-1;n+=2){if(i[n](t))return i[n+1](t)}return i[e-1](t)}}(t);case Dt:return function(t){const e=t.args.length,i=new Array(e);for(let n=0;n<e;++n)i[n]=bn(t.args[n]);return t=>{const n=i[0](t);for(let r=1;r<e-1;r+=2)if(n===i[r](t))return i[r+1](t);return i[e-1](t)}}(t);case Tt:return function(t){const e=t.args.length,i=new Array(e);for(let n=0;n<e;++n)i[n]=bn(t.args[n]);return t=>{const n=i[0](t),r=i[1](t);let s,o;for(let a=2;a<e;a+=2){const e=i[a](t);let l=i[a+1](t);const h=Array.isArray(l);if(h&&(l=k(l)),e>=r)return 2===a?l:h?In(n,r,s,o,e,l):Mn(n,r,s,o,e,l);s=e,o=l}return o}}(t);case Vt:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);if(e===Vt)return e=>{const i=n[0](e);return t.args[0].type===X?T(i):i.toString()};throw new Error(`Unsupported convert operator ${e}`)}(t);default:throw new Error(`Unsupported operator ${i}`)}}function Mn(t,e,i,n,r,s){const o=r-i;if(0===o)return n;const a=e-i;return n+(1===t?a/o:(Math.pow(t,a)-1)/(Math.pow(t,o)-1))*(s-n)}function In(t,e,i,n,r,s){if(0===r-i)return n;const o=L(n),a=L(s);let l=a[2]-o[2];l>180?l-=360:l<-180&&(l+=360);return function(t){const e=(t[0]+16)/116,i=t[1],n=t[2]*Math.PI/180,r=P(e),s=P(e+i/500*Math.cos(n)),o=P(e-i/200*Math.sin(n)),a=A(3.021973625*s-1.617392459*r-.404875592*o),l=A(-.943766287*s+1.916279586*r+.027607165*o),h=A(.069407491*s-.22898585*r+1.159737864*o);return[c(a+.5|0,0,255),c(l+.5|0,0,255),c(h+.5|0,0,255),t[3]]}([Mn(t,e,i,o[0],r,a[0]),Mn(t,e,i,o[1],r,a[1]),o[2]+Mn(t,e,i,0,r,l),Mn(t,e,i,n[3],r,s[3])])}class En{constructor(t){this.opacity_=t.opacity,this.rotateWithView_=t.rotateWithView,this.rotation_=t.rotation,this.scale_=t.scale,this.scaleArray_=z(t.scale),this.displacement_=t.displacement,this.declutterMode_=t.declutterMode}clone(){const t=this.getScale();return new En({opacity:this.getOpacity(),scale:Array.isArray(t)?t.slice():t,rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()})}getOpacity(){return this.opacity_}getRotateWithView(){return this.rotateWithView_}getRotation(){return this.rotation_}getScale(){return this.scale_}getScaleArray(){return this.scaleArray_}getDisplacement(){return this.displacement_}getDeclutterMode(){return this.declutterMode_}getAnchor(){return Qe()}getImage(t){return Qe()}getHitDetectionImage(){return Qe()}getPixelRatio(t){return 1}getImageState(){return Qe()}getImageSize(){return Qe()}getOrigin(){return Qe()}getSize(){return Qe()}setDisplacement(t){this.displacement_=t}setOpacity(t){this.opacity_=t}setRotateWithView(t){this.rotateWithView_=t}setRotation(t){this.rotation_=t}setScale(t){this.scale_=t,this.scaleArray_=z(t)}listenImageChange(t){Qe()}load(){Qe()}unlistenImageChange(t){Qe()}ready(){return Promise.resolve()}}class kn extends En{constructor(t){super({opacity:1,rotateWithView:void 0!==t.rotateWithView&&t.rotateWithView,rotation:void 0!==t.rotation?t.rotation:0,scale:void 0!==t.scale?t.scale:1,displacement:void 0!==t.displacement?t.displacement:[0,0],declutterMode:t.declutterMode}),this.hitDetectionCanvas_=null,this.fill_=void 0!==t.fill?t.fill:null,this.origin_=[0,0],this.points_=t.points,this.radius=t.radius,this.radius2_=t.radius2,this.angle_=void 0!==t.angle?t.angle:0,this.stroke_=void 0!==t.stroke?t.stroke:null,this.size_,this.renderOptions_,this.imageState_=this.fill_&&this.fill_.loading()?Yi:Bi,this.imageState_===Yi&&this.ready().then(()=>this.imageState_=Bi),this.render()}clone(){const t=this.getScale(),e=new kn({fill:this.getFill()?this.getFill().clone():void 0,points:this.getPoints(),radius:this.getRadius(),radius2:this.getRadius2(),angle:this.getAngle(),stroke:this.getStroke()?this.getStroke().clone():void 0,rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),scale:Array.isArray(t)?t.slice():t,displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()});return e.setOpacity(this.getOpacity()),e}getAnchor(){const t=this.size_,e=this.getDisplacement(),i=this.getScaleArray();return[t[0]/2-e[0]/i[0],t[1]/2+e[1]/i[1]]}getAngle(){return this.angle_}getFill(){return this.fill_}setFill(t){this.fill_=t,this.render()}getHitDetectionImage(){return this.hitDetectionCanvas_||(this.hitDetectionCanvas_=this.createHitDetectionCanvas_(this.renderOptions_)),this.hitDetectionCanvas_}getImage(t){const e=this.fill_?.getKey(),i=`${t},${this.angle_},${this.radius},${this.radius2_},${this.points_},${e}`+Object.values(this.renderOptions_).join(\",\");let n=qi.get(i,null)?.getImage(1);if(!n){const e=this.renderOptions_,r=Math.ceil(e.size*t),s=a(r,r);this.draw_(e,s,t),n=s.canvas;const o=new Ki(n,void 0,null,Bi,null);qi.set(i,null,o),createImageBitmap(n).then(t=>{o.setImage(t)})}return n}getPixelRatio(t){return t}getImageSize(){return this.size_}getImageState(){return this.imageState_}getOrigin(){return this.origin_}getPoints(){return this.points_}getRadius(){return this.radius}setRadius(t){this.radius!==t&&(this.radius=t,this.render())}getRadius2(){return this.radius2_}setRadius2(t){this.radius2_!==t&&(this.radius2_=t,this.render())}getSize(){return this.size_}getStroke(){return this.stroke_}setStroke(t){this.stroke_=t,this.render()}listenImageChange(t){}load(){}unlistenImageChange(t){}calculateLineJoinSize_(t,e,i){if(0===e||this.points_===1/0||\"bevel\"!==t&&\"miter\"!==t)return e;let n=this.radius,r=void 0===this.radius2_?n:this.radius2_;if(n<r){const t=n;n=r,r=t}const s=void 0===this.radius2_?this.points_:2*this.points_,o=2*Math.PI/s,a=r*Math.sin(o),l=n-Math.sqrt(r*r-a*a),h=Math.sqrt(a*a+l*l),c=h/a;if(\"miter\"===t&&c<=i)return c*e;const u=e/2/c,f=e/2*(l/h),d=Math.sqrt((n+u)*(n+u)+f*f)-n;if(void 0===this.radius2_||\"bevel\"===t)return 2*d;const g=n*Math.sin(o),p=r-Math.sqrt(n*n-g*g),_=Math.sqrt(g*g+p*p)/g;if(_<=i){const t=_*e/2-r-n;return 2*Math.max(d,t)}return 2*d}createRenderOptions(){let t,e=hi,i=ui,n=0,r=null,s=0,o=0;this.stroke_&&(t=Zi(this.stroke_.getColor()??fi),o=this.stroke_.getWidth()??1,r=this.stroke_.getLineDash(),s=this.stroke_.getLineDashOffset()??0,i=this.stroke_.getLineJoin()??ui,e=this.stroke_.getLineCap()??hi,n=this.stroke_.getMiterLimit()??10);const a=this.calculateLineJoinSize_(i,o,n),l=Math.max(this.radius,this.radius2_||0);return{strokeStyle:t,strokeWidth:o,size:Math.ceil(2*l+a),lineCap:e,lineDash:r,lineDashOffset:s,lineJoin:i,miterLimit:n}}render(){this.renderOptions_=this.createRenderOptions();const t=this.renderOptions_.size;this.hitDetectionCanvas_=null,this.size_=[t,t]}draw_(t,e,i){if(e.scale(i,i),e.translate(t.size/2,t.size/2),this.createPath_(e),this.fill_){let t=this.fill_.getColor();null===t&&(t=li),e.fillStyle=Zi(t),e.fill()}t.strokeStyle&&(e.strokeStyle=t.strokeStyle,e.lineWidth=t.strokeWidth,t.lineDash&&(e.setLineDash(t.lineDash),e.lineDashOffset=t.lineDashOffset),e.lineCap=t.lineCap,e.lineJoin=t.lineJoin,e.miterLimit=t.miterLimit,e.stroke())}createHitDetectionCanvas_(t){let e;if(this.fill_){let i=this.fill_.getColor(),n=0;\"string\"==typeof i&&(i=F(i)),null===i?n=1:Array.isArray(i)&&(n=4===i.length?i[3]:1),0===n&&(e=a(t.size,t.size),this.drawHitDetectionCanvas_(t,e))}return e?e.canvas:this.getImage(1)}createPath_(t){let e=this.points_;const i=this.radius;if(e===1/0)t.arc(0,0,i,0,2*Math.PI);else{const n=void 0===this.radius2_?i:this.radius2_;void 0!==this.radius2_&&(e*=2);const r=this.angle_-Math.PI/2,s=2*Math.PI/e;for(let o=0;o<e;o++){const e=r+o*s,a=o%2==0?i:n;t.lineTo(a*Math.cos(e),a*Math.sin(e))}t.closePath()}}drawHitDetectionCanvas_(t,e){e.translate(t.size/2,t.size/2),this.createPath_(e),e.fillStyle=li,e.fill(),t.strokeStyle&&(e.strokeStyle=t.strokeStyle,e.lineWidth=t.strokeWidth,t.lineDash&&(e.setLineDash(t.lineDash),e.lineDashOffset=t.lineDashOffset),e.lineJoin=t.lineJoin,e.miterLimit=t.miterLimit,e.stroke())}ready(){return this.fill_?this.fill_.ready():Promise.resolve()}}class An extends kn{constructor(t){super({points:1/0,fill:(t=t||{radius:5}).fill,radius:t.radius,stroke:t.stroke,scale:void 0!==t.scale?t.scale:1,rotation:void 0!==t.rotation?t.rotation:0,rotateWithView:void 0!==t.rotateWithView&&t.rotateWithView,displacement:void 0!==t.displacement?t.displacement:[0,0],declutterMode:t.declutterMode})}clone(){const t=this.getScale(),e=new An({fill:this.getFill()?this.getFill().clone():void 0,stroke:this.getStroke()?this.getStroke().clone():void 0,radius:this.getRadius(),scale:Array.isArray(t)?t.slice():t,rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()});return e.setOpacity(this.getOpacity()),e}}class Pn{constructor(t){t=t||{},this.patternImage_=null,this.color_=null,void 0!==t.color&&this.setColor(t.color)}clone(){const t=this.getColor();return new Pn({color:Array.isArray(t)?t.slice():t||void 0})}getColor(){return this.color_}setColor(t){if(null!==t&&\"object\"==typeof t&&\"src\"in t){const e=Hi(null,t.src,{crossOrigin:\"anonymous\"},void 0,t.offset?null:t.color?t.color:null,!(t.offset&&t.size));e.ready().then(()=>{this.patternImage_=null}),e.getImageState()===Xi&&e.load(),e.getImageState()===Yi&&(this.patternImage_=e)}this.color_=t}getKey(){const t=this.getColor();return t?t instanceof CanvasPattern||t instanceof CanvasGradient?ei(t):\"object\"==typeof t&&\"src\"in t?t.src+\":\"+t.offset:F(t).toString():\"\"}loading(){return!!this.patternImage_}ready(){return this.patternImage_?this.patternImage_.ready():Promise.resolve()}}function On(t,e,i,n){return void 0!==i&&void 0!==n?[i/t,n/e]:void 0!==i?i/t:void 0!==n?n/e:1}class Rn extends En{constructor(t){const e=void 0!==(t=t||{}).opacity?t.opacity:1,i=void 0!==t.rotation?t.rotation:0,n=void 0!==t.scale?t.scale:1,r=void 0!==t.rotateWithView&&t.rotateWithView;super({opacity:e,rotation:i,scale:n,displacement:void 0!==t.displacement?t.displacement:[0,0],rotateWithView:r,declutterMode:t.declutterMode}),this.anchor_=void 0!==t.anchor?t.anchor:[.5,.5],this.normalizedAnchor_=null,this.anchorOrigin_=void 0!==t.anchorOrigin?t.anchorOrigin:\"top-left\",this.anchorXUnits_=void 0!==t.anchorXUnits?t.anchorXUnits:\"fraction\",this.anchorYUnits_=void 0!==t.anchorYUnits?t.anchorYUnits:\"fraction\",this.crossOrigin_=void 0!==t.crossOrigin?t.crossOrigin:null,this.referrerPolicy_=t.referrerPolicy;const s=void 0!==t.img?t.img:null;let o,a=t.src;if(Le(!(void 0!==a&&s),\"`image` and `src` cannot be provided at the same time\"),void 0!==a&&0!==a.length||!s||(a=s.src||ei(s)),Le(void 0!==a&&a.length>0,\"A defined and non-empty `src` or `image` must be provided\"),Le(!((void 0!==t.width||void 0!==t.height)&&void 0!==t.scale),\"`width` or `height` cannot be provided together with `scale`\"),void 0!==t.src?o=Xi:void 0!==s&&(o=\"complete\"in s?s.complete?s.src?Bi:Xi:Yi:Bi),this.color_=void 0!==t.color?F(t.color):null,this.iconImage_=Hi(s,a,{crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_},o,this.color_),this.offset_=void 0!==t.offset?t.offset:[0,0],this.offsetOrigin_=void 0!==t.offsetOrigin?t.offsetOrigin:\"top-left\",this.origin_=null,this.size_=void 0!==t.size?t.size:null,this.initialOptions_,void 0!==t.width||void 0!==t.height){let e,i;if(t.size)[e,i]=t.size;else{const n=this.getImage(1);if(n.width&&n.height)e=n.width,i=n.height;else if(n instanceof HTMLImageElement){this.initialOptions_=t;const e=()=>{if(this.unlistenImageChange(e),!this.initialOptions_)return;const i=this.iconImage_.getSize();this.setScale(On(i[0],i[1],t.width,t.height))};return void this.listenImageChange(e)}}void 0!==e&&this.setScale(On(e,i,t.width,t.height))}}clone(){let t,e,i;return this.initialOptions_?(e=this.initialOptions_.width,i=this.initialOptions_.height):(t=this.getScale(),t=Array.isArray(t)?t.slice():t),new Rn({anchor:this.anchor_.slice(),anchorOrigin:this.anchorOrigin_,anchorXUnits:this.anchorXUnits_,anchorYUnits:this.anchorYUnits_,color:this.color_&&this.color_.slice?this.color_.slice():this.color_||void 0,crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_,offset:this.offset_.slice(),offsetOrigin:this.offsetOrigin_,opacity:this.getOpacity(),rotateWithView:this.getRotateWithView(),rotation:this.getRotation(),scale:t,width:e,height:i,size:null!==this.size_?this.size_.slice():void 0,src:this.getSrc(),displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()})}getAnchor(){let t=this.normalizedAnchor_;if(!t){t=this.anchor_;const e=this.getSize();if(\"fraction\"==this.anchorXUnits_||\"fraction\"==this.anchorYUnits_){if(!e)return null;t=this.anchor_.slice(),\"fraction\"==this.anchorXUnits_&&(t[0]*=e[0]),\"fraction\"==this.anchorYUnits_&&(t[1]*=e[1])}if(\"top-left\"!=this.anchorOrigin_){if(!e)return null;t===this.anchor_&&(t=this.anchor_.slice()),\"top-right\"!=this.anchorOrigin_&&\"bottom-right\"!=this.anchorOrigin_||(t[0]=-t[0]+e[0]),\"bottom-left\"!=this.anchorOrigin_&&\"bottom-right\"!=this.anchorOrigin_||(t[1]=-t[1]+e[1])}this.normalizedAnchor_=t}const e=this.getDisplacement(),i=this.getScaleArray();return[t[0]-e[0]/i[0],t[1]+e[1]/i[1]]}setAnchor(t){this.anchor_=t,this.normalizedAnchor_=null}getColor(){return this.color_}setColor(t){const e=t?F(t):null;if(this.color_===e||this.color_&&e&&this.color_.length===e.length&&this.color_.every((t,i)=>t===e[i]))return;this.color_=e;const i=this.getSrc(),n=void 0!==i?null:this.getHitDetectionImage(),r=void 0!==i?Xi:this.iconImage_.getImageState();this.iconImage_=Hi(n,i,{crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_},r,this.color_)}getImage(t){return this.iconImage_.getImage(t)}getPixelRatio(t){return this.iconImage_.getPixelRatio(t)}getImageSize(){return this.iconImage_.getSize()}getImageState(){return this.iconImage_.getImageState()}getHitDetectionImage(){return this.iconImage_.getHitDetectionImage()}getOrigin(){if(this.origin_)return this.origin_;let t=this.offset_;if(\"top-left\"!=this.offsetOrigin_){const e=this.getSize(),i=this.iconImage_.getSize();if(!e||!i)return null;t=t.slice(),\"top-right\"!=this.offsetOrigin_&&\"bottom-right\"!=this.offsetOrigin_||(t[0]=i[0]-e[0]-t[0]),\"bottom-left\"!=this.offsetOrigin_&&\"bottom-right\"!=this.offsetOrigin_||(t[1]=i[1]-e[1]-t[1])}return this.origin_=t,this.origin_}getSrc(){return this.iconImage_.getSrc()}setSrc(t){this.iconImage_=Hi(null,t,{crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_},Xi,this.color_)}getSize(){return this.size_?this.size_:this.iconImage_.getSize()}getWidth(){const t=this.getScaleArray();return this.size_?this.size_[0]*t[0]:this.iconImage_.getImageState()==Bi?this.iconImage_.getSize()[0]*t[0]:void 0}getHeight(){const t=this.getScaleArray();return this.size_?this.size_[1]*t[1]:this.iconImage_.getImageState()==Bi?this.iconImage_.getSize()[1]*t[1]:void 0}setScale(t){delete this.initialOptions_,super.setScale(t)}listenImageChange(t){this.iconImage_.addEventListener(je,t)}load(){this.iconImage_.load()}unlistenImageChange(t){this.iconImage_.removeEventListener(je,t)}ready(){return this.iconImage_.ready()}}class Ln{constructor(t){t=t||{},this.color_=void 0!==t.color?t.color:null,this.lineCap_=t.lineCap,this.lineDash_=void 0!==t.lineDash?t.lineDash:null,this.lineDashOffset_=t.lineDashOffset,this.lineJoin_=t.lineJoin,this.miterLimit_=t.miterLimit,this.offset_=t.offset,this.width_=t.width}clone(){const t=this.getColor();return new Ln({color:Array.isArray(t)?t.slice():t||void 0,lineCap:this.getLineCap(),lineDash:this.getLineDash()?this.getLineDash().slice():void 0,lineDashOffset:this.getLineDashOffset(),lineJoin:this.getLineJoin(),miterLimit:this.getMiterLimit(),offset:this.getOffset(),width:this.getWidth()})}getColor(){return this.color_}getLineCap(){return this.lineCap_}getLineDash(){return this.lineDash_}getLineDashOffset(){return this.lineDashOffset_}getLineJoin(){return this.lineJoin_}getMiterLimit(){return this.miterLimit_}getOffset(){return this.offset_}getWidth(){return this.width_}setColor(t){this.color_=t}setLineCap(t){this.lineCap_=t}setLineDash(t){this.lineDash_=t}setLineDashOffset(t){this.lineDashOffset_=t}setLineJoin(t){this.lineJoin_=t}setMiterLimit(t){this.miterLimit_=t}setOffset(t){this.offset_=t}setWidth(t){this.width_=t}}class Dn{constructor(t){t=t||{},this.geometry_=null,this.geometryFunction_=Fn,void 0!==t.geometry&&this.setGeometry(t.geometry),this.fill_=void 0!==t.fill?t.fill:null,this.image_=void 0!==t.image?t.image:null,this.renderer_=void 0!==t.renderer?t.renderer:null,this.hitDetectionRenderer_=void 0!==t.hitDetectionRenderer?t.hitDetectionRenderer:null,this.stroke_=void 0!==t.stroke?t.stroke:null,this.text_=void 0!==t.text?t.text:null,this.zIndex_=t.zIndex}clone(){let t=this.getGeometry();return t&&\"object\"==typeof t&&(t=t.clone()),new Dn({geometry:t??void 0,fill:this.getFill()?this.getFill().clone():void 0,image:this.getImage()?this.getImage().clone():void 0,renderer:this.getRenderer()??void 0,stroke:this.getStroke()?this.getStroke().clone():void 0,text:this.getText()?this.getText().clone():void 0,zIndex:this.getZIndex()})}getRenderer(){return this.renderer_}setRenderer(t){this.renderer_=t}setHitDetectionRenderer(t){this.hitDetectionRenderer_=t}getHitDetectionRenderer(){return this.hitDetectionRenderer_}getGeometry(){return this.geometry_}getGeometryFunction(){return this.geometryFunction_}getFill(){return this.fill_}setFill(t){this.fill_=t}getImage(){return this.image_}setImage(t){this.image_=t}getStroke(){return this.stroke_}setStroke(t){this.stroke_=t}getText(){return this.text_}setText(t){this.text_=t}getZIndex(){return this.zIndex_}setGeometry(t){\"function\"==typeof t?this.geometryFunction_=t:\"string\"==typeof t?this.geometryFunction_=function(e){return e.get(t)}:t?void 0!==t&&(this.geometryFunction_=function(){return t}):this.geometryFunction_=Fn,this.geometry_=t}setZIndex(t){this.zIndex_=t}}function Fn(t){return t.getGeometry()}class Tn{constructor(t){t=t||{},this.font_=t.font,this.rotation_=t.rotation,this.rotateWithView_=t.rotateWithView,this.keepUpright_=t.keepUpright,this.scale_=t.scale,this.scaleArray_=z(void 0!==t.scale?t.scale:1),this.text_=t.text,this.textAlign_=t.textAlign,this.justify_=t.justify,this.repeat_=t.repeat,this.textBaseline_=t.textBaseline,this.fill_=void 0!==t.fill?t.fill:new Pn({color:\"#333\"}),this.maxAngle_=void 0!==t.maxAngle?t.maxAngle:Math.PI/4,this.placement_=void 0!==t.placement?t.placement:\"point\",this.overflow_=!!t.overflow,this.stroke_=void 0!==t.stroke?t.stroke:null,this.offsetX_=void 0!==t.offsetX?t.offsetX:0,this.offsetY_=void 0!==t.offsetY?t.offsetY:0,this.backgroundFill_=t.backgroundFill?t.backgroundFill:null,this.backgroundStroke_=t.backgroundStroke?t.backgroundStroke:null,this.padding_=void 0===t.padding?null:t.padding,this.declutterMode_=t.declutterMode}clone(){const t=this.getScale();return new Tn({font:this.getFont(),placement:this.getPlacement(),repeat:this.getRepeat(),maxAngle:this.getMaxAngle(),overflow:this.getOverflow(),rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),keepUpright:this.getKeepUpright(),scale:Array.isArray(t)?t.slice():t,text:this.getText(),textAlign:this.getTextAlign(),justify:this.getJustify(),textBaseline:this.getTextBaseline(),fill:this.getFill()instanceof Pn?this.getFill().clone():this.getFill(),stroke:this.getStroke()?this.getStroke().clone():void 0,offsetX:this.getOffsetX(),offsetY:this.getOffsetY(),backgroundFill:this.getBackgroundFill()?this.getBackgroundFill().clone():void 0,backgroundStroke:this.getBackgroundStroke()?this.getBackgroundStroke().clone():void 0,padding:this.getPadding()||void 0,declutterMode:this.getDeclutterMode()})}getOverflow(){return this.overflow_}getFont(){return this.font_}getMaxAngle(){return this.maxAngle_}getPlacement(){return this.placement_}getRepeat(){return this.repeat_}getOffsetX(){return this.offsetX_}getOffsetY(){return this.offsetY_}getFill(){return this.fill_}getRotateWithView(){return this.rotateWithView_}getKeepUpright(){return this.keepUpright_}getRotation(){return this.rotation_}getScale(){return this.scale_}getScaleArray(){return this.scaleArray_}getStroke(){return this.stroke_}getText(){return this.text_}getTextAlign(){return this.textAlign_}getJustify(){return this.justify_}getTextBaseline(){return this.textBaseline_}getBackgroundFill(){return this.backgroundFill_}getBackgroundStroke(){return this.backgroundStroke_}getPadding(){return this.padding_}getDeclutterMode(){return this.declutterMode_}setOverflow(t){this.overflow_=t}setFont(t){this.font_=t}setMaxAngle(t){this.maxAngle_=t}setOffsetX(t){this.offsetX_=t}setOffsetY(t){this.offsetY_=t}setPlacement(t){this.placement_=t}setRepeat(t){this.repeat_=t}setRotateWithView(t){this.rotateWithView_=t}setKeepUpright(t){this.keepUpright_=t}setFill(t){this.fill_=t}setRotation(t){this.rotation_=t}setScale(t){this.scale_=t,this.scaleArray_=z(void 0!==t?t:1)}setStroke(t){this.stroke_=t}setText(t){this.text_=t}setTextAlign(t){this.textAlign_=t}setJustify(t){this.justify_=t}setTextBaseline(t){this.textBaseline_=t}setBackgroundFill(t){this.backgroundFill_=t}setBackgroundStroke(t){this.backgroundStroke_=t}setPadding(t){this.padding_=t}}function zn(t){return!0}function $n(t,e){const i=function(t,e){const i=t.length,n=new Array(i);for(let r=0;r<i;++r){const i=t[r],s=\"filter\"in i?Cn(i.filter,W,e):zn;let o;if(Array.isArray(i.style)){const t=i.style.length;o=new Array(t);for(let n=0;n<t;++n)o[n]=Gn(i.style[n],e)}else o=[Gn(i.style,e)];n[r]={filter:s,styles:o}}return function(e){const r=[];let s=!1;for(let o=0;o<i;++o){if((0,n[o].filter)(e)&&(!t[o].else||!s)){s=!0;for(const t of n[o].styles){const i=t(e);i&&r.push(i)}}}return r}}(t,e=e??tt()),n={variables:{},properties:{},resolution:NaN,featureId:null,geometryType:\"\"};return function(t,r){if(n.properties=t.getPropertiesInternal(),n.resolution=r,e.featureId){const e=t.getId();n.featureId=void 0!==e?e:null}return e.geometryType&&(n.geometryType=se(t.getGeometry())),i(n)}}function Wn(t,e){e=e??tt();const i=t.length,n=new Array(i);for(let r=0;r<i;++r)n[r]=Gn(t[r],e);const r={variables:{},properties:{},resolution:NaN,featureId:null,geometryType:\"\"},s=new Array(i);return function(t,o){if(r.properties=t.getPropertiesInternal(),r.resolution=o,e.featureId){const e=t.getId();r.featureId=void 0!==e?e:null}e.geometryType&&(r.geometryType=se(t.getGeometry()));let a=0;for(let t=0;t<i;++t){const e=n[t](r);e&&(s[a]=e,a+=1)}return s.length=a,s}}function Gn(t,e){const i=Nn(t,\"\",e),n=Xn(t,\"\",e),r=function(t,e){const i=\"text-\",n=Un(t,i+\"value\",e);if(!n)return null;const r=Nn(t,i,e),s=Nn(t,i+\"background-\",e),o=Xn(t,i,e),a=Xn(t,i+\"background-\",e),l=Un(t,i+\"font\",e),h=Bn(t,i+\"max-angle\",e),c=Bn(t,i+\"offset-x\",e),u=Bn(t,i+\"offset-y\",e),f=jn(t,i+\"overflow\",e),d=Un(t,i+\"placement\",e),g=Bn(t,i+\"repeat\",e),p=Hn(t,i+\"scale\",e),_=jn(t,i+\"rotate-with-view\",e),m=Bn(t,i+\"rotation\",e),y=Un(t,i+\"align\",e),w=Un(t,i+\"justify\",e),x=Un(t,i+\"baseline\",e),v=jn(t,i+\"keep-upright\",e),S=qn(t,i+\"padding\",e),C=er(t,i+\"declutter-mode\"),b=new Tn({declutterMode:C});return function(t){if(b.setText(n(t)),r&&b.setFill(r(t)),s&&b.setBackgroundFill(s(t)),o&&b.setStroke(o(t)),a&&b.setBackgroundStroke(a(t)),l&&b.setFont(l(t)),h&&b.setMaxAngle(h(t)),c&&b.setOffsetX(c(t)),u&&b.setOffsetY(u(t)),f&&b.setOverflow(f(t)),d){const e=d(t);if(\"point\"!==e&&\"line\"!==e)throw new Error(\"Expected point or line for text-placement\");b.setPlacement(e)}if(g&&b.setRepeat(g(t)),p&&b.setScale(p(t)),_&&b.setRotateWithView(_(t)),m&&b.setRotation(m(t)),y){const e=y(t);if(\"left\"!==e&&\"center\"!==e&&\"right\"!==e&&\"end\"!==e&&\"start\"!==e)throw new Error(\"Expected left, right, center, start, or end for text-align\");b.setTextAlign(e)}if(w){const e=w(t);if(\"left\"!==e&&\"right\"!==e&&\"center\"!==e)throw new Error(\"Expected left, right, or center for text-justify\");b.setJustify(e)}if(x){const e=x(t);if(\"bottom\"!==e&&\"top\"!==e&&\"middle\"!==e&&\"alphabetic\"!==e&&\"hanging\"!==e)throw new Error(\"Expected bottom, top, middle, alphabetic, or hanging for text-baseline\");b.setTextBaseline(e)}return S&&b.setPadding(S(t)),v&&b.setKeepUpright(v(t)),b}}(t,e),s=function(t,e){if(\"icon-src\"in t)return function(t,e){const i=\"icon-\",n=i+\"src\",r=nr(t[n],n),s=Jn(t,i+\"anchor\",e),o=Hn(t,i+\"scale\",e),a=Bn(t,i+\"opacity\",e),l=Jn(t,i+\"displacement\",e),h=Bn(t,i+\"rotation\",e),c=jn(t,i+\"rotate-with-view\",e),u=Qn(t,i+\"anchor-origin\"),f=tr(t,i+\"anchor-x-units\"),d=tr(t,i+\"anchor-y-units\"),g=Yn(t,i+\"color\");let p,_=null;if(void 0!==g){Array.isArray(g)&&g.length>0&&\"string\"==typeof g[0]?_=Vn(t,i+\"color\",e):p=sr(g,i+\"color\")}const m=function(t,e){const i=t[e];if(void 0===i)return;if(\"string\"!=typeof i)throw new Error(`Expected a string for ${e}`);return i}(t,i+\"cross-origin\"),y=function(t,e){const i=t[e];if(void 0===i)return;return ir(i,e)}(t,i+\"offset\"),w=Qn(t,i+\"offset-origin\"),x=Zn(t,i+\"width\"),v=Zn(t,i+\"height\"),S=function(t,e){const i=t[e];if(void 0===i)return;if(\"number\"==typeof i)return z(i);if(!Array.isArray(i))throw new Error(`Expected a number or size array for ${e}`);if(2!==i.length||\"number\"!=typeof i[0]||\"number\"!=typeof i[1])throw new Error(`Expected a number or size array for ${e}`);return i}(t,i+\"size\"),C=er(t,i+\"declutter-mode\"),b={src:r,anchorOrigin:u,anchorXUnits:f,anchorYUnits:d,crossOrigin:m,offset:y,offsetOrigin:w,height:v,width:x,size:S,declutterMode:C};let M=null;return function(t){if(M)_&&M.setColor(_(t));else{const e=_?_(t):p;M=new Rn(void 0!==e?Object.assign({},b,{color:e}):Object.assign({},b))}return a&&M.setOpacity(a(t)),l&&M.setDisplacement(l(t)),h&&M.setRotation(h(t)),c&&M.setRotateWithView(c(t)),o&&M.setScale(o(t)),s&&M.setAnchor(s(t)),M}}(t,e);if(\"shape-points\"in t)return function(t,e){const i=\"shape-\",n=i+\"points\",r=i+\"radius\",s=rr(t[n],n);if(!(r in t))throw new Error(`Expected a number for ${r}`);const o=Bn(t,r,e),a=\"number\"==typeof t[r]?t[r]:5,l=i+\"radius2\",h=Bn(t,l,e),c=\"number\"==typeof t[l]?t[l]:void 0,u=Nn(t,i,e),f=Xn(t,i,e),d=Hn(t,i+\"scale\",e),g=Jn(t,i+\"displacement\",e),p=Bn(t,i+\"rotation\",e),_=jn(t,i+\"rotate-with-view\",e),m=Zn(t,i+\"angle\"),y=er(t,i+\"declutter-mode\"),w=new kn({points:s,radius:a,radius2:c,angle:m,declutterMode:y});return function(t){return o&&w.setRadius(o(t)),h&&w.setRadius2(h(t)),u&&w.setFill(u(t)),f&&w.setStroke(f(t)),g&&w.setDisplacement(g(t)),p&&w.setRotation(p(t)),_&&w.setRotateWithView(_(t)),d&&w.setScale(d(t)),w}}(t,e);if(\"circle-radius\"in t)return function(t,e){const i=\"circle-\",n=Nn(t,i,e),r=Xn(t,i,e),s=Bn(t,i+\"radius\",e),o=Hn(t,i+\"scale\",e),a=Jn(t,i+\"displacement\",e),l=Bn(t,i+\"rotation\",e),h=jn(t,i+\"rotate-with-view\",e),c=er(t,i+\"declutter-mode\"),u=new An({radius:5,declutterMode:c});return function(t){return s&&u.setRadius(s(t)),n&&u.setFill(n(t)),r&&u.setStroke(r(t)),a&&u.setDisplacement(a(t)),l&&u.setRotation(l(t)),h&&u.setRotateWithView(h(t)),o&&u.setScale(o(t)),u}}(t,e);return null}(t,e),o=Bn(t,\"z-index\",e);if(!(i||n||r||s||Xe(t)))throw new Error(\"No fill, stroke, point, or text symbolizer properties in style: \"+JSON.stringify(t));const a=new Dn;return function(t){let e=!0;if(i){const n=i(t);n&&(e=!1),a.setFill(n)}if(n){const i=n(t);i&&(e=!1),a.setStroke(i)}if(r){const i=r(t);i&&(e=!1),a.setText(i)}if(s){const i=s(t);i&&(e=!1),a.setImage(i)}return o&&a.setZIndex(o(t)),e?null:a}}function Nn(t,e,i){let n;if(e+\"fill-pattern-src\"in t)n=function(t,e,i){const n=Un(t,e+\"pattern-src\",i),r=Kn(t,e+\"pattern-offset\",i),s=Kn(t,e+\"pattern-size\",i),o=Vn(t,e+\"color\",i);return function(t){return{src:n(t),offset:r&&r(t),size:s&&s(t),color:o&&o(t)}}}(t,e+\"fill-\",i);else{if(\"none\"===t[e+\"fill-color\"])return t=>null;n=Vn(t,e+\"fill-color\",i)}if(!n)return null;const r=new Pn;return function(t){const e=n(t);return e===m?null:(r.setColor(e),r)}}function Xn(t,e,i){const n=Bn(t,e+\"stroke-width\",i),r=Vn(t,e+\"stroke-color\",i);if(!n&&!r)return null;const s=Un(t,e+\"stroke-line-cap\",i),o=Un(t,e+\"stroke-line-join\",i),a=qn(t,e+\"stroke-line-dash\",i),l=Bn(t,e+\"stroke-line-dash-offset\",i),h=Bn(t,e+\"stroke-miter-limit\",i),c=Bn(t,e+\"stroke-offset\",i),u=new Ln;return function(t){if(r){const e=r(t);if(e===m)return null;u.setColor(e)}if(n&&u.setWidth(n(t)),s){const e=s(t);if(\"butt\"!==e&&\"round\"!==e&&\"square\"!==e)throw new Error(\"Expected butt, round, or square line cap\");u.setLineCap(e)}if(o){const e=o(t);if(\"bevel\"!==e&&\"round\"!==e&&\"miter\"!==e)throw new Error(\"Expected bevel, round, or miter line join\");u.setLineJoin(e)}return a&&u.setLineDash(a(t)),l&&u.setLineDashOffset(l(t)),h&&u.setMiterLimit(h(t)),c&&u.setOffset(c(t)),u}}function Yn(t,e){if(!(e in t))return;const i=t[e];return void 0===i?void 0:i}function Bn(t,e,i){const n=Yn(t,e);if(void 0===n)return;const r=Cn(n,G,i);return function(t){return rr(r(t),e)}}function Un(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,N,i);return function(t){return nr(r(t),e)}}function jn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,W,i);return function(t){const i=r(t);if(\"boolean\"!=typeof i)throw new Error(`Expected a boolean for ${e}`);return i}}function Vn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,X,i);return function(t){return sr(r(t),e)}}function qn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;if(Array.isArray(n)&&(0===n.length||\"string\"!=typeof n[0])){const t=n.map((t,n)=>{if(\"number\"==typeof t)return()=>t;const r=Cn(t,G,i);return function(t){return rr(r(t),`${e}[${n}]`)}});return function(e){const i=new Array(t.length);for(let n=0;n<t.length;++n)i[n]=t[n](e);return i}}const r=Cn(n,Y,i);return function(t){return ir(r(t),e)}}function Jn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,Y,i);return function(t){const i=ir(r(t),e);if(2!==i.length)throw new Error(`Expected two numbers for ${e}`);return i}}function Kn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,Y,i);return function(t){return or(r(t),e)}}function Hn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,Y|G,i);return function(t){return function(t,e){if(\"number\"==typeof t)return t;return or(t,e)}(r(t),e)}}function Zn(t,e){const i=t[e];if(void 0!==i){if(\"number\"!=typeof i)throw new Error(`Expected a number for ${e}`);return i}}function Qn(t,e){const i=t[e];if(void 0!==i){if(\"bottom-left\"!==i&&\"bottom-right\"!==i&&\"top-left\"!==i&&\"top-right\"!==i)throw new Error(`Expected bottom-left, bottom-right, top-left, or top-right for ${e}`);return i}}function tr(t,e){const i=t[e];if(void 0!==i){if(\"pixels\"!==i&&\"fraction\"!==i)throw new Error(`Expected pixels or fraction for ${e}`);return i}}function er(t,e){const i=t[e];if(void 0!==i){if(\"string\"!=typeof i)throw new Error(`Expected a string for ${e}`);if(\"declutter\"!==i&&\"obstacle\"!==i&&\"none\"!==i)throw new Error(`Expected declutter, obstacle, or none for ${e}`);return i}}function ir(t,e){if(!Array.isArray(t))throw new Error(`Expected an array for ${e}`);const i=t.length;for(let n=0;n<i;++n)if(\"number\"!=typeof t[n])throw new Error(`Expected an array of numbers for ${e}`);return t}function nr(t,e){if(\"string\"!=typeof t)throw new Error(`Expected a string for ${e}`);return t}function rr(t,e){if(\"number\"!=typeof t)throw new Error(`Expected a number for ${e}`);return t}function sr(t,e){if(\"string\"==typeof t)return t;const i=ir(t,e),n=i.length;if(n<3||n>4)throw new Error(`Expected a color with 3 or 4 values for ${e}`);return i}function or(t,e){const i=ir(t,e);if(2!==i.length)throw new Error(`Expected an array of two numbers for ${e}`);return i}const ar=\"BUILD_INSTRUCTIONS\",lr=\"DISPOSE_INSTRUCTIONS\",hr=\"RENDER\",cr={radians:6370997/(2*Math.PI),degrees:2*Math.PI*6370997/360,ft:.3048,m:1,\"us-ft\":1200/3937};class ur{constructor(t){this.code_=t.code,this.units_=t.units,this.extent_=void 0!==t.extent?t.extent:null,this.worldExtent_=void 0!==t.worldExtent?t.worldExtent:null,this.axisOrientation_=void 0!==t.axisOrientation?t.axisOrientation:\"enu\",this.global_=void 0!==t.global&&t.global,this.canWrapX_=!(!this.global_||!this.extent_),this.getPointResolutionFunc_=t.getPointResolution,this.defaultTileGrid_=null,this.metersPerUnit_=t.metersPerUnit}canWrapX(){return this.canWrapX_}getCode(){return this.code_}getExtent(){return this.extent_}getUnits(){return this.units_}getMetersPerUnit(){return this.metersPerUnit_||cr[this.units_]}getWorldExtent(){return this.worldExtent_}getAxisOrientation(){return this.axisOrientation_}isGlobal(){return this.global_}setGlobal(t){this.global_=t,this.canWrapX_=!(!t||!this.extent_)}getDefaultTileGrid(){return this.defaultTileGrid_}setDefaultTileGrid(t){this.defaultTileGrid_=t}setExtent(t){this.extent_=t,this.canWrapX_=!(!this.global_||!t)}setWorldExtent(t){this.worldExtent_=t}setGetPointResolution(t){this.getPointResolutionFunc_=t}getPointResolutionFunc(){return this.getPointResolutionFunc_}}const fr=6378137,dr=Math.PI*fr,gr=[-dr,-dr,dr,dr],pr=[-180,-85,180,85],_r=fr*Math.log(Math.tan(Math.PI/2));class mr extends ur{constructor(t){super({code:t,units:\"m\",extent:gr,global:!0,worldExtent:pr,getPointResolution:function(t,e){return t/Math.cosh(e[1]/fr)}})}}const yr=[new mr(\"EPSG:3857\"),new mr(\"EPSG:102100\"),new mr(\"EPSG:102113\"),new mr(\"EPSG:900913\"),new mr(\"http://www.opengis.net/def/crs/EPSG/0/3857\"),new mr(\"http://www.opengis.net/gml/srs/epsg.xml#3857\")];function wr(t,e,i,n){const r=t.length;i=i>1?i:2,n=n??i,void 0===e&&(e=i>2?t.slice():new Array(r));for(let i=0;i<r;i+=n){e[i]=dr*t[i]/180;let n=fr*Math.log(Math.tan(Math.PI*(+t[i+1]+90)/360));n>_r?n=_r:n<-_r&&(n=-_r),e[i+1]=n}return e}function xr(t,e,i,n){const r=t.length;i=i>1?i:2,n=n??i,void 0===e&&(e=i>2?t.slice():new Array(r));for(let i=0;i<r;i+=n)e[i]=180*t[i]/dr,e[i+1]=360*Math.atan(Math.exp(t[i+1]/fr))/Math.PI-90;return e}const vr=[-180,-90,180,90],Sr=6378137*Math.PI/180;class Cr extends ur{constructor(t,e){super({code:t,units:\"degrees\",extent:vr,axisOrientation:e,global:!0,metersPerUnit:Sr,worldExtent:vr})}}const br=[new Cr(\"CRS:84\"),new Cr(\"EPSG:4326\",\"neu\"),new Cr(\"urn:ogc:def:crs:OGC:1.3:CRS84\"),new Cr(\"urn:ogc:def:crs:OGC:2:84\"),new Cr(\"http://www.opengis.net/def/crs/OGC/1.3/CRS84\"),new Cr(\"http://www.opengis.net/gml/srs/epsg.xml#4326\",\"neu\"),new Cr(\"http://www.opengis.net/def/crs/EPSG/0/4326\",\"neu\")];let Mr={};let Ir={};function Er(t,e,i){const n=t.getCode(),r=e.getCode();n in Ir||(Ir[n]={}),Ir[n][r]=i}function kr(t,e){return t in Ir&&e in Ir[t]?Ir[t][e]:null}const Ar=.9996,Pr=.00669438,Or=Pr*Pr,Rr=Or*Pr,Lr=Pr/(1-Pr),Dr=Math.sqrt(1-Pr),Fr=(1-Dr)/(1+Dr),Tr=Fr*Fr,zr=Tr*Fr,$r=zr*Fr,Wr=$r*Fr,Gr=.9983242984503243,Nr=15*Or/256+45*Rr/1024,Xr=35*Rr/3072,Yr=1.5*Fr-27/32*zr+269/512*Wr,Br=21/16*Tr-55/32*$r,Ur=151/96*zr-417/128*Wr,jr=1097/512*$r,Vr=6378137;function qr(t,e,i){const n=t-5e5,r=(i.north?e:e-1e7)/Ar/(Vr*Gr),s=r+Yr*Math.sin(2*r)+Br*Math.sin(4*r)+Ur*Math.sin(6*r)+jr*Math.sin(8*r),o=Math.sin(s),a=o*o,l=Math.cos(s),h=o/l,c=h*h,u=c*c,f=1-Pr*a,p=Math.sqrt(1-Pr*a),m=Lr*l**2,y=m*m,w=n/(Vr/p*Ar),x=w*w,v=x*w,S=v*w,C=S*w,b=s-h/((1-Pr)/f)*(x/2-S/24*(5+3*c+10*m-4*y-9*Lr))+C*w/720*(61+90*c+298*m+45*u-252*Lr-3*y);let M=(w-v/6*(1+2*c+m)+C/120*(5-2*m+28*c-3*y+8*Lr+24*u))/l;return M=_(M+g(Kr(i.number)),-Math.PI,Math.PI),[d(M),d(b)]}function Jr(t,e,i){t=_(t,-180,180),e<-80?e=-80:e>84&&(e=84);const n=g(e),r=Math.sin(n),s=Math.cos(n),o=r/s,a=o*o,l=a*a,h=g(t),c=g(Kr(i.number)),u=Vr/Math.sqrt(1-Pr*r**2),f=Lr*s**2,d=s*_(h-c,-Math.PI,Math.PI),p=d*d,m=p*d,y=m*d,w=y*d,x=w*d,v=Vr*(Gr*n-.002514607064228144*Math.sin(2*n)+Nr*Math.sin(4*n)-Xr*Math.sin(6*n)),S=Ar*u*(d+m/6*(1-a+f)+w/120*(5-18*a+l+72*f-58*Lr))+5e5;let C=Ar*(v+u*o*(p/2+y/24*(5-a+9*f+4*f**2)+x/720*(61-58*a+l+600*f-330*Lr)));return i.north||(C+=1e7),[S,C]}function Kr(t){return 6*(t-1)-180+3}const Hr=[/^EPSG:(\\d+)$/,/^urn:ogc:def:crs:EPSG::(\\d+)$/,/^http:\\/\\/www\\.opengis\\.net\\/def\\/crs\\/EPSG\\/0\\/(\\d+)$/];function Zr(t){let e=0;for(const i of Hr){const n=t.match(i);if(n){e=parseInt(n[1]);break}}if(!e)return null;let i=0,n=!1;return e>32700&&e<32761?i=e-32700:e>32600&&e<32661&&(n=!0,i=e-32600),i?{number:i,north:n}:null}function Qr(t,e){return function(i,n,r,s){const o=i.length;r=r>1?r:2,s=s??r,n||(n=r>2?i.slice():new Array(o));for(let r=0;r<o;r+=s){const s=i[r],o=i[r+1],a=t(s,o,e);n[r]=a[0],n[r+1]=a[1]}return n}}const ts=[function(t){const e=Zr(t.getCode());return e?{forward:Qr(Jr,e),inverse:Qr(qr,e)}:null}],es=[function(t){return Zr(t)?new ur({code:t,units:\"m\"}):null}];function is(t,e){if(void 0!==e)for(let i=0,n=t.length;i<n;++i)e[i]=t[i];else e=t.slice();return e}function ns(t){!function(t,e){Mr[t]=e}(t.getCode(),t),Er(t,t,is)}function rs(t){if(\"string\"!=typeof t)return t;const e=Mr[i=t]||Mr[i.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\\w+)$/,\"EPSG:$3\")]||null;var i;if(e)return e;for(const e of es){const i=e(t);if(i)return i}return null}function ss(t){!function(t){t.forEach(ns)}(t),t.forEach(function(e){t.forEach(function(t){e!==t&&Er(e,t,is)})})}function os(t,e){return function(i,n,r,s){return n=t(i,n,r,s),e(n,n,r,s)}}function as(t,e){return function(t,e){const i=t.getCode(),n=e.getCode();let r=kr(i,n);if(r)return r;let s=null,o=null;for(const i of ts)s||(s=i(t)),o||(o=i(e));if(!s&&!o)return null;const a=\"EPSG:4326\";if(o)if(s)r=os(s.inverse,o.forward);else{const t=kr(i,a);t&&(r=os(t,o.forward))}else{const t=kr(a,n);t&&(r=os(s.inverse,t))}return r&&(ns(t),ns(e),Er(t,e,r)),r}(rs(t),rs(e))}var ls,hs,cs;ss(yr),ss(br),ls=yr,hs=wr,cs=xr,br.forEach(function(t){ls.forEach(function(e){Er(t,e,hs),Er(e,t,cs)})});const us=Fe(),fs=[NaN,NaN];class ds extends ni{constructor(){super(),this.extent_=[1/0,1/0,-1/0,-1/0],this.extentRevision_=-1,this.simplifiedGeometryMaxMinSquaredTolerance=0,this.simplifiedGeometryRevision=0,this.simplifyTransformedInternal=Je((t,e,i)=>{if(!i)return this.getSimplifiedGeometry(e);const n=this.clone();return n.applyTransform(i),n.getSimplifiedGeometry(e)})}simplifyTransformed(t,e){return this.simplifyTransformedInternal(this.getRevision(),t,e)}clone(){return Qe()}closestPointXY(t,e,i,n){return Qe()}containsXY(t,e){return 0===this.closestPointXY(t,e,fs,Number.MIN_VALUE)}getClosestPoint(t,e){return e=e||[NaN,NaN],this.closestPointXY(t[0],t[1],e,1/0),e}intersectsCoordinate(t){return this.containsXY(t[0],t[1])}computeExtent(t){return Qe()}getExtent(t){if(this.extentRevision_!=this.getRevision()){const t=this.computeExtent(this.extent_);(isNaN(t[0])||isNaN(t[1]))&&me(t),this.extentRevision_=this.getRevision()}return function(t,e){return e?(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e):t}(this.extent_,t)}rotate(t,e){Qe()}scale(t,e,i){Qe()}simplify(t){return this.getSimplifiedGeometry(t*t)}getSimplifiedGeometry(t){return Qe()}getType(){return Qe()}applyTransform(t){Qe()}intersectsExtent(t){return Qe()}translate(t,e){Qe()}transform(t,e){const i=rs(t),n=\"tile-pixels\"==i.getUnits()?function(t,n,r){const s=i.getExtent(),o=i.getWorldExtent(),a=Ce(o)/Ce(s);$e(us,o[0],o[3],a,-a,0,0,0);const l=Ae(t,0,t.length,r,us,n),h=as(i,e);return h?h(l,l,r):l}:as(i,e);return this.applyTransform(n),this}}class gs extends ds{constructor(){super(),this.layout=\"XY\",this.stride=2,this.flatCoordinates}computeExtent(t){return we(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t)}getCoordinates(){return Qe()}getFirstCoordinate(){return this.flatCoordinates.slice(0,this.stride)}getFlatCoordinates(){return this.flatCoordinates}getLastCoordinate(){return this.flatCoordinates.slice(this.flatCoordinates.length-this.stride)}getLayout(){return this.layout}getSimplifiedGeometry(t){if(this.simplifiedGeometryRevision!==this.getRevision()&&(this.simplifiedGeometryMaxMinSquaredTolerance=0,this.simplifiedGeometryRevision=this.getRevision()),t<0||0!==this.simplifiedGeometryMaxMinSquaredTolerance&&t<=this.simplifiedGeometryMaxMinSquaredTolerance)return this;const e=this.getSimplifiedGeometryInternal(t);return e.getFlatCoordinates().length<this.flatCoordinates.length?e:(this.simplifiedGeometryMaxMinSquaredTolerance=t,this)}getSimplifiedGeometryInternal(t){return this}getStride(){return this.stride}setFlatCoordinates(t,e){this.stride=ps(t),this.layout=t,this.flatCoordinates=e}setCoordinates(t,e){Qe()}setLayout(t,e,i){let n;if(t)n=ps(t);else{for(let t=0;t<i;++t){if(0===e.length)return this.layout=\"XY\",void(this.stride=2);e=e[0]}n=e.length,t=function(t){let e;2==t?e=\"XY\":3==t?e=\"XYZ\":4==t&&(e=\"XYZM\");return e}(n)}this.layout=t,this.stride=n}applyTransform(t){this.flatCoordinates&&(t(this.flatCoordinates,this.flatCoordinates,this.layout.startsWith(\"XYZ\")?3:2,this.stride),this.changed())}rotate(t,e){const i=this.getFlatCoordinates();if(i){const n=this.getStride();Pe(i,0,i.length,n,t,e,i),this.changed()}}scale(t,e,i){void 0===e&&(e=t),i||(i=Se(this.getExtent()));const n=this.getFlatCoordinates();if(n){const r=this.getStride();!function(t,e,i,n,r,s,o,a){a=a||[];const l=o[0],h=o[1];let c=0;for(let o=e;o<i;o+=n){const e=t[o]-l,i=t[o+1]-h;a[c++]=l+r*e,a[c++]=h+s*i;for(let e=o+2;e<o+n;++e)a[c++]=t[e]}a&&a.length!=c&&(a.length=c)}(n,0,n.length,r,t,e,i,n),this.changed()}}translate(t,e){const i=this.getFlatCoordinates();if(i){const n=this.getStride();!function(t,e,i,n,r,s,o){o=o||[];let a=0;for(let l=e;l<i;l+=n){o[a++]=t[l]+r,o[a++]=t[l+1]+s;for(let e=l+2;e<l+n;++e)o[a++]=t[e]}o&&o.length!=a&&(o.length=a)}(i,0,i.length,n,t,e,i),this.changed()}}}function ps(t){let e;return\"XY\"==t?e=2:\"XYZ\"==t||\"XYM\"==t?e=3:\"XYZM\"==t&&(e=4),e}function _s(t,e,i,n,r,s,o){const a=t[e],l=t[e+1],h=t[i]-a,c=t[i+1]-l;let u;if(0===h&&0===c)u=e;else{const f=((r-a)*h+(s-l)*c)/(h*h+c*c);if(f>1)u=i;else{if(f>0){for(let r=0;r<n;++r)o[r]=p(t[e+r],t[i+r],f);return void(o.length=n)}u=e}}for(let e=0;e<n;++e)o[e]=t[u+e];o.length=n}function ms(t,e,i,n,r){let s=t[e],o=t[e+1];for(e+=n;e<i;e+=n){const i=t[e],n=t[e+1],a=f(s,o,i,n);a>r&&(r=a),s=i,o=n}return r}function ys(t,e,i,n,r,s,o,a,l,h,c){if(e==i)return h;let u,d;if(0===r){if(d=f(o,a,t[e],t[e+1]),d<h){for(u=0;u<n;++u)l[u]=t[e+u];return l.length=n,d}return h}c=c||[NaN,NaN];let g=e+n;for(;g<i;)if(_s(t,g-n,g,n,o,a,c),d=f(o,a,c[0],c[1]),d<h){for(h=d,u=0;u<n;++u)l[u]=c[u];l.length=n,g+=n}else g+=n*Math.max((Math.sqrt(d)-Math.sqrt(h))/r|0,1);if(s&&(_s(t,i-n,e,n,o,a,c),d=f(o,a,c[0],c[1]),d<h)){for(h=d,u=0;u<n;++u)l[u]=c[u];l.length=n}return h}function ws(t,e,i,n){for(let r=0,s=i.length;r<s;++r){const s=i[r];for(let i=0;i<n;++i)t[e++]=s[i]}return e}function xs(e,i,n,r,s,o,a){let l,h;const c=(n-i)/r;if(1===c)l=i;else if(2===c)l=i,h=s;else if(0!==c){let o=e[i],a=e[i+1],c=0;const u=[0];for(let t=i+r;t<n;t+=r){const i=e[t],n=e[t+1];c+=Math.sqrt((i-o)*(i-o)+(n-a)*(n-a)),u.push(c),o=i,a=n}const f=s*c,d=function(e,i,n){let r,s;n=n||t;let o=0,a=e.length,l=!1;for(;o<a;)r=o+(a-o>>1),s=+n(e[r],i),s<0?o=r+1:(a=r,l=!s);return l?o:~o}(u,f);d<0?(h=(f-u[-d-2])/(u[-d-1]-u[-d-2]),l=i+(-d-2)*r):l=i+d*r}a=a>1?a:2,o=o||new Array(a);for(let t=0;t<a;++t)o[t]=void 0===l?NaN:void 0===h?e[l+t]:p(e[l+t],e[l+r+t],h);return o}function vs(t,e,i,n,r){const s=function(t,e){let i;return i=e(function(t){return[t[0],t[1]]}(t)),i||(i=e(function(t){return[t[2],t[1]]}(t)),i||(i=e(function(t){return[t[2],t[3]]}(t)),i||(i=e(function(t){return[t[0],t[3]]}(t)),i||!1)))}(r,function(r){return!Ss(t,e,i,n,r[0],r[1])});return!s}function Ss(t,e,i,n,r,s){let o=0,a=t[i-n],l=t[i-n+1];for(;e<i;e+=n){const i=t[e],n=t[e+1];l<=s?n>s&&(i-a)*(s-l)-(r-a)*(n-l)>0&&o++:n<=s&&(i-a)*(s-l)-(r-a)*(n-l)<0&&o--,a=i,l=n}return 0!==o}function Cs(t,e,i,n,r,s){if(0===i.length)return!1;if(!Ss(t,e,i[0],n,r,s))return!1;for(let e=1,o=i.length;e<o;++e)if(Ss(t,i[e-1],i[e],n,r,s))return!1;return!0}function bs(t,e,i,n,r){let s;for(e+=n;e<i;e+=n)if(s=r(t.slice(e-n,e),t.slice(e,e+n)),s)return s;return!1}function Ms(t,e,i,n,r,s){return s=s??xe([1/0,1/0,-1/0,-1/0],t,e,i,n),!!be(r,s)&&(s[0]>=r[0]&&s[2]<=r[2]||s[1]>=r[1]&&s[3]<=r[3]||bs(t,e,i,n,function(t,e){return function(t,e,i){let n=!1;const r=pe(t,e),s=pe(t,i);if(r===ae||s===ae)n=!0;else{const o=t[0],a=t[1],l=t[2],h=t[3],c=e[0],u=e[1],f=i[0],d=i[1],g=(d-u)/(f-c);let p,_;s&le&&!(r&le)&&(p=f-(d-h)/g,n=p>=o&&p<=l),n||!(s&he)||r&he||(_=d-(f-l)*g,n=_>=a&&_<=h),n||!(s&ce)||r&ce||(p=f-(d-a)/g,n=p>=o&&p<=l),n||!(s&ue)||r&ue||(_=d-(f-o)*g,n=_>=a&&_<=h)}return n}(r,t,e)}))}function Is(t,e,i,n,r){if(!function(t,e,i,n,r){return!!(Ms(t,e,i,n,r)||Ss(t,e,i,n,r[0],r[1])||Ss(t,e,i,n,r[0],r[3])||Ss(t,e,i,n,r[2],r[1])||Ss(t,e,i,n,r[2],r[3]))}(t,e,i[0],n,r))return!1;if(1===i.length)return!0;for(let e=1,s=i.length;e<s;++e)if(vs(t,i[e-1],i[e],n,r)&&!Ms(t,i[e-1],i[e],n,r))return!1;return!0}function Es(t,e,i,n,r,s,o){const a=(i-e)/n;if(a<3){for(;e<i;e+=n)s[o++]=t[e],s[o++]=t[e+1];return o}const l=new Array(a);l[0]=1,l[a-1]=1;const h=[e,i-n];let c=0;for(;h.length>0;){const i=h.pop(),s=h.pop();let o=0;const a=t[s],f=t[s+1],d=t[i],g=t[i+1];for(let e=s+n;e<i;e+=n){const i=u(t[e],t[e+1],a,f,d,g);i>o&&(c=e,o=i)}o>r&&(l[(c-e)/n]=1,s+n<c&&h.push(s,c),c+n<i&&h.push(c,i))}for(let i=0;i<a;++i)l[i]&&(s[o++]=t[e+i*n],s[o++]=t[e+i*n+1]);return o}function ks(t,e){return e*Math.round(t/e)}function As(t,e,i,n,r,s,o){if(e==i)return o;let a,l,h=ks(t[e],r),c=ks(t[e+1],r);e+=n,s[o++]=h,s[o++]=c;do{if(a=ks(t[e],r),l=ks(t[e+1],r),(e+=n)==i)return s[o++]=a,s[o++]=l,o}while(a==h&&l==c);for(;e<i;){const i=ks(t[e],r),u=ks(t[e+1],r);if(e+=n,i==a&&u==l)continue;const f=a-h,d=l-c,g=i-h,p=u-c;f*p==d*g&&(f<0&&g<f||f==g||f>0&&g>f)&&(d<0&&p<d||d==p||d>0&&p>d)?(a=i,l=u):(s[o++]=a,s[o++]=l,h=a,c=l,a=i,l=u)}return s[o++]=a,s[o++]=l,o}function Ps(t,e,i,n,r,s,o,a){for(let l=0,h=i.length;l<h;++l){const h=i[l];o=As(t,e,h,n,r,s,o),a.push(o),e=h}return o}class Os extends gs{constructor(t,e){super(),this.flatMidpoint_=null,this.flatMidpointRevision_=-1,this.maxDelta_=-1,this.maxDeltaRevision_=-1,void 0===e||Array.isArray(t[0])?this.setCoordinates(t,e):this.setFlatCoordinates(e,t)}appendCoordinate(t){i(this.flatCoordinates,t),this.changed()}clone(){const t=new Os(this.flatCoordinates.slice(),this.layout);return t.applyProperties(this),t}closestPointXY(t,e,i,n){return n<fe(this.getExtent(),t,e)?n:(this.maxDeltaRevision_!=this.getRevision()&&(this.maxDelta_=Math.sqrt(ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,0)),this.maxDeltaRevision_=this.getRevision()),ys(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,this.maxDelta_,!1,t,e,i,n))}forEachSegment(t){return bs(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t)}getCoordinateAtM(t,e){return\"XYM\"!=this.layout&&\"XYZM\"!=this.layout?null:(e=void 0!==e&&e,function(t,e,i,n,r,s){if(i==e)return null;let o;if(r<t[e+n-1])return s?(o=t.slice(e,e+n),o[n-1]=r,o):null;if(t[i-1]<r)return s?(o=t.slice(i-n,i),o[n-1]=r,o):null;if(r==t[e+n-1])return t.slice(e,e+n);let a=e/n,l=i/n;for(;a<l;){const e=a+l>>1;r<t[(e+1)*n-1]?l=e:a=e+1}const h=t[a*n-1];if(r==h)return t.slice((a-1)*n,(a-1)*n+n);const c=(r-h)/(t[(a+1)*n-1]-h);o=[];for(let e=0;e<n-1;++e)o.push(p(t[(a-1)*n+e],t[a*n+e],c));return o.push(r),o}(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e))}getCoordinates(){return sn(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getCoordinateAt(t,e){return xs(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e,this.stride)}getLength(){return Me(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getFlatMidpoint(){return this.flatMidpointRevision_!=this.getRevision()&&(this.flatMidpoint_=this.getCoordinateAt(.5,this.flatMidpoint_??void 0),this.flatMidpointRevision_=this.getRevision()),this.flatMidpoint_}getSimplifiedGeometryInternal(t){const e=[];return e.length=Es(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e,0),new Os(e,\"XY\")}getType(){return\"LineString\"}intersectsExtent(t){return Ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,this.getExtent())}setCoordinates(t,e){this.setLayout(e,t,1),this.flatCoordinates||(this.flatCoordinates=[]),this.flatCoordinates.length=ws(this.flatCoordinates,0,t,this.stride),this.changed()}}class Rs extends gs{constructor(t,e){super(),this.setCoordinates(t,e)}clone(){const t=new Rs(this.flatCoordinates.slice(),this.layout);return t.applyProperties(this),t}closestPointXY(t,e,i,n){const r=this.flatCoordinates,s=f(t,e,r[0],r[1]);if(s<n){const t=this.stride;for(let e=0;e<t;++e)i[e]=r[e];return i.length=t,s}return n}getCoordinates(){return this.flatCoordinates.slice()}computeExtent(t){return ye(this.flatCoordinates,t)}getType(){return\"Point\"}intersectsExtent(t){return ge(t,this.flatCoordinates[0],this.flatCoordinates[1])}setCoordinates(t,e){this.setLayout(e,t,0),this.flatCoordinates||(this.flatCoordinates=[]),this.flatCoordinates.length=function(t,e,i){for(let n=0,r=i.length;n<r;++n)t[e++]=i[n];return e}(this.flatCoordinates,0,t,this.stride),this.changed()}}function Ls(t,e,i,n){let r=0;const s=t[i-n],o=t[i-n+1];let a=0,l=0;for(;e<i;e+=n){const i=t[e]-s,n=t[e+1]-o;r+=l*i-a*n,a=i,l=n}return r/2}class Ds extends gs{constructor(t,e){super(),this.maxDelta_=-1,this.maxDeltaRevision_=-1,void 0===e||Array.isArray(t[0])?this.setCoordinates(t,e):this.setFlatCoordinates(e,t)}clone(){return new Ds(this.flatCoordinates.slice(),this.layout)}closestPointXY(t,e,i,n){return n<fe(this.getExtent(),t,e)?n:(this.maxDeltaRevision_!=this.getRevision()&&(this.maxDelta_=Math.sqrt(ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,0)),this.maxDeltaRevision_=this.getRevision()),ys(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,this.maxDelta_,!0,t,e,i,n))}getArea(){return Ls(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getCoordinates(){return sn(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getSimplifiedGeometryInternal(t){const e=[];return e.length=Es(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e,0),new Ds(e,\"XY\")}getType(){return\"LinearRing\"}intersectsExtent(t){return Ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t)}setCoordinates(t,e){this.setLayout(e,t,1),this.flatCoordinates||(this.flatCoordinates=[]),this.flatCoordinates.length=ws(this.flatCoordinates,0,t,this.stride),this.changed()}}function Fs(e,i,n,r,s,o,a){let l,h,c,u,f,d,g;const p=s[o+1],_=[];for(let t=0,s=n.length;t<s;++t){const s=n[t];for(u=e[s-r],d=e[s-r+1],l=i;l<s;l+=r)f=e[l],g=e[l+1],(p<=d&&g<=p||d<=p&&p<=g)&&(c=(p-d)/(g-d)*(f-u)+u,_.push(c)),u=f,d=g}let m=NaN,y=-1/0;for(_.sort(t),u=_[0],l=1,h=_.length;l<h;++l){f=_[l];const t=Math.abs(f-u);t>y&&(c=(u+f)/2,Cs(e,i,n,r,c,p)&&(m=c,y=t)),u=f}return isNaN(m)&&(m=s[o]),a?(a.push(m,p,y),a):[m,p,y]}function Ts(t,e,i,n){for(;e<i-n;){for(let r=0;r<n;++r){const s=t[e+r];t[e+r]=t[i-n+r],t[i-n+r]=s}e+=n,i-=n}}function zs(t,e,i,n){let r=0,s=t[i-n],o=t[i-n+1];for(;e<i;e+=n){const i=t[e],n=t[e+1];r+=(i-s)*(n+o),s=i,o=n}return 0===r?void 0:r>0}function $s(t,e,i,n,r){r=void 0!==r&&r;for(let s=0,o=i.length;s<o;++s){const o=i[s],a=zs(t,e,o,n);(0===s?r&&a||!r&&!a:r&&!a||!r&&a)&&Ts(t,e,o,n),e=o}return e}class Ws extends gs{constructor(t,e,i){super(),this.ends_=[],this.flatInteriorPointRevision_=-1,this.flatInteriorPoint_=null,this.maxDelta_=-1,this.maxDeltaRevision_=-1,this.orientedRevision_=-1,this.orientedFlatCoordinates_=null,void 0!==e&&i?(this.setFlatCoordinates(e,t),this.ends_=i):this.setCoordinates(t,e)}appendLinearRing(t){this.flatCoordinates?i(this.flatCoordinates,t.getFlatCoordinates()):this.flatCoordinates=t.getFlatCoordinates().slice(),this.ends_.push(this.flatCoordinates.length),this.changed()}clone(){const t=new Ws(this.flatCoordinates.slice(),this.layout,this.ends_.slice());return t.applyProperties(this),t}closestPointXY(t,e,i,n){return n<fe(this.getExtent(),t,e)?n:(this.maxDeltaRevision_!=this.getRevision()&&(this.maxDelta_=Math.sqrt(function(t,e,i,n,r){for(let s=0,o=i.length;s<o;++s){const o=i[s];r=ms(t,e,o,n,r),e=o}return r}(this.flatCoordinates,0,this.ends_,this.stride,0)),this.maxDeltaRevision_=this.getRevision()),function(t,e,i,n,r,s,o,a,l,h,c){c=c||[NaN,NaN];for(let u=0,f=i.length;u<f;++u){const f=i[u];h=ys(t,e,f,n,r,s,o,a,l,h,c),e=f}return h}(this.flatCoordinates,0,this.ends_,this.stride,this.maxDelta_,!0,t,e,i,n))}containsXY(t,e){return Cs(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride,t,e)}getArea(){return function(t,e,i,n){let r=0;for(let s=0,o=i.length;s<o;++s){const o=i[s];r+=Ls(t,e,o,n),e=o}return r}(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride)}getCoordinates(t){let e;return void 0!==t?(e=this.getOrientedFlatCoordinates().slice(),$s(e,0,this.ends_,this.stride,t)):e=this.flatCoordinates,on(e,0,this.ends_,this.stride)}getEnds(){return this.ends_}getFlatInteriorPoint(){if(this.flatInteriorPointRevision_!=this.getRevision()){const t=Se(this.getExtent());this.flatInteriorPoint_=Fs(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride,t,0),this.flatInteriorPointRevision_=this.getRevision()}return this.flatInteriorPoint_}getInteriorPoint(){return new Rs(this.getFlatInteriorPoint(),\"XYM\")}getLinearRingCount(){return this.ends_.length}getLinearRing(t){return t<0||this.ends_.length<=t?null:new Ds(this.flatCoordinates.slice(0===t?0:this.ends_[t-1],this.ends_[t]),this.layout)}getLinearRings(){const t=this.layout,e=this.flatCoordinates,i=this.ends_,n=[];let r=0;for(let s=0,o=i.length;s<o;++s){const o=i[s],a=new Ds(e.slice(r,o),t);n.push(a),r=o}return n}getOrientedFlatCoordinates(){if(this.orientedRevision_!=this.getRevision()){const t=this.flatCoordinates;!function(t,e,i,n,r){r=void 0!==r&&r;for(let s=0,o=i.length;s<o;++s){const o=i[s],a=zs(t,e,o,n);if(0===s){if(r&&a||!r&&!a)return!1}else if(r&&!a||!r&&a)return!1;e=o}return!0}(t,0,this.ends_,this.stride)?(this.orientedFlatCoordinates_=t.slice(),this.orientedFlatCoordinates_.length=$s(this.orientedFlatCoordinates_,0,this.ends_,this.stride)):this.orientedFlatCoordinates_=t,this.orientedRevision_=this.getRevision()}return this.orientedFlatCoordinates_}getSimplifiedGeometryInternal(t){const e=[],i=[];return e.length=Ps(this.flatCoordinates,0,this.ends_,this.stride,Math.sqrt(t),e,0,i),new Ws(e,\"XY\",i)}getType(){return\"Polygon\"}intersectsExtent(t){return Is(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride,t)}setCoordinates(t,e){this.setLayout(e,t,2),this.flatCoordinates||(this.flatCoordinates=[]);const i=function(t,e,i,n,r){r=r||[];let s=0;for(let o=0,a=i.length;o<a;++o){const a=ws(t,e,i[o],n);r[s++]=a,e=a}return r.length=s,r}(this.flatCoordinates,0,t,this.stride,this.ends_);this.flatCoordinates.length=0===i.length?0:i[i.length-1],this.changed()}}const Gs=Fe();class Ns{constructor(t,e,i,n,r,s){this.styleFunction,this.extent_,this.id_=s,this.type_=t,this.flatCoordinates_=e,this.flatInteriorPoints_=null,this.flatMidpoints_=null,this.ends_=i||null,this.properties_=r,this.squaredTolerance_,this.stride_=n,this.simplifiedGeometry_}get(t){return this.properties_[t]}getExtent(){return this.extent_||(this.extent_=\"Point\"===this.type_?ye(this.flatCoordinates_):we(this.flatCoordinates_,0,this.flatCoordinates_.length,this.stride_)),this.extent_}getFlatInteriorPoint(){if(!this.flatInteriorPoints_){const t=Se(this.getExtent());this.flatInteriorPoints_=Fs(this.flatCoordinates_,0,this.ends_,this.stride_,t,0)}return this.flatInteriorPoints_}getFlatInteriorPoints(){if(!this.flatInteriorPoints_){const t=function(t,e){const i=[];let n,r=0,s=0;for(let o=0,a=e.length;o<a;++o){const a=e[o],l=zs(t,r,a,2);if(void 0===n&&(n=l),l===n)i.push(e.slice(s,o+1));else{if(0===i.length)continue;i[i.length-1].push(e[s])}s=o+1,r=a}return i}(this.flatCoordinates_,this.ends_),e=function(t,e,i,n){const r=[];let s=[1/0,1/0,-1/0,-1/0];for(let o=0,a=i.length;o<a;++o){const a=i[o];s=we(t,e,a[0],n),r.push((s[0]+s[2])/2,(s[1]+s[3])/2),e=a[a.length-1]}return r}(this.flatCoordinates_,0,t,this.stride_);this.flatInteriorPoints_=function(t,e,i,n,r){let s=[];for(let o=0,a=i.length;o<a;++o){const a=i[o];s=Fs(t,e,a,n,r,2*o,s),e=a[a.length-1]}return s}(this.flatCoordinates_,0,t,this.stride_,e)}return this.flatInteriorPoints_}getFlatMidpoint(){return this.flatMidpoints_||(this.flatMidpoints_=xs(this.flatCoordinates_,0,this.flatCoordinates_.length,this.stride_,.5)),this.flatMidpoints_}getFlatMidpoints(){if(!this.flatMidpoints_){this.flatMidpoints_=[];const t=this.flatCoordinates_;let e=0;const n=this.ends_;for(let r=0,s=n.length;r<s;++r){const s=n[r],o=xs(t,e,s,this.stride_,.5);i(this.flatMidpoints_,o),e=s}}return this.flatMidpoints_}getId(){return this.id_}getOrientedFlatCoordinates(){return this.flatCoordinates_}getGeometry(){return this}getSimplifiedGeometry(t){return this}simplifyTransformed(t,e){return this}getProperties(){return this.properties_}getPropertiesInternal(){return this.properties_}getStride(){return this.stride_}getStyleFunction(){return this.styleFunction}getType(){return this.type_}transform(t){const e=(t=rs(t)).getExtent(),i=t.getWorldExtent();if(e&&i){const t=Ce(i)/Ce(e);$e(Gs,i[0],i[3],t,-t,0,0,0),Ae(this.flatCoordinates_,0,this.flatCoordinates_.length,this.stride_,Gs,this.flatCoordinates_)}}applyTransform(t){t(this.flatCoordinates_,this.flatCoordinates_,this.stride_)}clone(){return new Ns(this.type_,this.flatCoordinates_.slice(),this.ends_?.slice(),this.stride_,Object.assign({},this.properties_),this.id_)}getEnds(){return this.ends_}enableSimplifyTransformed(){return this.simplifyTransformed=Je((t,e)=>{if(t===this.squaredTolerance_)return this.simplifiedGeometry_;this.simplifiedGeometry_=this.clone(),e&&this.simplifiedGeometry_.applyTransform(e);const i=this.simplifiedGeometry_.getFlatCoordinates();let n;switch(this.type_){case\"LineString\":i.length=Es(i,0,this.simplifiedGeometry_.flatCoordinates_.length,this.simplifiedGeometry_.stride_,t,i,0),n=[i.length];break;case\"MultiLineString\":n=[],i.length=function(t,e,i,n,r,s,o,a){for(let l=0,h=i.length;l<h;++l){const h=i[l];o=Es(t,e,h,n,r,s,o),a.push(o),e=h}return o}(i,0,this.simplifiedGeometry_.ends_,this.simplifiedGeometry_.stride_,t,i,0,n);break;case\"Polygon\":n=[],i.length=Ps(i,0,this.simplifiedGeometry_.ends_,this.simplifiedGeometry_.stride_,Math.sqrt(t),i,0,n)}return n&&(this.simplifiedGeometry_=new Ns(this.type_,i,n,this.stride_,this.properties_,this.id_)),this.squaredTolerance_=t,this.simplifiedGeometry_}),this}}function Xs(t){return Object.keys(t).reduce((e,i)=>e+(t[i].size||1),0)}Ns.prototype.getFlatCoordinates=Ns.prototype.getOrientedFlatCoordinates;const Ys={},Bs=new Ns(\"Point\",[0,0],[],2,Ys,\"dummy\"),Us=new TextDecoder;function js(t,e,i,n,r,s){const o=`prop_${t}`,a=i.findIndex(t=>t===o),l=i.slice(0,a).reduce((t,e)=>t+n[e].size,0),h=n[o].size;if(e===N){const t=r[l+1],e=r[l+2],i=s.slice(t,t+e);return Us.decode(i)}if(e===X){const t=(c=Array.from(r.slice(l,l+2)),[Math.min(Math.floor(c[0]/256)/255,1),Math.min(c[0]%256/255,1),Math.min(Math.floor(c[1]/256)/255,1),Math.min(c[1]%256/255,1)]);return t[0]*=255,t[1]*=255,t[2]*=255,t}var c;return h>1?Array.from(r.slice(l,l+h)):r[l]}function Vs(t,e,i,n,r){let s=t(i,1);if(s){s=Array.isArray(s)?s:[s];for(let t=0,o=s.length;t<o;t++){const o=s[t].getText();if(!o)continue;const a=o.getPlacement(),l=\"LineString\"===n.getType();\"line\"===a&&!l||\"line\"!==a&&l||(e.setTextStyle(o,r),e.drawText(n,i))}}}const qs=self;let Js=0;const Ks=new OffscreenCanvas(1,1),Hs=Ks.getContext(\"2d\"),Zs=new Map,Qs=Fe();function to(t,e,i,n,r,s){const o=n/2,a=r/2,l=1/e,h=-l,c=-t[0]+s,u=-t[1];return $e(Qs,o,a,l,h,-i,c,u)}qs.onmessage=t=>{const e=t.data;switch(e.type){case hr:{const t=(i=e.frameState,{...i,viewState:{...i.viewState,projection:rs(i.viewState.projection)}}),n=t.viewState,r=e.batchesToRender;Js&&cancelAnimationFrame(Js),Js=requestAnimationFrame(()=>{Js=0,t.size[0]!==Ks.width||t.size[1]!==Ks.height?(Ks.width=t.size[0],Ks.height=t.size[1]):Hs.clearRect(0,0,Ks.width,Ks.height);for(const i of r.values()){if(!Zs.has(i)){const t={type:hr,imageData:null,frameState:e.frameState,id:e.id};return void qs.postMessage(t)}const r=Zs.get(i);if(!r)continue;const s=to(n.center,n.resolution,n.rotation,Ks.width,Ks.height,0);Te(s,r.inverseTransform),r.executor.execute(Hs,t.size,s,t.viewState.rotation,!1)}const i=Ks.transferToImageBitmap(),s={type:hr,imageData:i,frameState:e.frameState,id:e.id};qs.postMessage(s,[i])});break}case ar:{const{polygonRenderInstructions:t,lineStringRenderInstructions:i,pointRenderInstructions:n,style:r,customAttributesSizes:s,renderInstructionsTransform:o,id:a,resolution:l}=e,h=1,c=l*o[0],u=Date.now().toString(),f=new Uint8Array(e.labelsArray),d=new un(1,[-1/0,-1/0,1/0,1/0],c,h),g=Object.keys(s).reduce((t,e)=>({...t,[e]:{size:s[e]}}),{}),p=tt();!function(t){function e(t){for(const e in t)e.startsWith(\"text-\")||\"z-index\"===e||delete t[e]}if(Array.isArray(t)){for(let i=0,n=t.length;i<n;i++){const n=t[i];if(\"style\"in n&&Array.isArray(n.style))for(let t=0,i=n.style.length;t<i;t++)e(n.style[t]);else e(\"style\"in n?n.style:n)}return t}e(t)}(r);const _=function(t,e){if(e=e??tt(),!Array.isArray(t))return Wn([t],e);const i=t.length;if(\"style\"in t[0]){const n=new Array(i);for(let e=0;e<i;++e){const i=t[e];if(!(\"style\"in i))throw new Error(\"Expected a list of rules with a style property\");n[e]=i}return $n(n,e)}return Wn(t,e)}(r,p);!function(t,e,i,n,r,s){const o=Object.keys(n),a=Xs(n),l={};let h=0;for(;h<t.length;){const c=new Float32Array(t.buffer,h*Float32Array.BYTES_PER_ELEMENT,a);h+=a;const u=t[h++];let f=0;const d=new Array(u);for(let e=0;e<u;e++)f+=t[h++],d[e]=2*f;const g=h+2*f,p=Array.from(new Float32Array(t.buffer,h*Float32Array.BYTES_PER_ELEMENT,2*f)),_=new Ws(p,\"XY\",d),m=Array.from(i.entries());for(let t=0;t<m.length;t++){const[i,r]=m[t];Ys[i]=js(i,r,o,n,c,e)}Vs(s,r,Bs,_,l),h=g}}(new Float32Array(t),f,p.properties,g,d,_),function(t,e,i,n,r,s){const o=Object.keys(n),a=Xs(n),l={};let h,c=0;for(;c<t.length;){const u=new Float32Array(t.buffer,c*Float32Array.BYTES_PER_ELEMENT,a);c+=a,h=t[c++];const f=Array.from(new Float32Array(t.buffer,c*Float32Array.BYTES_PER_ELEMENT,3*h)),d=new Os(f,\"XYM\"),g=Array.from(i.entries());for(let t=0;t<g.length;t++){const[i,r]=g[t];Ys[i]=js(i,r,o,n,u,e)}Vs(s,r,Bs,d,l),c+=3*h}}(new Float32Array(i),f,p.properties,g,d,_),function(t,e,i,n,r,s){const o=Object.keys(n),a=Xs(n),l={};let h=0;for(;h<t.length;){const c=[t.at(h),t.at(h+1)];h+=2;const u=new Float32Array(t.buffer,h*Float32Array.BYTES_PER_ELEMENT,a),f=new Rs(c,\"XY\"),d=Array.from(i.entries());for(let t=0;t<d.length;t++){const[i,r]=d[t];Ys[i]=js(i,r,o,n,u,e)}Vs(s,r,Bs,f,l),h+=a}}(new Float32Array(n),f,p.properties,g,d,_);const m=d.finish();if(0===m.instructions.length)Zs.set(u,null);else{const t=We(o),e=new Sn(c,h,!1,m);Zs.set(u,{inverseTransform:t,executor:e})}const y={type:ar,instructionsSetKey:u,id:a};qs.postMessage(y);break}case lr:{const{instructionsSetKey:t}=e;Zs.has(t)&&Zs.delete(t);break}}var i};";
	return new Worker(typeof Blob > "u" ? "data:application/javascript;base64," + Buffer.from(e, "binary").toString("base64") : URL.createObjectURL(new Blob([e], { type: "application/javascript" })));
}
//#endregion
//#region node_modules/ol/worker/webgl.js
function um() {
	let e = "const t=new Set;let e=!1;function n(e,n,i=2){const f=n&&n.length,u=f?n[0]*i:e.length;t.size&&t.clear();let s=r(e,0,u,i,!0);const v=[];if(!s||s.next===s.prev)return v;let b=0,A=0,m=0;if(f&&(s=function(e,n,x,i){const f=[];for(let o=0,x=n.length;o<x;o++){const u=r(e,n[o]*i,o<x-1?n[o+1]*i:e.length,i,!1);u===u.next&&t.add(u),f.push(B(u))}f.sort(l),function(t,e){const n=Math.ceil((t+2*e)/y)+e+2;h.length<4*n&&(h=new Float64Array(4*n));p=0}(e.length/i,n.length),M(x,x),c=!0;for(let t=0;t<f.length;t++)x=a(f[t],x);return c=!1,o(x)}(e,n,s,i)),e.length>80*i){b=e[0],A=e[1];let t=b,n=A;for(let r=i;r<u;r+=i){const o=e[r],x=e[r+1];o<b&&(b=o),x<A&&(A=x),o>t&&(t=o),x>n&&(n=x)}m=Math.max(t-b,n-A),m=0!==m?32767/m:0}return x(s,v,b,A,m),v}function r(t,e,n,r,o){let x=null;if(o===function(t,e,n,r){let o=0;for(let x=e,i=n-r;x<n;x+=r)o+=(t[i]-t[x])*(t[x+1]+t[i+1]),i=x;return o}(t,e,n,r)>0)for(let o=e;o<n;o+=r)x=G(o/r|0,t[o],t[o+1],x);else for(let o=n-r;o>=e;o-=r)x=G(o/r|0,t[o],t[o+1],x);return x&&N(x,x.next)&&(k(x),x=x.next),x}function o(n,r=n){const o=r===n;let x,i=n;do{x=!1,i===i.next||0!==t.size&&t.has(i)||!N(i,i.next)&&0!==S(i.prev,i,i.next)?(o||i!==r)&&(i=i.next,x=!o):((o||i===r)&&(r=i.prev),e=!0,k(i),i=i.prev,x=!0)}while(x||i!==r);return r}function x(t,n,r,x,c){c&&function(t,e,n,r){let o=t,x=0;do{o.z=I(o.x,o.y,e,n,r),w[x++]=o,o=o.next}while(o!==t);!function(t){if(t<=32){for(let e=1;e<t;e++){const t=w[e],n=t.z;let r=e-1;for(;r>=0&&w[r].z>n;)w[r+1]=w[r],r--;w[r+1]=t}return}F.length<t&&(F=new Uint32Array(t),Z=new Uint32Array(t),d=new Array(t));for(let e=0;e<t;e++)F[e]=w[e].z;E(t,w,F,d,Z,0),E(t,d,Z,w,F,8),E(t,w,F,d,Z,16),E(t,d,Z,w,F,24)}(x);let i=null;for(let t=0;t<x;t++){const e=w[t];e.prevZ=i,i&&(i.nextZ=e),i=e}i.nextZ=null}(t,r,x,c);let l=t,a=!1;for(;t.prev!==t.next;){const y=t.prev,h=t.next;if(S(y,t,h)<0&&(c?f(t,r,x,c):i(t)))n.push(y.i,t.i,h.i),k(t),t=h,l=h;else if((t=h)===l){if(e=!1,t=o(t),e){l=t;continue}if(!a){l=t=u(t,n),a=!0;continue}s(t,n,r,x,c);break}}}function i(t){const e=t.prev,n=t,r=t.next,o=e.x,x=n.x,i=r.x,f=e.y,u=n.y,s=r.y,c=Math.min(o,x,i),l=Math.min(f,u,s),a=Math.max(o,x,i),y=Math.max(f,u,s);let h=r.next;for(;h!==e;){if(h.x>=c&&h.x<=a&&h.y>=l&&h.y<=y&&(o!==h.x||f!==h.y)&&U(o,f,x,u,i,s,h.x,h.y)&&S(h.prev,h,h.next)>=0)return!1;h=h.next}return!0}function f(t,e,n,r){const o=t.prev,x=t,i=t.next,f=o.x,u=x.x,s=i.x,c=o.y,l=x.y,a=i.y,y=Math.min(f,u,s),h=Math.min(c,l,a),p=Math.max(f,u,s),v=Math.max(c,l,a),b=I(y,h,e,n,r),M=I(p,v,e,n,r);let A=t.prevZ;for(;A&&A.z>=b;){if(A.x>=y&&A.x<=p&&A.y>=h&&A.y<=v&&A!==i&&(f!==A.x||c!==A.y)&&U(f,c,u,l,s,a,A.x,A.y)&&S(A.prev,A,A.next)>=0)return!1;A=A.prevZ}let m=t.nextZ;for(;m&&m.z<=M;){if(m.x>=y&&m.x<=p&&m.y>=h&&m.y<=v&&m!==i&&(f!==m.x||c!==m.y)&&U(f,c,u,l,s,a,m.x,m.y)&&S(m.prev,m,m.next)>=0)return!1;m=m.nextZ}return!0}function u(t,e){let n=t,r=!1;do{const o=n.prev,x=n.next.next;R(o,n,n.next,x,!1)&&_(o,x)&&_(x,o)&&(e.push(o.i,n.i,x.i),k(n),k(n.next),n=t=x,r=!0),n=n.next}while(n!==t);return r?o(n):n}function s(t,e,n,r,i){let f=t;do{let t=f.next.next;for(;t!==f.prev;){if(f.i!==t.i&&P(f,t)){let u=O(f,t);return f=o(f,f.next),u=o(u,u.next),x(f,e,n,r,i),void x(u,e,n,r,i)}t=t.next}f=f.next}while(f!==t)}let c=!1;function l(t,e){return t.x-e.x||t.y-e.y||(t.next.y-t.y)/(t.next.x-t.x)-(e.next.y-e.y)/(e.next.x-e.x)}function a(t,e){const n=function(t,e){let n=e;const r=t.x,o=t.y;let x,i=-1/0;if(N(t,n))return n;for(let e=0,f=0;e<p;e++,f+=4){if(o<h[f+1]||o>h[f+3]||h[f]>r||h[f+2]<=i)continue;const u=A(e);n=m(e);do{if(n.prev.next===n){if(N(t,n.next))return n.next;if(o<=n.y&&o>=n.next.y&&n.next.y!==n.y){const t=n.x+(o-n.y)*(n.next.x-n.x)/(n.next.y-n.y);if(t<=r&&t>i&&(i=t,x=n.x<n.next.x?n:n.next,t===r))return x}}n=n.next}while(n!==u)}if(!x)return null;const f=x.x,u=x.y,s=Math.min(o,u),c=Math.max(o,u);let l=1/0;for(let e=0,a=0;e<p;e++,a+=4){if(h[a+2]<f||h[a]>r||h[a+3]<s||h[a+1]>c)continue;const y=A(e);n=m(e);do{if(n.prev.next===n&&r>=n.x&&n.x>=f&&r!==n.x&&U(o<u?r:i,o,f,u,o<u?i:r,o,n.x,n.y)){const e=Math.abs(o-n.y)/(r-n.x);(_(n,t)||n.y===o&&n.next.y===o&&n.next.x>r)&&(e<l||e===l&&(n.x>x.x||n.x===x.x&&g(x,n)))&&(x=n,l=e)}n=n.next}while(n!==y)}return x}(t,e);if(!n)return e;const r=O(n,t);return M(n,r.next.next),o(r,r.next),o(n,n.next)}const y=16;let h=new Float64Array(0),p=0;const v=[],b=[];function M(t,e){let n=t;do{const t=p++;v[t]=n;let r=1/0,o=1/0,x=-1/0,i=-1/0,f=0;do{const e=n.next;n.z=t,n.x<r&&(r=n.x),n.x>x&&(x=n.x),n.y<o&&(o=n.y),n.y>i&&(i=n.y),e.x<r&&(r=e.x),e.x>x&&(x=e.x),e.y<o&&(o=e.y),e.y>i&&(i=e.y),n=e}while(++f<y&&n!==e);b[t]=n;const u=4*t;h[u]=r,h[u+1]=o,h[u+2]=x,h[u+3]=i}while(n!==e)}function A(t){let e=b[t];for(;e.prev.next!==e;)e=e.next;return b[t]=e,e}function m(t){let e=v[t];for(;e.prev.next!==e;)e=e.next;return v[t]=e,e}function g(t,e){return S(t.prev,t,e.prev)<0&&S(e.next,t,t.next)<0}const w=[];let d=[],F=new Uint32Array(0),Z=new Uint32Array(0);const z=new Uint32Array(256);function E(t,e,n,r,o,x){z.fill(0);for(let e=0;e<t;e++)z[n[e]>>>x&255]++;let i=0;for(let t=0;t<256;t++){const e=z[t];z[t]=i,i+=e}for(let i=0;i<t;i++){const t=n[i],f=z[t>>>x&255]++;r[f]=e[i],o[f]=t}}function I(t,e,n,r,o){return(t=1431655765&((t=858993459&((t=252645135&((t=16711935&((t=(t-n)*o|0)|t<<8))|t<<4))|t<<2))|t<<1))|(e=1431655765&((e=858993459&((e=252645135&((e=16711935&((e=(e-r)*o|0)|e<<8))|e<<4))|e<<2))|e<<1))<<1}function B(t){let e=t,n=t;do{(e.x<n.x||e.x===n.x&&e.y<n.y)&&(n=e),e=e.next}while(e!==t);return n}function U(t,e,n,r,o,x,i,f){return(o-i)*(e-f)>=(t-i)*(x-f)&&(t-i)*(r-f)>=(n-i)*(e-f)&&(n-i)*(x-f)>=(o-i)*(r-f)}function P(t,e){const n=N(t,e)&&S(t.prev,t,t.next)>0&&S(e.prev,e,e.next)>0;return t.next.i!==e.i&&(n||_(t,e)&&_(e,t)&&(0!==S(t.prev,t,e.prev)||0!==S(t,e.prev,e)))&&!function(t,e){const n=Math.min(t.x,e.x),r=Math.max(t.x,e.x),o=Math.min(t.y,e.y),x=Math.max(t.y,e.y);let i=t;do{const f=i.next;if(i.x>r&&f.x>r||i.x<n&&f.x<n||i.y>x&&f.y>x||i.y<o&&f.y<o)i=f;else{if(i.i!==t.i&&f.i!==t.i&&i.i!==e.i&&f.i!==e.i&&R(i,f,t,e))return!0;i=f}}while(i!==t);return!1}(t,e)&&(n||function(t,e){let n=t,r=!1;const o=(t.x+e.x)/2,x=(t.y+e.y)/2;do{const t=n.next;n.y>x!=t.y>x&&o<(t.x-n.x)*(x-n.y)/(t.y-n.y)+n.x&&(r=!r),n=t}while(n!==t);return r}(t,e))}function S(t,e,n){return(e.y-t.y)*(n.x-e.x)-(e.x-t.x)*(n.y-e.y)}function N(t,e){return t.x===e.x&&t.y===e.y}function R(t,e,n,r,o=!0){const x=S(t,e,n),i=S(t,e,r),f=S(n,r,t),u=S(n,r,e);return(x>0&&i<0||x<0&&i>0)&&(f>0&&u<0||f<0&&u>0)||!!o&&(!(0!==x||!T(t,n,e))||(!(0!==i||!T(t,r,e))||(!(0!==f||!T(n,t,r))||!(0!==u||!T(n,e,r)))))}function T(t,e,n){return e.x<=Math.max(t.x,n.x)&&e.x>=Math.min(t.x,n.x)&&e.y<=Math.max(t.y,n.y)&&e.y>=Math.min(t.y,n.y)}function _(t,e){return S(t.prev,t,t.next)<0?S(t,e,t.next)>=0&&S(t,t.prev,e)>=0:S(t,e,t.prev)<0||S(t,t.next,e)<0}function O(t,e){const n=j(t.i,t.x,t.y),r=j(e.i,e.x,e.y),o=t.next,x=e.prev;return t.next=e,e.prev=t,n.next=o,o.prev=n,r.next=n,n.prev=r,x.next=r,r.prev=x,r}function G(t,e,n,r){const o=j(t,e,n);return r?(o.next=r.next,o.prev=r,r.next.prev=o,r.next=o):(o.prev=o,o.next=o),o}function k(t){t.next.prev=t.prev,t.prev.next=t.next,t.prevZ&&(t.prevZ.nextZ=t.nextZ),t.nextZ&&(t.nextZ.prevZ=t.prevZ),c&&function(t,e){const n=4*t.z;e.x<h[n]&&(h[n]=e.x),e.y<h[n+1]&&(h[n+1]=e.y),e.x>h[n+2]&&(h[n+2]=e.x),e.y>h[n+3]&&(h[n+3]=e.y)}(t.prev,t.next)}function j(t,e,n){return{i:t,x:e,y:n,prev:null,next:null,z:0,prevZ:null,nextZ:null}}function q(t,e,n){const r=Math.sqrt((e[0]-t[0])*(e[0]-t[0])+(e[1]-t[1])*(e[1]-t[1])),o=[(e[0]-t[0])/r,(e[1]-t[1])/r],x=[-o[1],o[0]],i=Math.sqrt((n[0]-t[0])*(n[0]-t[0])+(n[1]-t[1])*(n[1]-t[1])),f=[(n[0]-t[0])/i,(n[1]-t[1])/i];let u=0===r||0===i?0:Math.acos((s=f[0]*o[0]+f[1]*o[1],c=-1,l=1,Math.min(Math.max(s,c),l)));var s,c,l;u=Math.max(u,1e-5);return f[0]*x[0]+f[1]*x[1]>0?u:2*Math.PI-u}const L=[1,0,0,1,0,0];function Y(t,e){const n=e[0],r=e[1];return e[0]=t[0]*n+t[2]*r+t[4],e[1]=t[1]*n+t[3]*r+t[5],e}function C(t,e){const n=(r=e)[0]*r[3]-r[1]*r[2];var r;!function(t,e){if(!t)throw new Error(e)}(0!==n,\"Transformation matrix cannot be inverted\");const o=e[0],x=e[1],i=e[2],f=e[3],u=e[4],s=e[5];return t[0]=f/n,t[1]=-x/n,t[2]=-i/n,t[3]=o/n,t[4]=(i*s-f*u)/n,t[5]=-(o*s-x*u)/n,t}new Array(6);const D=[],H={vertexAttributesPosition:0,instanceAttributesPosition:0,indicesPosition:0};function J(t,e,n,r,o){const x=t[e++],i=t[e++],f=D;f.length=r;for(let n=0;n<f.length;n++)f[n]=t[e+n];let u=o?o.instanceAttributesPosition:0;return n[u++]=x,n[u++]=i,f.length&&(n.set(f,u),u+=f.length),H.instanceAttributesPosition=u,H}function K(t,e,n,r,o,x,i,f,u,s){const c=[t[e],t[e+1]],l=[t[n],t[n+1]],a=t[e+2],y=t[n+2],h=Y(f,[...c]),p=Y(f,[...l]);let v=-1,b=-1,M=s;const A=null!==o;if(null!==r){v=q(h,p,Y(f,[...[t[r],t[r+1]]])),Math.cos(v)<=.985&&(M+=Math.tan((v-Math.PI)/2))}if(A){b=q(p,h,Y(f,[...[t[o],t[o+1]]])),Math.cos(b)<=.985&&(M+=Math.tan((Math.PI-b)/2))}const m=Math.pow(2,24),g=u%m,w=Math.floor(u/m)*m;return x.push(c[0],c[1],a,l[0],l[1],y,v,b,g,w,s),x.push(...i),{length:u+Math.sqrt((p[0]-h[0])*(p[0]-h[0])+(p[1]-h[1])*(p[1]-h[1])),angle:M}}function Q(t,e,r,o,x){const i=2+x;let f=e;const u=t.slice(f,f+x);f+=x;const s=t[f++];let c=0;const l=new Array(s-1);for(let e=0;e<s;e++)c+=t[f++],e<s-1&&(l[e]=c);const a=t.slice(f,f+2*c),y=n(a,l,2);for(let t=0;t<y.length;t++)o.push(y[t]+r.length/i);for(let t=0;t<a.length;t+=2)r.push(a[t],a[t+1],...u);return f+2*c}const V=\"GENERATE_POLYGON_BUFFERS\",W=\"GENERATE_POINT_BUFFERS\",X=\"GENERATE_LINE_STRING_BUFFERS\",$=self;$.onmessage=t=>{const e=t.data;switch(e.type){case W:{const t=2,n=2,r=e.customAttributesSize,o=n+r,x=new Float32Array(e.renderInstructions),i=x.length/o*(t+r),f=Uint32Array.from([0,1,3,1,2,3]),u=Float32Array.from([-1,-1,1,-1,1,1,-1,1]),s=new Float32Array(i);let c;for(let t=0;t<x.length;t+=o)c=J(x,t,s,r,c);const l=Object.assign({indicesBuffer:f.buffer,vertexAttributesBuffer:u.buffer,instanceAttributesBuffer:s.buffer,renderInstructions:x.buffer},e);$.postMessage(l,[u.buffer,s.buffer,f.buffer,x.buffer]);break}case X:{const t=[],n=e.customAttributesSize,r=3,o=new Float32Array(e.renderInstructions);let x=0;const i=e.renderInstructionsTransform,f=L.slice(0);let u,s;for(C(f,i);x<o.length;){s=Array.from(o.slice(x,x+n)),x+=n,u=o[x++];const e=x,i=x+(u-1)*r,c=o[e]===o[i]&&o[e+1]===o[i+1];let l=0,a=0;for(let n=0;n<u-1;n++){let y=null;n>0?y=x+(n-1)*r:c&&(y=i-r);let h=null;n<u-2?h=x+(n+2)*r:c&&(h=e+r);const p=K(o,x+n*r,x+(n+1)*r,y,h,t,s,f,l,a);l=p.length,a=p.angle}x+=u*r}const c=Uint32Array.from([0,1,3,1,2,3]),l=Float32Array.from([-1,-1,1,-1,1,1,-1,1]),a=Float32Array.from(t),y=Object.assign({indicesBuffer:c.buffer,vertexAttributesBuffer:l.buffer,instanceAttributesBuffer:a.buffer,renderInstructions:o.buffer},e);$.postMessage(y,[l.buffer,a.buffer,c.buffer,o.buffer]);break}case V:{const t=[],n=[],r=e.customAttributesSize,o=new Float32Array(e.renderInstructions);let x=0;for(;x<o.length;)x=Q(o,x,t,n,r);const i=Uint32Array.from(n),f=Float32Array.from(t),u=Float32Array.from([]),s=Object.assign({indicesBuffer:i.buffer,vertexAttributesBuffer:f.buffer,instanceAttributesBuffer:u.buffer,renderInstructions:o.buffer},e);$.postMessage(s,[f.buffer,u.buffer,i.buffer,o.buffer]);break}}};";
	return new Worker(typeof Blob > "u" ? "data:application/javascript;base64," + Buffer.from(e, "binary").toString("base64") : URL.createObjectURL(new Blob([e], { type: "application/javascript" })));
}
//#endregion
//#region node_modules/ol/render/webgl/constants.js
var dm = {
	GENERATE_POLYGON_BUFFERS: "GENERATE_POLYGON_BUFFERS",
	GENERATE_POINT_BUFFERS: "GENERATE_POINT_BUFFERS",
	GENERATE_LINE_STRING_BUFFERS: "GENERATE_LINE_STRING_BUFFERS"
}, fm = {
	BUILD_INSTRUCTIONS: "BUILD_INSTRUCTIONS",
	DISPOSE_INSTRUCTIONS: "DISPOSE_INSTRUCTIONS",
	RENDER: "RENDER"
};
//#endregion
//#region node_modules/ol/render/webgl/encodeUtil.js
function pm(e, t) {
	t ||= [];
	let n = Math.floor(e / 256 / 256 / 256) / 255, r = Math.floor(e / 256 / 256) % 256 / 255, i = Math.floor(e / 256) % 256 / 255, a = e % 256 / 255;
	return t[0] = n * 256 * 255 + r * 255, t[1] = i * 256 * 255 + a * 255, t;
}
function mm(e) {
	let t = 0;
	return t += Math.round(e[0] * 256 * 256 * 256 * 255), t += Math.round(e[1] * 256 * 256 * 255), t += Math.round(e[2] * 256 * 255), t += Math.round(e[3] * 255), t;
}
//#endregion
//#region node_modules/ol/render/webgl/renderinstructions.js
function hm(e, t, n, r, i) {
	let a = 0;
	for (let o in n) {
		let s = n[o], c = s.callback.call(r, r.feature);
		if (typeof c == "string") {
			let [n, r] = t.push(c);
			e[i + a++] = Dp(c), e[i + a++] = n, e[i + a++] = r;
			continue;
		}
		let l = c?.[0] ?? c;
		l === -9999999 && console.warn("The \"has\" operator might return false positives."), l === void 0 ? l = Fp : l === null && (l = 0), e[i + a++] = l, !(!s.size || s.size === 1) && (e[i + a++] = c?.[1] ?? -9999999, !(s.size < 3) && (e[i + a++] = c?.[2] ?? -9999999, !(s.size < 4) && (e[i + a++] = c?.[3] ?? -9999999)));
	}
	return a;
}
function gm(e) {
	return Object.keys(e).reduce((t, n) => t + (e[n].size || 1), 0);
}
function _m(e, t, n, r, i) {
	let a = (2 + gm(r)) * e.geometriesCount;
	(!t || t.length !== a) && (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) o[0] = c.flatCoordss[e][0], o[1] = c.flatCoordss[e][1], z(i, o), t[s++] = o[0], t[s++] = o[1], s += hm(t, n, r, c, s);
	}
	return t;
}
function vm(e, t, n, r, i) {
	let a = 3 * e.verticesCount + (1 + gm(r)) * e.geometriesCount;
	(!t || t.length !== a) && (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) {
			o.length = c.flatCoordss[e].length, si(c.flatCoordss[e], 0, o.length, 3, i, o, 3), s += hm(t, n, r, c, s), t[s++] = o.length / 3;
			for (let e = 0, n = o.length; e < n; e += 3) t[s++] = o[e], t[s++] = o[e + 1], t[s++] = o[e + 2];
		}
	}
	return t;
}
function ym(e, t, n, r, i) {
	let a = 2 * e.verticesCount + (1 + gm(r)) * e.geometriesCount + e.ringsCount;
	(!t || t.length !== a) && (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) {
			o.length = c.flatCoordss[e].length, si(c.flatCoordss[e], 0, o.length, 2, i, o), s += hm(t, n, r, c, s), t[s++] = c.ringsVerticesCounts[e].length;
			for (let n = 0, r = c.ringsVerticesCounts[e].length; n < r; n++) t[s++] = c.ringsVerticesCounts[e][n];
			for (let e = 0, n = o.length; e < n; e += 2) t[s++] = o[e], t[s++] = o[e + 1];
		}
	}
	return t;
}
//#endregion
//#region node_modules/ol/render/webgl/serialize.js
function bm(e) {
	let t = e.viewState;
	return {
		viewState: {
			...t,
			projection: t.projection.getCode()
		},
		viewHints: e.viewHints,
		pixelRatio: e.pixelRatio,
		size: e.size,
		extent: e.extent,
		coordinateToPixelTransform: e.coordinateToPixelTransform,
		pixelToCoordinateTransform: e.pixelToCoordinateTransform,
		layerStatesArray: e.layerStatesArray.map((e) => ({
			zIndex: e.zIndex,
			visible: e.visible,
			extent: e.extent,
			maxResolution: e.maxResolution,
			minResolution: e.minResolution,
			managed: e.managed,
			opacity: e.opacity
		})),
		time: e.time,
		layerIndex: e.layerIndex
	};
}
//#endregion
//#region node_modules/ol/render/webgl/style.js
function xm(e) {
	return (JSON.stringify(e).split("").reduce((e, t) => (e << 5) - e + t.charCodeAt(0), 0) >>> 0).toString();
}
function Sm(e, t, n, r) {
	if (`${r}radius` in e && r !== "icon-") {
		let i = Z(n, e[`${r}radius`], V);
		if (`${r}radius2` in e) {
			let t = Z(n, e[`${r}radius2`], V);
			i = `max(${i}, ${t})`;
		}
		`${r}stroke-width` in e && (i = `(${i} + ${Z(n, e[`${r}stroke-width`], V)} * 0.5)`), t.setSymbolSizeExpression(`vec2(${i} * 2. + 0.5)`);
	}
	if (`${r}scale` in e) {
		let i = Z(n, e[`${r}scale`], ss);
		t.setSymbolSizeExpression(`${t.getSymbolSizeExpression()} * ${i}`);
	}
	`${r}displacement` in e && t.setSymbolOffsetExpression(Z(n, e[`${r}displacement`], os)), `${r}rotation` in e && t.setSymbolRotationExpression(Z(n, e[`${r}rotation`], V)), `${r}rotate-with-view` in e && t.setSymbolRotateWithView(!!e[`${r}rotate-with-view`]);
}
function Cm(e, t, n, r, i) {
	let a = "vec4(0.)";
	if (t !== null && (a = t), n !== null && r !== null) {
		let t = `smoothstep(-${r} + 0.63, -${r} - 0.58, ${e})`;
		a = `mix(${n}, ${a}, ${t})`;
	}
	let o = `(1.0 - smoothstep(-0.63, 0.58, ${e}))`, s = `${a} * vec4(1.0, 1.0, 1.0, ${o})`;
	return i !== null && (s = `${s} * vec4(1.0, 1.0, 1.0, ${i})`), s;
}
function wm(e, t, n, r, i) {
	let a = new Image();
	a.crossOrigin = e[`${r}cross-origin`] === void 0 ? "anonymous" : e[`${r}cross-origin`], R(typeof e[`${r}src`] == "string", `WebGL layers do not support expressions for the ${r}src style property`), a.src = e[`${r}src`], n[`u_texture${i}_size`] = () => a.complete ? [a.width, a.height] : [0, 0], t.addUniform(`u_texture${i}_size`, "vec2");
	let o = `u_texture${i}_size`;
	return n[`u_texture${i}`] = a, t.addUniform(`u_texture${i}`, "sampler2D"), o;
}
function Tm(e, t, n, r, i) {
	let a = Z(n, e[`${t}offset`], ss);
	if (`${t}offset-origin` in e) switch (e[`${t}offset-origin`]) {
		case "top-right":
			a = `vec2(${r}.x, 0.) + ${i} * vec2(-1., 0.) + ${a} * vec2(-1., 1.)`;
			break;
		case "bottom-left":
			a = `vec2(0., ${r}.y) + ${i} * vec2(0., -1.) + ${a} * vec2(1., -1.)`;
			break;
		case "bottom-right": a = `${r} - ${i} - ${a}`;
	}
	return a;
}
function Em(e, t, n, r) {
	r.functions.circleDistanceField = "float circleDistanceField(vec2 point, float radius) {\n  return length(point) - radius;\n}", Sm(e, t, r, "circle-");
	let i = null;
	"circle-opacity" in e && (i = Z(r, e["circle-opacity"], V));
	let a = "coordsPx";
	"circle-scale" in e && (a = `coordsPx / ${Z(r, e["circle-scale"], ss)}`);
	let o = null;
	"circle-fill-color" in e && (o = Z(r, e["circle-fill-color"], U));
	let s = null;
	"circle-stroke-color" in e && (s = Z(r, e["circle-stroke-color"], U));
	let c = Z(r, e["circle-radius"], V), l = null;
	"circle-stroke-width" in e && (l = Z(r, e["circle-stroke-width"], V), c = `(${c} + ${l} * 0.5)`);
	let u = Cm(`circleDistanceField(${a}, ${c})`, o, s, l, i);
	t.setSymbolColorExpression(u);
}
function Dm(e, t, n, r) {
	r.functions.round = "float round(float v) {\n  return sign(v) * floor(abs(v) + 0.5);\n}", r.functions.starDistanceField = "float starDistanceField(vec2 point, float numPoints, float radius, float radius2, float angle) {\n  float startAngle = -PI * 0.5 + angle; // tip starts upwards and rotates clockwise with angle\n  float c = cos(startAngle);\n  float s = sin(startAngle);\n  vec2 pointRotated = vec2(c * point.x - s * point.y, s * point.x + c * point.y);\n  float alpha = TWO_PI / numPoints; // the angle of one sector\n  float beta = atan(pointRotated.y, pointRotated.x);\n  float gamma = round(beta / alpha) * alpha; // angle in sector\n  c = cos(-gamma);\n  s = sin(-gamma);\n  vec2 inSector = vec2(c * pointRotated.x - s * pointRotated.y, abs(s * pointRotated.x + c * pointRotated.y));\n  vec2 tipToPoint = inSector + vec2(-radius, 0.);\n  vec2 edgeNormal = vec2(radius2 * sin(alpha * 0.5), -radius2 * cos(alpha * 0.5) + radius);\n  return dot(normalize(edgeNormal), tipToPoint);\n}", r.functions.regularDistanceField = "float regularDistanceField(vec2 point, float numPoints, float radius, float angle) {\n  float startAngle = -PI * 0.5 + angle; // tip starts upwards and rotates clockwise with angle\n  float c = cos(startAngle);\n  float s = sin(startAngle);\n  vec2 pointRotated = vec2(c * point.x - s * point.y, s * point.x + c * point.y);\n  float alpha = TWO_PI / numPoints; // the angle of one sector\n  float radiusIn = radius * cos(PI / numPoints);\n  float beta = atan(pointRotated.y, pointRotated.x);\n  float gamma = round((beta - alpha * 0.5) / alpha) * alpha + alpha * 0.5; // angle in sector from mid\n  c = cos(-gamma);\n  s = sin(-gamma);\n  vec2 inSector = vec2(c * pointRotated.x - s * pointRotated.y, abs(s * pointRotated.x + c * pointRotated.y));\n  return inSector.x - radiusIn;\n}", Sm(e, t, r, "shape-");
	let i = null;
	"shape-opacity" in e && (i = Z(r, e["shape-opacity"], V));
	let a = "coordsPx";
	"shape-scale" in e && (a = `coordsPx / ${Z(r, e["shape-scale"], ss)}`);
	let o = null;
	"shape-fill-color" in e && (o = Z(r, e["shape-fill-color"], U));
	let s = null;
	"shape-stroke-color" in e && (s = Z(r, e["shape-stroke-color"], U));
	let c = null;
	"shape-stroke-width" in e && (c = Z(r, e["shape-stroke-width"], V));
	let l = Z(r, e["shape-points"], V), u = "0.";
	"shape-angle" in e && (u = Z(r, e["shape-angle"], V));
	let d, f = Z(r, e["shape-radius"], V);
	if (c !== null && (f = `${f} + ${c} * 0.5`), "shape-radius2" in e) {
		let t = Z(r, e["shape-radius2"], V);
		c !== null && (t = `${t} + ${c} * 0.5`), d = `starDistanceField(${a}, ${l}, ${f}, ${t}, ${u})`;
	} else d = `regularDistanceField(${a}, ${l}, ${f}, ${u})`;
	let p = Cm(d, o, s, c, i);
	t.setSymbolColorExpression(p);
}
function Om(e, t, n, r) {
	let i = "vec4(1.0)";
	"icon-color" in e && (i = Z(r, e["icon-color"], U)), "icon-opacity" in e && (i = `${i} * vec4(1.0, 1.0, 1.0, ${Z(r, e["icon-opacity"], V)})`);
	let a = xm(e["icon-src"]), o = wm(e, t, n, "icon-", a);
	if (t.setSymbolColorExpression(`${i} * texture2D(u_texture${a}, v_texCoord)`).setSymbolSizeExpression(o), "icon-width" in e && "icon-height" in e && t.setSymbolSizeExpression(`vec2(${Z(r, e["icon-width"], V)}, ${Z(r, e["icon-height"], V)})`), "icon-offset" in e && "icon-size" in e) {
		let n = Z(r, e["icon-size"], os), i = t.getSymbolSizeExpression();
		t.setSymbolSizeExpression(n);
		let a = Tm(e, "icon-", r, "v_quadSizePx", n);
		t.setTextureCoordinateExpression(`(vec4((${a}).xyxy) + vec4(0., 0., ${n})) / (${i}).xyxy`);
	}
	if (Sm(e, t, r, "icon-"), "icon-anchor" in e) {
		let n = Z(r, e["icon-anchor"], os), i = "1.0";
		"icon-scale" in e && (i = Z(r, e["icon-scale"], ss));
		let a;
		a = e["icon-anchor-x-units"] === "pixels" && e["icon-anchor-y-units"] === "pixels" ? `${n} * ${i}` : e["icon-anchor-x-units"] === "pixels" ? `${n} * vec2(vec2(${i}).x, v_quadSizePx.y)` : e["icon-anchor-y-units"] === "pixels" ? `${n} * vec2(v_quadSizePx.x, vec2(${i}).x)` : `${n} * v_quadSizePx`;
		let o = `v_quadSizePx * vec2(0.5, -0.5) + ${a} * vec2(-1., 1.)`;
		if ("icon-anchor-origin" in e) switch (e["icon-anchor-origin"]) {
			case "top-right":
				o = `v_quadSizePx * -0.5 + ${a}`;
				break;
			case "bottom-left":
				o = `v_quadSizePx * 0.5 - ${a}`;
				break;
			case "bottom-right": o = `v_quadSizePx * vec2(-0.5, 0.5) + ${a} * vec2(1., -1.)`;
		}
		t.setSymbolOffsetExpression(`${t.getSymbolOffsetExpression()} + ${o}`);
	}
}
function km(e, t, n, r) {
	if ("stroke-color" in e && t.setStrokeColorExpression(Z(r, e["stroke-color"], U)), "stroke-pattern-src" in e) {
		let i = xm(e["stroke-pattern-src"]), a = wm(e, t, n, "stroke-pattern-", i), o = a, s = "vec2(0.)";
		"stroke-pattern-offset" in e && "stroke-pattern-size" in e && (o = Z(r, e["stroke-pattern-size"], os), s = Tm(e, "stroke-pattern-", r, a, o));
		let c = "0.";
		"stroke-pattern-spacing" in e && (c = Z(r, e["stroke-pattern-spacing"], V));
		let l = "0.";
		"stroke-pattern-start-offset" in e && (l = Z(r, e["stroke-pattern-start-offset"], V)), r.functions.sampleStrokePattern = "vec4 sampleStrokePattern(sampler2D texture, vec2 textureSize, vec2 textureOffset, vec2 sampleSize, float spacingPx, float startOffsetPx, float currentLengthPx, float currentRadiusRatio, float lineWidth) {\n  float currentLengthScaled = (currentLengthPx - startOffsetPx) * sampleSize.y / lineWidth;\n  float spacingScaled = spacingPx * sampleSize.y / lineWidth;\n  float uCoordPx = mod(currentLengthScaled, (sampleSize.x + spacingScaled));\n  float isInsideOfPattern = step(uCoordPx, sampleSize.x);\n  float vCoordPx = (-currentRadiusRatio * 0.5 + 0.5) * sampleSize.y;\n  // make sure that we're not sampling too close to the borders to avoid interpolation with outside pixels\n  uCoordPx = clamp(uCoordPx, 0.5, sampleSize.x - 0.5);\n  vCoordPx = clamp(vCoordPx, 0.5, sampleSize.y - 0.5);\n  vec2 texCoord = (vec2(uCoordPx, vCoordPx) + textureOffset) / textureSize;\n  return texture2D(texture, texCoord) * vec4(1.0, 1.0, 1.0, isInsideOfPattern);\n}";
		let u = `u_texture${i}`, d = "1.";
		"stroke-color" in e && (d = t.getStrokeColorExpression()), t.setStrokeColorExpression(`${d} * sampleStrokePattern(${u}, ${a}, ${s}, ${o}, ${c}, ${l}, currentLengthPx, currentRadiusRatio, v_width)`), r.functions.computeStrokePatternLength = "float computeStrokePatternLength(vec2 sampleSize, float spacingPx, float lineWidth) {\n  float patternLengthPx = sampleSize.x / sampleSize.y * lineWidth;\n  return patternLengthPx + spacingPx;\n}", t.setStrokePatternLengthExpression(`computeStrokePatternLength(${o}, ${c}, v_width)`);
	}
	if ("stroke-width" in e && t.setStrokeWidthExpression(Z(r, e["stroke-width"], V)), "stroke-offset" in e && t.setStrokeOffsetExpression(Z(r, e["stroke-offset"], V)), "stroke-line-cap" in e && t.setStrokeCapExpression(Z(r, e["stroke-line-cap"], H)), "stroke-line-join" in e && t.setStrokeJoinExpression(Z(r, e["stroke-line-join"], H)), "stroke-miter-limit" in e && t.setStrokeMiterLimitExpression(Z(r, e["stroke-miter-limit"], V)), "stroke-line-dash" in e) {
		r.functions.getSingleDashDistance = `float getSingleDashDistance(float distance, float radius, float dashOffset, float dashLength, float dashLengthTotal, float capType, float lineWidth) {
  float localDistance = mod(distance, dashLengthTotal);
  float distanceSegment = abs(localDistance - dashOffset - dashLength * 0.5) - dashLength * 0.5;
  distanceSegment = min(distanceSegment, dashLengthTotal - localDistance);
  if (capType == ${Op("square")}) {
    distanceSegment -= lineWidth * 0.5;
  } else if (capType == ${Op("round")}) {
    distanceSegment = min(distanceSegment, sqrt(distanceSegment * distanceSegment + radius * radius) - lineWidth * 0.5);
  }
  return distanceSegment;
}`;
		let n = e["stroke-line-dash"].map((e) => Z(r, e, V));
		n.length % 2 == 1 && (n = [...n, ...n]);
		let i = "0.";
		"stroke-line-dash-offset" in e && (i = Z(r, e["stroke-line-dash-offset"], V));
		let a = `dashDistanceField_${xm(e["stroke-line-dash"])}`, o = n.map((e, t) => `float dashLength${t}`).join(", "), s = n.map((e, t) => `dashLength${t}`).join(" + "), c = "0.", l = `getSingleDashDistance(distance, radius, ${c}, dashLength0, totalDashLength, capType, lineWidth)`;
		for (let e = 2; e < n.length; e += 2) c = `${c} + dashLength${e - 2} + dashLength${e - 1}`, l = `min(${l}, getSingleDashDistance(distance, radius, ${c}, dashLength${e}, totalDashLength, capType, lineWidth))`;
		r.functions[a] = `float ${a}(float distance, float radius, float capType, float lineWidth, ${o}) {
  float totalDashLength = ${s};
  return ${l};
}`;
		let u = n.map((e, t) => `${e}`).join(", ");
		t.setStrokeDistanceFieldExpression(`${a}(currentLengthPx + ${i}, currentRadiusPx, capType, v_width, ${u})`);
		let d = n.join(" + ");
		t.getStrokePatternLengthExpression() && (r.functions.combinePatternLengths = "float combinePatternLengths(float patternLength1, float patternLength2) {\n  return patternLength1 * patternLength2;\n}", d = `combinePatternLengths(${t.getStrokePatternLengthExpression()}, ${d})`), t.setStrokePatternLengthExpression(d);
	}
}
function Am(e, t, n, r) {
	if ("fill-color" in e && t.setFillColorExpression(Z(r, e["fill-color"], U)), "fill-pattern-src" in e) {
		let i = xm(e["fill-pattern-src"]), a = wm(e, t, n, "fill-pattern-", i);
		t.setFillPatternSizeExpression(a);
		let o = "vec2(0.)";
		if ("fill-pattern-offset" in e && "fill-pattern-size" in e) {
			let n = Z(r, e["fill-pattern-size"], os);
			t.setFillPatternSizeExpression(n), o = Tm(e, "fill-pattern-", r, a, "v_patternSizePx");
		}
		r.functions.sampleFillPattern = "vec4 sampleFillPattern(sampler2D texture, vec2 textureSize, vec2 textureOffset, vec2 sampleSize, vec2 patternOriginPx, vec2 pxPosition, float sampleScaleRatio) {\n  vec2 pxRelativePos = pxPosition - patternOriginPx;\n\n  // rotate the relative position from origin by the current view rotation\n  pxRelativePos = vec2(pxRelativePos.x * cos(u_rotation) - pxRelativePos.y * sin(u_rotation), pxRelativePos.x * sin(u_rotation) + pxRelativePos.y * cos(u_rotation));\n  // sample position is computed according to the sample offset & size\n  vec2 samplePos = mod(pxRelativePos / sampleScaleRatio, sampleSize);\n  // also make sure that we're not sampling too close to the borders to avoid interpolation with outside pixels\n  samplePos = clamp(samplePos, vec2(0.5), sampleSize - vec2(0.5));\n  samplePos.y = sampleSize.y - samplePos.y; // invert y axis so that images appear upright\n  return texture2D(texture, (samplePos + textureOffset) / textureSize);\n}";
		let s = `u_texture${i}`, c = "1.";
		"fill-color" in e && (c = t.getFillColorExpression()), t.setFillColorExpression(`${c} * sampleFillPattern(${s}, ${a}, ${o}, v_patternSizePx, v_patternOriginPx, pxPos, df_float(u_df_patternScaleRatio))`);
	}
}
function jm(e, t, n, r) {
	function i(...e) {
		try {
			Z(...e);
		} catch {}
	}
	"text-value" in e && i(r, e["text-value"], H), "text-font" in e && i(r, e["text-font"], H), "text-max-angle" in e && i(r, e["text-max-angle"], V), "text-offset-x" in e && i(r, e["text-offset-x"], V), "text-offset-y" in e && i(r, e["text-offset-y"], V), "text-overflow" in e && i(r, e["text-overflow"], as), "text-placement" in e && i(r, e["text-placement"], H), "text-repeat" in e && i(r, e["text-repeat"], V), "text-scale" in e && i(r, e["text-scale"], ss), "text-rotate-with-view" in e && i(r, e["text-rotate-with-view"], as), "text-rotation" in e && i(r, e["text-rotation"], V), "text-align" in e && i(r, e["text-align"], H), "text-justify" in e && i(r, e["text-justify"], H), "text-baseline" in e && i(r, e["text-baseline"], H), "text-padding" in e && i(r, e["text-padding"], os), "text-fill-color" in e && i(r, e["text-fill-color"], U), "text-stroke-color" in e && i(r, e["text-stroke-color"], U), "text-stroke-line-cap" in e && i(r, e["text-stroke-line-cap"], H), "text-stroke-line-join" in e && i(r, e["text-stroke-line-join"], H), "text-stroke-line-dash" in e && i(r, e["text-stroke-line-dash"], os), "text-stroke-line-dash-offset" in e && i(r, e["text-stroke-line-dash-offset"], V), "text-stroke-miter-limit" in e && i(r, e["text-stroke-miter-limit"], V), "text-stroke-width" in e && i(r, e["text-stroke-width"], V), "text-background-fill-color" in e && i(r, e["text-background-fill-color"], U), "text-background-stroke-color" in e && i(r, e["text-background-stroke-color"], U), "text-background-stroke-line-cap" in e && i(r, e["text-background-stroke-line-cap"], H), "text-background-stroke-line-join" in e && i(r, e["text-background-stroke-line-join"], H), "text-background-stroke-line-dash" in e && i(r, e["text-background-stroke-line-dash"], os), "text-background-stroke-line-dash-offset" in e && i(r, e["text-background-stroke-line-dash-offset"], V), "text-background-stroke-miter-limit" in e && i(r, e["text-background-stroke-miter-limit"], V), "text-background-stroke-width" in e && i(r, e["text-background-stroke-width"], V), "z-index" in e && i(r, e["z-index"], V);
}
function Mm(e, t, n) {
	let r = Ap(t), i = new Qp(), a = {};
	if ("icon-src" in e ? Om(e, i, a, r) : "shape-points" in e ? Dm(e, i, a, r) : "circle-radius" in e && Em(e, i, a, r), km(e, i, a, r), Am(e, i, a, r), jm(e, i, a, r), n) {
		let e = vs(t), a = Z(r, n, as, e);
		e.mCoordinate ? i.setFragmentDiscardExpression(`!${a}`) : i.setShapeDiscardExpression(`!${a}`);
	}
	let o = {};
	function s(e, t, n, a) {
		if (!r[e]) return;
		let s = Wp(n), c = Up(n);
		i.addAttribute(`a_${t}`, s), o[t] = {
			size: c,
			callback: a
		};
	}
	return s("geometryType", Pp, H, (e) => Dp(Rs(e.getGeometry()))), s("featureId", Np, H | V, (e) => {
		let t = e.getId() ?? null;
		return typeof t == "string" ? Dp(t) : t;
	}), Gp(i, r), {
		builder: i,
		attributes: {
			...o,
			...qp(r)
		},
		uniforms: {
			...a,
			...Kp(r, t)
		}
	};
}
//#endregion
//#region node_modules/ol/render/webgl/textUtil.js
var Nm = {
	TEXT_OVERLAY_TEXTURE: "u_textOverlay",
	TEXT_OVERLAY_MATRIX: "u_textOverlayMatrix"
};
function Pm(e) {
	let t = !1;
	function n(e) {
		for (let n in e) if (n === "text-value") {
			t = !0;
			return;
		}
	}
	if (Array.isArray(e)) {
		for (let r = 0, i = e.length; r < i; r++) {
			let i = e[r];
			if ("style" in i && Array.isArray(i.style)) for (let e = 0, t = i.style.length; e < t; e++) n(i.style[e]);
			else "style" in i ? n(i.style) : n(i);
			if (t) return t;
		}
		return t;
	}
	return n(e), t;
}
function Fm(e, t) {
	let n = If();
	return {
		fragmentShader: `
      precision mediump float;

      uniform sampler2D u_image;
      uniform sampler2D ${Nm.TEXT_OVERLAY_TEXTURE};
      uniform mat4 ${Nm.TEXT_OVERLAY_MATRIX};

      varying vec2 v_texCoord;

      void main() {
        vec4 color = texture2D(u_image, v_texCoord);

        vec2 coords = v_texCoord * 2. - vec2(1.);
        coords = (${Nm.TEXT_OVERLAY_MATRIX} * vec4(coords.xy, 0., 1.)).xy;
        coords = coords * 0.5 + vec2(0.5);
        float outOfBounds = clamp(step(1., coords.x) + step(1., coords.y) + step(0., -coords.x) + step(0., -coords.y), 0., 1.);

        vec4 textColor = texture2D(${Nm.TEXT_OVERLAY_TEXTURE}, vec2(coords.x, 1. - coords.y));
        textColor.a *= 1. - outOfBounds; // if we're sampling out of the text overlay, make alpha 0 to avoid drawing anything

        gl_FragColor = textColor.a * textColor + (1. - textColor.a) * color;
      }`,
		uniforms: {
			[Nm.TEXT_OVERLAY_TEXTURE]: e,
			[Nm.TEXT_OVERLAY_MATRIX]: (r) => {
				let i = e(), a = t();
				if (!i || !a) return n;
				let o = a.viewState, s = r.viewState, c = s.center, l = s.resolution, u = s.rotation, d = r.size, f = o.center, p = o.resolution, m = o.rotation, h = i.width, g = i.height;
				return Lf(n), zf(n, 1 / p / (h / 2), 1 / p / (g / 2), 1, n), Vf(n, m, n), Bf(n, c[0] - f[0], c[1] - f[1], 0, n), Vf(n, -u, n), zf(n, l * d[0] / 2, l * d[1] / 2, 1, n), n;
			}
		}
	};
}
new nm("Point", [0, 0], [], 2, {}, "dummy"), new TextDecoder();
//#endregion
//#region node_modules/ol/render/webgl/VectorStyleRenderer.js
var Im = [], Lm;
function Rm() {
	return Lm ||= um(), Lm;
}
var zm = 0;
function Bm(e, t, n) {
	let r = zm++;
	return n ? e.postMessage({
		...t,
		id: r
	}, n) : e.postMessage({
		...t,
		id: r
	}), new Promise((t) => {
		let n = (i) => {
			let a = i.data;
			a.id === r && (e.removeEventListener("message", n), t(a));
		};
		e.addEventListener("message", n);
	});
}
var Vm = {
	POSITION: "a_position",
	LOCAL_POSITION: "a_localPosition",
	SEGMENT_START: "a_segmentStart",
	SEGMENT_END: "a_segmentEnd",
	MEASURE_START: "a_measureStart",
	MEASURE_END: "a_measureEnd",
	ANGLE_TANGENT_SUM: "a_angleTangentSum",
	JOIN_ANGLES: "a_joinAngles",
	DISTANCE_LOW: "a_distanceLow",
	DISTANCE_HIGH: "a_distanceHigh"
}, Hm = class extends ie {
	constructor(e, t, n, r) {
		super(), this.helper_, this.hitDetectionEnabled_ = !!r, this.flatStyle = Um(e), this.styleShaders = Wm(e, t), this.customAttributes_ = {}, this.uniforms_ = {}, this.hitDetectionEnabled_ && (this.customAttributes_.hitColor = {
			callback() {
				return pm(this.ref, Im);
			},
			size: 2
		});
		for (let e of this.styleShaders) {
			for (let t in e.attributes) t in this.customAttributes_ || (this.customAttributes_[t] = e.attributes[t]);
			for (let t in e.uniforms) t in this.uniforms_ || (this.uniforms_[t] = e.uniforms[t]);
		}
		this.renderPasses_ = this.styleShaders.map((e) => {
			let t = {}, n = Object.entries(this.customAttributes_).map(([t, n]) => ({
				name: t in e.attributes || t === "hitColor" ? `a_${t}` : null,
				size: n.size || 1,
				type: sp.FLOAT
			}));
			return e.builder.getFillVertexShader() && (t.fillRenderPass = {
				vertexShader: e.builder.getFillVertexShader(),
				fragmentShader: e.builder.getFillFragmentShader(),
				attributesDesc: [{
					name: Vm.POSITION,
					size: 2,
					type: sp.FLOAT
				}, ...n],
				instancedAttributesDesc: [],
				instancePrimitiveVertexCount: 3
			}), e.builder.getStrokeVertexShader() && (t.strokeRenderPass = {
				vertexShader: e.builder.getStrokeVertexShader(),
				fragmentShader: e.builder.getStrokeFragmentShader(),
				attributesDesc: [{
					name: Vm.LOCAL_POSITION,
					size: 2,
					type: sp.FLOAT
				}],
				instancedAttributesDesc: [
					{
						name: Vm.SEGMENT_START,
						size: 2,
						type: sp.FLOAT
					},
					{
						name: Vm.MEASURE_START,
						size: 1,
						type: sp.FLOAT
					},
					{
						name: Vm.SEGMENT_END,
						size: 2,
						type: sp.FLOAT
					},
					{
						name: Vm.MEASURE_END,
						size: 1,
						type: sp.FLOAT
					},
					{
						name: Vm.JOIN_ANGLES,
						size: 2,
						type: sp.FLOAT
					},
					{
						name: Vm.DISTANCE_LOW,
						size: 1,
						type: sp.FLOAT
					},
					{
						name: Vm.DISTANCE_HIGH,
						size: 1,
						type: sp.FLOAT
					},
					{
						name: Vm.ANGLE_TANGENT_SUM,
						size: 1,
						type: sp.FLOAT
					},
					...n
				],
				instancePrimitiveVertexCount: 6
			}), e.builder.getSymbolVertexShader() && (t.symbolRenderPass = {
				vertexShader: e.builder.getSymbolVertexShader(),
				fragmentShader: e.builder.getSymbolFragmentShader(),
				attributesDesc: [{
					name: Vm.LOCAL_POSITION,
					size: 2,
					type: sp.FLOAT
				}],
				instancedAttributesDesc: [{
					name: Vm.POSITION,
					size: 2,
					type: sp.FLOAT
				}, ...n],
				instancePrimitiveVertexCount: 6
			}), t;
		}), this.hasFill_ = this.renderPasses_.some((e) => e.fillRenderPass), this.hasStroke_ = this.renderPasses_.some((e) => e.strokeRenderPass), this.hasSymbol_ = this.renderPasses_.some((e) => e.symbolRenderPass), this.hasText_ = this.flatStyle && Pm(this.flatStyle), this.hasText_ && (this.textOverlayCanvas_ = y().canvas, this.textOverlayContext_ = this.textOverlayCanvas_.getContext("2d"), this.textOverlayRenderFrameState_ = null, this.textOverlayWorker_ = lm(), this.textOverlayRenderList_ = /* @__PURE__ */ new Set()), this.setHelper(n);
	}
	async generateBuffers(e, t, n) {
		let r = ti(qr(), t);
		if (e.isEmpty()) return {
			polygonBuffers: null,
			lineStringBuffers: null,
			pointBuffers: null,
			invertVerticesTransform: r,
			textInstructionsKey: null
		};
		let i = new cm(), a = this.generateRenderInstructions_(e, i, t), [o, s, c, l] = await Promise.all([
			this.hasText_ ? this.generateTextInstructions_(a, i, t, n) : null,
			this.hasFill_ ? this.generateBuffersForType_(a.polygonInstructions, "Polygon", t) : null,
			this.hasStroke_ ? this.generateBuffersForType_(a.lineStringInstructions, "LineString", t) : null,
			this.hasSymbol_ ? this.generateBuffersForType_(a.pointInstructions, "Point", t) : null
		]);
		return {
			polygonBuffers: s,
			lineStringBuffers: c,
			pointBuffers: l,
			invertVerticesTransform: r,
			textInstructionsKey: o
		};
	}
	generateRenderInstructions_(e, t, n) {
		return {
			polygonInstructions: this.hasFill_ || this.hasText_ ? ym(e.polygonBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null,
			lineStringInstructions: this.hasStroke_ || this.hasText_ ? vm(e.lineStringBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null,
			pointInstructions: this.hasSymbol_ || this.hasText_ ? _m(e.pointBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null
		};
	}
	generateBuffersForType_(e, t, n) {
		if (e === null) return null;
		let r;
		switch (t) {
			case "Polygon":
				r = dm.GENERATE_POLYGON_BUFFERS;
				break;
			case "LineString":
				r = dm.GENERATE_LINE_STRING_BUFFERS;
				break;
			case "Point": r = dm.GENERATE_POINT_BUFFERS;
		}
		let i = {
			type: r,
			renderInstructions: e.buffer,
			renderInstructionsTransform: n,
			customAttributesSize: gm(this.customAttributes_)
		};
		return Bm(Rm(), i, [e.buffer]).then((e) => {
			if (!this.helper_.getGL()) return;
			let t = e, n = new ep(Uf, Kf).fromArrayBuffer(t.indicesBuffer), r = new ep(Hf, Kf).fromArrayBuffer(t.vertexAttributesBuffer), i = new ep(Hf, Kf).fromArrayBuffer(t.instanceAttributesBuffer);
			return this.helper_.flushBufferData(n), this.helper_.flushBufferData(r), this.helper_.flushBufferData(i), [
				n,
				r,
				i
			];
		});
	}
	generateTextInstructions_(e, t, n, r) {
		let i = [t.getArray().buffer], a = null, o = null, s = null;
		e.polygonInstructions && (a = new Float32Array(e.polygonInstructions).buffer, i.push(a)), e.lineStringInstructions && (o = new Float32Array(e.lineStringInstructions).buffer, i.push(o)), e.pointInstructions && (s = new Float32Array(e.pointInstructions).buffer, i.push(s));
		let c = Object.keys(this.customAttributes_).reduce((e, t) => ({
			...e,
			[t]: this.customAttributes_[t].size || 1
		}), {}), l = {
			type: fm.BUILD_INSTRUCTIONS,
			polygonRenderInstructions: a,
			lineStringRenderInstructions: o,
			pointRenderInstructions: s,
			labelsArray: t.getArray(),
			style: this.flatStyle,
			customAttributesSizes: c,
			renderInstructionsTransform: n,
			resolution: r
		};
		return Bm(this.textOverlayWorker_, l, i).then((e) => e.instructionsSetKey);
	}
	render(e, t, n) {
		for (let r of this.renderPasses_) r.fillRenderPass && e.polygonBuffers && this.renderInternal_(e.polygonBuffers[0], e.polygonBuffers[1], e.polygonBuffers[2], r.fillRenderPass, t, n), r.strokeRenderPass && e.lineStringBuffers && this.renderInternal_(e.lineStringBuffers[0], e.lineStringBuffers[1], e.lineStringBuffers[2], r.strokeRenderPass, t, n), r.symbolRenderPass && e.pointBuffers && this.renderInternal_(e.pointBuffers[0], e.pointBuffers[1], e.pointBuffers[2], r.symbolRenderPass, t, n);
		e.textInstructionsKey && this.renderText_(e);
	}
	renderInternal_(e, t, n, r, i, a) {
		let o = e.getSize();
		if (o === 0) return;
		let s = r.instancedAttributesDesc.length;
		if (this.helper_.useProgram(r.program, i), this.helper_.bindBuffer(t), this.helper_.bindBuffer(e), this.helper_.enableAttributes(r.attributesDesc), this.helper_.bindBuffer(n), this.helper_.enableAttributesInstanced(r.instancedAttributesDesc), a(), s) {
			let e = r.instancedAttributesDesc.reduce((e, t) => e + (t.size || 1), 0), t = n.getSize() / e;
			this.helper_.drawElementsInstanced(0, o, t);
		} else this.helper_.drawElements(0, o);
	}
	renderText_(e) {
		this.textOverlayRenderList_.add(e.textInstructionsKey);
	}
	finalizeTextRender(e) {
		if (!this.hasText_) return Promise.resolve();
		let t = {
			type: fm.RENDER,
			frameState: bm(e),
			batchesToRender: this.textOverlayRenderList_
		};
		return Bm(this.textOverlayWorker_, t).then((e) => {
			let t = e;
			if (t.imageData) {
				this.textOverlayRenderFrameState_ = t.frameState;
				let e = t.imageData;
				e.width !== this.textOverlayCanvas_.width || e.height !== this.textOverlayCanvas_.height ? (this.textOverlayCanvas_.width = e.width, this.textOverlayCanvas_.height = e.height) : this.textOverlayContext_.clearRect(0, 0, this.textOverlayCanvas_.width, this.textOverlayCanvas_.height), this.textOverlayContext_.drawImage(e, 0, 0), e.close();
			}
			this.textOverlayRenderList_.clear();
		});
	}
	setHelper(e, t = null) {
		this.helper_ = e;
		for (let e of this.renderPasses_) e.fillRenderPass && (e.fillRenderPass.program = this.helper_.getProgram(e.fillRenderPass.fragmentShader, e.fillRenderPass.vertexShader)), e.strokeRenderPass && (e.strokeRenderPass.program = this.helper_.getProgram(e.strokeRenderPass.fragmentShader, e.strokeRenderPass.vertexShader)), e.symbolRenderPass && (e.symbolRenderPass.program = this.helper_.getProgram(e.symbolRenderPass.fragmentShader, e.symbolRenderPass.vertexShader));
		this.helper_.addUniforms(this.uniforms_), t && (t.polygonBuffers && (this.helper_.flushBufferData(t.polygonBuffers[0]), this.helper_.flushBufferData(t.polygonBuffers[1]), this.helper_.flushBufferData(t.polygonBuffers[2])), t.lineStringBuffers && (this.helper_.flushBufferData(t.lineStringBuffers[0]), this.helper_.flushBufferData(t.lineStringBuffers[1]), this.helper_.flushBufferData(t.lineStringBuffers[2])), t.pointBuffers && (this.helper_.flushBufferData(t.pointBuffers[0]), this.helper_.flushBufferData(t.pointBuffers[1]), this.helper_.flushBufferData(t.pointBuffers[2])));
	}
	getTextOverlayCanvas() {
		return this.textOverlayCanvas_;
	}
	getTextOverlayFrameState() {
		return this.textOverlayRenderFrameState_;
	}
	disposeTextInstructions(e) {
		this.textOverlayWorker_?.postMessage({
			type: fm.DISPOSE_INSTRUCTIONS,
			instructionsSetKey: e
		});
	}
	disposeInternal() {
		this.textOverlayWorker_?.terminate(), super.disposeInternal();
	}
};
function Um(e) {
	return Array.isArray(e) ? e.some((e) => "builder" in e && !("sourceRule" in e)) ? null : e.some((e) => "builder" in e) ? e.map((e) => e.sourceRule) : e : "builder" in e ? "sourceRule" in e ? [e.sourceRule] : null : e;
}
function Wm(e, t) {
	let n = Array.isArray(e) ? e : [e];
	if ("style" in n[0]) {
		let e = [], r = n, i = [];
		for (let n of r) {
			let r = Array.isArray(n.style) ? n.style : [n.style], a = n.filter;
			n.else && i.length && (a = ["all", ...i.map((e) => ["!", e])], n.filter && a.push(n.filter), a.length < 3 && (a = a[1])), n.filter && i.push(n.filter);
			let o = r.map((e) => ({
				...Mm(e, t, a),
				sourceRule: n
			}));
			e.push(...o);
		}
		return e;
	}
	return "builder" in n[0] ? n : n.map((e) => ({
		...Mm(e, t, null),
		sourceRule: { style: e }
	}));
}
//#endregion
//#region node_modules/ol/source/VectorEventType.js
var Gm = {
	ADDFEATURE: "addfeature",
	CHANGEFEATURE: "changefeature",
	CLEAR: "clear",
	REMOVEFEATURE: "removefeature",
	FEATURESLOADSTART: "featuresloadstart",
	FEATURESLOADEND: "featuresloadend",
	FEATURESLOADERROR: "featuresloaderror"
}, Km = /* @__PURE__ */ new Uint8Array(4), qm = class {
	constructor(e, t) {
		this.helper_ = e;
		let n = e.getGL();
		this.texture_ = n.createTexture(), this.framebuffer_ = n.createFramebuffer(), this.depthbuffer_ = n.createRenderbuffer(), this.size_ = t || [1, 1], this.data_ = /* @__PURE__ */ new Uint8Array(), this.dataCacheDirty_ = !0, this.updateSize_();
	}
	setSize(e) {
		de(e, this.size_) || (this.size_[0] = e[0], this.size_[1] = e[1], this.updateSize_());
	}
	getSize() {
		return this.size_;
	}
	clearCachedData() {
		this.dataCacheDirty_ = !0;
	}
	readAll() {
		if (this.dataCacheDirty_) {
			let e = this.size_, t = this.helper_.getGL();
			t.bindFramebuffer(t.FRAMEBUFFER, this.framebuffer_), t.readPixels(0, 0, e[0], e[1], t.RGBA, t.UNSIGNED_BYTE, this.data_), this.dataCacheDirty_ = !1;
		}
		return this.data_;
	}
	readPixel(e, t) {
		if (e < 0 || t < 0 || e > this.size_[0] || t >= this.size_[1]) return Km[0] = 0, Km[1] = 0, Km[2] = 0, Km[3] = 0, Km;
		this.readAll();
		let n = Math.floor(e) + (this.size_[1] - Math.floor(t) - 1) * this.size_[0];
		return Km[0] = this.data_[n * 4], Km[1] = this.data_[n * 4 + 1], Km[2] = this.data_[n * 4 + 2], Km[3] = this.data_[n * 4 + 3], Km;
	}
	getTexture() {
		return this.texture_;
	}
	getFramebuffer() {
		return this.framebuffer_;
	}
	getDepthbuffer() {
		return this.depthbuffer_;
	}
	updateSize_() {
		let e = this.size_, t = this.helper_.getGL();
		this.texture_ = this.helper_.createTexture(e, null, this.texture_), t.bindFramebuffer(t.FRAMEBUFFER, this.framebuffer_), t.viewport(0, 0, e[0], e[1]), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, this.texture_, 0), t.bindRenderbuffer(t.RENDERBUFFER, this.depthbuffer_), t.renderbufferStorage(t.RENDERBUFFER, t.DEPTH_COMPONENT16, e[0], e[1]), t.framebufferRenderbuffer(t.FRAMEBUFFER, t.DEPTH_ATTACHMENT, t.RENDERBUFFER, this.depthbuffer_), this.data_ = new Uint8Array(e[0] * e[1] * 4);
	}
}, Jm = {
	PATTERN_ORIGIN_X_DOUBLE: "u_df_patternOriginX",
	PATTERN_ORIGIN_Y_DOUBLE: "u_df_patternOriginY",
	PATTERN_SCALE_RATIO_DOUBLE: "u_df_patternScaleRatio",
	ONE: "u_one"
}, Ym = [0, 0], Xm = [0, 0], Zm = qr(), Qm = If();
function $m(e, t, n, r) {
	Zr(Zm, t), Yr(Zm, n), e.setUniformMatrixValue(op.PROJECTION_MATRIX, Rf(Qm, Zm)), ti(Zm, Zm), e.setUniformMatrixValue(op.INVERT_PROJECTION_MATRIX, Rf(Qm, Zm)), Ym[0] = 0, Ym[1] = 0;
	let i = r.size, a = r.viewState.resolution, o = r.viewState.center;
	ei(Zm, i[0] / 2, i[1] / 2, 1 / a, 1 / a, 0, -o[0], -o[1]), z(Zm, Ym), Xm[0] = Yp(Ym[0]), Xm[1] = Jp(Ym[0]), e.setUniformFloatVec2(Jm.PATTERN_ORIGIN_X_DOUBLE, Xm), Xm[0] = Yp(Ym[1]), Xm[1] = Jp(Ym[1]), e.setUniformFloatVec2(Jm.PATTERN_ORIGIN_Y_DOUBLE, Xm);
	let s = 2 ** ((r.viewState.zoom + .5) % 1 - .5);
	Ym[0] = Yp(s), Ym[1] = Jp(s), e.setUniformFloatVec2(Jm.PATTERN_SCALE_RATIO_DOUBLE, Ym);
}
//#endregion
//#region node_modules/ol/renderer/webgl/worldUtil.js
function eh(e, t) {
	let n = e.viewState.projection, r = t.getSource().getWrapX() && n.canWrapX(), i = n.getExtent(), a = e.extent, o = r ? I(i) : null, s = r ? Math.ceil((a[2] - i[2]) / o) + 1 : 1;
	return [
		r ? Math.floor((a[0] - i[0]) / o) : 0,
		s,
		o
	];
}
//#endregion
//#region node_modules/ol/renderer/webgl/VectorLayer.js
var th = {
	...op,
	...Jm,
	...Nm,
	RENDER_EXTENT: "u_renderExtent",
	GLOBAL_ALPHA: "u_globalAlpha"
}, nh = class extends _p {
	constructor(e, t) {
		let n = {
			[th.RENDER_EXTENT]: [
				0,
				0,
				0,
				0
			],
			[th.GLOBAL_ALPHA]: 1,
			[th.ONE]: 1
		};
		super(e, {
			uniforms: n,
			postProcesses: t.postProcesses ?? []
		}), this.hitDetectionEnabled_ = !t.disableHitDetection, this.hitRenderTarget_, this.sourceRevision_ = -1, this.layerRevision_ = -1, this.skipNextTextRender_ = !1, this.previousExtent_ = We(), this.currentTransform_ = qr(), this.currentFrameStateTransform_ = qr(), this.styleVariables_ = {}, this.style_ = [], this.hasText_ = !1, this.styleRenderer_ = null, this.buffers_ = null, this.batch_ = new rm(), this.initialFeaturesAdded_ = !1, this.sourceListenKeys_ = null, this.applyOptions_(t);
	}
	addInitialFeatures_(e) {
		let t = this.getLayer().getSource(), n = hr(), r;
		n && (r = lr(n, e.viewState.projection)), this.batch_.addFeatures(t.getFeatures(), r), this.sourceListenKeys_ = [
			A(t, Gm.ADDFEATURE, this.handleSourceFeatureAdded_.bind(this, r)),
			A(t, Gm.CHANGEFEATURE, this.handleSourceFeatureChanged_.bind(this, r), this),
			A(t, Gm.REMOVEFEATURE, this.handleSourceFeatureDelete_, this),
			A(t, Gm.CLEAR, this.handleSourceFeatureClear_, this)
		];
	}
	applyOptions_(e) {
		this.styleVariables_ = e.variables, this.style_ = e.style;
		let t = Um(this.style_), n = !!t && Pm(t);
		n && !this.hasText_ ? this.setPostProcesses([Fm(() => this.styleRenderer_.getTextOverlayCanvas(), () => this.styleRenderer_.getTextOverlayFrameState()), ...this.getPostProcesses()]) : !n && this.hasText_ && this.setPostProcesses(this.getPostProcesses().slice(1)), this.hasText_ = n;
	}
	createRenderers_() {
		this.buffers_ = null, this.styleRenderer_ = new Hm(this.style_, this.styleVariables_, this.helper, this.hitDetectionEnabled_);
	}
	reset(e) {
		this.applyOptions_(e), this.helper && this.createRenderers_(), super.reset(e);
	}
	afterHelperCreated() {
		this.styleRenderer_ ? this.styleRenderer_.setHelper(this.helper, this.buffers_) : this.createRenderers_(), this.hitDetectionEnabled_ && (this.hitRenderTarget_ = new qm(this.helper));
	}
	handleSourceFeatureAdded_(e, t) {
		let n = t.feature;
		this.batch_.addFeature(n, e);
	}
	handleSourceFeatureChanged_(e, t) {
		let n = t.feature;
		this.batch_.changeFeature(n, e);
	}
	handleSourceFeatureDelete_(e) {
		let t = e.feature;
		this.batch_.removeFeature(t);
	}
	handleSourceFeatureClear_() {
		this.batch_.clear();
	}
	applyUniforms_(e, t) {
		$m(this.helper, this.currentFrameStateTransform_, e, t);
	}
	renderFrame(e) {
		let t = this.helper.getGL();
		this.preRender(t, e);
		let n = this.getLayer(), [r, i, a] = eh(e, n);
		this.helper.prepareDraw(e), this.renderWorlds(e, !1, r, i, a), this.hasText_ && this.styleRenderer_.finalizeTextRender(e).then(() => {
			if (this.skipNextTextRender_) {
				this.skipNextTextRender_ = !1;
				return;
			}
			this.skipNextTextRender_ = !0, this.layerRevision_++, n.changed();
		}), this.helper.finalizeDraw(e, this.dispatchPreComposeEvent, this.dispatchPostComposeEvent);
		let o = this.helper.getCanvas();
		return this.hitDetectionEnabled_ && (this.renderWorlds(e, !0, r, i, a), this.hitRenderTarget_.clearCachedData()), this.postRender(t, e), o;
	}
	prepareFrameInternal(e) {
		this.initialFeaturesAdded_ ||= (this.addInitialFeatures_(e), !0);
		let t = this.getLayer(), n = t.getSource(), r = e.viewState, i = !e.viewHints[Lr.ANIMATING] && !e.viewHints[Lr.INTERACTING], a = !Ye(this.previousExtent_, e.extent), o = this.sourceRevision_ < n.getRevision(), s = this.layerRevision_ < t.getRevision();
		if (this.sourceRevision_ = n.getRevision(), this.layerRevision_ = t.getRevision(), (s || a || o) && (this.skipNextTextRender_ = !1), i && (a || o)) {
			let i = r.projection, a = r.resolution, o = t instanceof Al ? t.getRenderBuffer() : 0, s = Le(e.extent, o * a), c = hr();
			c ? n.loadFeatures(vr(s, c), br(a, i), c) : n.loadFeatures(s, a, i), this.ready = !1;
			let l = this.helper.makeProjectionTransform(e, qr(), !0);
			this.styleRenderer_.generateBuffers(this.batch_, l, e.viewState.resolution).then((e) => {
				this.buffers_ && this.disposeBuffers(this.buffers_), this.buffers_ = e, this.ready = !0, this.getLayer()?.changed();
			}), this.previousExtent_ = e.extent.slice();
		}
		return !0;
	}
	renderWorlds(e, t, n, r, i) {
		let a = n;
		t && (this.hitRenderTarget_.setSize([Math.floor(e.size[0] / 2), Math.floor(e.size[1] / 2)]), this.helper.prepareDrawToRenderTarget(e, this.hitRenderTarget_, !0));
		do
			this.helper.makeProjectionTransform(e, this.currentFrameStateTransform_), $r(this.currentFrameStateTransform_, a * i, 0), this.buffers_ && this.styleRenderer_.render(this.buffers_, e, () => {
				this.applyUniforms_(this.buffers_.invertVerticesTransform, e), this.helper.applyHitDetectionUniform(t);
			});
		while (++a < r);
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		if (R(this.hitDetectionEnabled_, "`forEachFeatureAtCoordinate` cannot be used on a WebGL layer if the hit detection logic has been disabled using the `disableHitDetection: true` option."), !this.styleRenderer_ || !this.hitDetectionEnabled_) return;
		let a = z(t.coordinateToPixelTransform, e.slice()), o = this.hitRenderTarget_.readPixel(a[0] / 2, a[1] / 2), s = mm([
			o[0] / 255,
			o[1] / 255,
			o[2] / 255,
			o[3] / 255
		]), c = this.batch_.getFeatureFromRef(s);
		if (c) return r(c, this.getLayer(), null);
	}
	disposeBuffers(e) {
		if (!this.helper) return;
		let t = (e) => {
			for (let t of e) t && this.helper.deleteBuffer(t);
		};
		e.pointBuffers && t(e.pointBuffers), e.lineStringBuffers && t(e.lineStringBuffers), e.polygonBuffers && t(e.polygonBuffers), e.textInstructionsKey && this.styleRenderer_.disposeTextInstructions(e.textInstructionsKey);
	}
	disposeInternal() {
		this.buffers_ && this.disposeBuffers(this.buffers_), this.sourceListenKeys_ &&= (this.sourceListenKeys_.forEach(function(e) {
			j(e);
		}), null), this.styleRenderer_ && this.styleRenderer_.dispose(), super.disposeInternal();
	}
	renderDeclutter() {}
}, rh = {
	BLUR: "blur",
	GRADIENT: "gradient",
	RADIUS: "radius"
}, ih = [
	"#00f",
	"#0ff",
	"#0f0",
	"#ff0",
	"#f00"
], ah = class extends Al {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.gradient, delete t.radius, delete t.blur, delete t.weight, super(t), this.on, this.once, this.un, this.filter_ = e.filter ?? !0, this.styleVariables_ = e.variables || {}, this.gradient_ = null, this.addChangeListener(rh.GRADIENT, this.handleGradientChanged_), this.setGradient(e.gradient ? e.gradient : ih), this.setBlur(e.blur === void 0 ? 15 : e.blur), this.setRadius(e.radius === void 0 ? 8 : e.radius);
		let n = e.weight ? e.weight : "weight";
		this.weight_ = n, this.setRenderOrder(null);
	}
	getBlur() {
		return this.get(rh.BLUR);
	}
	getGradient() {
		return this.get(rh.GRADIENT);
	}
	getRadius() {
		return this.get(rh.RADIUS);
	}
	handleGradientChanged_() {
		this.gradient_ = oh(this.getGradient());
	}
	setBlur(e) {
		let t = this.get(rh.BLUR);
		if (this.set(rh.BLUR, e), typeof e == "number" && typeof t == "number") {
			this.changed();
			return;
		}
		this.clearRenderer();
	}
	setGradient(e) {
		this.set(rh.GRADIENT, e);
	}
	setRadius(e) {
		let t = this.get(rh.RADIUS);
		if (this.set(rh.RADIUS, e), typeof e == "number" && typeof t == "number") {
			this.changed();
			return;
		}
		this.clearRenderer();
	}
	setFilter(e) {
		this.filter_ = e, this.changed(), this.clearRenderer();
	}
	setWeight(e) {
		this.weight_ = e, this.changed(), this.clearRenderer();
	}
	createRenderer() {
		let e = new Qp(), t = Ap(this.styleVariables_), n = vs(this.styleVariables_), r = Z(t, this.filter_, as, n), i = Z(t, this.getRadius(), V), a = Z(t, this.getBlur(), V), o = {};
		typeof this.getBlur() == "number" && (a = "a_blur", o.a_blur = () => this.getBlur(), e.addUniform("a_blur", "float")), typeof this.getRadius() == "number" && (i = "a_radius", o.a_radius = () => this.getRadius(), e.addUniform("a_radius", "float"));
		let s = {}, c = null;
		if (typeof this.weight_ == "string" || typeof this.weight_ == "function") {
			let t = typeof this.weight_ == "string" ? (e) => e.get(this.weight_) : this.weight_;
			s.prop_weight = {
				size: 1,
				callback: (e) => {
					let n = t(e);
					return n === void 0 ? 1 : St(n, 0, 1);
				}
			}, c = "a_prop_weight", e.addAttribute("a_prop_weight", "float");
		} else c = Z(t, [
			"clamp",
			this.weight_,
			0,
			1
		], V);
		let l = `(${i} / max(1., ${a}))`;
		e.setSymbolSizeExpression(`vec2(${i} + ${a}) * 2.`).setSymbolColorExpression(`vec4(smoothstep(0., 1., (1. - length(coordsPx * 2. / v_quadSizePx)) * ${l}) * ${c})`).setStrokeColorExpression(`vec4(smoothstep(0., 1., (1. - length(currentRadiusPx * 2. / v_width)) * ${l}) * ${c})`).setStrokeWidthExpression(`(${i} + ${a}) * 2.`).setFillColorExpression(`vec4(${c})`), n.mCoordinate ? e.setFragmentDiscardExpression(`!${r}`) : e.setShapeDiscardExpression(`!${r}`), Gp(e, t);
		let u = qp(t), d = Kp(t, this.styleVariables_);
		return new nh(this, {
			className: this.getClassName(),
			variables: this.styleVariables_,
			style: {
				builder: e,
				attributes: {
					...u,
					...s
				},
				uniforms: {
					...d,
					...o
				}
			},
			disableHitDetection: !1,
			postProcesses: [{
				fragmentShader: "\n            precision mediump float;\n\n            uniform sampler2D u_image;\n            uniform sampler2D u_gradientTexture;\n            uniform float u_opacity;\n\n            varying vec2 v_texCoord;\n\n            void main() {\n              vec4 color = texture2D(u_image, v_texCoord);\n              gl_FragColor.a = color.a * u_opacity;\n              gl_FragColor.rgb = texture2D(u_gradientTexture, vec2(0.5, color.a)).rgb;\n              gl_FragColor.rgb *= gl_FragColor.a;\n            }",
				uniforms: {
					u_gradientTexture: () => this.gradient_,
					u_opacity: () => this.getOpacity()
				}
			}]
		});
	}
	updateStyleVariables(e) {
		Object.assign(this.styleVariables_, e), this.changed();
	}
	renderDeclutter() {}
};
function oh(e) {
	let t = y(1, 256), n = t.createLinearGradient(0, 0, 1, 256), r = 1 / (e.length - 1);
	for (let t = 0, i = e.length; t < i; ++t) n.addColorStop(t * r, e[t]);
	return t.fillStyle = n, t.fillRect(0, 0, 1, 256), t.canvas;
}
//#endregion
//#region node_modules/ol-ext/util/input/Base.js
var sh = class extends we {
	constructor(e) {
		e ||= {}, super();
		var t = this.input = e.input;
		t || (t = this.input = document.createElement("input"), e.type && t.setAttribute("type", e.type), e.min !== void 0 && t.setAttribute("min", e.min), e.max !== void 0 && t.setAttribute("max", e.max), e.step !== void 0 && t.setAttribute("step", e.step), e.parent && e.parent.appendChild(t)), e.disabled && (t.disabled = !0), e.checked !== void 0 && (t.checked = !!e.checked), e.val !== void 0 && (t.value = e.val), e.hidden && t.classList.add("ol-input-hidden"), t.addEventListener("focus", function() {
			this.element && this.element.classList.add("ol-focus");
		}.bind(this));
		var n;
		t.addEventListener("focusout", function() {
			this.element && (n && clearTimeout(n), n = setTimeout(function() {
				this.element.classList.remove("ol-focus");
			}.bind(this), 0));
		}.bind(this));
	}
	_listenDrag(e, t) {
		var n = function(n) {
			this.moving = !0, this.element.classList.add("ol-moving");
			var r = function(n) {
				n.type === "pointerup" && (document.removeEventListener("pointermove", r), document.removeEventListener("pointerup", r), document.removeEventListener("pointercancel", r), setTimeout(function() {
					this.moving = !1, this.element.classList.remove("ol-moving");
				}.bind(this))), n.target === e && t(n), n.stopPropagation(), n.preventDefault();
			}.bind(this);
			document.addEventListener("pointermove", r, !1), document.addEventListener("pointerup", r, !1), document.addEventListener("pointercancel", r, !1), n.stopPropagation(), n.preventDefault();
		}.bind(this);
		e.addEventListener("mousedown", n, !1), e.addEventListener("touchstart", n, !1);
	}
	setValue(e) {
		e !== void 0 && (this.input.value = e), this.input.dispatchEvent(new Event("change"));
	}
	getValue() {
		return this.input.value;
	}
	getInputElement() {
		return this.input;
	}
}, ch = class extends sh {
	constructor(e) {
		e ||= {}, super(e);
		var t = this.element = document.createElement("label");
		e.html instanceof Element ? t.appendChild(e.html) : e.html !== void 0 && (t.innerHTML = e.html), t.className = ("ol-ext-check ol-ext-checkbox " + (e.className || "")).trim(), this.input.parentNode && this.input.parentNode.insertBefore(t, this.input), t.appendChild(this.input), t.appendChild(document.createElement("span")), e.after && t.appendChild(document.createTextNode(e.after)), this.input.addEventListener("change", function() {
			this.dispatchEvent({
				type: "check",
				checked: this.input.checked,
				value: this.input.value
			});
		}.bind(this));
	}
	isChecked() {
		return this.input.checked;
	}
}, lh = class extends ch {
	constructor(e) {
		e ||= {}, super(e), this.element.className = ("ol-ext-toggle-switch " + (e.className || "")).trim();
	}
}, uh = class extends ch {
	constructor(e) {
		e ||= {}, super(e), this.element.className = ("ol-ext-check ol-ext-radio " + (e.className || "")).trim();
	}
}, Q = {};
Q.create = function(e, t) {
	t ||= {};
	var n;
	if (e === "TEXT") n = document.createTextNode(t.html || ""), t.parent && t.parent.appendChild(n);
	else for (var r in n = document.createElement(e.toLowerCase()), /button/i.test(e) && n.setAttribute("type", "button"), t) switch (r) {
		case "className":
			t.className && t.className.trim && n.setAttribute("class", t.className.trim());
			break;
		case "text":
			n.innerText = t.text;
			break;
		case "html":
			t.html instanceof Element ? n.appendChild(t.html) : t.html !== void 0 && (n.innerHTML = t.html);
			break;
		case "parent":
			t.parent && t.parent.appendChild(n);
			break;
		case "options":
			if (/select/i.test(e)) for (var i in t.options) Q.create("OPTION", {
				html: i,
				value: t.options[i],
				parent: n
			});
			break;
		case "style":
			Q.setStyle(n, t.style);
			break;
		case "change":
		case "click":
			Q.addListener(n, r, t[r]);
			break;
		case "on":
			for (var a in t.on) Q.addListener(n, a, t.on[a]);
			break;
		case "checked":
			n.checked = !!t.checked;
			break;
		default: n.setAttribute(r, t[r]);
	}
	return n;
}, Q.createSwitch = function(e) {
	var t = Q.create("INPUT", {
		type: "checkbox",
		on: e.on,
		click: e.click,
		change: e.change,
		parent: e.parent
	});
	return new lh(Object.assign({ input: t }, e || {})), t;
}, Q.createCheck = function(e) {
	var t = Q.create("INPUT", {
		name: e.name,
		type: e.type === "radio" ? "radio" : "checkbox",
		on: e.on,
		parent: e.parent
	}), n = Object.assign({ input: t }, e || {});
	return e.type === "radio" ? new uh(n) : new ch(n), t;
}, Q.setHTML = function(e, t) {
	t instanceof Element ? e.appendChild(t) : t !== void 0 && (e.innerHTML = t);
}, Q.appendText = function(e, t) {
	e.appendChild(document.createTextNode(t || ""));
}, Q.addListener = function(e, t, n, r) {
	typeof t == "string" && (t = t.split(" ")), t.forEach(function(t) {
		e.addEventListener(t, n, r);
	});
}, Q.removeListener = function(e, t, n) {
	typeof t == "string" && (t = t.split(" ")), t.forEach(function(t) {
		e.removeEventListener(t, n);
	});
}, Q.show = function(e) {
	e.style.display = "";
}, Q.hide = function(e) {
	e.style.display = "none";
}, Q.hidden = function(e) {
	return Q.getStyle(e, "display") === "none";
}, Q.toggle = function(e) {
	e.style.display = e.style.display === "none" ? "" : "none";
}, Q.setStyle = function(e, t) {
	for (var n in t) switch (n) {
		case "top":
		case "left":
		case "bottom":
		case "right":
		case "minWidth":
		case "maxWidth":
		case "width":
		case "height":
			typeof t[n] == "number" ? e.style[n] = t[n] + "px" : e.style[n] = t[n];
			break;
		default: e.style[n] = t[n];
	}
}, Q.getStyle = function(e, t) {
	var n, r = (e.ownerDocument || document).defaultView;
	if (r && r.getComputedStyle) t = t.replace(/([A-Z])/g, "-$1").toLowerCase(), n = r.getComputedStyle(e, null).getPropertyValue(t);
	else if (e.currentStyle && (t = t.replace(/-(\w)/g, function(e, t) {
		return t.toUpperCase();
	}), n = e.currentStyle[t], /^\d+(em|pt|%|ex)?$/i.test(n))) return (function(t) {
		var n = e.style.left, r = e.runtimeStyle.left;
		return e.runtimeStyle.left = e.currentStyle.left, e.style.left = t || 0, t = e.style.pixelLeft + "px", e.style.left = n, e.runtimeStyle.left = r, t;
	})(n);
	return /px$/.test(n) ? parseInt(n) : n;
}, Q.outerHeight = function(e) {
	return e.offsetHeight + Q.getStyle(e, "marginBottom");
}, Q.outerWidth = function(e) {
	return e.offsetWidth + Q.getStyle(e, "marginLeft");
}, Q.offsetRect = function(e) {
	var t = e.getBoundingClientRect();
	return {
		top: t.top + (window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0),
		left: t.left + (window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft || 0),
		height: t.height || t.bottom - t.top,
		width: t.width || t.right - t.left
	};
}, Q.getFixedOffset = function(e) {
	var t = {
		left: 0,
		top: 0
	}, n = function(e) {
		if (!e) return t;
		if (Q.getStyle(e, "position") === "absolute" && Q.getStyle(e, "transform") !== "none") {
			var r = e.getBoundingClientRect();
			return t.left += r.left, t.top += r.top, t;
		}
		return n(e.offsetParent);
	};
	return n(e.offsetParent);
}, Q.positionRect = function(e, t) {
	var n = 0, r = 0, i = function(a) {
		if (a) return n += a.offsetLeft, r += a.offsetTop, i(a.offsetParent);
		var o = {
			top: e.offsetTop + r,
			left: e.offsetLeft + n
		};
		return t && (o.top -= window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0, o.left -= window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft || 0), o.bottom = o.top + e.offsetHeight, o.right = o.top + e.offsetWidth, o;
	};
	return i(e.offsetParent);
}, Q.scrollDiv = function(e, t) {
	t ||= {};
	var n = !1, r = 0, i, a = 0, o = typeof t.onmove == "function" ? t.onmove : function() {}, s = t.vertical ? "screenY" : "screenX", c = t.vertical ? "scrollTop" : "scrollLeft", l = !1, u, d, f = 0, p = function() {
		y && (f++, setTimeout(m));
	}, m = function() {
		if (y) {
			if (f--, f) return;
			var t = e.clientHeight, n = e.scrollHeight;
			u = t / n, y.style.height = u * 100 + "%", y.style.top = e.scrollTop / n * 100 + "%", v.style.height = t + "px", t > n - .5 ? v.classList.add("ol-100pc") : v.classList.remove("ol-100pc");
		}
	}, h = function(t) {
		t.target.classList.contains("ol-noscroll") || (l = !1, n = t[s], a = /* @__PURE__ */ new Date(), e.classList.add("ol-move"), t.preventDefault(), window.addEventListener("pointermove", g), Q.addListener(window, ["pointerup", "pointercancel"], x));
	}, g = function(t) {
		if (n !== !1) {
			var f = (d ? -1 / u : 1) * (n - t[s]);
			l ||= Math.round(f), e[c] += f, i = /* @__PURE__ */ new Date(), i - a && (r = (r + f / (i - a)) / 2), n = t[s], a = i, f && o(!0);
		} else l = !0;
	}, _ = function(t) {
		var n = t > 0 ? Math.min(100, t / 2) : Math.max(-100, t / 2);
		t -= n, e[c] += n, -1 < t && t < 1 ? (l ? setTimeout(function() {
			e.classList.remove("ol-move");
		}) : e.classList.remove("ol-move"), l = !1, o(!1)) : setTimeout(function() {
			_(t);
		}, 40);
	}, v, y;
	if (t.vertical && t.minibar) {
		var b = function(n) {
			e.removeEventListener("pointermove", b), e.parentNode.classList.add("ol-miniscroll"), y = Q.create("DIV"), v = Q.create("DIV", {
				className: "ol-scroll",
				html: y
			}), e.parentNode.insertBefore(v, e), y.addEventListener("pointerdown", function(e) {
				d = !0, h(e);
			}), t.mousewheel && (Q.addListener(v, [
				"mousewheel",
				"DOMMouseScroll",
				"onmousewheel"
			], function(e) {
				S(e);
			}), Q.addListener(y, [
				"mousewheel",
				"DOMMouseScroll",
				"onmousewheel"
			], function(e) {
				S(e);
			})), e.parentNode.addEventListener("pointerenter", p), window.addEventListener("resize", p), n !== !1 && p();
		};
		e.parentNode ? b(!1) : e.addEventListener("pointermove", b), e.addEventListener("scroll", function() {
			p();
		});
	}
	e.style["touch-action"] = "none", e.style.overflow = "hidden", e.classList.add("ol-scrolldiv"), Q.addListener(e, ["pointerdown"], function(e) {
		d = !1, h(e);
	}), e.addEventListener("click", function(t) {
		e.classList.contains("ol-move") && (t.preventDefault(), t.stopPropagation());
	}, !0);
	var x = function(i) {
		a = /* @__PURE__ */ new Date() - a, a > 100 || d ? r = 0 : a > 0 && (r = ((r || 0) + (n - i[s]) / a) / 2), _(t.animate === !1 ? 0 : r * 200), n = !1, r = 0, a = 0, e.classList.contains("ol-move") ? e.classList.remove("ol-hasClick") : (e.classList.add("ol-hasClick"), setTimeout(function() {
			e.classList.remove("ol-hasClick");
		}, 500)), d = !1, window.removeEventListener("pointermove", g), Q.removeListener(window, ["pointerup", "pointercancel"], x);
	}, S = function(t) {
		var n = Math.max(-1, Math.min(1, t.wheelDelta || -t.detail));
		return e.classList.add("ol-move"), e[c] -= n * 30, e.classList.remove("ol-move"), !1;
	};
	return t.mousewheel && Q.addListener(e, [
		"mousewheel",
		"DOMMouseScroll",
		"onmousewheel"
	], S), { refresh: p };
}, Q.dispatchEvent = function(e, t) {
	var n;
	try {
		n = new CustomEvent(e);
	} catch {
		n = document.createEvent("CustomEvent"), n.initCustomEvent(e, !0, !0, {});
	}
	t.dispatchEvent(n);
}, Q.setCursor = function(e, t) {
	e instanceof Rl && (e = e.getTargetElement()), !("ontouchstart" in window) && e instanceof Element && (e.style.cursor = t);
};
//#endregion
//#region node_modules/ol-ext/control/LayerSwitcher.js
var dh = class extends Te {
	constructor(e) {
		e ||= {};
		var t = Q.create("DIV", { className: e.switcherClass || "ol-layerswitcher" });
		super({
			element: t,
			target: e.target
		});
		var n = this;
		this.dcount = 0, this.show_progress = e.show_progress, this.oninfo = typeof e.oninfo == "function" ? e.oninfo : null, this.onextent = typeof e.onextent == "function" ? e.onextent : null, this.hasextent = e.extent || e.onextent, this.hastrash = e.trash, this.reordering = e.reordering !== !1, this._layers = [], this._layerGroup = e.layerGroup && e.layerGroup.getLayers ? e.layerGroup : null, this.onchangeCheck = typeof e.onchangeCheck == "function" ? e.onchangeCheck : null, typeof e.displayInLayerSwitcher == "function" && (this.displayInLayerSwitcher = e.displayInLayerSwitcher), e.target || (t.classList.add("ol-unselectable"), t.classList.add("ol-control"), t.classList.add(e.collapsed === !1 ? "ol-forceopen" : "ol-collapsed"), e.counter && t.classList.add("ol-counter"), this.counter = Q.create("SPAN", {
			class: "ol-counter",
			text: 0,
			parent: t
		}), this.button = Q.create("BUTTON", {
			type: "button",
			parent: t
		}), this.button.addEventListener("touchstart", function(e) {
			t.classList.toggle("ol-forceopen"), t.classList.add("ol-collapsed"), n.dispatchEvent({
				type: "toggle",
				collapsed: t.classList.contains("ol-collapsed")
			}), e.preventDefault(), n.overflow();
		}), this.button.addEventListener("click", function() {
			t.classList.toggle("ol-forceopen"), t.classList.add("ol-collapsed"), n.dispatchEvent({
				type: "toggle",
				collapsed: !t.classList.contains("ol-forceopen")
			}), n.overflow();
		}), e.mouseover && (t.addEventListener("mouseleave", function() {
			t.classList.add("ol-collapsed"), n.dispatchEvent({
				type: "toggle",
				collapsed: !0
			});
		}), t.addEventListener("mouseover", function() {
			t.classList.remove("ol-collapsed"), n.dispatchEvent({
				type: "toggle",
				collapsed: !1
			});
		})), e.minibar && (e.noScroll = !0), e.noScroll || (this.topv = Q.create("DIV", {
			className: "ol-switchertopdiv",
			parent: t,
			click: function() {
				n.overflow("+50%");
			}
		}), this.botv = Q.create("DIV", {
			className: "ol-switcherbottomdiv",
			parent: t,
			click: function() {
				n.overflow("-50%");
			}
		})), this._noScroll = e.noScroll), this.panel_ = Q.create("UL", { className: "panel" }), this.panelContainer_ = Q.create("DIV", {
			className: "panel-container",
			html: this.panel_,
			parent: t
		}), !e.target && !e.noScroll && Q.addListener(this.panel_, "mousewheel DOMMouseScroll onmousewheel", function(e) {
			n.overflow(Math.max(-1, Math.min(1, e.wheelDelta || -e.detail))) && (e.stopPropagation(), e.preventDefault());
		}), this.header_ = Q.create("LI", {
			className: "ol-header",
			parent: this.panel_
		}), this.set("drawDelay", e.drawDelay || 0), this.set("selection", e.selection), e.minibar && setTimeout(function() {
			var e = Q.scrollDiv(this.panelContainer_, {
				mousewheel: !0,
				vertical: !0,
				minibar: !0
			});
			this.on(["drawlist", "toggle"], function() {
				e.refresh();
			});
		}.bind(this));
	}
	displayInLayerSwitcher(e) {
		return e.get("displayInLayerSwitcher") !== !1;
	}
	setMap(e) {
		if (super.setMap(e), this.drawPanel(), this._listener) for (var t in this._listener) xe(this._listener[t]);
		this._listener = null, e && (this._listener = {
			moveend: e.on("moveend", this.viewChange.bind(this)),
			size: e.on("change:size", this.overflow.bind(this))
		}, this._layerGroup ? this._listener.change = this._layerGroup.getLayers().on("change:length", this.drawPanel.bind(this)) : this._listener.change = e.getLayerGroup().getLayers().on("change:length", this.drawPanel.bind(this)));
	}
	show() {
		this.element.classList.add("ol-forceopen"), this.overflow(), this.dispatchEvent({
			type: "toggle",
			collapsed: !1
		});
	}
	hide() {
		this.element.classList.remove("ol-forceopen"), this.overflow(), this.dispatchEvent({
			type: "toggle",
			collapsed: !0
		});
	}
	toggle() {
		this.element.classList.toggle("ol-forceopen"), this.overflow(), this.dispatchEvent({
			type: "toggle",
			collapsed: !this.isOpen()
		});
	}
	isOpen() {
		return this.element.classList.contains("ol-forceopen");
	}
	setHeader(e) {
		Q.setHTML(this.header_, e);
	}
	overflow(e) {
		if (this.button && !this._noScroll) {
			if (Q.hidden(this.panel_)) {
				Q.setStyle(this.element, { height: "auto" });
				return;
			}
			var t = Q.outerHeight(this.element), n = Q.outerHeight(this.panel_), r = this.button.offsetTop + Q.outerHeight(this.button), i = this.panel_.offsetTop - r;
			if (n > t - r) {
				Q.setStyle(this.element, { height: "100%" });
				var a = this.panel_.querySelectorAll("li.ol-visible .li-content")[0], o = a ? 2 * Q.getStyle(a, "height") : 0;
				switch (e) {
					case 1:
						i += o;
						break;
					case -1:
						i -= o;
						break;
					case "+50%":
						i += Math.round(t / 2);
						break;
					case "-50%": i -= Math.round(t / 2);
				}
				return i + n <= t - 3 * r / 2 ? (i = t - 3 * r / 2 - n, Q.hide(this.botv)) : Q.show(this.botv), i >= 0 ? (i = 0, Q.hide(this.topv)) : Q.show(this.topv), Q.setStyle(this.panel_, { top: i + "px" }), !0;
			}
			return Q.setStyle(this.element, { height: "auto" }), Q.setStyle(this.panel_, { top: 0 }), Q.hide(this.botv), Q.hide(this.topv), !1;
		}
		return !1;
	}
	_setLayerForLI(e, t) {
		var n = [];
		t.getLayers && n.push(t.getLayers().on("change:length", this.drawPanel.bind(this))), e && (n.push(t.on("change:opacity", (function() {
			this.setLayerOpacity(t, e);
		}).bind(this))), n.push(t.on("change:visible", (function() {
			this.setLayerVisibility(t, e);
		}).bind(this)))), n.push(t.on("propertychange", (function(e) {
			(e.key === "displayInLayerSwitcher" || e.key === "openInLayerSwitcher" || e.key === "title" || e.key === "name") && this.drawPanel(e);
		}).bind(this))), this._layers.push({
			li: e,
			layer: t,
			listeners: n
		});
	}
	setLayerOpacity(e, t) {
		var n = t.querySelector(".layerswitcher-opacity-cursor");
		n && (n.style.left = e.getOpacity() * 100 + "%"), this.dispatchEvent({
			type: "layer:opacity",
			layer: e
		});
	}
	setLayerVisibility(e, t) {
		var n = t.querySelector(".ol-visibility");
		n && (n.checked = e.getVisible()), e.getVisible() ? t.classList.add("ol-visible") : t.classList.remove("ol-visible"), this.dispatchEvent({
			type: "layer:visible",
			layer: e
		});
	}
	_clearLayerForLI() {
		this._layers.forEach(function(e) {
			e.listeners.forEach(function(e) {
				xe(e);
			});
		}), this._layers = [];
	}
	_getLayerForLI(e) {
		for (var t = 0, n; n = this._layers[t]; t++) if (n.li === e) return n.layer;
		return null;
	}
	viewChange() {
		this.panel_.querySelectorAll("li").forEach(function(e) {
			var t = this._getLayerForLI(e);
			t && (this.testLayerVisibility(t) ? e.classList.remove("ol-layer-hidden") : e.classList.add("ol-layer-hidden"));
		}.bind(this));
	}
	getPanel() {
		return this.panelContainer_;
	}
	drawPanel() {
		if (this.getMap()) {
			var e = this;
			this.dcount++, setTimeout(function() {
				e.drawPanel_();
			}, this.get("drawDelay") || 0);
		}
	}
	drawPanel_() {
		if (!(--this.dcount || this.dragging_)) {
			var e = this.panelContainer_.scrollTop;
			this._clearLayerForLI(), this.panel_.querySelectorAll("li").forEach(function(e) {
				e.classList.contains("ol-header") || e.remove();
			}.bind(this)), this._layerGroup ? this.drawList(this.panel_, this._layerGroup.getLayers()) : this.getMap() && this.drawList(this.panel_, this.getMap().getLayers()), this.panelContainer_.scrollTop = e, this.counter && (this.counter.innerHTML = this.panel_.parentNode.querySelectorAll("ul.panel > li:not(.ol-header)").length);
		}
	}
	switchLayerVisibility(e, t) {
		e.get("baseLayer") ? (e.getVisible() || e.setVisible(!0), t.forEach(function(t) {
			e !== t && t.get("baseLayer") && t.getVisible() && t.setVisible(!1);
		})) : e.setVisible(!e.getVisible());
	}
	testLayerVisibility(e) {
		if (!this.getMap()) return !0;
		var t = this.getMap().getView().getResolution(), n = this.getMap().getView().getZoom();
		if (e.getMaxResolution() <= t || e.getMinResolution() >= t || e.getMinZoom && (e.getMinZoom() >= n || e.getMaxZoom() < n)) return !1;
		var r = e.getExtent();
		return !r || pt(this.getMap().getView().calculateExtent(this.getMap().getSize()), r);
	}
	dragOrdering_(e) {
		e.stopPropagation(), e.preventDefault();
		var t = this, n = e.currentTarget.parentNode.parentNode, r = !0, i = this.panel_, a, o = e.pageY || e.touches && e.touches.length && e.touches[0].pageY || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageY, s, c, l, u;
		n.parentNode.classList.add("drag");
		function d() {
			if (s) {
				var e = l, r = t.getSelection() === e;
				if (e && s) {
					for (var i = u ? u.getLayers() : t._layerGroup ? t._layerGroup.getLayers() : t.getMap().getLayers(), a = i.getArray(), o = 0; o < a.length; o++) if (a[o] == e) {
						i.removeAt(o);
						break;
					}
					for (var p = 0; p < a.length; p++) if (a[p] === s) {
						o > p ? i.insertAt(p, e) : i.insertAt(p + 1, e);
						break;
					}
				}
				r && t.selectLayer(e), t.dispatchEvent({
					type: "reorder-end",
					layer: e,
					group: u
				});
			}
			n.parentNode.querySelectorAll("li").forEach(function(e) {
				e.classList.remove("dropover"), e.classList.remove("dropover-after"), e.classList.remove("dropover-before");
			}), n.classList.remove("drag"), n.parentNode.classList.remove("drag"), t.element.classList.remove("drag"), c && c.remove(), Q.removeListener(document, "mousemove touchmove", f), Q.removeListener(document, "mouseup touchend touchcancel", d);
		}
		function f(e) {
			if (a = e.pageY || e.touches && e.touches.length && e.touches[0].pageY || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageY, r && Math.abs(o - a) > 2 && (r = !1, n.classList.add("drag"), l = t._getLayerForLI(n), s = !1, u = t._getLayerForLI(n.parentNode.parentNode), c = Q.create("LI", {
				className: "ol-dragover",
				html: n.innerHTML,
				style: {
					position: "absolute",
					"z-index": 1e4,
					left: n.offsetLeft,
					opacity: .5,
					width: Q.outerWidth(n),
					height: Q.getStyle(n, "height")
				},
				parent: i
			}), t.element.classList.add("drag"), t.dispatchEvent({
				type: "reorder-start",
				layer: l,
				group: u
			})), !r) {
				e.preventDefault(), e.stopPropagation(), Q.setStyle(c, { top: a - Q.offsetRect(i).top + i.scrollTop + 5 });
				var d;
				if (!e.touches) d = e.target, e.target.shadowRoot && (d = e.composedPath()[0]);
				else for (d = document.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY); d.shadowRoot;) d = d.shadowRoot.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY);
				for (d.classList.contains("ol-switcherbottomdiv") ? t.overflow(-1) : d.classList.contains("ol-switchertopdiv") && t.overflow(1); d && d.tagName !== "LI";) d = d.parentNode;
				(!d || !d.classList.contains("dropover")) && n.parentNode.querySelectorAll("li").forEach(function(e) {
					e.classList.remove("dropover"), e.classList.remove("dropover-after"), e.classList.remove("dropover-before");
				}), d && d.parentNode.classList.contains("drag") && d !== n ? (s = t._getLayerForLI(d), s && !s.get("allwaysOnTop") == !l.get("allwaysOnTop") ? (d.classList.add("dropover"), d.classList.add(n.offsetTop < d.offsetTop ? "dropover-after" : "dropover-before")) : s = !1, Q.show(c)) : (s = !1, d === n ? Q.hide(c) : Q.show(c)), s ? c.classList.remove("forbidden") : c.classList.add("forbidden");
			}
		}
		Q.addListener(document, "mousemove touchmove", f), Q.addListener(document, "mouseup touchend touchcancel", d);
	}
	dragOpacity_(e) {
		e.stopPropagation(), e.preventDefault();
		var t = this, n = e.target, r = this._getLayerForLI(n.parentNode.parentNode.parentNode);
		if (!r) return;
		var i = e.pageX || e.touches && e.touches.length && e.touches[0].pageX || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageX, a = Q.getStyle(n, "left") - i;
		t.dragging_ = !0;
		function o() {
			Q.removeListener(document, "mouseup touchend touchcancel", o), Q.removeListener(document, "mousemove touchmove", s), t.dragging_ = !1;
		}
		function s(e) {
			var t = (a + (e.pageX || e.touches && e.touches.length && e.touches[0].pageX || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageX)) / Q.getStyle(n.parentNode, "width"), i = Math.max(0, Math.min(1, t));
			Q.setStyle(n, { left: i * 100 + "%" }), n.parentNode.nextElementSibling.innerHTML = Math.round(i * 100), r.setOpacity(i);
		}
		Q.addListener(document, "mouseup touchend touchcancel", o), Q.addListener(document, "mousemove touchmove", s);
	}
	drawList(e, t) {
		var n = this, r = t.getArray(), i = function(e) {
			e.stopPropagation(), e.preventDefault();
			var r = n._getLayerForLI(this.parentNode.parentNode);
			n.switchLayerVisibility(r, t), n.get("selection") && r.getVisible() && n.selectLayer(r), n.onchangeCheck && n.onchangeCheck(r);
		};
		function a(e) {
			e.stopPropagation(), e.preventDefault();
			var t = n._getLayerForLI(this.parentNode.parentNode);
			n.oninfo(t), n.dispatchEvent({
				type: "info",
				layer: t
			});
		}
		function o(e) {
			e.stopPropagation(), e.preventDefault();
			var t = n._getLayerForLI(this.parentNode.parentNode);
			n.onextent ? n.onextent(t) : n.getMap().getView().fit(t.getExtent(), n.getMap().getSize()), n.dispatchEvent({
				type: "extent",
				layer: t
			});
		}
		function s(e) {
			e.stopPropagation(), e.preventDefault();
			var t = this.parentNode.parentNode.parentNode.parentNode, r, i = n._getLayerForLI(t);
			i ? (r = n._getLayerForLI(this.parentNode.parentNode), i.getLayers().remove(r), i.getLayers().getLength() == 0 && !i.get("noSwitcherDelete") && s.call(t.querySelectorAll(".layerTrash")[0], e)) : (t = this.parentNode.parentNode, n.getMap().removeLayer(n._getLayerForLI(t)));
		}
		function c(c) {
			if (!this.displayInLayerSwitcher(c)) {
				this._setLayerForLI(null, c);
				return;
			}
			var u = Q.create("LI", {
				className: (c.getVisible() ? "ol-visible " : " ") + (c.get("baseLayer") ? "baselayer" : ""),
				parent: e
			});
			this._setLayerForLI(u, c), this._selectedLayer === c && u.classList.add("ol-layer-select");
			var d = Q.create("DIV", {
				className: "ol-layerswitcher-buttons",
				parent: u
			}), f = Q.create("DIV", {
				className: "li-content",
				parent: u
			}), p = Q.create("INPUT", {
				type: c.get("baseLayer") ? "radio" : "checkbox",
				className: "ol-visibility",
				checked: c.getVisible(),
				click: function(e) {
					i.bind(this)(e), setTimeout(function() {
						e.target.checked = c.getVisible();
					});
				},
				on: { keydown: function(r) {
					switch (r.key) {
						case "ArrowLeft":
						case "ArrowRight":
							r.preventDefault(), r.stopPropagation();
							var i = r.key === "ArrowLeft" ? -.1 : .1, a = Math.min(1, Math.max(0, c.getOpacity() + i));
							c.setOpacity(a);
							break;
						case "Enter":
							n.get("selection") && (r.preventDefault(), r.stopPropagation(), n.selectLayer(c));
							break;
						case "-":
						case "+": c.getLayers && (this._focus = c, c.set("openInLayerSwitcher", !c.get("openInLayerSwitcher")));
						case "ArrowUp":
						case "ArrowDown":
							if (r.ctrlKey && this.reordering) {
								r.preventDefault(), r.stopPropagation();
								var o = t.getArray().indexOf(c);
								o > -1 && (r.key === "ArrowDown" ? o > 0 && (t.remove(c), t.insertAt(o - 1, c), n._focus = c, n.dispatchEvent({
									type: "reorder-end",
									layer: c
								})) : o < t.getLength() - 1 && (t.remove(c), t.insertAt(o + 1, c), n._focus = c, n.dispatchEvent({
									type: "reorder-end",
									layer: c
								})));
							}
							break;
						default:
							var s = this._getLayerForLI(e.parentNode);
							this.dispatchEvent({
								type: "layer:keydown",
								key: r.key,
								group: s,
								li: u,
								layer: c,
								originalEvent: r
							});
					}
				}.bind(this) },
				parent: f
			});
			c === n._focus && (p.focus(), n.overflow());
			var m = Q.create("LABEL", {
				title: c.get("title") || c.get("name"),
				click: i,
				style: { userSelect: "none" },
				parent: f
			});
			if (m.addEventListener("selectstart", function() {
				return !1;
			}), Q.create("SPAN", {
				html: c.get("title") || c.get("name"),
				click: function(e) {
					this.get("selection") && (e.stopPropagation(), this.selectLayer(c));
				}.bind(this),
				parent: m
			}), this.reordering && (l < r.length - 1 && (c.get("allwaysOnTop") || !r[l + 1].get("allwaysOnTop")) || l > 0 && (!c.get("allwaysOnTop") || r[l - 1].get("allwaysOnTop"))) && Q.create("DIV", {
				className: "layerup ol-noscroll",
				title: this.tip.up,
				on: { "mousedown touchstart": function(e) {
					n.dragOrdering_(e);
				} },
				parent: d
			}), c.getLayers) {
				var h = 0;
				c.getLayers().forEach(function(e) {
					n.displayInLayerSwitcher(e) && h++;
				}), h && Q.create("DIV", {
					className: c.get("openInLayerSwitcher") ? "collapse-layers" : "expend-layers",
					title: this.tip.plus,
					click: function() {
						var e = n._getLayerForLI(this.parentNode.parentNode);
						e.set("openInLayerSwitcher", !e.get("openInLayerSwitcher"));
					},
					parent: d
				});
			}
			if (this.oninfo && Q.create("DIV", {
				className: "layerInfo",
				title: this.tip.info,
				click: a,
				parent: d
			}), this.hastrash && !c.get("noSwitcherDelete") && Q.create("DIV", {
				className: "layerTrash",
				title: this.tip.trash,
				click: s,
				parent: d
			}), this.hasextent && r[l].getExtent()) {
				var g = r[l].getExtent();
				g.length == 4 && g[0] < g[2] && g[1] < g[3] && Q.create("DIV", {
					className: "layerExtent",
					title: this.tip.extent,
					click: o,
					parent: d
				});
			}
			if (this.show_progress && c instanceof Lu) {
				var _ = Q.create("DIV", {
					className: "layerswitcher-progress",
					parent: f
				});
				this.setprogress_(c), c.layerswitcher_progress = Q.create("DIV", { parent: _ });
			}
			var v = Q.create("DIV", {
				className: "layerswitcher-opacity",
				click: function(e) {
					if (e.target === this) {
						e.stopPropagation(), e.preventDefault();
						var t = Math.max(0, Math.min(1, e.offsetX / Q.getStyle(this, "width")));
						n._getLayerForLI(this.parentNode.parentNode).setOpacity(t), this.parentNode.querySelectorAll(".layerswitcher-opacity-label")[0].innerHTML = Math.round(t * 100);
					}
				},
				parent: f
			});
			if (Q.create("DIV", {
				className: "layerswitcher-opacity-cursor ol-noscroll",
				style: { left: c.getOpacity() * 100 + "%" },
				on: { "mousedown touchstart": function(e) {
					n.dragOpacity_(e);
				} },
				parent: v
			}), Q.create("DIV", {
				className: "layerswitcher-opacity-label",
				html: Math.round(c.getOpacity() * 100),
				parent: f
			}), c.getLayers && (u.classList.add("ol-layer-group"), c.get("openInLayerSwitcher") === !0)) {
				var y = Q.create("UL", { parent: u });
				this.drawList(y, c.getLayers());
			}
			u.classList.add(this.getLayerClass(c)), this.dispatchEvent({
				type: "drawlist",
				layer: c,
				li: u
			});
		}
		for (var l = r.length - 1; l >= 0; l--) c.call(this, r[l]);
		this.viewChange(), e === this.panel_ && (this.overflow(""), this._focus = null);
	}
	getLayerClass(e) {
		return e ? e.getLayers ? "ol-layer-group" : e instanceof Of ? "ol-layer-vector" : e instanceof Mf ? "ol-layer-vectortile" : e instanceof Lu ? "ol-layer-tile" : e instanceof Ff ? "ol-layer-image" : e instanceof ah ? "ol-layer-heatmap" : e.getFeatures ? "ol-layer-vectorimage" : "unknown" : "none";
	}
	selectLayer(e, t) {
		if (!e) {
			if (!this.getMap()) return;
			e = this.getMap().getLayers().item(this.getMap().getLayers().getLength() - 1);
		}
		this._selectedLayer = e, this.element.querySelector("input.ol-visibility:focus") && (this._focus = e), this.drawPanel(), t || this.dispatchEvent({
			type: "select",
			layer: e
		});
	}
	getSelection() {
		return this._selectedLayer;
	}
	setprogress_(e) {
		if (!e.layerswitcher_progress) {
			var t = 0, n = 0, r = function() {
				n === t ? (n = t = 0, Q.setStyle(e.layerswitcher_progress, { width: 0 })) : Q.setStyle(e.layerswitcher_progress, { width: (t / n * 100).toFixed(1) + "%" });
			};
			e.getSource().on("tileloadstart", function() {
				n++, r();
			}), e.getSource().on("tileloadend", function() {
				t++, r();
			}), e.getSource().on("tileloaderror", function() {
				t++, r();
			});
		}
	}
};
dh.prototype.tip = {
	up: "up/down",
	down: "down",
	info: "informations...",
	extent: "zoom to extent",
	trash: "remove layer",
	plus: "expand/shrink"
};
//#endregion
//#region node_modules/ol-ext/control/Button.js
var fh = class extends Te {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = (e.className || "") + " ol-button ol-unselectable ol-control", super({
			element: t,
			target: e.target
		});
		var n = this, r = this.button_ = document.createElement(/ol-text-button/.test(e.className) ? "div" : "button");
		for (var i in e.id ? r.setAttribute("id", e.id) : r.setAttribute("id", "ol-button-" + F(this)), this.button_.className = e.classButton || "", r.type = "button", e.title && (r.title = e.title), e.name && (r.name = e.name), e.html instanceof Element ? r.appendChild(e.html) : r.innerHTML = e.html || "", r.addEventListener("click", function(t) {
			t && t.preventDefault && (t.preventDefault(), t.stopPropagation()), e.handleClick && e.handleClick.call(n, t);
		}), t.appendChild(r), !e.title && r.firstElementChild && (r.title = r.firstElementChild.title), e.title && this.set("title", e.title), e.title && this.set("title", e.title), e.name && this.set("name", e.name), e.attributes || {}) this.button_.setAttribute(i, e.attributes[i]);
	}
	setVisible(e) {
		e ? Q.show(this.element) : Q.hide(this.element);
	}
	getDisable() {
		var e = this.element.querySelector("button");
		return e && e.disabled;
	}
	setDisable(e) {
		this.getDisable() != e && (this.element.querySelector("button").disabled = e);
	}
	setTitle(e) {
		this.button_.setAttribute("title", e);
	}
	setHtml(e) {
		Q.setHTML(this.button_, e);
	}
	getButtonElement() {
		return this.button_;
	}
}, ph = class extends fh {
	constructor(e) {
		e ||= {}, e.toggleFn && (e.onToggle = e.toggleFn), e.handleClick = function() {
			t.toggle(), e.onToggle && e.onToggle.call(t, t.getActive());
		}, e.className = (e.className || "") + " ol-toggle", super(e);
		var t = this;
		this.interaction_ = e.interaction, this.interaction_ && (this.interaction_.setActive(e.active), this.interaction_.on("change:active", function() {
			t.setActive(t.interaction_.getActive());
		})), this.set("title", e.title), this.set("autoActivate", e.autoActivate), e.bar && this.setSubBar(e.bar), this.setActive(e.active), this.setDisable(e.disable);
	}
	setMap(e) {
		!e && this.getMap() && (this.interaction_ && this.getMap().removeInteraction(this.interaction_), this.subbar_ && this.getMap().removeControl(this.subbar_)), super.setMap(e), e && (this.interaction_ && e.addInteraction(this.interaction_), this.subbar_ && e.addControl(this.subbar_));
	}
	getSubBar() {
		return this.subbar_;
	}
	setSubBar(e) {
		var t = this.getMap();
		t && this.subbar_ && t.removeControl(this.subbar_), this.subbar_ = e, e && (this.subbar_.setTarget(this.element), this.subbar_.element.classList.add("ol-option-bar"), t && t.addControl(this.subbar_), e.element.id && (this.getButtonElement().setAttribute("aria-controls", e.element.id), e.element.setAttribute("aria-labelledby", this.getButtonElement().id), this.on("change:active", function(e) {
			this.getButtonElement().setAttribute("aria-expanded", !!e.active);
		}.bind(this))));
	}
	getDisable() {
		var e = this.element.querySelector("button");
		return e && e.disabled;
	}
	setDisable(e) {
		this.getDisable() != e && (this.element.querySelector("button").disabled = e, e && this.getActive() && this.setActive(!1), this.dispatchEvent({
			type: "change:disable",
			key: "disable",
			oldValue: !e,
			disable: e
		}));
	}
	getActive() {
		return this.element.classList.contains("ol-active");
	}
	toggle() {
		this.getActive() ? this.setActive(!1) : this.setActive(!0);
	}
	setActive(e) {
		this.interaction_ && this.interaction_.setActive(e), this.subbar_ && this.subbar_.setActive(e), this.getActive() !== e && (e ? this.element.classList.add("ol-active") : this.element.classList.remove("ol-active"), this.button_.setAttribute("aria-pressed", e), this.dispatchEvent({
			type: "change:active",
			key: "active",
			oldValue: !e,
			active: e
		}));
	}
	setInteraction(e) {
		this.interaction_ = e;
	}
	getInteraction() {
		return this.interaction_;
	}
};
//#endregion
//#region src/main.js
async function mh() {
	let e = await fetch("/tileserver/session_id");
	if (!e.ok) throw Error("Failed to create TileServer session.");
	return (await e.json()).session_id;
}
async function hh(e) {
	let t = new FormData();
	if (t.append("slide_path", e), !(await fetch("/tileserver/slide", {
		method: "PUT",
		body: t
	})).ok) throw Error(`Failed to load slide: ${e}`);
	let n = await fetch("/tileserver/slide");
	if (!n.ok) throw Error("Failed to retrieve slide metadata.");
	return n.json();
}
function gh(e, t, n) {
	return new dd({
		url: `/tileserver/layer/slide/${e}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${n}`,
		size: t.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
}
var _h = document.getElementById("map"), vh = document.querySelector(".viewer-app"), yh = document.getElementById("viewer-panel"), bh = document.getElementById("viewer-panel-toggle"), xh = document.getElementById("current-slide"), Sh = document.getElementById("layer-editor"), Ch = document.getElementById("layer-editor-toggle"), wh = document.getElementById("layer-editor-list");
if (_h === null || vh === null) throw Error("The OpenLayers viewer could not be found.");
if (yh === null || bh === null || Sh === null || Ch === null || wh === null || xh === null) throw Error("The OpenLayers viewer controls could not be found.");
bh.addEventListener("click", () => {
	let e = yh.classList.toggle("hidden");
	bh.classList.toggle("active", !e);
}), Ch.addEventListener("click", () => {
	let e = Sh.classList.toggle("hidden");
	Ch.classList.toggle("active", !e);
});
var Th = JSON.parse(_h.dataset.layers ?? "[]"), Eh = null, Dh = Date.now(), Oh = Date.now(), kh = null, Ah = null, jh = {}, Mh = /* @__PURE__ */ new Set();
function Nh(e) {
	let t = e.split(/[\\/]/).pop() ?? e, n = t.lastIndexOf(".");
	return n <= 0 ? t : t.slice(0, n);
}
function Ph() {
	if (Ah === null) {
		xh.textContent = "No slide selected", xh.removeAttribute("title");
		return;
	}
	xh.textContent = Ah.split(/[\\/]/).pop() || Ah, xh.title = Ah;
}
var Fh = new URLSearchParams(window.location.search).get("slide");
if (Fh !== null) {
	Ah = Fh, Eh = await mh();
	let e = await hh(Fh);
	kh = e, Th = [{
		name: "slide",
		url: `/tileserver/layer/slide/${Eh}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${Dh}`,
		size: e.slide_dimensions,
		mpp: e.mpp[0]
	}];
} else Th.length === 0 && (Eh = await mh());
Ph();
var Ih = Th.map((e) => {
	let t = new dd({
		url: e.url,
		size: e.size,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	return new Lu({
		title: e.name,
		source: t
	});
}), Lh = Ih[0];
Lh === void 0 && (Lh = new Lu({ title: "slide" }), Ih.push(Lh)), Lh.setZIndex(0);
var Rh = Lh.getSource(), zh, Bh, Vh;
if (Rh !== null) {
	let e = Rh.getTileGrid();
	zh = e.getResolutions(), Bh = e.getExtent(), Vh = new qt({
		code: "ZoomifyProjection",
		units: "pixels",
		extent: Bh,
		metersPerUnit: Th[0].mpp * 1e-6,
		getPointResolution(e) {
			return e;
		}
	});
} else zh = [1], Bh = [
	0,
	-1,
	1,
	0
], Vh = new qt({
	code: "ZoomifyProjection",
	units: "pixels",
	extent: Bh,
	metersPerUnit: 1,
	getPointResolution(e) {
		return e;
	}
});
var Hh = .1;
function Uh(e) {
	let t = e[2] - e[0], n = e[3] - e[1], r = t * Hh, i = n * Hh;
	return [
		e[0] - r,
		e[1] - i,
		e[2] + r,
		e[3] + i
	];
}
er(Vh);
var Wh = new ua({
	projection: Vh,
	resolutions: zh,
	extent: Uh(Bh),
	constrainOnlyCenter: !0,
	smoothExtentConstraint: !0,
	smoothResolutionConstraint: !1,
	center: [.5, -.5],
	resolution: zh[0]
}), $ = new Rl({
	target: _h,
	layers: Ih,
	view: Wh
}), Gh = _h.querySelector(".ol-zoom"), Kh = _h.querySelector(".ol-zoom-out");
if (Gh === null || Kh === null) throw Error("The OpenLayers zoom control could not be found.");
var qh = document.createElement("div");
qh.className = "ol-zoom-level", Gh.insertBefore(qh, Kh);
function Jh() {
	let e = $.getView().getZoom();
	e !== void 0 && (qh.textContent = `${Number.isInteger(e) ? e.toString() : e.toFixed(1)}x`);
}
Jh();
var Yh = new Jl({
	units: "metric",
	minWidth: 100
});
$.addControl(Yh);
var Xh = new Lu();
Rh !== null && Xh.setSource(Rh);
var Zh = 300, Qh = 250;
function $h(e, t) {
	let n = [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2], r = t[2] - t[0], i = t[3] - t[1], a = Math.max(r / Zh, i / Qh), o = new ua({
		projection: e,
		center: n,
		resolution: a,
		resolutions: [a],
		constrainOnlyCenter: !0
	});
	return o.on("change:center", () => {
		let e = o.getCenter();
		e !== void 0 && (e[0] !== n[0] || e[1] !== n[1]) && o.setCenter(n);
	}), o;
}
var eg = document.createElement("span");
eg.className = "overview-toggle-icon", eg.innerHTML = "<i class=\"fas fa-chevron-up\"></i>";
var tg = document.createElement("span");
tg.className = "overview-toggle-icon", tg.innerHTML = "<i class=\"fas fa-chevron-down\"></i>";
var ng = new Wl({
	className: "ol-overviewmap ol-custom-overviewmap",
	layers: [Xh],
	collapsed: !1,
	collapsible: !0,
	collapseLabel: eg,
	label: tg,
	rotateWithView: !1,
	tipLabel: "Toggle overview map",
	view: $h(Vh, Bh)
});
$.addControl(ng);
var rg = ng.getOverviewMap(), ig = new wr({
	coordinateFormat: (e) => Ft([e[0], -e[1]], "{x}, {y}", 0),
	className: "ol-mouse-position",
	placeholder: "\xA0"
});
$.addControl(ig);
var ag = new va({
	autoHide: !1,
	className: "ol-rotate"
});
$.addControl(ag);
var og = new Oe({ source: vh });
$.addControl(og);
var sg = new dh();
$.addControl(sg);
var cg = 64, lg = 64, ug = new Bc({
	stroke: new zc({
		color: "rgba(0, 0, 0, 0.5)",
		width: 1
	}),
	text: new Kc({
		font: "12px Calibri,sans-serif",
		fill: new Ic({ color: "rgba(0, 0, 0, 1)" }),
		stroke: new zc({
			color: "rgba(255, 255, 255, 1)",
			width: 3
		})
	})
});
function dg(e) {
	return new md({
		projection: e,
		margin: lg,
		style: ug,
		spacing: cg,
		formatCoord: (e, t) => (e = t === "left" || t === "right" ? -Math.floor(e) : Math.floor(e), e >= 1e6 && (e = e.toExponential(3), e = e.replace("+", "")), e)
	});
}
var fg = dg(Vh), pg = cg, mg = lg;
function hg(e) {
	return new md({
		projection: e.getCode(),
		spacing: pg,
		margin: mg,
		style: ug,
		formatCoord(e, t) {
			let n = $.getView().calculateExtent($.getSize()), r = $.getView().getResolution(), i = n[0] + r * mg, a = n[3] - r * mg, o;
			if (o = t === "left" || t === "right" ? -(e - a) : e - i, o = Math.floor(o / r / pg), t === "left" || t === "right") {
				let e = "";
				do
					e += String.fromCharCode(65 + o % 26), o = Math.floor(o / 26);
				while (o > 0);
				return e.split("").reverse().join("");
			}
			return o;
		}
	});
}
var gg = hg(Vh), _g = new ph({
	html: "<i class=\"fas fa-ruler-combined\"></i>",
	className: "ol-graticule",
	title: "Toggle Graticule",
	onToggle(e) {
		_g.element.classList.toggle("active", e), e ? (vg.setActive(!1), vg.element.classList.remove("active"), gg.setMap(null), fg.setMap($)) : fg.setMap(null);
	}
});
$.addControl(_g);
var vg = new ph({
	html: "<i class=\"fas fa-border-all\"></i>",
	className: "ol-screen-space-graticule",
	title: "Toggle Screen Space Graticule",
	onToggle(e) {
		vg.element.classList.toggle("active", e), e ? (_g.setActive(!1), _g.element.classList.remove("active"), fg.setMap(null), gg.setMap($)) : gg.setMap(null);
	}
});
$.addControl(vg);
function yg(e) {
	let t = _h.querySelector(".ol-zoom-in"), n = _h.querySelector(".ol-zoom-out"), r = ag.element.querySelector("button"), i = _g.element.querySelector("button"), a = vg.element.querySelector("button");
	for (let o of [
		t,
		n,
		r,
		i,
		a
	]) o !== null && (o.disabled = !e);
	Yh.element.classList.toggle("viewer-control-hidden", !e), ig.element.classList.toggle("viewer-control-hidden", !e), ng.element.classList.toggle("viewer-control-hidden", !e), e || (_g.setActive(!1), vg.setActive(!1), _g.element.classList.remove("active"), vg.element.classList.remove("active"), fg.setMap(null), gg.setMap(null)), e && requestAnimationFrame(() => {
		rg.updateSize(), rg.renderSync();
	});
}
if (yg(Rh !== null), Rh !== null) {
	$.getView().fit(Bh);
	let e = Sg();
	e !== null && ($.getView().setCenter(e.center), $.getView().setZoom(e.zoom));
}
$.on("moveend", () => {
	Cg(), Jh();
});
function bg() {
	for (let e of Object.values(jh)) {
		e.setSource(null), $.removeLayer(e);
		let t = Ih.indexOf(e);
		t !== -1 && Ih.splice(t, 1);
	}
	for (let e of Object.keys(jh)) delete jh[e];
	Og();
}
async function xg() {
	if (!(await fetch("/tileserver/clear_overlays", { method: "PUT" })).ok) throw Error("Failed to clear overlays.");
	bg();
}
function Sg() {
	let e = new URLSearchParams(window.location.search), t = Number(e.get("x")), n = Number(e.get("y")), r = Number(e.get("zoom"));
	return e.get("x") === null || e.get("y") === null || e.get("zoom") === null || !Number.isFinite(t) || !Number.isFinite(n) || !Number.isFinite(r) ? null : {
		center: [t, n],
		zoom: r
	};
}
function Cg() {
	if (Ah === null) return;
	let e = $.getView(), t = e.getCenter(), n = e.getZoom();
	if (t === void 0 || n === void 0) return;
	let r = new URL(window.location.href);
	r.searchParams.set("slide", Ah), r.searchParams.set("x", t[0].toFixed(2)), r.searchParams.set("y", t[1].toFixed(2)), r.searchParams.set("zoom", n.toString());
	let i = r.searchParams.toString().replace(/%2F/gi, "/");
	window.history.replaceState({}, "", `${r.pathname}?${i}${r.hash}`);
}
async function wg() {
	if (Eh === null) throw Error("No TileServer session is available.");
	if (!(await fetch("/tileserver/slide", { method: "DELETE" })).ok) throw Error("Failed to remove the current slide.");
	bg(), Ah = null, kh = null, Th.length = 0, Ph(), Dh += 1, Oh += 1, Lh.setSource(null), Xh.setSource(null), Og();
	let e = [
		0,
		-1,
		1,
		0
	], t = [1], n = new qt({
		code: "ZoomifyProjectionEmpty",
		units: "pixels",
		extent: e,
		metersPerUnit: 1,
		getPointResolution(e) {
			return e;
		}
	});
	er(n);
	let r = new ua({
		projection: n,
		resolutions: t,
		constrainOnlyCenter: !0,
		center: [.5, -.5],
		resolution: t[0]
	});
	$.setView(r), rg.setView($h(n, e)), _g.setActive(!1), vg.setActive(!1), _g.element.classList.remove("active"), vg.element.classList.remove("active"), fg.setMap(null), gg.setMap(null), fg = dg(n), gg = hg(n), window.graticule = fg, window.screenSpaceGraticule = gg, window.projection = n, window.resolutions = t, window.extent = e, window.view = r;
	let i = new URL(window.location.href);
	i.search = "", i.hash = "", window.history.replaceState({}, "", i), yg(!1), Jh();
}
async function Tg(e) {
	if (Eh === null) throw Error("Dynamic slide switching requires a TileServer session.");
	bg();
	let t = await hh(e);
	Ah = e, Ph(), kh = t, Dh += 1;
	let n = gh(Eh, t, Dh), r = n.getTileGrid(), i = r.getExtent(), a = r.getResolutions(), o = new qt({
		code: "ZoomifyProjection",
		units: "pixels",
		extent: i,
		metersPerUnit: t.mpp[0] * 1e-6
	});
	er(o);
	let s = [(i[0] + i[2]) / 2, (i[1] + i[3]) / 2], c = new ua({
		projection: o,
		resolutions: a,
		extent: Uh(i),
		constrainOnlyCenter: !0,
		smoothExtentConstraint: !0,
		smoothResolutionConstraint: !1,
		center: s,
		resolution: a[0]
	});
	c.fit(i, { size: $.getSize() }), $.setView(c), rg.setView($h(o, i));
	let l = _g.getActive(), u = vg.getActive();
	fg.setMap(null), gg.setMap(null), fg = dg(o), gg = hg(o), l && fg.setMap($), u && gg.setMap($), Lh.setSource(n), Xh.setSource(n), Og(), window.graticule = fg, window.screenSpaceGraticule = gg, yg(!0), Cg(), Jh();
}
function Eg() {
	let e = [];
	Lh.getSource() !== null && e.push({
		id: "slide",
		name: Nh(Ah ?? "slide"),
		layer: Lh
	});
	for (let [t, n] of Object.entries(jh)) e.push({
		id: t,
		name: t,
		layer: n
	});
	return e.sort((e, t) => (t.layer.getZIndex() ?? 0) - (e.layer.getZIndex() ?? 0));
}
function Dg(e, t) {
	let n = Eg(), r = n.findIndex((t) => t.id === e);
	if (r === -1) return;
	let i = t === "up" ? r - 1 : r + 1;
	if (i < 0 || i >= n.length) return;
	let a = n[r].layer, o = n[i].layer, s = a.getZIndex() ?? 0, c = o.getZIndex() ?? 0;
	a.setZIndex(c), o.setZIndex(s), Og();
}
function Og() {
	wh.replaceChildren();
	let e = Eg();
	if (e.length === 0) {
		let e = document.createElement("div");
		e.className = "layer-editor-empty", e.textContent = "No layers loaded", wh.appendChild(e);
		return;
	}
	e.forEach(({ id: t, name: n, layer: r }, i) => {
		let a = document.createElement("div");
		a.className = "layer-editor-item";
		let o = document.createElement("div");
		o.className = "layer-editor-item-header";
		let s = document.createElement("input");
		s.className = "layer-editor-visibility", s.type = "checkbox", s.checked = r.getVisible(), s.title = `Toggle ${n}`, s.addEventListener("change", () => {
			r.setVisible(s.checked);
		});
		let c = document.createElement("span");
		c.className = "layer-editor-name", c.textContent = n;
		let l = document.createElement("div");
		l.className = "layer-editor-order";
		let u = document.createElement("button");
		u.type = "button", u.title = "Move layer up", u.innerHTML = "<i class=\"fas fa-chevron-up\"></i>", u.disabled = i === 0, u.addEventListener("click", () => {
			Dg(t, "up");
		});
		let d = document.createElement("button");
		d.type = "button", d.title = "Move layer down", d.innerHTML = "<i class=\"fas fa-chevron-down\"></i>", d.disabled = i === e.length - 1, d.addEventListener("click", () => {
			Dg(t, "down");
		}), l.append(u, d), o.append(s, c, l);
		let f = document.createElement("div");
		f.className = "layer-editor-opacity";
		let p = document.createElement("input");
		p.className = "layer-editor-slider", p.type = "range", p.min = "0", p.max = "1", p.step = "0.05", p.value = r.getOpacity().toString();
		let m = document.createElement("span");
		m.className = "layer-editor-value", m.textContent = `${Math.round(r.getOpacity() * 100)}%`, p.addEventListener("input", () => {
			let e = Number(p.value);
			r.setOpacity(e), m.textContent = `${Math.round(e * 100)}%`;
		}), f.append(p, m), a.append(o, f), wh.appendChild(a);
	});
}
Og();
async function kg(e) {
	if (Eh === null || kh === null) throw Error("Dynamic overlay loading requires a loaded slide.");
	let t = e.split(".").pop().toLowerCase();
	if (t === "npy" || t === "mha") throw Error("Registration overlays are not supported yet.");
	let n = [
		"db",
		"dat",
		"geojson"
	].includes(t), r = Nh(e);
	if (r === "slide") throw Error("The overlay name \"slide\" is reserved.");
	let i = new FormData();
	i.append("overlay_path", e), i.append("layer_name", r);
	let a = await fetch("/tileserver/overlay", {
		method: "PUT",
		body: i
	});
	if (!a.ok) throw Error(`Failed to load overlay: ${e}`);
	let o = await a.json();
	n ? Mh.add(r) : Mh.delete(r), Oh += 1;
	let s = new dd({
		url: `/tileserver/layer/${encodeURIComponent(r)}/${Eh}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${Oh}`,
		size: kh.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	if (jh[r] !== void 0) jh[r].setSource(s), jh[r].setVisible(!0);
	else {
		let e = [Lh, ...Object.values(jh)], t = Math.max(...e.map((e) => e.getZIndex() ?? 0)), n = new Lu({
			title: r,
			source: s,
			opacity: .75
		});
		n.setZIndex(t + 1), jh[r] = n, $.addLayer(n), Ih.push(n);
	}
	return Og(), o;
}
async function Ag(e) {
	let t = jh[e];
	if (t === void 0) throw Error(`Overlay is not loaded: ${e}`);
	if (!(await fetch(`/tileserver/overlay/${encodeURIComponent(e)}`, { method: "DELETE" })).ok) throw Error(`Failed to remove overlay: ${e}`);
	$.removeLayer(t);
	let n = Ih.indexOf(t);
	n !== -1 && Ih.splice(n, 1), Mh.delete(e), delete jh[e], Og();
}
async function jg(e) {
	if (Mh.size === 0) throw Error("No annotation overlay is loaded.");
	let t = new FormData();
	if (t.append("cmap", JSON.stringify({
		keys: Object.keys(e),
		values: Object.values(e)
	})), !(await fetch("/tileserver/cmap", {
		method: "PUT",
		body: t
	})).ok) throw Error("Failed to update annotation colours.");
	Oh += 1;
	for (let e of Mh) {
		let t = jh[e];
		if (t === void 0) continue;
		let n = new dd({
			url: `/tileserver/layer/${encodeURIComponent(e)}/${Eh}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${Oh}`,
			size: kh.slide_dimensions,
			crossOrigin: "anonymous",
			zDirection: -1
		});
		t.setSource(n);
	}
}
Object.assign(window, {
	clearOverlays: xg,
	extent: Bh,
	fullscreen: og,
	graticule: fg,
	graticuleToggle: _g,
	layerSwitcher: sg,
	layers: Ih,
	layersData: Th,
	loadOverlay: kg,
	map: $,
	mousePositionControl: ig,
	overlayLayers: jh,
	overviewMapControl: ng,
	projection: Vh,
	removeOverlay: Ag,
	removeSlide: wg,
	resolutions: zh,
	rotate: ag,
	scaleLineControl: Yh,
	screenSpaceGraticule: gg,
	screenSpaceGraticuleToggle: vg,
	setAnnotationColors: jg,
	switchSlide: Tg,
	view: Wh
});
//#endregion
