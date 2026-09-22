//#region node_modules/ol/CollectionEventType.js
var e = {
	ADD: "add",
	REMOVE: "remove"
}, t = { PROPERTYCHANGE: "propertychange" };
//#endregion
//#region node_modules/ol/obj.js
function n(e) {
	for (let t in e) delete e[t];
}
function r(e) {
	let t;
	for (t in e) return !1;
	return !t;
}
//#endregion
//#region node_modules/ol/events.js
function i(e, t, n, r, i) {
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
function a(e, t, n, r) {
	return i(e, t, n, r, !0);
}
function o(e) {
	e && e.target && (e.target.removeEventListener(e.type, e.listener), n(e));
}
//#endregion
//#region node_modules/ol/events/EventType.js
var s = {
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
}, c = class {
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
function l(e, t, n) {
	let r, i;
	n ||= u;
	let a = 0, o = e.length, s = !1;
	for (; a < o;) r = a + (o - a >> 1), i = +n(e[r], t), i < 0 ? a = r + 1 : (o = r, s = !i);
	return s ? a : ~a;
}
function u(e, t) {
	return e > t ? 1 : e < t ? -1 : 0;
}
function d(e, t) {
	return e < t ? 1 : e > t ? -1 : 0;
}
function f(e, t, n) {
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
function p(e, t, n) {
	for (; t < n;) {
		let r = e[t];
		e[t] = e[n], e[n] = r, ++t, --n;
	}
}
function m(e, t) {
	let n = Array.isArray(t) ? t : [t], r = n.length;
	for (let t = 0; t < r; t++) e[e.length] = n[t];
}
function h(e, t) {
	let n = e.length;
	if (n !== t.length) return !1;
	for (let r = 0; r < n; r++) if (e[r] !== t[r]) return !1;
	return !0;
}
function g(e, t, n) {
	let r = t || u;
	return e.every(function(t, i) {
		if (i === 0) return !0;
		let a = r(e[i - 1], t);
		return !(a > 0 || n && a === 0);
	});
}
//#endregion
//#region node_modules/ol/functions.js
function _() {
	return !0;
}
function v() {
	return !1;
}
function y() {}
function b(e) {
	let t, n, r;
	return function() {
		let i = Array.prototype.slice.call(arguments);
		return (!n || this !== r || !h(i, n)) && (r = this, n = i, t = e.apply(this, arguments)), t;
	};
}
function x(e) {
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
var S = class {
	constructor(e) {
		this.propagationStopped, this.defaultPrevented, this.type = e, this.target = null;
	}
	preventDefault() {
		this.defaultPrevented = !0;
	}
	stopPropagation() {
		this.propagationStopped = !0;
	}
}, C = class extends c {
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
		let i = t ? new S(e) : e;
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
			for (delete o[n]; e--;) this.removeEventListener(n, y);
			delete a[n];
		}
		return s;
	}
	disposeInternal() {
		this.listeners_ && n(this.listeners_);
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
		r !== -1 && (this.pendingRemovals_ && e in this.pendingRemovals_ ? (n[r] = y, ++this.pendingRemovals_[e]) : (n.splice(r, 1), n.length === 0 && delete this.listeners_[e]));
	}
}, w = class extends C {
	constructor() {
		super(), this.on = this.onInternal, this.once = this.onceInternal, this.un = this.unInternal, this.revision_ = 0;
	}
	changed() {
		++this.revision_, this.dispatchEvent(s.CHANGE);
	}
	getRevision() {
		return this.revision_;
	}
	onInternal(e, t) {
		if (Array.isArray(e)) {
			let n = e.length, r = Array(n);
			for (let a = 0; a < n; ++a) r[a] = i(this, e[a], t);
			return r;
		}
		return i(this, e, t);
	}
	onceInternal(e, t) {
		let n;
		if (Array.isArray(e)) {
			let r = e.length;
			n = Array(r);
			for (let i = 0; i < r; ++i) n[i] = a(this, e[i], t);
		} else n = a(this, e, t);
		return t.ol_key = n, n;
	}
	unInternal(e, t) {
		let n = t.ol_key;
		if (n) T(n);
		else if (Array.isArray(e)) for (let n = 0, r = e.length; n < r; ++n) this.removeEventListener(e[n], t);
		else this.removeEventListener(e, t);
	}
};
w.prototype.on, w.prototype.once, w.prototype.un;
function T(e) {
	if (Array.isArray(e)) for (let t = 0, n = e.length; t < n; ++t) o(e[t]);
	else o(e);
}
//#endregion
//#region node_modules/ol/util.js
function E() {
	throw Error("Unimplemented abstract method.");
}
var D = 0;
function O(e) {
	return e.ol_uid ||= String(++D);
}
//#endregion
//#region node_modules/ol/Object.js
var k = class extends S {
	constructor(e, t, n) {
		super(e), this.key = t, this.oldValue = n;
	}
}, A = class extends w {
	constructor(e) {
		super(), this.on, this.once, this.un, O(this), this.values_ = null, e !== void 0 && this.setProperties(e);
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
	notify(e, n) {
		let r;
		r = `change:${e}`, this.hasListener(r) && this.dispatchEvent(new k(r, e, n)), r = t.PROPERTYCHANGE, this.hasListener(r) && this.dispatchEvent(new k(r, e, n));
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
			delete this.values_[e], r(this.values_) && (this.values_ = null), t || this.notify(e, n);
		}
	}
}, j = { LENGTH: "length" }, M = class extends S {
	constructor(e, t, n) {
		super(e), this.element = t, this.index = n;
	}
}, N = class extends A {
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
		return this.get(j.LENGTH);
	}
	insertAt(t, n) {
		if (t < 0 || t > this.getLength()) throw Error("Index out of bounds: " + t);
		this.unique_ && this.assertUnique_(n), this.array_.splice(t, 0, n), this.updateLength_(), this.dispatchEvent(new M(e.ADD, n, t));
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
	removeAt(t) {
		if (t < 0 || t >= this.getLength()) return;
		let n = this.array_[t];
		return this.array_.splice(t, 1), this.updateLength_(), this.dispatchEvent(new M(e.REMOVE, n, t)), n;
	}
	setAt(t, n) {
		if (t >= this.getLength()) {
			this.insertAt(t, n);
			return;
		}
		if (t < 0) throw Error("Index out of bounds: " + t);
		this.unique_ && this.assertUnique_(n, t);
		let r = this.array_[t];
		this.array_[t] = n, this.dispatchEvent(new M(e.REMOVE, r, t)), this.dispatchEvent(new M(e.ADD, n, t));
	}
	updateLength_() {
		this.set(j.LENGTH, this.array_.length);
	}
	assertUnique_(e, t) {
		let n = this.array_;
		for (let r = 0, i = n.length; r < i; ++r) if (n[r] === e && r !== t) throw Error("Duplicate item added to a unique collection");
	}
}, ee = "ol-hidden", P = "ol-selectable", F = "ol-unselectable", te = "ol-unsupported", ne = "ol-control", re = "ol-collapsed", ie = new RegExp([
	"^\\s*(?=(?:(?:[-a-z]+\\s*){0,2}(italic|oblique))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(small-caps))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)",
	"(?:(?:normal|\\1|\\2|\\3)\\s*){0,3}((?:xx?-)?",
	"(?:small|large)|medium|smaller|larger|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx]))",
	"(?:\\s*\\/\\s*(normal|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx])?))",
	"?\\s*([-,\\\"\\'\\sa-z0-9]+?)\\s*$"
].join(""), "i"), ae = [
	"style",
	"variant",
	"weight",
	"size",
	"lineHeight",
	"family"
], oe = {
	normal: 400,
	bold: 700
}, se = function(e) {
	let t = e.match(ie);
	if (!t) return null;
	let n = {
		lineHeight: "normal",
		size: "1.2em",
		style: "normal",
		weight: "400",
		variant: "normal"
	};
	for (let e = 0, r = ae.length; e < r; ++e) {
		let r = t[e + 1];
		r !== void 0 && (n[ae[e]] = typeof r == "string" ? r.trim() : r);
	}
	return isNaN(Number(n.weight)) && n.weight in oe && (n.weight = oe[n.weight]), n.families = n.family.split(/,\s?/).map((e) => e.trim().replace(/^['"]|['"]$/g, "")), n;
}, ce = typeof navigator < "u" && navigator.userAgent !== void 0 ? navigator.userAgent.toLowerCase() : "", le = ce.includes("safari") && !ce.includes("chrom") && (ce.includes("version/15.4") || /cpu (os|iphone os) 15_4 like mac os x/.test(ce)), ue = ce.includes("webkit") && !ce.includes("edge"), de = ce.includes("macintosh"), fe = typeof devicePixelRatio < "u" ? devicePixelRatio : 1, pe = typeof WorkerGlobalScope < "u" && typeof OffscreenCanvas < "u" && self instanceof WorkerGlobalScope, me = typeof Image < "u" && Image.prototype.decode, he = (function() {
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
function I(e, t, n, r) {
	let i;
	return i = n && n.length ? n.shift() : pe ? new class extends OffscreenCanvas {
		style = {};
	}(e ?? 300, t ?? 150) : document.createElement("canvas"), e && (i.width = e), t && (i.height = t), i.getContext("2d", r);
}
var ge;
function _e() {
	return ge ||= I(1, 1), ge;
}
function ve(e) {
	let t = e.canvas;
	t.width = 1, t.height = 1, e.clearRect(0, 0, 1, 1);
}
function ye(e) {
	let t = e.offsetWidth, n = getComputedStyle(e);
	return t += parseInt(n.marginLeft, 10) + parseInt(n.marginRight, 10), t;
}
function be(e) {
	let t = e.offsetHeight, n = getComputedStyle(e);
	return t += parseInt(n.marginTop, 10) + parseInt(n.marginBottom, 10), t;
}
function xe(e, t) {
	let n = t.parentNode;
	n && n.replaceChild(e, t);
}
function Se(e) {
	for (; e.lastChild;) e.lastChild.remove();
}
function Ce(e, t) {
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
function we() {
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
function Te(e) {
	return typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof OffscreenCanvas < "u" && e instanceof OffscreenCanvas;
}
//#endregion
//#region node_modules/ol/MapEventType.js
var Ee = {
	POSTRENDER: "postrender",
	MOVESTART: "movestart",
	MOVEEND: "moveend",
	LOADSTART: "loadstart",
	LOADEND: "loadend"
}, De = class extends A {
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
		for (let e = 0, t = this.listenerKeys.length; e < t; ++e) o(this.listenerKeys[e]);
		if (this.listenerKeys.length = 0, this.map_ = e, e) {
			let t = this.target_ ?? e.getOverlayContainerStopEvent();
			this.element && t.appendChild(this.element), this.render !== y && this.listenerKeys.push(i(e, Ee.POSTRENDER, this.render, this)), e.render();
		}
	}
	render(e) {}
	setTarget(e) {
		this.target_ = typeof e == "string" ? document.getElementById(e) : e;
	}
}, Oe = class extends De {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		}), this.ulElement_ = document.createElement("ul"), this.collapsed_ = e.collapsed === void 0 || e.collapsed, this.userCollapsed_ = this.collapsed_, this.overrideCollapsible_ = e.collapsible !== void 0, this.collapsible_ = e.collapsible === void 0 || e.collapsible, this.collapsible_ || (this.collapsed_ = !1), this.attributions_ = e.attributions;
		let t = e.className === void 0 ? "ol-attribution" : e.className, n = e.tipLabel === void 0 ? "Attributions" : e.tipLabel, r = e.expandClassName === void 0 ? t + "-expand" : e.expandClassName, i = e.collapseLabel === void 0 ? "›" : e.collapseLabel, a = e.collapseClassName === void 0 ? t + "-collapse" : e.collapseClassName;
		typeof i == "string" ? (this.collapseLabel_ = document.createElement("span"), this.collapseLabel_.textContent = i, this.collapseLabel_.className = a) : this.collapseLabel_ = i;
		let o = e.label === void 0 ? "i" : e.label;
		typeof o == "string" ? (this.label_ = document.createElement("span"), this.label_.textContent = o, this.label_.className = r) : this.label_ = o;
		let c = this.collapsible_ && !this.collapsed_ ? this.collapseLabel_ : this.label_;
		this.toggleButton_ = document.createElement("button"), this.toggleButton_.setAttribute("type", "button"), this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_)), this.toggleButton_.title = n, this.toggleButton_.appendChild(c), this.toggleButton_.addEventListener(s.CLICK, this.handleClick_.bind(this), !1);
		let l = t + " " + F + " " + ne + (this.collapsed_ && this.collapsible_ ? " " + re : "") + (this.collapsible_ ? "" : " ol-uncollapsible"), u = this.element;
		u.className = l, u.appendChild(this.toggleButton_), u.appendChild(this.ulElement_), this.renderedAttributions_ = [], this.renderedVisible_ = !0;
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
		let t = await Promise.all(this.collectSourceAttributions_(e).map((e) => x(() => e))), n = t.length > 0;
		if (this.renderedVisible_ != n && (this.element.style.display = n ? "" : "none", this.renderedVisible_ = n), !h(t, this.renderedAttributions_)) {
			Se(this.ulElement_);
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
		this.element.classList.toggle(re), this.collapsed_ ? xe(this.collapseLabel_, this.label_) : xe(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_, this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_));
	}
	getCollapsible() {
		return this.collapsible_;
	}
	setCollapsible(e) {
		this.collapsible_ !== e && (this.collapsible_ = e, this.element.classList.toggle("ol-uncollapsible"), this.userCollapsed_ && this.handleToggle_());
	}
	setCollapsed(e) {
		this.userCollapsed_ = e, this.collapsible_ && this.collapsed_ !== e && this.handleToggle_();
	}
	getCollapsed() {
		return this.collapsed_;
	}
	render(e) {
		this.updateElement_(e.frameState);
	}
};
//#endregion
//#region node_modules/ol/easing.js
function ke(e) {
	return e ** 3;
}
function Ae(e) {
	return 1 - ke(1 - e);
}
function je(e) {
	return 3 * e * e - 2 * e * e * e;
}
function Me(e) {
	return e;
}
//#endregion
//#region node_modules/ol/control/Rotate.js
var Ne = class extends De {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		});
		let t = e.className === void 0 ? "ol-rotate" : e.className, n = e.label === void 0 ? "⇧" : e.label, r = e.compassClassName === void 0 ? "ol-compass" : e.compassClassName;
		this.label_ = null, typeof n == "string" ? (this.label_ = document.createElement("span"), this.label_.className = r, this.label_.textContent = n) : (this.label_ = n, this.label_.classList.add(r));
		let i = e.tipLabel ? e.tipLabel : "Reset rotation", a = document.createElement("button");
		a.className = t + "-reset", a.setAttribute("type", "button"), a.title = i, a.appendChild(this.label_), a.addEventListener(s.CLICK, this.handleClick_.bind(this), !1);
		let o = t + " " + F + " " + ne, c = this.element;
		c.className = o, c.appendChild(a), this.callResetNorth_ = e.resetNorth ? e.resetNorth : void 0, this.duration_ = e.duration === void 0 ? 250 : e.duration, this.autoHide_ = e.autoHide === void 0 || e.autoHide, this.rotation_ = void 0, this.autoHide_ && this.element.classList.add(ee);
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
			easing: Ae
		}) : e.setRotation(0));
	}
	render(e) {
		let t = e.frameState;
		if (!t) return;
		let n = t.viewState.rotation;
		if (n != this.rotation_) {
			let e = "rotate(" + n + "rad)";
			if (this.autoHide_) {
				let e = this.element.classList.contains(ee);
				!e && n === 0 ? this.element.classList.add(ee) : e && n !== 0 && this.element.classList.remove(ee);
			}
			this.label_.style.transform = e;
		}
		this.rotation_ = n;
	}
}, Pe = class extends De {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			target: e.target
		});
		let t = e.className === void 0 ? "ol-zoom" : e.className, n = e.delta === void 0 ? 1 : e.delta, r = e.zoomInClassName === void 0 ? t + "-in" : e.zoomInClassName, i = e.zoomOutClassName === void 0 ? t + "-out" : e.zoomOutClassName, a = e.zoomInLabel === void 0 ? "+" : e.zoomInLabel, o = e.zoomOutLabel === void 0 ? "–" : e.zoomOutLabel, c = e.zoomInTipLabel === void 0 ? "Zoom in" : e.zoomInTipLabel, l = e.zoomOutTipLabel === void 0 ? "Zoom out" : e.zoomOutTipLabel, u = document.createElement("button");
		u.className = r, u.setAttribute("type", "button"), u.title = c, u.appendChild(typeof a == "string" ? document.createTextNode(a) : a), u.addEventListener(s.CLICK, this.handleClick_.bind(this, n), !1);
		let d = document.createElement("button");
		d.className = i, d.setAttribute("type", "button"), d.title = l, d.appendChild(typeof o == "string" ? document.createTextNode(o) : o), d.addEventListener(s.CLICK, this.handleClick_.bind(this, -n), !1);
		let f = t + " " + F + " " + ne, p = this.element;
		p.className = f, p.appendChild(u), p.appendChild(d), this.duration_ = e.duration === void 0 ? 250 : e.duration;
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
				easing: Ae
			})) : t.setZoom(r);
		}
	}
};
//#endregion
//#region node_modules/ol/control/defaults.js
function Fe(e) {
	e ||= {};
	let t = new N();
	return (e.zoom === void 0 || e.zoom) && t.push(new Pe(e.zoomOptions)), (e.rotate === void 0 || e.rotate) && t.push(new Ne(e.rotateOptions)), (e.attribution === void 0 || e.attribution) && t.push(new Oe(e.attributionOptions)), t;
}
//#endregion
//#region node_modules/ol/extent/Relationship.js
var Ie = {
	UNKNOWN: 0,
	INTERSECTING: 1,
	ABOVE: 2,
	RIGHT: 4,
	BELOW: 8,
	LEFT: 16
};
//#endregion
//#region node_modules/ol/extent.js
function Le(e) {
	let t = Ke();
	for (let n = 0, r = e.length; n < r; ++n) $e(t, e[n]);
	return t;
}
function Re(e, t, n) {
	return qe(Math.min.apply(null, e), Math.min.apply(null, t), Math.max.apply(null, e), Math.max.apply(null, t), n);
}
function ze(e, t, n) {
	return n ? (n[0] = e[0] - t, n[1] = e[1] - t, n[2] = e[2] + t, n[3] = e[3] + t, n) : [
		e[0] - t,
		e[1] - t,
		e[2] + t,
		e[3] + t
	];
}
function Be(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e.slice();
}
function Ve(e, t, n) {
	let r, i;
	return r = t < e[0] ? e[0] - t : e[2] < t ? t - e[2] : 0, i = n < e[1] ? e[1] - n : e[3] < n ? n - e[3] : 0, r * r + i * i;
}
function He(e, t) {
	return We(e, t[0], t[1]);
}
function Ue(e, t) {
	return e[0] <= t[0] && t[2] <= e[2] && e[1] <= t[1] && t[3] <= e[3];
}
function We(e, t, n) {
	return e[0] <= t && t <= e[2] && e[1] <= n && n <= e[3];
}
function Ge(e, t) {
	let n = e[0], r = e[1], i = e[2], a = e[3], o = t[0], s = t[1], c = Ie.UNKNOWN;
	return o < n ? c |= Ie.LEFT : o > i && (c |= Ie.RIGHT), s < r ? c |= Ie.BELOW : s > a && (c |= Ie.ABOVE), c === Ie.UNKNOWN && (c = Ie.INTERSECTING), c;
}
function Ke() {
	return [
		Infinity,
		Infinity,
		-Infinity,
		-Infinity
	];
}
function qe(e, t, n, r, i) {
	return i ? (i[0] = e, i[1] = t, i[2] = n, i[3] = r, i) : [
		e,
		t,
		n,
		r
	];
}
function Je(e) {
	return qe(Infinity, Infinity, -Infinity, -Infinity, e);
}
function Ye(e, t) {
	let n = e[0], r = e[1];
	return qe(n, r, n, r, t);
}
function Xe(e, t, n, r, i) {
	return et(Je(i), e, t, n, r);
}
function Ze(e, t) {
	return e[0] == t[0] && e[2] == t[2] && e[1] == t[1] && e[3] == t[3];
}
function Qe(e, t) {
	return t[0] < e[0] && (e[0] = t[0]), t[2] > e[2] && (e[2] = t[2]), t[1] < e[1] && (e[1] = t[1]), t[3] > e[3] && (e[3] = t[3]), e;
}
function $e(e, t) {
	t[0] < e[0] && (e[0] = t[0]), t[0] > e[2] && (e[2] = t[0]), t[1] < e[1] && (e[1] = t[1]), t[1] > e[3] && (e[3] = t[1]);
}
function et(e, t, n, r, i) {
	for (; n < r; n += i) tt(e, t[n], t[n + 1]);
	return e;
}
function tt(e, t, n) {
	e[0] = Math.min(e[0], t), e[1] = Math.min(e[1], n), e[2] = Math.max(e[2], t), e[3] = Math.max(e[3], n);
}
function nt(e, t) {
	let n;
	return n = t(it(e)), n || (n = t(at(e)), n) || (n = t(mt(e)), n) || (n = t(pt(e)), n) ? n : !1;
}
function rt(e) {
	let t = 0;
	return gt(e) || (t = L(e) * ut(e)), t;
}
function it(e) {
	return [e[0], e[1]];
}
function at(e) {
	return [e[2], e[1]];
}
function ot(e) {
	return [(e[0] + e[2]) / 2, (e[1] + e[3]) / 2];
}
function st(e, t) {
	let n;
	if (t === "bottom-left") n = it(e);
	else if (t === "bottom-right") n = at(e);
	else if (t === "top-left") n = pt(e);
	else if (t === "top-right") n = mt(e);
	else throw Error("Invalid corner");
	return n;
}
function ct(e, t, n, r, i) {
	let [a, o, s, c, l, u, d, f] = lt(e, t, n, r);
	return qe(Math.min(a, s, l, d), Math.min(o, c, u, f), Math.max(a, s, l, d), Math.max(o, c, u, f), i);
}
function lt(e, t, n, r) {
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
function ut(e) {
	return e[3] - e[1];
}
function dt(e, t, n) {
	let r = n || Ke();
	return ht(e, t) ? (r[0] = e[0] > t[0] ? e[0] : t[0], r[1] = e[1] > t[1] ? e[1] : t[1], r[2] = e[2] < t[2] ? e[2] : t[2], r[3] = e[3] < t[3] ? e[3] : t[3]) : Je(r), r;
}
function ft(e, t) {
	if (!ht(e, t)) return [e.slice()];
	if (Ue(t, e)) return [];
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
function pt(e) {
	return [e[0], e[3]];
}
function mt(e) {
	return [e[2], e[3]];
}
function L(e) {
	return e[2] - e[0];
}
function ht(e, t) {
	return e[0] <= t[2] && e[2] >= t[0] && e[1] <= t[3] && e[3] >= t[1];
}
function gt(e) {
	return e[2] < e[0] || e[3] < e[1];
}
function _t(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e;
}
function vt(e, t) {
	let n = (e[2] - e[0]) / 2 * (t - 1), r = (e[3] - e[1]) / 2 * (t - 1);
	e[0] -= n, e[2] += n, e[1] -= r, e[3] += r;
}
function yt(e, t, n) {
	let r = !1, i = Ge(e, t), a = Ge(e, n);
	if (i === Ie.INTERSECTING || a === Ie.INTERSECTING) r = !0;
	else {
		let o = e[0], s = e[1], c = e[2], l = e[3], u = t[0], d = t[1], f = n[0], p = n[1], m = (p - d) / (f - u), h, g;
		a & Ie.ABOVE && !(i & Ie.ABOVE) && (h = f - (p - l) / m, r = h >= o && h <= c), !r && a & Ie.RIGHT && !(i & Ie.RIGHT) && (g = p - (f - c) * m, r = g >= s && g <= l), !r && a & Ie.BELOW && !(i & Ie.BELOW) && (h = f - (p - s) / m, r = h >= o && h <= c), !r && a & Ie.LEFT && !(i & Ie.LEFT) && (g = p - (f - o) * m, r = g >= s && g <= l);
	}
	return r;
}
function bt(e, t, n, r) {
	if (gt(e)) return Je(n);
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
	return Re(a, o, n);
}
function xt(e, t) {
	let n = t.getExtent(), r = ot(e);
	if (t.canWrapX() && (r[0] < n[0] || r[0] >= n[2])) {
		let t = L(n), i = Math.floor((r[0] - n[0]) / t) * t;
		e[0] -= i, e[2] -= i;
	}
	return e;
}
function St(e, t, n) {
	if (t.canWrapX()) {
		let r = t.getExtent();
		if (!isFinite(e[0]) || !isFinite(e[2])) return [[
			r[0],
			e[1],
			r[2],
			e[3]
		]];
		xt(e, t);
		let i = L(r);
		if (L(e) > i && !n) return [[
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
function Ct(e, t) {
	let n = [e];
	for (let e = 0, r = t.length; e < r && n.length > 0; ++e) {
		let r = [];
		for (let i = 0, a = n.length; i < a; ++i) r.push(...ft(n[i], t[e]));
		n = r;
	}
	return n;
}
//#endregion
//#region node_modules/ol/console.js
var wt = {
	info: 1,
	warn: 2,
	error: 3,
	none: 4
}, Tt = wt.info;
function Et(...e) {
	Tt > wt.warn || console.warn(...e);
}
//#endregion
//#region node_modules/ol/math.js
function Dt(e, t, n) {
	return Math.min(Math.max(e, t), n);
}
function Ot(e, t, n, r, i, a) {
	let o = i - n, s = a - r;
	if (o !== 0 || s !== 0) {
		let c = ((e - n) * o + (t - r) * s) / (o * o + s * s);
		c > 1 ? (n = i, r = a) : c > 0 && (n += o * c, r += s * c);
	}
	return kt(e, t, n, r);
}
function kt(e, t, n, r) {
	let i = n - e, a = r - t;
	return i * i + a * a;
}
function At(e) {
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
function jt(e) {
	return e * 180 / Math.PI;
}
function Mt(e) {
	return e * Math.PI / 180;
}
function Nt(e, t) {
	let n = e % t;
	return n * t < 0 ? n + t : n;
}
function Pt(e, t, n) {
	return e + n * (t - e);
}
function Ft(e, t) {
	let n = 10 ** t;
	return Math.round(e * n) / n;
}
function It(e, t) {
	return Math.floor(Ft(e, t));
}
function Lt(e, t) {
	return Math.ceil(Ft(e, t));
}
function Rt(e, t, n) {
	if (e >= t && e < n) return e;
	let r = n - t;
	return ((e - t) % r + r) % r + t;
}
//#endregion
//#region node_modules/ol/coordinate.js
function zt(e, t) {
	return e[0] += +t[0], e[1] += +t[1], e;
}
function Bt(e, t, n) {
	return e ? t.replace("{x}", e[0].toFixed(n)).replace("{y}", e[1].toFixed(n)) : "";
}
function Vt(e, t) {
	let n = !0;
	for (let r = e.length - 1; r >= 0; --r) if (e[r] != t[r]) {
		n = !1;
		break;
	}
	return n;
}
function Ht(e, t) {
	let n = Math.cos(t), r = Math.sin(t), i = e[0] * n - e[1] * r, a = e[1] * n + e[0] * r;
	return e[0] = i, e[1] = a, e;
}
function Ut(e, t) {
	return e[0] *= t, e[1] *= t, e;
}
function Wt(e, t) {
	if (t.canWrapX()) {
		let n = L(t.getExtent()), r = Gt(e, t, n);
		r && (e[0] -= r * n);
	}
	return e;
}
function Gt(e, t, n) {
	let r = t.getExtent(), i = 0;
	return t.canWrapX() && (e[0] < r[0] || e[0] > r[2]) && (n ||= L(r), i = Math.floor((e[0] - r[0]) / n)), i;
}
function Kt(e, t, n) {
	let r = Math.sqrt((t[0] - e[0]) * (t[0] - e[0]) + (t[1] - e[1]) * (t[1] - e[1])), i = [(t[0] - e[0]) / r, (t[1] - e[1]) / r], a = [-i[1], i[0]], o = Math.sqrt((n[0] - e[0]) * (n[0] - e[0]) + (n[1] - e[1]) * (n[1] - e[1])), s = [(n[0] - e[0]) / o, (n[1] - e[1]) / o], c = r === 0 || o === 0 ? 0 : Math.acos(Dt(s[0] * i[0] + s[1] * i[1], -1, 1));
	return c = Math.max(c, 1e-5), s[0] * a[0] + s[1] * a[1] > 0 ? c : Math.PI * 2 - c;
}
//#endregion
//#region node_modules/ol/proj/Units.js
var qt = {
	radians: 6370997 / (2 * Math.PI),
	degrees: 2 * Math.PI * 6370997 / 360,
	ft: .3048,
	m: 1,
	"us-ft": 1200 / 3937
}, Jt = class {
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
		return this.metersPerUnit_ || qt[this.units_];
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
}, Yt = 6378137, Xt = Math.PI * Yt, Zt = [
	-Xt,
	-Xt,
	Xt,
	Xt
], Qt = [
	-180,
	-85,
	180,
	85
], $t = Yt * Math.log(Math.tan(Math.PI / 2)), en = class extends Jt {
	constructor(e) {
		super({
			code: e,
			units: "m",
			extent: Zt,
			global: !0,
			worldExtent: Qt,
			getPointResolution: function(e, t) {
				return e / Math.cosh(t[1] / Yt);
			}
		});
	}
}, tn = [
	new en("EPSG:3857"),
	new en("EPSG:102100"),
	new en("EPSG:102113"),
	new en("EPSG:900913"),
	new en("http://www.opengis.net/def/crs/EPSG/0/3857"),
	new en("http://www.opengis.net/gml/srs/epsg.xml#3857")
];
function nn(e, t, n, r) {
	let i = e.length;
	n = n > 1 ? n : 2, r ??= n, t === void 0 && (t = n > 2 ? e.slice() : Array(i));
	for (let n = 0; n < i; n += r) {
		t[n] = Xt * e[n] / 180;
		let r = Yt * Math.log(Math.tan(Math.PI * (+e[n + 1] + 90) / 360));
		r > $t ? r = $t : r < -$t && (r = -$t), t[n + 1] = r;
	}
	return t;
}
function rn(e, t, n, r) {
	let i = e.length;
	n = n > 1 ? n : 2, r ??= n, t === void 0 && (t = n > 2 ? e.slice() : Array(i));
	for (let n = 0; n < i; n += r) t[n] = 180 * e[n] / Xt, t[n + 1] = 360 * Math.atan(Math.exp(e[n + 1] / Yt)) / Math.PI - 90;
	return t;
}
//#endregion
//#region node_modules/ol/proj/epsg4326.js
var an = 6378137, on = [
	-180,
	-90,
	180,
	90
], sn = Math.PI * an / 180, cn = class extends Jt {
	constructor(e, t) {
		super({
			code: e,
			units: "degrees",
			extent: on,
			axisOrientation: t,
			global: !0,
			metersPerUnit: sn,
			worldExtent: on
		});
	}
}, ln = [
	new cn("CRS:84"),
	new cn("EPSG:4326", "neu"),
	new cn("urn:ogc:def:crs:OGC:1.3:CRS84"),
	new cn("urn:ogc:def:crs:OGC:2:84"),
	new cn("http://www.opengis.net/def/crs/OGC/1.3/CRS84"),
	new cn("http://www.opengis.net/gml/srs/epsg.xml#4326", "neu"),
	new cn("http://www.opengis.net/def/crs/EPSG/0/4326", "neu")
], un = {};
function dn(e) {
	return un[e] || un[e.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\w+)$/, "EPSG:$3")] || null;
}
function fn(e, t) {
	un[e] = t;
}
//#endregion
//#region node_modules/ol/proj/transforms.js
var pn = {};
function mn(e, t, n) {
	let r = e.getCode(), i = t.getCode();
	r in pn || (pn[r] = {}), pn[r][i] = n;
}
function hn(e, t) {
	return e in pn && t in pn[e] ? pn[e][t] : null;
}
//#endregion
//#region node_modules/ol/proj/utm.js
var gn = .9996, _n = .00669438, vn = _n * _n, yn = vn * _n, bn = _n / .99330562, xn = Math.sqrt(.99330562), Sn = (1 - xn) / (1 + xn), Cn = Sn * Sn, wn = Cn * Sn, Tn = wn * Sn, En = Tn * Sn, Dn = 1 - _n / 4 - 3 * vn / 64 - 5 * yn / 256, On = .002514607064228144, kn = 26390466021299826e-22, An = 35 * yn / 3072, jn = 3 / 2 * Sn - 27 / 32 * wn + 269 / 512 * En, Mn = 21 / 16 * Cn - 55 / 32 * Tn, Nn = 151 / 96 * wn - 417 / 128 * En, Pn = 1097 / 512 * Tn, Fn = 6378137;
function In(e, t, n) {
	let r = e - 5e5, i = (n.north ? t : t - 1e7) / gn / (Fn * Dn), a = i + jn * Math.sin(2 * i) + Mn * Math.sin(4 * i) + Nn * Math.sin(6 * i) + Pn * Math.sin(8 * i), o = Math.sin(a), s = o * o, c = Math.cos(a), l = o / c, u = l * l, d = u * u, f = 1 - _n * s, p = Fn / Math.sqrt(1 - _n * s), m = .99330562 / f, h = bn * c ** 2, g = h * h, _ = r / (p * gn), v = _ * _, y = v * _, b = y * _, x = b * _, S = x * _, C = a - l / m * (v / 2 - b / 24 * (5 + 3 * u + 10 * h - 4 * g - 9 * bn)) + S / 720 * (61 + 90 * u + 298 * h + 45 * d - 252 * bn - 3 * g), w = (_ - y / 6 * (1 + 2 * u + h) + x / 120 * (5 - 2 * h + 28 * u - 3 * g + 8 * bn + 24 * d)) / c;
	return w = Rt(w + Mt(Hn(n.number)), -Math.PI, Math.PI), [jt(w), jt(C)];
}
var Ln = -80, Rn = 84, zn = -180, Bn = 180;
function Vn(e, t, n) {
	e = Rt(e, zn, Bn), t < Ln ? t = Ln : t > Rn && (t = Rn);
	let r = Mt(t), i = Math.sin(r), a = Math.cos(r), o = i / a, s = o * o, c = s * s, l = Mt(e), u = Mt(Hn(n.number)), d = Fn / Math.sqrt(1 - _n * i ** 2), f = bn * a ** 2, p = a * Rt(l - u, -Math.PI, Math.PI), m = p * p, h = m * p, g = h * p, _ = g * p, v = _ * p, y = Fn * (Dn * r - On * Math.sin(2 * r) + kn * Math.sin(4 * r) - An * Math.sin(6 * r)), b = gn * d * (p + h / 6 * (1 - s + f) + _ / 120 * (5 - 18 * s + c + 72 * f - 58 * bn)) + 5e5, x = gn * (y + d * o * (m / 2 + g / 24 * (5 - s + 9 * f + 4 * f ** 2) + v / 720 * (61 - 58 * s + c + 600 * f - 330 * bn)));
	return n.north || (x += 1e7), [b, x];
}
function Hn(e) {
	return (e - 1) * 6 - 180 + 3;
}
var Un = [
	/^EPSG:(\d+)$/,
	/^urn:ogc:def:crs:EPSG::(\d+)$/,
	/^http:\/\/www\.opengis\.net\/def\/crs\/EPSG\/0\/(\d+)$/
];
function Wn(e) {
	let t = 0;
	for (let n of Un) {
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
function Gn(e, t) {
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
function Kn(e) {
	return Wn(e) ? new Jt({
		code: e,
		units: "m"
	}) : null;
}
function qn(e) {
	let t = Wn(e.getCode());
	return t ? {
		forward: Gn(Vn, t),
		inverse: Gn(In, t)
	} : null;
}
function Jn(e, t, n) {
	n ||= 6371008.8;
	let r = Mt(e[1]), i = Mt(t[1]), a = (i - r) / 2, o = Mt(t[0] - e[0]) / 2, s = Math.sin(a) * Math.sin(a) + Math.sin(o) * Math.sin(o) * Math.cos(r) * Math.cos(i);
	return 2 * n * Math.atan2(Math.sqrt(s), Math.sqrt(1 - s));
}
//#endregion
//#region node_modules/ol/proj.js
var Yn = [qn], Xn = [Kn], Zn = !0;
function Qn(e) {
	Zn = !(e === void 0 || e);
}
function $n(e, t) {
	if (t !== void 0) {
		for (let n = 0, r = e.length; n < r; ++n) t[n] = e[n];
		t = t;
	} else t = e.slice();
	return t;
}
function er(e, t) {
	if (t !== void 0 && e !== t) {
		for (let n = 0, r = e.length; n < r; ++n) t[n] = e[n];
		e = t;
	}
	return e;
}
function tr(e) {
	fn(e.getCode(), e), mn(e, e, $n);
}
function nr(e) {
	e.forEach(tr);
}
function rr(e) {
	if (typeof e != "string") return e;
	let t = dn(e);
	if (t) return t;
	for (let t of Xn) {
		let n = t(e);
		if (n) return n;
	}
	return null;
}
function ir(e, t, n, r) {
	e = rr(e);
	let i, a = e.getPointResolutionFunc();
	if (a) {
		if (i = a(t, n), r && r !== e.getUnits()) {
			let t = e.getMetersPerUnit();
			t && (i = i * t / qt[r]);
		}
	} else {
		let a = e.getUnits();
		if (a == "degrees" && !r || r == "degrees") i = t;
		else {
			let o = ur(e, rr("EPSG:4326"));
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
				e = o(e, e, 2), i = (Jn(e.slice(0, 2), e.slice(2, 4)) + Jn(e.slice(4, 6), e.slice(6, 8))) / 2;
			}
			let s = r ? qt[r] : e.getMetersPerUnit();
			s !== void 0 && (i /= s);
		}
	}
	return i;
}
function ar(e) {
	nr(e), e.forEach(function(t) {
		e.forEach(function(e) {
			t !== e && mn(t, e, $n);
		});
	});
}
function or(e, t, n, r) {
	e.forEach(function(e) {
		t.forEach(function(t) {
			mn(e, t, n), mn(t, e, r);
		});
	});
}
function sr(e, t) {
	return e ? typeof e == "string" ? rr(e) : e : rr(t);
}
function cr(e) {
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
function lr(e, t) {
	if (e === t) return !0;
	let n = e.getUnits() === t.getUnits();
	return (e.getCode() === t.getCode() || ur(e, t) === $n) && n;
}
function ur(e, t) {
	let n = e.getCode(), r = t.getCode(), i = hn(n, r);
	if (i) return i;
	let a = null, o = null;
	for (let n of Yn) a ||= n(e), o ||= n(t);
	if (!a && !o) return null;
	let s = "EPSG:4326";
	if (!o) {
		let e = hn(s, r);
		e && (i = dr(a.inverse, e));
	} else if (a) i = dr(a.inverse, o.forward);
	else {
		let e = hn(n, s);
		e && (i = dr(e, o.forward));
	}
	return i && (tr(e), tr(t), mn(e, t, i)), i;
}
function dr(e, t) {
	return function(n, r, i, a) {
		return r = e(n, r, i, a), t(r, r, i, a);
	};
}
function fr(e, t) {
	return ur(rr(e), rr(t));
}
function pr(e, t, n) {
	let r = fr(t, n);
	if (!r) {
		let e = rr(t).getCode(), r = rr(n).getCode();
		throw Error(`No transform available between ${e} and ${r}`);
	}
	return r(e, void 0, e.length);
}
function mr(e, t, n, r) {
	return bt(e, fr(t, n), void 0, r);
}
var hr = null;
function gr() {
	return hr;
}
function _r(e, t) {
	return e;
}
function vr(e, t) {
	return Zn && !Vt(e, [0, 0]) && e[0] >= -180 && e[0] <= 180 && e[1] >= -90 && e[1] <= 90 && (Zn = !1, Et("Call useGeographic() from ol/proj once to work with [longitude, latitude] coordinates.")), e;
}
function yr(e, t) {
	return e;
}
function br(e, t) {
	return e;
}
function xr(e, t) {
	return e;
}
function Sr() {
	ar(tn), ar(ln), or(ln, tn, nn, rn);
}
Sr();
//#endregion
//#region node_modules/ol/asserts.js
function R(e, t) {
	if (!e) throw Error(t);
}
//#endregion
//#region node_modules/ol/transform.js
var Cr = [
	1,
	0,
	0,
	1,
	0,
	0
], wr = [
	,
	,
	,
	,
	,
	,
];
function Tr() {
	return Cr.slice(0);
}
function Er(e) {
	return Or(e, 1, 0, 0, 1, 0, 0);
}
function Dr(e, t) {
	let n = e[0], r = e[1], i = e[2], a = e[3], o = e[4], s = e[5], c = t[0], l = t[1], u = t[2], d = t[3], f = t[4], p = t[5];
	return e[0] = n * c + i * l, e[1] = r * c + a * l, e[2] = n * u + i * d, e[3] = r * u + a * d, e[4] = n * f + i * p + o, e[5] = r * f + a * p + s, e;
}
function Or(e, t, n, r, i, a, o) {
	return e[0] = t, e[1] = n, e[2] = r, e[3] = i, e[4] = a, e[5] = o, e;
}
function kr(e, t) {
	return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e;
}
function z(e, t) {
	let n = t[0], r = t[1];
	return t[0] = e[0] * n + e[2] * r + e[4], t[1] = e[1] * n + e[3] * r + e[5], t;
}
function Ar(e, t, n) {
	return Dr(e, Or(wr, t, 0, 0, n, 0, 0));
}
function jr(e, t, n) {
	return Dr(e, Or(wr, 1, 0, 0, 1, t, n));
}
function Mr(e, t, n, r, i, a, o, s) {
	let c = Math.sin(a), l = Math.cos(a);
	return e[0] = r * l, e[1] = i * c, e[2] = -r * c, e[3] = i * l, e[4] = o * r * l - s * r * c + t, e[5] = o * i * c + s * i * l + n, e;
}
function Nr(e, t) {
	let n = Pr(t);
	R(n !== 0, "Transformation matrix cannot be inverted");
	let r = t[0], i = t[1], a = t[2], o = t[3], s = t[4], c = t[5];
	return e[0] = o / n, e[1] = -i / n, e[2] = -a / n, e[3] = r / n, e[4] = (a * c - o * s) / n, e[5] = -(r * c - i * s) / n, e;
}
function Pr(e) {
	return e[0] * e[3] - e[1] * e[2];
}
var Fr = [
	1e5,
	1e5,
	1e5,
	1e5,
	2,
	2
];
function Ir(e) {
	return "matrix(" + e.join(", ") + ")";
}
function Lr(e) {
	return e.substring(7, e.length - 1).split(",").map(parseFloat);
}
function Rr(e, t) {
	let n = Lr(e), r = Lr(t);
	for (let e = 0; e < 6; ++e) if (Math.round((n[e] - r[e]) * Fr[e]) !== 0) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/transform.js
function zr(e, t, n, r, i, a, o) {
	a ||= [], o ||= 2;
	let s = 0;
	for (let c = t; c < n; c += r) {
		let t = e[c], n = e[c + 1];
		a[s++] = i[0] * t + i[2] * n + i[4], a[s++] = i[1] * t + i[3] * n + i[5];
		for (let t = 2; t < o; t++) a[s++] = e[c + t];
	}
	return a && a.length != s && (a.length = s), a;
}
function Br(e, t, n, r, i, a, o) {
	o ||= [];
	let s = Math.cos(i), c = Math.sin(i), l = a[0], u = a[1], d = 0;
	for (let i = t; i < n; i += r) {
		let t = e[i] - l, n = e[i + 1] - u;
		o[d++] = l + t * s - n * c, o[d++] = u + t * c + n * s;
		for (let t = i + 2; t < i + r; ++t) o[d++] = e[t];
	}
	return o && o.length != d && (o.length = d), o;
}
function Vr(e, t, n, r, i, a, o, s) {
	s ||= [];
	let c = o[0], l = o[1], u = 0;
	for (let o = t; o < n; o += r) {
		let t = e[o] - c, n = e[o + 1] - l;
		s[u++] = c + i * t, s[u++] = l + a * n;
		for (let t = o + 2; t < o + r; ++t) s[u++] = e[t];
	}
	return s && s.length != u && (s.length = u), s;
}
function Hr(e, t, n, r, i, a, o) {
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
var Ur = Tr(), Wr = [NaN, NaN], Gr = class extends A {
	constructor() {
		super(), this.extent_ = Ke(), this.extentRevision_ = -1, this.simplifiedGeometryMaxMinSquaredTolerance = 0, this.simplifiedGeometryRevision = 0, this.simplifyTransformedInternal = b((e, t, n) => {
			if (!n) return this.getSimplifiedGeometry(t);
			let r = this.clone();
			return r.applyTransform(n), r.getSimplifiedGeometry(t);
		});
	}
	simplifyTransformed(e, t) {
		return this.simplifyTransformedInternal(this.getRevision(), e, t);
	}
	clone() {
		return E();
	}
	closestPointXY(e, t, n, r) {
		return E();
	}
	containsXY(e, t) {
		return this.closestPointXY(e, t, Wr, Number.MIN_VALUE) === 0;
	}
	getClosestPoint(e, t) {
		return t ||= [NaN, NaN], this.closestPointXY(e[0], e[1], t, Infinity), t;
	}
	intersectsCoordinate(e) {
		return this.containsXY(e[0], e[1]);
	}
	computeExtent(e) {
		return E();
	}
	getExtent(e) {
		if (this.extentRevision_ != this.getRevision()) {
			let e = this.computeExtent(this.extent_);
			(isNaN(e[0]) || isNaN(e[1])) && Je(e), this.extentRevision_ = this.getRevision();
		}
		return _t(this.extent_, e);
	}
	rotate(e, t) {
		E();
	}
	scale(e, t, n) {
		E();
	}
	simplify(e) {
		return this.getSimplifiedGeometry(e * e);
	}
	getSimplifiedGeometry(e) {
		return E();
	}
	getType() {
		return E();
	}
	applyTransform(e) {
		E();
	}
	intersectsExtent(e) {
		return E();
	}
	translate(e, t) {
		E();
	}
	transform(e, t) {
		let n = rr(e), r = n.getUnits() == "tile-pixels" ? function(e, r, i) {
			let a = n.getExtent(), o = n.getWorldExtent(), s = ut(o) / ut(a);
			Mr(Ur, o[0], o[3], s, -s, 0, 0, 0);
			let c = zr(e, 0, e.length, i, Ur, r), l = fr(n, t);
			return l ? l(c, c, i) : c;
		} : fr(n, t);
		return this.applyTransform(r), this;
	}
}, Kr = class extends Gr {
	constructor() {
		super(), this.layout = "XY", this.stride = 2, this.flatCoordinates;
	}
	computeExtent(e) {
		return Xe(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	getCoordinates() {
		return E();
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
		this.stride = Jr(e), this.layout = e, this.flatCoordinates = t;
	}
	setCoordinates(e, t) {
		E();
	}
	setLayout(e, t, n) {
		let r;
		if (e) r = Jr(e);
		else {
			for (let e = 0; e < n; ++e) {
				if (t.length === 0) {
					this.layout = "XY", this.stride = 2;
					return;
				}
				t = t[0];
			}
			r = t.length, e = qr(r);
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
			Br(n, 0, n.length, r, e, t, n), this.changed();
		}
	}
	scale(e, t, n) {
		t === void 0 && (t = e), n ||= ot(this.getExtent());
		let r = this.getFlatCoordinates();
		if (r) {
			let i = this.getStride();
			Vr(r, 0, r.length, i, e, t, n, r), this.changed();
		}
	}
	translate(e, t) {
		let n = this.getFlatCoordinates();
		if (n) {
			let r = this.getStride();
			Hr(n, 0, n.length, r, e, t, n), this.changed();
		}
	}
};
function qr(e) {
	let t;
	return e == 2 ? t = "XY" : e == 3 ? t = "XYZ" : e == 4 && (t = "XYZM"), t;
}
function Jr(e) {
	let t;
	return e == "XY" ? t = 2 : e == "XYZ" || e == "XYM" ? t = 3 : e == "XYZM" && (t = 4), t;
}
function Yr(e, t, n) {
	let r = e.getFlatCoordinates();
	if (!r) return null;
	let i = e.getStride();
	return zr(r, 0, r.length, i, t, n);
}
//#endregion
//#region node_modules/ol/geom/flat/deflate.js
function Xr(e, t, n, r) {
	for (let r = 0, i = n.length; r < i; ++r) e[t++] = n[r];
	return t;
}
function Zr(e, t, n, r) {
	for (let i = 0, a = n.length; i < a; ++i) {
		let a = n[i];
		for (let n = 0; n < r; ++n) e[t++] = a[n];
	}
	return t;
}
function Qr(e, t, n, r, i) {
	i ||= [];
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = Zr(e, t, n[o], r);
		i[a++] = s, t = s;
	}
	return i.length = a, i;
}
function $r(e, t, n, r, i) {
	i ||= [];
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = Qr(e, t, n[o], r, i[a]);
		s.length === 0 && (s[0] = t), i[a++] = s, t = s[s.length - 1];
	}
	return i.length = a, i;
}
//#endregion
//#region node_modules/ol/Feature.js
var ei = class e extends A {
	constructor(e) {
		if (super(), this.on, this.once, this.un, this.id_ = void 0, this.geometryName_ = "geometry", this.style_ = null, this.styleFunction_ = void 0, this.geometryChangeKey_ = null, this.addChangeListener(this.geometryName_, this.handleGeometryChanged_), e) {
			if (typeof e.getSimplifiedGeometry == "function") {
				let t = e;
				this.setGeometry(t);
			} else {
				let t = e;
				this.setProperties(t);
			}
		}
	}
	clone() {
		let t = new e(), n = this.geometryName_;
		t.setGeometryName(n);
		let r = this.getPropertiesInternal();
		if (r) {
			let e = this.getGeometry();
			for (let i in r) i === n && e ? t.set(i, e.clone()) : t.set(i, r[i], !0);
		}
		let i = this.getStyle();
		return i && t.setStyle(i), t;
	}
	getGeometry() {
		return this.get(this.geometryName_);
	}
	getId() {
		return this.id_;
	}
	getGeometryName() {
		return this.geometryName_;
	}
	getStyle() {
		return this.style_;
	}
	getStyleFunction() {
		return this.styleFunction_;
	}
	handleGeometryChange_() {
		this.changed();
	}
	handleGeometryChanged_() {
		this.geometryChangeKey_ &&= (o(this.geometryChangeKey_), null);
		let e = this.getGeometry();
		e && (this.geometryChangeKey_ = i(e, s.CHANGE, this.handleGeometryChange_, this)), this.changed();
	}
	setGeometry(e) {
		this.set(this.geometryName_, e);
	}
	setStyle(e) {
		this.style_ = e, this.styleFunction_ = e ? ti(e) : void 0, this.changed();
	}
	setId(e) {
		this.id_ = e, this.changed();
	}
	setGeometryName(e) {
		e !== this.geometryName_ && (this.removeChangeListener(this.geometryName_, this.handleGeometryChanged_), this.geometryName_ = e, this.addChangeListener(this.geometryName_, this.handleGeometryChanged_), this.handleGeometryChanged_());
	}
};
function ti(e) {
	if (typeof e == "function") return e;
	let t;
	return Array.isArray(e) ? t = e : (R(typeof e.getZIndex == "function", "Expected an `ol/style/Style` or an array of `ol/style/Style.js`"), t = [e]), function() {
		return t;
	};
}
//#endregion
//#region node_modules/ol/geom/flat/closest.js
function ni(e, t, n, r, i, a, o) {
	let s = e[t], c = e[t + 1], l = e[n] - s, u = e[n + 1] - c, d;
	if (l === 0 && u === 0) d = t;
	else {
		let f = ((i - s) * l + (a - c) * u) / (l * l + u * u);
		if (f > 1) d = n;
		else if (f > 0) {
			for (let i = 0; i < r; ++i) o[i] = Pt(e[t + i], e[n + i], f);
			o.length = r;
			return;
		} else d = t;
	}
	for (let t = 0; t < r; ++t) o[t] = e[d + t];
	o.length = r;
}
function ri(e, t, n, r, i) {
	let a = e[t], o = e[t + 1];
	for (t += r; t < n; t += r) {
		let n = e[t], r = e[t + 1], s = kt(a, o, n, r);
		s > i && (i = s), a = n, o = r;
	}
	return i;
}
function ii(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i = ri(e, t, o, r, i), t = o;
	}
	return i;
}
function ai(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i = ii(e, t, o, r, i), t = o[o.length - 1];
	}
	return i;
}
function oi(e, t, n, r, i, a, o, s, c, l, u) {
	if (t == n) return l;
	let d, f;
	if (i === 0) {
		if (f = kt(o, s, e[t], e[t + 1]), f < l) {
			for (d = 0; d < r; ++d) c[d] = e[t + d];
			return c.length = r, f;
		}
		return l;
	}
	u ||= [NaN, NaN];
	let p = t + r;
	for (; p < n;) if (ni(e, p - r, p, r, o, s, u), f = kt(o, s, u[0], u[1]), f < l) {
		for (l = f, d = 0; d < r; ++d) c[d] = u[d];
		c.length = r, p += r;
	} else p += r * Math.max((Math.sqrt(f) - Math.sqrt(l)) / i | 0, 1);
	if (a && (ni(e, n - r, t, r, o, s, u), f = kt(o, s, u[0], u[1]), f < l)) {
		for (l = f, d = 0; d < r; ++d) c[d] = u[d];
		c.length = r;
	}
	return l;
}
function si(e, t, n, r, i, a, o, s, c, l, u) {
	u ||= [NaN, NaN];
	for (let d = 0, f = n.length; d < f; ++d) {
		let f = n[d];
		l = oi(e, t, f, r, i, a, o, s, c, l, u), t = f;
	}
	return l;
}
function ci(e, t, n, r, i, a, o, s, c, l, u) {
	u ||= [NaN, NaN];
	for (let d = 0, f = n.length; d < f; ++d) {
		let f = n[d];
		l = si(e, t, f, r, i, a, o, s, c, l, u), t = f[f.length - 1];
	}
	return l;
}
//#endregion
//#region node_modules/ol/geom/flat/inflate.js
function li(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = t; o < n; o += r) i[a++] = e.slice(o, o + r);
	return i.length = a, i;
}
function ui(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		i[a++] = li(e, t, s, r, i[a]), t = s;
	}
	return i.length = a, i;
}
function di(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		i[a++] = s.length === 1 && s[0] === t ? [] : ui(e, t, s, r, i[a]), t = s[s.length - 1];
	}
	return i.length = a, i;
}
//#endregion
//#region node_modules/ol/geom/flat/interpolate.js
function fi(e, t, n, r, i, a, o) {
	let s, c, u = (n - t) / r;
	if (u === 1) s = t;
	else if (u === 2) s = t, c = i;
	else if (u !== 0) {
		let a = e[t], o = e[t + 1], u = 0, d = [0];
		for (let i = t + r; i < n; i += r) {
			let t = e[i], n = e[i + 1];
			u += Math.sqrt((t - a) * (t - a) + (n - o) * (n - o)), d.push(u), a = t, o = n;
		}
		let f = i * u, p = l(d, f);
		p < 0 ? (c = (f - d[-p - 2]) / (d[-p - 1] - d[-p - 2]), s = t + (-p - 2) * r) : s = t + p * r;
	}
	o = o > 1 ? o : 2, a ||= Array(o);
	for (let t = 0; t < o; ++t) a[t] = s === void 0 ? NaN : c === void 0 ? e[s + t] : Pt(e[s + t], e[s + r + t], c);
	return a;
}
function pi(e, t, n, r, i, a) {
	if (n == t) return null;
	let o;
	if (i < e[t + r - 1]) return a ? (o = e.slice(t, t + r), o[r - 1] = i, o) : null;
	if (e[n - 1] < i) return a ? (o = e.slice(n - r, n), o[r - 1] = i, o) : null;
	if (i == e[t + r - 1]) return e.slice(t, t + r);
	let s = t / r, c = n / r;
	for (; s < c;) {
		let t = s + c >> 1;
		i < e[(t + 1) * r - 1] ? c = t : s = t + 1;
	}
	let l = e[s * r - 1];
	if (i == l) return e.slice((s - 1) * r, (s - 1) * r + r);
	let u = e[(s + 1) * r - 1], d = (i - l) / (u - l);
	o = [];
	for (let t = 0; t < r - 1; ++t) o.push(Pt(e[(s - 1) * r + t], e[s * r + t], d));
	return o.push(i), o;
}
function mi(e, t, n, r, i, a, o) {
	if (o) return pi(e, t, n[n.length - 1], r, i, a);
	let s;
	if (i < e[r - 1]) return a ? (s = e.slice(0, r), s[r - 1] = i, s) : null;
	if (e[e.length - 1] < i) return a ? (s = e.slice(e.length - r), s[r - 1] = i, s) : null;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		if (t != o) {
			if (i < e[t + r - 1]) return null;
			if (i <= e[o - 1]) return pi(e, t, o, r, i, !1);
			t = o;
		}
	}
	return null;
}
//#endregion
//#region node_modules/ol/geom/flat/contains.js
function hi(e, t, n, r, i) {
	return !nt(i, function(i) {
		return !gi(e, t, n, r, i[0], i[1]);
	});
}
function gi(e, t, n, r, i, a) {
	let o = 0, s = e[n - r], c = e[n - r + 1];
	for (; t < n; t += r) {
		let n = e[t], r = e[t + 1];
		c <= a ? r > a && (n - s) * (a - c) - (i - s) * (r - c) > 0 && o++ : r <= a && (n - s) * (a - c) - (i - s) * (r - c) < 0 && o--, s = n, c = r;
	}
	return o !== 0;
}
function _i(e, t, n, r, i, a) {
	if (n.length === 0 || !gi(e, t, n[0], r, i, a)) return !1;
	for (let t = 1, o = n.length; t < o; ++t) if (gi(e, n[t - 1], n[t], r, i, a)) return !1;
	return !0;
}
function vi(e, t, n, r, i, a) {
	if (n.length === 0) return !1;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		if (_i(e, t, s, r, i, a)) return !0;
		t = s[s.length - 1];
	}
	return !1;
}
//#endregion
//#region node_modules/ol/geom/flat/segments.js
function yi(e, t, n, r, i) {
	let a;
	for (t += r; t < n; t += r) if (a = i(e.slice(t - r, t), e.slice(t, t + r)), a) return a;
	return !1;
}
//#endregion
//#region node_modules/ol/geom/flat/intersectsextent.js
function bi(e, t, n, r, i, a) {
	return a ??= et(Ke(), e, t, n, r), ht(i, a) ? a[0] >= i[0] && a[2] <= i[2] || a[1] >= i[1] && a[3] <= i[3] || yi(e, t, n, r, function(e, t) {
		return yt(i, e, t);
	}) : !1;
}
function xi(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		if (bi(e, t, n[a], r, i)) return !0;
		t = n[a];
	}
	return !1;
}
function Si(e, t, n, r, i) {
	return !!(bi(e, t, n, r, i) || gi(e, t, n, r, i[0], i[1]) || gi(e, t, n, r, i[0], i[3]) || gi(e, t, n, r, i[2], i[1]) || gi(e, t, n, r, i[2], i[3]));
}
function Ci(e, t, n, r, i) {
	if (!Si(e, t, n[0], r, i)) return !1;
	if (n.length === 1) return !0;
	for (let t = 1, a = n.length; t < a; ++t) if (hi(e, n[t - 1], n[t], r, i) && !bi(e, n[t - 1], n[t], r, i)) return !1;
	return !0;
}
function wi(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		if (Ci(e, t, o, r, i)) return !0;
		t = o[o.length - 1];
	}
	return !1;
}
//#endregion
//#region node_modules/ol/geom/flat/length.js
function Ti(e, t, n, r) {
	let i = e[t], a = e[t + 1], o = 0;
	for (let s = t + r; s < n; s += r) {
		let t = e[s], n = e[s + 1];
		o += Math.sqrt((t - i) * (t - i) + (n - a) * (n - a)), i = t, a = n;
	}
	return o;
}
//#endregion
//#region node_modules/ol/geom/flat/simplify.js
function Ei(e, t, n, r, i, a, o) {
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
			let n = e[t], r = e[t + 1], i = Ot(n, r, s, d, f, p);
			i > o && (u = t, o = i);
		}
		o > i && (c[(u - t) / r] = 1, a + r < u && l.push(a, u), u + r < n && l.push(u, n));
	}
	for (let n = 0; n < s; ++n) c[n] && (a[o++] = e[t + n * r], a[o++] = e[t + n * r + 1]);
	return o;
}
function Di(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c];
		o = Ei(e, t, l, r, i, a, o), s.push(o), t = l;
	}
	return o;
}
function Oi(e, t) {
	return t * Math.round(e / t);
}
function ki(e, t, n, r, i, a, o) {
	if (t == n) return o;
	let s = Oi(e[t], i), c = Oi(e[t + 1], i);
	t += r, a[o++] = s, a[o++] = c;
	let l, u;
	do
		if (l = Oi(e[t], i), u = Oi(e[t + 1], i), t += r, t == n) return a[o++] = l, a[o++] = u, o;
	while (l == s && u == c);
	for (; t < n;) {
		let n = Oi(e[t], i), d = Oi(e[t + 1], i);
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
function Ai(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c];
		o = ki(e, t, l, r, i, a, o), s.push(o), t = l;
	}
	return o;
}
function ji(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c], u = [];
		o = Ai(e, t, l, r, i, a, o, u), s.push(u), t = l[l.length - 1];
	}
	return o;
}
//#endregion
//#region node_modules/ol/geom/LineString.js
var Mi = class e extends Kr {
	constructor(e, t) {
		super(), this.flatMidpoint_ = null, this.flatMidpointRevision_ = -1, this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, t !== void 0 && !Array.isArray(e[0]) ? this.setFlatCoordinates(t, e) : this.setCoordinates(e, t);
	}
	appendCoordinate(e) {
		m(this.flatCoordinates, e), this.changed();
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout);
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		return r < Ve(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(ri(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), oi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !1, e, t, n, r));
	}
	forEachSegment(e) {
		return yi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	getCoordinateAtM(e, t) {
		return this.layout != "XYM" && this.layout != "XYZM" ? null : (t = t !== void 0 && t, pi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, t));
	}
	getCoordinates() {
		return li(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getCoordinateAt(e, t) {
		return fi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, t, this.stride);
	}
	getLength() {
		return Ti(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getFlatMidpoint() {
		return this.flatMidpointRevision_ != this.getRevision() && (this.flatMidpoint_ = this.getCoordinateAt(.5, this.flatMidpoint_ ?? void 0), this.flatMidpointRevision_ = this.getRevision()), this.flatMidpoint_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [];
		return n.length = Ei(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, t, n, 0), new e(n, "XY");
	}
	getType() {
		return "LineString";
	}
	intersectsExtent(e) {
		return bi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, this.getExtent());
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = Zr(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
}, Ni = class e extends Kr {
	constructor(e, t, n) {
		if (super(), this.ends_ = [], this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, Array.isArray(e[0])) this.setCoordinates(e, t);
		else if (t !== void 0 && n) this.setFlatCoordinates(t, e), this.ends_ = n;
		else {
			let t = e, n = [], r = [];
			for (let e = 0, i = t.length; e < i; ++e) {
				let i = t[e];
				m(n, i.getFlatCoordinates()), r.push(n.length);
			}
			let i = t.length === 0 ? this.getLayout() : t[0].getLayout();
			this.setFlatCoordinates(i, n), this.ends_ = r;
		}
	}
	appendLineString(e) {
		m(this.flatCoordinates, e.getFlatCoordinates().slice()), this.ends_.push(this.flatCoordinates.length), this.changed();
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout, this.ends_.slice());
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		return r < Ve(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(ii(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), si(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !1, e, t, n, r));
	}
	getCoordinateAtM(e, t, n) {
		return this.layout != "XYM" && this.layout != "XYZM" || this.flatCoordinates.length === 0 ? null : (t = t !== void 0 && t, n = n !== void 0 && n, mi(this.flatCoordinates, 0, this.ends_, this.stride, e, t, n));
	}
	getCoordinates() {
		return ui(this.flatCoordinates, 0, this.ends_, this.stride);
	}
	getEnds() {
		return this.ends_;
	}
	getLineString(e) {
		return e < 0 || this.ends_.length <= e ? null : new Mi(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}
	getLineStrings() {
		let e = this.flatCoordinates, t = this.ends_, n = this.layout, r = [], i = 0;
		for (let a = 0, o = t.length; a < o; ++a) {
			let o = t[a], s = new Mi(e.slice(i, o), n);
			r.push(s), i = o;
		}
		return r;
	}
	getLength() {
		let e = this.ends_, t = 0, n = 0;
		for (let r = 0, i = e.length; r < i; ++r) n += Ti(this.flatCoordinates, t, e[r], this.stride), t = e[r];
		return n;
	}
	getFlatMidpoints() {
		let e = [], t = this.flatCoordinates, n = 0, r = this.ends_, i = this.stride;
		for (let a = 0, o = r.length; a < o; ++a) {
			let o = r[a];
			m(e, fi(t, n, o, i, .5)), n = o;
		}
		return e;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = Di(this.flatCoordinates, 0, this.ends_, this.stride, t, n, 0, r), new e(n, "XY", r);
	}
	getType() {
		return "MultiLineString";
	}
	intersectsExtent(e) {
		return xi(this.flatCoordinates, 0, this.ends_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		let n = Qr(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}
}, Pi = class e extends Kr {
	constructor(e, t) {
		super(), this.setCoordinates(e, t);
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout);
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		let i = this.flatCoordinates, a = kt(e, t, i[0], i[1]);
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
		return Ye(this.flatCoordinates, e);
	}
	getType() {
		return "Point";
	}
	intersectsExtent(e) {
		return We(e, this.flatCoordinates[0], this.flatCoordinates[1]);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 0), this.flatCoordinates ||= [], this.flatCoordinates.length = Xr(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
}, Fi = class e extends Kr {
	constructor(e, t) {
		super(), t && !Array.isArray(e[0]) ? this.setFlatCoordinates(t, e) : this.setCoordinates(e, t);
	}
	appendPoint(e) {
		m(this.flatCoordinates, e.getFlatCoordinates()), this.changed();
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout);
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		if (r < Ve(this.getExtent(), e, t)) return r;
		let i = this.flatCoordinates, a = this.stride;
		for (let o = 0, s = i.length; o < s; o += a) {
			let s = kt(e, t, i[o], i[o + 1]);
			if (s < r) {
				r = s;
				for (let e = 0; e < a; ++e) n[e] = i[o + e];
				n.length = a;
			}
		}
		return r;
	}
	getCoordinates() {
		return li(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getPoint(e) {
		let t = this.flatCoordinates.length / this.stride;
		return e < 0 || t <= e ? null : new Pi(this.flatCoordinates.slice(e * this.stride, (e + 1) * this.stride), this.layout);
	}
	getPoints() {
		let e = this.flatCoordinates, t = this.layout, n = this.stride, r = [];
		for (let i = 0, a = e.length; i < a; i += n) {
			let a = new Pi(e.slice(i, i + n), t);
			r.push(a);
		}
		return r;
	}
	getType() {
		return "MultiPoint";
	}
	intersectsExtent(e) {
		let t = this.flatCoordinates, n = this.stride;
		for (let r = 0, i = t.length; r < i; r += n) {
			let n = t[r], i = t[r + 1];
			if (We(e, n, i)) return !0;
		}
		return !1;
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = Zr(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
};
//#endregion
//#region node_modules/ol/geom/flat/area.js
function Ii(e, t, n, r) {
	let i = 0, a = e[n - r], o = e[n - r + 1], s = 0, c = 0;
	for (; t < n; t += r) {
		let n = e[t] - a, r = e[t + 1] - o;
		i += c * n - s * r, s = n, c = r;
	}
	return i / 2;
}
function Li(e, t, n, r) {
	let i = 0;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i += Ii(e, t, o, r), t = o;
	}
	return i;
}
function Ri(e, t, n, r) {
	let i = 0;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i += Li(e, t, o, r), t = o[o.length - 1];
	}
	return i;
}
//#endregion
//#region node_modules/ol/geom/LinearRing.js
var zi = class e extends Kr {
	constructor(e, t) {
		super(), this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, t !== void 0 && !Array.isArray(e[0]) ? this.setFlatCoordinates(t, e) : this.setCoordinates(e, t);
	}
	clone() {
		return new e(this.flatCoordinates.slice(), this.layout);
	}
	closestPointXY(e, t, n, r) {
		return r < Ve(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(ri(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), oi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	getArea() {
		return Ii(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getCoordinates() {
		return li(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getSimplifiedGeometryInternal(t) {
		let n = [];
		return n.length = Ei(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, t, n, 0), new e(n, "XY");
	}
	getType() {
		return "LinearRing";
	}
	intersectsExtent(e) {
		return bi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = Zr(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
};
//#endregion
//#region node_modules/ol/geom/flat/interiorpoint.js
function Bi(e, t, n, r, i, a, o) {
	let s, c, l, d, f, p, m, h = i[a + 1], g = [];
	for (let i = 0, a = n.length; i < a; ++i) {
		let a = n[i];
		for (d = e[a - r], p = e[a - r + 1], s = t; s < a; s += r) f = e[s], m = e[s + 1], (h <= p && m <= h || p <= h && h <= m) && (l = (h - p) / (m - p) * (f - d) + d, g.push(l)), d = f, p = m;
	}
	let _ = NaN, v = -Infinity;
	for (g.sort(u), d = g[0], s = 1, c = g.length; s < c; ++s) {
		f = g[s];
		let i = Math.abs(f - d);
		i > v && (l = (d + f) / 2, _i(e, t, n, r, l, h) && (_ = l, v = i)), d = f;
	}
	return isNaN(_) && (_ = i[a]), o ? (o.push(_, h, v), o) : [
		_,
		h,
		v
	];
}
function Vi(e, t, n, r, i) {
	let a = [];
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		a = Bi(e, t, s, r, i, 2 * o, a), t = s[s.length - 1];
	}
	return a;
}
//#endregion
//#region node_modules/ol/geom/flat/reverse.js
function Hi(e, t, n, r) {
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
function Ui(e, t, n, r) {
	let i = 0, a = e[n - r], o = e[n - r + 1];
	for (; t < n; t += r) {
		let n = e[t], r = e[t + 1];
		i += (n - a) * (r + o), a = n, o = r;
	}
	return i === 0 ? void 0 : i > 0;
}
function Wi(e, t, n, r, i) {
	i = i !== void 0 && i;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a], s = Ui(e, t, o, r);
		if (a === 0) {
			if (i && s || !i && !s) return !1;
		} else if (i && !s || !i && s) return !1;
		t = o;
	}
	return !0;
}
function Gi(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		if (!Wi(e, t, o, r, i)) return !1;
		o.length && (t = o[o.length - 1]);
	}
	return !0;
}
function Ki(e, t, n, r, i) {
	i = i !== void 0 && i;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a], s = Ui(e, t, o, r);
		(a === 0 ? i && s || !i && !s : i && !s || !i && s) && Hi(e, t, o, r), t = o;
	}
	return t;
}
function qi(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) t = Ki(e, t, n[a], r, i);
	return t;
}
function Ji(e, t) {
	let n = [], r = 0, i = 0, a;
	for (let o = 0, s = t.length; o < s; ++o) {
		let s = t[o], c = Ui(e, r, s, 2);
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
var Yi = class e extends Kr {
	constructor(e, t, n) {
		super(), this.ends_ = [], this.flatInteriorPointRevision_ = -1, this.flatInteriorPoint_ = null, this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, this.orientedRevision_ = -1, this.orientedFlatCoordinates_ = null, t !== void 0 && n ? (this.setFlatCoordinates(t, e), this.ends_ = n) : this.setCoordinates(e, t);
	}
	appendLinearRing(e) {
		this.flatCoordinates ? m(this.flatCoordinates, e.getFlatCoordinates()) : this.flatCoordinates = e.getFlatCoordinates().slice(), this.ends_.push(this.flatCoordinates.length), this.changed();
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout, this.ends_.slice());
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		return r < Ve(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(ii(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), si(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	containsXY(e, t) {
		return _i(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, t);
	}
	getArea() {
		return Li(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride);
	}
	getCoordinates(e) {
		let t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), Ki(t, 0, this.ends_, this.stride, e)), ui(t, 0, this.ends_, this.stride);
	}
	getEnds() {
		return this.ends_;
	}
	getFlatInteriorPoint() {
		if (this.flatInteriorPointRevision_ != this.getRevision()) {
			let e = ot(this.getExtent());
			this.flatInteriorPoint_ = Bi(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, 0), this.flatInteriorPointRevision_ = this.getRevision();
		}
		return this.flatInteriorPoint_;
	}
	getInteriorPoint() {
		return new Pi(this.getFlatInteriorPoint(), "XYM");
	}
	getLinearRingCount() {
		return this.ends_.length;
	}
	getLinearRing(e) {
		return e < 0 || this.ends_.length <= e ? null : new zi(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}
	getLinearRings() {
		let e = this.layout, t = this.flatCoordinates, n = this.ends_, r = [], i = 0;
		for (let a = 0, o = n.length; a < o; ++a) {
			let o = n[a], s = new zi(t.slice(i, o), e);
			r.push(s), i = o;
		}
		return r;
	}
	getOrientedFlatCoordinates() {
		if (this.orientedRevision_ != this.getRevision()) {
			let e = this.flatCoordinates;
			Wi(e, 0, this.ends_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = Ki(this.orientedFlatCoordinates_, 0, this.ends_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = Ai(this.flatCoordinates, 0, this.ends_, this.stride, Math.sqrt(t), n, 0, r), new e(n, "XY", r);
	}
	getType() {
		return "Polygon";
	}
	intersectsExtent(e) {
		return Ci(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		let n = Qr(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}
};
function Xi(e) {
	if (gt(e)) throw Error("Cannot create polygon from empty extent");
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
	return new Yi(a, "XY", [a.length]);
}
//#endregion
//#region node_modules/ol/geom/flat/center.js
function Zi(e, t, n, r) {
	let i = [], a = Ke();
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		a = Xe(e, t, s[0], r), i.push((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), t = s[s.length - 1];
	}
	return i;
}
//#endregion
//#region node_modules/ol/geom/MultiPolygon.js
var Qi = class e extends Kr {
	constructor(e, t, n) {
		if (super(), this.endss_ = [], this.flatInteriorPointsRevision_ = -1, this.flatInteriorPoints_ = null, this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, this.orientedRevision_ = -1, this.orientedFlatCoordinates_ = null, !n && !Array.isArray(e[0])) {
			let r = e, i = [], a = [];
			for (let e = 0, t = r.length; e < t; ++e) {
				let t = r[e], n = i.length, o = t.getEnds();
				for (let e = 0, t = o.length; e < t; ++e) o[e] += n;
				m(i, t.getFlatCoordinates()), a.push(o);
			}
			t = r.length === 0 ? this.getLayout() : r[0].getLayout(), e = i, n = a;
		}
		t !== void 0 && n ? (this.setFlatCoordinates(t, e), this.endss_ = n) : this.setCoordinates(e, t);
	}
	appendPolygon(e) {
		let t;
		if (!this.flatCoordinates) this.flatCoordinates = e.getFlatCoordinates().slice(), t = e.getEnds().slice(), this.endss_.push();
		else {
			let n = this.flatCoordinates.length;
			m(this.flatCoordinates, e.getFlatCoordinates()), t = e.getEnds().slice();
			for (let e = 0, r = t.length; e < r; ++e) t[e] += n;
		}
		this.endss_.push(t), this.changed();
	}
	clone() {
		let t = this.endss_.length, n = Array(t);
		for (let e = 0; e < t; ++e) n[e] = this.endss_[e].slice();
		let r = new e(this.flatCoordinates.slice(), this.layout, n);
		return r.applyProperties(this), r;
	}
	closestPointXY(e, t, n, r) {
		return r < Ve(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(ai(this.flatCoordinates, 0, this.endss_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), ci(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	containsXY(e, t) {
		return vi(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, e, t);
	}
	getArea() {
		return Ri(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride);
	}
	getCoordinates(e) {
		let t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), qi(t, 0, this.endss_, this.stride, e)), di(t, 0, this.endss_, this.stride);
	}
	getEndss() {
		return this.endss_;
	}
	getFlatInteriorPoints() {
		if (this.flatInteriorPointsRevision_ != this.getRevision()) {
			let e = Zi(this.flatCoordinates, 0, this.endss_, this.stride);
			this.flatInteriorPoints_ = Vi(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, e), this.flatInteriorPointsRevision_ = this.getRevision();
		}
		return this.flatInteriorPoints_;
	}
	getInteriorPoints() {
		return new Fi(this.getFlatInteriorPoints().slice(), "XYM");
	}
	getOrientedFlatCoordinates() {
		if (this.orientedRevision_ != this.getRevision()) {
			let e = this.flatCoordinates;
			Gi(e, 0, this.endss_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = qi(this.orientedFlatCoordinates_, 0, this.endss_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = ji(this.flatCoordinates, 0, this.endss_, this.stride, Math.sqrt(t), n, 0, r), new e(n, "XY", r);
	}
	getPolygon(e) {
		if (e < 0 || this.endss_.length <= e) return null;
		let t;
		if (e === 0) t = 0;
		else {
			let n = this.endss_[e - 1];
			t = n[n.length - 1];
		}
		let n = this.endss_[e].slice(), r = n[n.length - 1];
		if (t !== 0) for (let e = 0, r = n.length; e < r; ++e) n[e] -= t;
		return new Yi(this.flatCoordinates.slice(t, r), this.layout, n);
	}
	getPolygons() {
		let e = this.layout, t = this.flatCoordinates, n = this.endss_, r = [], i = 0;
		for (let a = 0, o = n.length; a < o; ++a) {
			let o = n[a].slice(), s = o[o.length - 1];
			if (i !== 0) for (let e = 0, t = o.length; e < t; ++e) o[e] -= i;
			let c = new Yi(t.slice(i, s), e, o);
			r.push(c), i = s;
		}
		return r;
	}
	getType() {
		return "MultiPolygon";
	}
	intersectsExtent(e) {
		return wi(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 3), this.flatCoordinates ||= [];
		let n = $r(this.flatCoordinates, 0, e, this.stride, this.endss_);
		if (n.length === 0) this.flatCoordinates.length = 0;
		else {
			let e = n[n.length - 1];
			this.flatCoordinates.length = e.length === 0 ? 0 : e[e.length - 1];
		}
		this.changed();
	}
}, $i = Tr(), ea = class e {
	constructor(e, t, n, r, i, a) {
		this.styleFunction, this.extent_, this.id_ = a, this.type_ = e, this.flatCoordinates_ = t, this.flatInteriorPoints_ = null, this.flatMidpoints_ = null, this.ends_ = n || null, this.properties_ = i, this.squaredTolerance_, this.stride_ = r, this.simplifiedGeometry_;
	}
	get(e) {
		return this.properties_[e];
	}
	getExtent() {
		return this.extent_ ||= this.type_ === "Point" ? Ye(this.flatCoordinates_) : Xe(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_), this.extent_;
	}
	getFlatInteriorPoint() {
		if (!this.flatInteriorPoints_) {
			let e = ot(this.getExtent());
			this.flatInteriorPoints_ = Bi(this.flatCoordinates_, 0, this.ends_, this.stride_, e, 0);
		}
		return this.flatInteriorPoints_;
	}
	getFlatInteriorPoints() {
		if (!this.flatInteriorPoints_) {
			let e = Ji(this.flatCoordinates_, this.ends_), t = Zi(this.flatCoordinates_, 0, e, this.stride_);
			this.flatInteriorPoints_ = Vi(this.flatCoordinates_, 0, e, this.stride_, t);
		}
		return this.flatInteriorPoints_;
	}
	getFlatMidpoint() {
		return this.flatMidpoints_ ||= fi(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_, .5), this.flatMidpoints_;
	}
	getFlatMidpoints() {
		if (!this.flatMidpoints_) {
			this.flatMidpoints_ = [];
			let e = this.flatCoordinates_, t = 0, n = this.ends_;
			for (let r = 0, i = n.length; r < i; ++r) {
				let i = n[r], a = fi(e, t, i, this.stride_, .5);
				m(this.flatMidpoints_, a), t = i;
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
		e = rr(e);
		let t = e.getExtent(), n = e.getWorldExtent();
		if (t && n) {
			let e = ut(n) / ut(t);
			Mr($i, n[0], n[3], e, -e, 0, 0, 0), zr(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_, $i, this.flatCoordinates_);
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
		return this.simplifyTransformed = b((t, n) => {
			if (t === this.squaredTolerance_) return this.simplifiedGeometry_;
			this.simplifiedGeometry_ = this.clone(), n && this.simplifiedGeometry_.applyTransform(n);
			let r = this.simplifiedGeometry_.getFlatCoordinates(), i;
			switch (this.type_) {
				case "LineString":
					r.length = Ei(r, 0, this.simplifiedGeometry_.flatCoordinates_.length, this.simplifiedGeometry_.stride_, t, r, 0), i = [r.length];
					break;
				case "MultiLineString":
					i = [], r.length = Di(r, 0, this.simplifiedGeometry_.ends_, this.simplifiedGeometry_.stride_, t, r, 0, i);
					break;
				case "Polygon": i = [], r.length = Ai(r, 0, this.simplifiedGeometry_.ends_, this.simplifiedGeometry_.stride_, Math.sqrt(t), r, 0, i);
			}
			return i && (this.simplifiedGeometry_ = new e(this.type_, r, i, this.stride_, this.properties_, this.id_)), this.squaredTolerance_ = t, this.simplifiedGeometry_;
		}), this;
	}
};
ea.prototype.getFlatCoordinates = ea.prototype.getOrientedFlatCoordinates;
//#endregion
//#region node_modules/ol/geom/GeometryCollection.js
var ta = class e extends Gr {
	constructor(e) {
		super(), this.geometries_ = e, this.changeEventsKeys_ = [], this.listenGeometriesChange_();
	}
	unlistenGeometriesChange_() {
		this.changeEventsKeys_.forEach(o), this.changeEventsKeys_.length = 0;
	}
	listenGeometriesChange_() {
		let e = this.geometries_;
		for (let t = 0, n = e.length; t < n; ++t) this.changeEventsKeys_.push(i(e[t], s.CHANGE, this.changed, this));
	}
	clone() {
		let t = new e(na(this.geometries_));
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		if (r < Ve(this.getExtent(), e, t)) return r;
		let i = this.geometries_;
		for (let a = 0, o = i.length; a < o; ++a) r = i[a].closestPointXY(e, t, n, r);
		return r;
	}
	containsXY(e, t) {
		let n = this.geometries_;
		for (let r = 0, i = n.length; r < i; ++r) if (n[r].containsXY(e, t)) return !0;
		return !1;
	}
	computeExtent(e) {
		Je(e);
		let t = this.geometries_;
		for (let n = 0, r = t.length; n < r; ++n) Qe(e, t[n].getExtent());
		return e;
	}
	getGeometries() {
		return na(this.geometries_);
	}
	getGeometriesArray() {
		return this.geometries_;
	}
	getGeometriesArrayRecursive() {
		let e = [], t = this.geometries_;
		for (let n = 0, r = t.length; n < r; ++n) t[n].getType() === this.getType() ? e = e.concat(t[n].getGeometriesArrayRecursive()) : e.push(t[n]);
		return e;
	}
	getSimplifiedGeometry(t) {
		if (this.simplifiedGeometryRevision !== this.getRevision() && (this.simplifiedGeometryMaxMinSquaredTolerance = 0, this.simplifiedGeometryRevision = this.getRevision()), t < 0 || this.simplifiedGeometryMaxMinSquaredTolerance !== 0 && t < this.simplifiedGeometryMaxMinSquaredTolerance) return this;
		let n = [], r = this.geometries_, i = !1;
		for (let e = 0, a = r.length; e < a; ++e) {
			let a = r[e], o = a.getSimplifiedGeometry(t);
			n.push(o), o !== a && (i = !0);
		}
		return i ? new e(n) : (this.simplifiedGeometryMaxMinSquaredTolerance = t, this);
	}
	getType() {
		return "GeometryCollection";
	}
	intersectsExtent(e) {
		let t = this.geometries_;
		for (let n = 0, r = t.length; n < r; ++n) if (t[n].intersectsExtent(e)) return !0;
		return !1;
	}
	isEmpty() {
		return this.geometries_.length === 0;
	}
	rotate(e, t) {
		let n = this.geometries_;
		for (let r = 0, i = n.length; r < i; ++r) n[r].rotate(e, t);
		this.changed();
	}
	scale(e, t, n) {
		n ||= ot(this.getExtent());
		let r = this.geometries_;
		for (let i = 0, a = r.length; i < a; ++i) r[i].scale(e, t, n);
		this.changed();
	}
	setGeometries(e) {
		this.setGeometriesArray(na(e));
	}
	setGeometriesArray(e) {
		this.unlistenGeometriesChange_(), this.geometries_ = e, this.listenGeometriesChange_(), this.changed();
	}
	applyTransform(e) {
		let t = this.geometries_;
		for (let n = 0, r = t.length; n < r; ++n) t[n].applyTransform(e);
		this.changed();
	}
	translate(e, t) {
		let n = this.geometries_;
		for (let r = 0, i = n.length; r < i; ++r) n[r].translate(e, t);
		this.changed();
	}
	disposeInternal() {
		this.unlistenGeometriesChange_(), super.disposeInternal();
	}
};
function na(e) {
	return e.map((e) => e.clone());
}
//#endregion
//#region node_modules/ol/format/Feature.js
var ra = class {
	constructor() {
		this.dataProjection = void 0, this.defaultFeatureProjection = void 0, this.featureClass = ei, this.supportedMediaTypes = null;
	}
	getReadOptions(e, t) {
		if (t) {
			let n = t.dataProjection ? rr(t.dataProjection) : this.readProjection(e);
			t.extent && n && n.getUnits() === "tile-pixels" && (n = rr(n), n.setWorldExtent(t.extent)), t = {
				dataProjection: n,
				featureProjection: t.featureProjection
			};
		}
		return this.adaptOptions(t);
	}
	adaptOptions(e) {
		return Object.assign({
			dataProjection: this.dataProjection,
			featureProjection: this.defaultFeatureProjection,
			featureClass: this.featureClass
		}, e);
	}
	getType() {
		return E();
	}
	readFeature(e, t) {
		return E();
	}
	readFeatures(e, t) {
		return E();
	}
	readGeometry(e, t) {
		return E();
	}
	readProjection(e) {
		return E();
	}
	writeFeature(e, t) {
		return E();
	}
	writeFeatures(e, t) {
		return E();
	}
	writeGeometry(e, t) {
		return E();
	}
};
function ia(e, t, n) {
	let r = n ? rr(n.featureProjection) : null, i = n ? rr(n.dataProjection) : null, a = e;
	if (r && i && !lr(r, i)) {
		t && (a = e.clone());
		let n = t ? r : i, o = t ? i : r;
		n.getUnits() === "tile-pixels" ? a.transform(n, o) : a.applyTransform(fr(n, o));
	}
	if (t && n && n.decimals !== void 0) {
		let t = 10 ** n.decimals;
		a === e && (a = e.clone()), a.applyTransform(function(e) {
			for (let n = 0, r = e.length; n < r; ++n) e[n] = Math.round(e[n] * t) / t;
			return e;
		});
	}
	return a;
}
var aa = {
	Point: Pi,
	LineString: Mi,
	Polygon: Yi,
	MultiPoint: Fi,
	MultiLineString: Ni,
	MultiPolygon: Qi
};
function oa(e, t, n) {
	return Array.isArray(t[0]) ? (Gi(e, 0, t, n) || (e = e.slice(), qi(e, 0, t, n)), e) : (Wi(e, 0, t, n) || (e = e.slice(), Ki(e, 0, t, n)), e);
}
function sa(e, t) {
	let n = e.geometry;
	if (!n) return [];
	if (Array.isArray(n)) return n.map((t) => sa({
		...e,
		geometry: t
	})).flat();
	let r = n.type === "MultiPolygon" ? "Polygon" : n.type;
	if (r === "GeometryCollection" || r === "Circle") throw Error("Unsupported geometry type: " + r);
	let i = n.layout.length;
	return ia(new ea(r, r === "Polygon" ? oa(n.flatCoordinates, n.ends, i) : n.flatCoordinates, n.ends?.flat(), i, e.properties || {}, e.id).enableSimplifyTransformed(), !1, t);
}
function ca(e, t) {
	if (!e) return null;
	if (Array.isArray(e)) return new ta(e.map((e) => ca(e, t)));
	let n = aa[e.type];
	return ia(new n(e.flatCoordinates, e.layout || "XY", e.ends), !1, t);
}
//#endregion
//#region node_modules/ol/format/JSONFeature.js
var la = class extends ra {
	constructor() {
		super();
	}
	getType() {
		return "json";
	}
	readFeature(e, t) {
		return this.readFeatureFromObject(ua(e), this.getReadOptions(e, t));
	}
	readFeatures(e, t) {
		return this.readFeaturesFromObject(ua(e), this.getReadOptions(e, t));
	}
	readFeatureFromObject(e, t) {
		return E();
	}
	readFeaturesFromObject(e, t) {
		return E();
	}
	readGeometry(e, t) {
		return this.readGeometryFromObject(ua(e), this.getReadOptions(e, t));
	}
	readGeometryFromObject(e, t) {
		return E();
	}
	readProjection(e) {
		return this.readProjectionFromObject(ua(e));
	}
	readProjectionFromObject(e) {
		return E();
	}
	writeFeature(e, t) {
		return JSON.stringify(this.writeFeatureObject(e, t));
	}
	writeFeatureObject(e, t) {
		return E();
	}
	writeFeatures(e, t) {
		return JSON.stringify(this.writeFeaturesObject(e, t));
	}
	writeFeaturesObject(e, t) {
		return E();
	}
	writeGeometry(e, t) {
		return JSON.stringify(this.writeGeometryObject(e, t));
	}
	writeGeometryObject(e, t) {
		return E();
	}
};
function ua(e) {
	return typeof e == "string" ? JSON.parse(e) || null : e === null ? null : e;
}
//#endregion
//#region node_modules/ol/format/GeoJSON.js
var da = class extends la {
	constructor(e) {
		e ||= {}, super(), this.dataProjection = rr(e.dataProjection ? e.dataProjection : "EPSG:4326"), e.featureProjection && (this.defaultFeatureProjection = rr(e.featureProjection)), e.featureClass && (this.featureClass = e.featureClass), this.geometryName_ = e.geometryName, this.extractGeometryName_ = e.extractGeometryName, this.supportedMediaTypes = ["application/geo+json", "application/vnd.geo+json"];
	}
	readFeatureFromObject(e, t) {
		let n = null;
		n = e.type === "Feature" ? e : {
			type: "Feature",
			geometry: e,
			properties: null
		};
		let r = fa(n.geometry, t);
		if (this.featureClass === ea) return sa({
			geometry: r,
			id: n.id,
			properties: n.properties
		}, t);
		let i = this.featureClass, a = new i();
		return this.geometryName_ ? a.setGeometryName(this.geometryName_) : this.extractGeometryName_ && n.geometry_name && a.setGeometryName(n.geometry_name), a.setGeometry(ca(r, t)), "id" in n && a.setId(n.id), n.properties && a.setProperties(n.properties, !0), a;
	}
	readFeaturesFromObject(e, t) {
		let n = e, r = null;
		if (n.type === "FeatureCollection") {
			let n = e;
			r = [];
			let i = n.features;
			for (let e = 0, n = i.length; e < n; ++e) {
				let n = this.readFeatureFromObject(i[e], t);
				n && r.push(n);
			}
		} else r = [this.readFeatureFromObject(e, t)];
		return r.flat();
	}
	readGeometryFromObject(e, t) {
		return pa(e, t);
	}
	readProjectionFromObject(e) {
		let t = e.crs, n;
		if (t) {
			if (t.type == "name") n = rr(t.properties.name);
			else if (t.type === "EPSG") n = rr("EPSG:" + t.properties.code);
			else throw Error("Unknown SRS type");
		} else n = this.dataProjection;
		return n;
	}
	writeFeatureObject(e, t) {
		t = this.adaptOptions(t);
		let n = {
			type: "Feature",
			geometry: null,
			properties: null
		}, i = e.getId();
		if (i !== void 0 && (n.id = i), !e.hasProperties()) return n;
		let a = e.getProperties(), o = e.getGeometry();
		return o && (n.geometry = xa(o, t), delete a[e.getGeometryName()]), r(a) || (n.properties = a), n;
	}
	writeFeaturesObject(e, t) {
		t = this.adaptOptions(t);
		let n = [];
		for (let r = 0, i = e.length; r < i; ++r) n.push(this.writeFeatureObject(e[r], t));
		return {
			type: "FeatureCollection",
			features: n
		};
	}
	writeGeometryObject(e, t) {
		return xa(e, this.adaptOptions(t));
	}
};
function fa(e, t) {
	if (!e) return null;
	let n;
	switch (e.type) {
		case "Point":
			n = ha(e);
			break;
		case "LineString":
			n = ga(e);
			break;
		case "Polygon":
			n = ba(e);
			break;
		case "MultiPoint":
			n = va(e);
			break;
		case "MultiLineString":
			n = _a(e);
			break;
		case "MultiPolygon":
			n = ya(e);
			break;
		case "GeometryCollection":
			n = ma(e);
			break;
		default: throw Error("Unsupported GeoJSON type: " + e.type);
	}
	return n;
}
function pa(e, t) {
	return ca(fa(e, t), t);
}
function ma(e, t) {
	return e.geometries.map(function(e) {
		return fa(e, t);
	});
}
function ha(e) {
	let t = e.coordinates;
	return {
		type: "Point",
		flatCoordinates: t,
		layout: qr(t.length)
	};
}
function ga(e) {
	let t = e.coordinates, n = t.flat();
	return {
		type: "LineString",
		flatCoordinates: n,
		ends: [n.length],
		layout: qr(t[0]?.length || 2)
	};
}
function _a(e) {
	let t = e.coordinates, n = t[0]?.[0]?.length || 2, r = [];
	return {
		type: "MultiLineString",
		flatCoordinates: r,
		ends: Qr(r, 0, t, n),
		layout: qr(n)
	};
}
function va(e) {
	let t = e.coordinates;
	return {
		type: "MultiPoint",
		flatCoordinates: t.flat(),
		layout: qr(t[0]?.length || 2)
	};
}
function ya(e) {
	let t = e.coordinates, n = [], r = t[0]?.[0]?.[0].length || 2;
	return {
		type: "MultiPolygon",
		flatCoordinates: n,
		ends: $r(n, 0, t, r),
		layout: qr(r)
	};
}
function ba(e) {
	let t = e.coordinates, n = [], r = t[0]?.[0]?.length;
	return {
		type: "Polygon",
		flatCoordinates: n,
		ends: Qr(n, 0, t, r),
		layout: qr(r)
	};
}
function xa(e, t) {
	e = ia(e, !0, t);
	let n = e.getType(), r;
	switch (n) {
		case "Point":
			r = Da(e, t);
			break;
		case "LineString":
			r = Ca(e, t);
			break;
		case "Polygon":
			r = Oa(e, t);
			break;
		case "MultiPoint":
			r = Ta(e, t);
			break;
		case "MultiLineString":
			r = wa(e, t);
			break;
		case "MultiPolygon":
			r = Ea(e, t);
			break;
		case "GeometryCollection":
			r = Sa(e, t);
			break;
		case "Circle":
			r = {
				type: "GeometryCollection",
				geometries: []
			};
			break;
		default: throw Error("Unsupported geometry type: " + n);
	}
	return r;
}
function Sa(e, t) {
	return t = Object.assign({}, t), delete t.featureProjection, {
		type: "GeometryCollection",
		geometries: e.getGeometriesArray().map(function(e) {
			return xa(e, t);
		})
	};
}
function Ca(e, t) {
	return {
		type: "LineString",
		coordinates: e.getCoordinates()
	};
}
function wa(e, t) {
	return {
		type: "MultiLineString",
		coordinates: e.getCoordinates()
	};
}
function Ta(e, t) {
	return {
		type: "MultiPoint",
		coordinates: e.getCoordinates()
	};
}
function Ea(e, t) {
	let n;
	return t && (n = t.rightHanded), {
		type: "MultiPolygon",
		coordinates: e.getCoordinates(n)
	};
}
function Da(e, t) {
	return {
		type: "Point",
		coordinates: e.getCoordinates()
	};
}
function Oa(e, t) {
	let n;
	return t && (n = t.rightHanded), {
		type: "Polygon",
		coordinates: e.getCoordinates(n)
	};
}
//#endregion
//#region node_modules/ol/TileState.js
var B = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
}, ka = class extends C {
	constructor(e, t, n) {
		super(), n ||= {}, this.tileCoord = e, this.state = t, this.key = "", this.transition_ = n.transition === void 0 ? 250 : n.transition, this.transitionStarts_ = {}, this.interpolate = !!n.interpolate;
	}
	changed() {
		this.dispatchEvent(s.CHANGE);
	}
	release() {
		this.setState(B.EMPTY);
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
		if (this.state !== B.EMPTY) {
			if (this.state !== B.ERROR && this.state > e) throw Error("Tile load sequence violation");
			this.state = e, this.changed();
		}
	}
	load() {
		E();
	}
	getAlpha(e, t) {
		if (!this.transition_) return 1;
		let n = this.transitionStarts_[e];
		if (!n) n = t, this.transitionStarts_[e] = n;
		else if (n === -1) return 1;
		let r = t - n + 1e3 / 60;
		return r >= this.transition_ ? 1 : ke(r / this.transition_);
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
function Aa(e) {
	return e instanceof Image || e instanceof HTMLCanvasElement || e instanceof HTMLVideoElement || e instanceof ImageBitmap ? e : null;
}
var ja = /* @__PURE__ */ Error("disposed"), Ma = [256, 256], Na = class extends ka {
	constructor(e) {
		let t = B.IDLE;
		super(e.tileCoord, t, {
			transition: e.transition,
			interpolate: e.interpolate
		}), this.loader_ = e.loader, this.data_ = null, this.error_ = null, this.size_ = e.size || null, this.controller_ = e.controller || null;
	}
	getSize() {
		if (this.size_) return this.size_;
		let e = Aa(this.data_);
		return e ? [e.width, e.height] : Ma;
	}
	getData() {
		return this.data_;
	}
	getError() {
		return this.error_;
	}
	load() {
		if (this.state !== B.IDLE && this.state !== B.ERROR) return;
		this.state = B.LOADING, this.changed();
		let e = this;
		this.loader_().then(function(t) {
			e.data_ = t, e.state = B.LOADED, e.changed();
		}).catch(function(t) {
			e.error_ = t, e.state = B.ERROR, e.changed();
		});
	}
	disposeInternal() {
		this.controller_ &&= (this.controller_.abort(ja), null), super.disposeInternal();
	}
}, V = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
};
//#endregion
//#region node_modules/ol/Image.js
function Pa(e, t, n) {
	let r = e, i = !0, c = !1, l = !1, u = [a(r, s.LOAD, function() {
		l = !0, c || t();
	})];
	return r.src && me ? (c = !0, r.decode().then(function() {
		i && t();
	}).catch(function(e) {
		i && (l ? t() : n());
	})) : u.push(a(r, s.ERROR, n)), function() {
		i = !1, u.forEach(o);
	};
}
function Fa(e, t) {
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
function Ia(e, t) {
	return t && (e.src = t), e.src && me ? new Promise((t, n) => e.decode().then(() => t(e)).catch((r) => e.complete && e.width ? t(e) : n(r))) : Fa(e);
}
//#endregion
//#region node_modules/ol/ImageTile.js
var La = class extends ka {
	constructor(e, t, n, r, i, a) {
		super(e, t, a), this.crossOrigin_ = r?.crossOrigin, this.referrerPolicy_ = r?.referrerPolicy, this.src_ = n, this.key = n, this.image_, pe ? this.image_ = new OffscreenCanvas(1, 1) : (this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_)), this.unlisten_ = null, this.tileLoadFunction_ = i;
	}
	getImage() {
		return this.image_;
	}
	setImage(e) {
		this.image_ = e, this.state = B.LOADED, this.unlistenImage_(), this.changed();
	}
	getCrossOrigin() {
		return this.crossOrigin_;
	}
	getReferrerPolicy() {
		return this.referrerPolicy_;
	}
	handleImageError_() {
		this.state = B.ERROR, this.unlistenImage_(), this.image_ = Ra(), this.changed();
	}
	handleImageLoad_() {
		if (pe) this.state = B.LOADED;
		else {
			let e = this.image_;
			this.state = e.naturalWidth && e.naturalHeight ? B.LOADED : B.EMPTY;
		}
		this.unlistenImage_(), this.changed();
	}
	load() {
		this.state == B.ERROR && (this.state = B.IDLE, this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_)), this.state == B.IDLE && (this.state = B.LOADING, this.changed(), this.tileLoadFunction_(this, this.src_), this.unlisten_ = Pa(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
	}
	unlistenImage_() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}
	disposeInternal() {
		this.unlistenImage_(), this.image_ = null, super.disposeInternal();
	}
};
function Ra() {
	let e = I(1, 1);
	return e.fillStyle = "rgba(0,0,0,0)", e.fillRect(0, 0, 1, 1), e.canvas;
}
//#endregion
//#region node_modules/ol/TileRange.js
var za = class {
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
function Ba(e, t, n, r, i) {
	return i === void 0 ? new za(e, t, n, r) : (i.minX = e, i.maxX = t, i.minY = n, i.maxY = r, i);
}
//#endregion
//#region node_modules/ol/reproj.js
var Va, Ha = [];
function Ua(e, t, n, r, i) {
	e.beginPath(), e.moveTo(0, 0), e.lineTo(t, n), e.lineTo(r, i), e.closePath(), e.save(), e.clip(), e.fillRect(0, 0, Math.max(t, r) + 1, Math.max(n, i)), e.restore();
}
function Wa(e, t) {
	return Math.abs(e[t * 4] - 210) > 2 || Math.abs(e[t * 4 + 3] - 191.25) > 2;
}
function Ga() {
	if (Va === void 0) {
		let e = I(6, 6, Ha);
		e.globalCompositeOperation = "lighter", e.fillStyle = "rgba(210, 0, 0, 0.75)", Ua(e, 4, 5, 4, 0), Ua(e, 4, 5, 0, 5);
		let t = e.getImageData(0, 0, 3, 3).data;
		Va = Wa(t, 0) || Wa(t, 4) || Wa(t, 8), ve(e), Ha.push(e.canvas);
	}
	return Va;
}
function Ka(e, t, n, r) {
	let i = pr(n, t, e), a = ir(t, r, n), o = t.getMetersPerUnit();
	o !== void 0 && (a *= o);
	let s = e.getMetersPerUnit();
	s !== void 0 && (a /= s);
	let c = e.getExtent();
	if (!c || He(c, i)) {
		let t = ir(e, a, i) / a;
		isFinite(t) && t > 0 && (a /= t);
	}
	return a;
}
function qa(e, t, n, r) {
	let i = Ka(e, t, ot(n), r);
	return (!isFinite(i) || i <= 0) && nt(n, function(n) {
		return i = Ka(e, t, n, r), isFinite(i) && i > 0;
	}), i;
}
function Ja(e, t, n, r, i, a, o, s, c, l, u, d, f, p) {
	let m = I(Math.round(n * e), Math.round(n * t), Ha);
	if (d || (m.imageSmoothingEnabled = !1), c.length === 0) return m.canvas;
	m.scale(n, n);
	function h(e) {
		return Math.round(e * n) / n;
	}
	m.globalCompositeOperation = "lighter";
	let g = Ke();
	c.forEach(function(e, t, n) {
		Qe(g, e.extent);
	});
	let _, v = n / r, y = (d ? 1 : 1 + 2 ** -24) / v;
	if (!f || c.length !== 1 || l !== 0) {
		if (_ = I(Math.round(L(g) * v), Math.round(ut(g) * v), Ha), d || (_.imageSmoothingEnabled = !1), i && p) {
			let e = (i[0] - g[0]) * v, t = -(i[3] - g[3]) * v, n = L(i) * v, r = ut(i) * v;
			_.rect(e, t, n, r), _.clip();
		}
		c.forEach(function(e, t, n) {
			if (e.image.width > 0 && e.image.height > 0) {
				if (e.clipExtent) {
					_.save();
					let t = (e.clipExtent[0] - g[0]) * v, n = -(e.clipExtent[3] - g[3]) * v, r = L(e.clipExtent) * v, i = ut(e.clipExtent) * v;
					_.rect(d ? t : Math.round(t), d ? n : Math.round(n), d ? r : Math.round(t + r) - Math.round(t), d ? i : Math.round(n + i) - Math.round(n)), _.clip();
				}
				let t = (e.extent[0] - g[0]) * v, n = -(e.extent[3] - g[3]) * v, r = L(e.extent) * v, i = ut(e.extent) * v;
				_.drawImage(e.image, l, l, e.image.width - 2 * l, e.image.height - 2 * l, d ? t : Math.round(t), d ? n : Math.round(n), d ? r : Math.round(t + r) - Math.round(t), d ? i : Math.round(n + i) - Math.round(n)), e.clipExtent && _.restore();
			}
		});
	}
	let b = pt(o);
	return s.getTriangles().forEach(function(e, t, n) {
		let r = e.source, i = e.target, o = r[0][0], s = r[0][1], l = r[1][0], u = r[1][1], f = r[2][0], p = r[2][1], v = h((i[0][0] - b[0]) / a), x = h(-(i[0][1] - b[1]) / a), S = h((i[1][0] - b[0]) / a), C = h(-(i[1][1] - b[1]) / a), w = h((i[2][0] - b[0]) / a), T = h(-(i[2][1] - b[1]) / a), E = o, D = s;
		o = 0, s = 0, l -= E, u -= D, f -= E, p -= D;
		let O = At([
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
				C - x
			],
			[
				0,
				0,
				f,
				p,
				T - x
			]
		]);
		if (!O) return;
		if (m.save(), m.beginPath(), Ga() || !d) {
			m.moveTo(S, C);
			let e = v - S, t = x - C;
			for (let n = 0; n < 4; n++) m.lineTo(S + h((n + 1) * e / 4), C + h(n * t / 3)), n != 3 && m.lineTo(S + h((n + 1) * e / 4), C + h((n + 1) * t / 3));
			m.lineTo(w, T);
		} else m.moveTo(S, C), m.lineTo(v, x), m.lineTo(w, T);
		m.clip(), m.transform(O[0], O[2], O[1], O[3], v, x), m.translate(g[0] - E, g[3] - D);
		let k;
		if (_) k = _.canvas, m.scale(y, -y);
		else {
			let e = c[0], t = e.extent;
			k = e.image, m.scale(L(t) / k.width, -ut(t) / k.height);
		}
		m.drawImage(k, 0, 0), m.restore();
	}), _ && (ve(_), Ha.push(_.canvas)), u && (m.save(), m.globalCompositeOperation = "source-over", m.strokeStyle = "black", m.lineWidth = 1, s.getTriangles().forEach(function(e, t, n) {
		let r = e.target, i = (r[0][0] - b[0]) / a, o = -(r[0][1] - b[1]) / a, s = (r[1][0] - b[0]) / a, c = -(r[1][1] - b[1]) / a, l = (r[2][0] - b[0]) / a, u = -(r[2][1] - b[1]) / a;
		m.beginPath(), m.moveTo(s, c), m.lineTo(i, o), m.lineTo(l, u), m.closePath(), m.stroke();
	}), m.restore()), m.canvas;
}
//#endregion
//#region node_modules/ol/reproj/Triangulation.js
var Ya = 10, Xa = .25, Za = class {
	constructor(e, t, n, r, i, a, o) {
		this.sourceProj_ = e, this.targetProj_ = t;
		let s = {}, c = o ? cr((e) => z(o, pr(e, this.targetProj_, this.sourceProj_))) : fr(this.targetProj_, this.sourceProj_);
		this.transformInv_ = function(e) {
			let t = e[0] + "/" + e[1];
			return s[t] || (s[t] = c(e)), s[t];
		}, this.maxSourceExtent_ = r, this.errorThresholdSquared_ = i * i, this.triangles_ = [], this.wrapsXInSource_ = !1, this.canWrapXInSource_ = this.sourceProj_.canWrapX() && !!r && !!this.sourceProj_.getExtent() && L(r) >= L(this.sourceProj_.getExtent()), this.sourceWorldWidth_ = this.sourceProj_.getExtent() ? L(this.sourceProj_.getExtent()) : null, this.targetWorldWidth_ = this.targetProj_.getExtent() ? L(this.targetProj_.getExtent()) : null;
		let l = pt(n), u = mt(n), d = at(n), f = it(n), p = this.transformInv_(l), m = this.transformInv_(u), h = this.transformInv_(d), g = this.transformInv_(f), _ = Ya + (a ? Math.max(0, Math.ceil(Math.log2(rt(n) / (a * a * 256 * 256)))) : 0);
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
		let l = Le([
			i,
			a,
			o,
			s
		]), u = this.sourceWorldWidth_ ? L(l) / this.sourceWorldWidth_ : null, d = this.sourceWorldWidth_, f = this.sourceProj_.canWrapX() && u > .5 && u < 1, p = !1;
		if (c > 0 && (this.targetProj_.isGlobal() && this.targetWorldWidth_ && (p = L(Le([
			e,
			t,
			n,
			r
		])) / this.targetWorldWidth_ > Xa || p), !f && this.sourceProj_.isGlobal() && u && (p = u > Xa || p)), !p && this.maxSourceExtent_ && isFinite(l[0]) && isFinite(l[1]) && isFinite(l[2]) && isFinite(l[3]) && !ht(l, this.maxSourceExtent_)) return;
		let m = 0;
		if (!p && (!isFinite(i[0]) || !isFinite(i[1]) || !isFinite(a[0]) || !isFinite(a[1]) || !isFinite(o[0]) || !isFinite(o[1]) || !isFinite(s[0]) || !isFinite(s[1]))) {
			if (c > 0) p = !0;
			else if (m = (!isFinite(i[0]) || !isFinite(i[1]) ? 8 : 0) + (!isFinite(a[0]) || !isFinite(a[1]) ? 4 : 0) + (!isFinite(o[0]) || !isFinite(o[1]) ? 2 : 0) + +(!isFinite(s[0]) || !isFinite(s[1])), m != 1 && m != 2 && m != 4 && m != 8) return;
		}
		if (c > 0) {
			if (!p) {
				let t = [(e[0] + n[0]) / 2, (e[1] + n[1]) / 2], r = this.transformInv_(t), a;
				a = f ? (Nt(i[0], d) + Nt(o[0], d)) / 2 - Nt(r[0], d) : (i[0] + o[0]) / 2 - r[0];
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
		let e = Ke();
		return this.triangles_.forEach(function(t, n, r) {
			let i = t.source;
			$e(e, i[0]), $e(e, i[1]), $e(e, i[2]);
		}), e;
	}
	getTriangles() {
		return this.triangles_;
	}
}, Qa = .5, $a = class extends ka {
	constructor(e, t, n, r, i, a, o, s, c, l, u, d) {
		super(i, B.IDLE, d), this.renderEdges_ = u !== void 0 && u, this.pixelRatio_ = o, this.gutter_ = s, this.canvas_ = null, this.sourceTileGrid_ = t, this.targetTileGrid_ = r, this.wrappedTileCoord_ = a || i, this.sourceTiles_ = [], this.sourcesListenerKeys_ = null, this.sourceZ_ = 0, this.clipExtent_ = e.canWrapX() ? e.getExtent() : void 0;
		let f = r.getTileCoordExtent(this.wrappedTileCoord_), p = this.targetTileGrid_.getExtent(), m = this.sourceTileGrid_.getExtent(), h = p ? dt(f, p) : f;
		if (rt(h) === 0) {
			this.state = B.EMPTY;
			return;
		}
		let g = e.getExtent();
		g && (m = m ? dt(m, g) : g);
		let _ = r.getResolution(this.wrappedTileCoord_[0]), v = qa(e, n, h, _);
		if (!isFinite(v) || v <= 0) {
			this.state = B.EMPTY;
			return;
		}
		let y = l === void 0 ? Qa : l;
		if (this.triangulation_ = new Za(e, n, h, m, v * y, _), this.triangulation_.getTriangles().length === 0) {
			this.state = B.EMPTY;
			return;
		}
		this.sourceZ_ = t.getZForResolution(v);
		let b = this.triangulation_.calculateSourceExtent();
		if (m && (e.canWrapX() ? (b[1] = Dt(b[1], m[1], m[3]), b[3] = Dt(b[3], m[1], m[3])) : b = dt(b, m)), !rt(b)) this.state = B.EMPTY;
		else {
			let n = 0, r = 0;
			e.canWrapX() && (n = L(g), r = Math.floor((b[0] - g[0]) / n)), St(b.slice(), e, !0).forEach((e) => {
				let i = t.getTileRangeForExtentAndZ(e, this.sourceZ_);
				for (let e = i.minX; e <= i.maxX; e++) for (let t = i.minY; t <= i.maxY; t++) {
					let i = r * n;
					this.sourceTiles_.push({
						getTile: () => c(this.sourceZ_, e, t, o),
						offset: i
					});
				}
				++r;
			}), this.sourceTiles_.length === 0 && (this.state = B.EMPTY);
		}
	}
	getImage() {
		return this.canvas_;
	}
	reproject_() {
		let e = [];
		if (this.sourceTiles_.forEach((t) => {
			let n = t.tile;
			if (n && n.getState() == B.LOADED) {
				let r = this.sourceTileGrid_.getTileCoordExtent(n.tileCoord);
				r[0] += t.offset, r[2] += t.offset;
				let i = this.clipExtent_?.slice();
				i && (i[0] += t.offset, i[2] += t.offset), e.push({
					extent: r,
					clipExtent: i,
					image: n.getImage()
				});
			}
		}), this.sourceTiles_.length = 0, e.length === 0) this.state = B.ERROR;
		else {
			let t = this.wrappedTileCoord_[0], n = this.targetTileGrid_.getTileSize(t), r = typeof n == "number" ? n : n[0], i = typeof n == "number" ? n : n[1], a = this.targetTileGrid_.getResolution(t), o = this.sourceTileGrid_.getResolution(this.sourceZ_), s = this.targetTileGrid_.getTileCoordExtent(this.wrappedTileCoord_);
			this.canvas_ = Ja(r, i, this.pixelRatio_, o, this.sourceTileGrid_.getExtent(), a, s, this.triangulation_, e, this.gutter_, this.renderEdges_, this.interpolate), this.state = B.LOADED;
		}
		this.changed();
	}
	load() {
		for (let e of this.sourceTiles_) e.tile = e.getTile();
		if (this.state == B.IDLE) {
			this.state = B.LOADING, this.changed();
			let e = 0;
			this.sourcesListenerKeys_ = [], this.sourceTiles_.forEach(({ tile: t }) => {
				let n = t.getState();
				if (n == B.IDLE || n == B.LOADING) {
					e++;
					let n = i(t, s.CHANGE, (r) => {
						let i = t.getState();
						(i == B.LOADED || i == B.ERROR || i == B.EMPTY) && (o(n), e--, e === 0 && (this.unlistenSources_(), this.reproject_()));
					});
					this.sourcesListenerKeys_.push(n);
				}
			}), e === 0 ? setTimeout(this.reproject_.bind(this), 0) : this.sourceTiles_.forEach(function({ tile: e }, t, n) {
				e.getState() == B.IDLE && e.load();
			});
		}
	}
	unlistenSources_() {
		this.sourcesListenerKeys_.forEach(o), this.sourcesListenerKeys_ = null;
	}
	release() {
		this.canvas_ &&= (ve(this.canvas_.getContext("2d")), Ha.push(this.canvas_), null), this.sourceTiles_.length = 0, super.release();
	}
};
//#endregion
//#region node_modules/ol/size.js
function eo(e) {
	return e[0] > 0 && e[1] > 0;
}
function to(e, t, n) {
	return n === void 0 && (n = [0, 0]), n[0] = e[0] * t + .5 | 0, n[1] = e[1] * t + .5 | 0, n;
}
function no(e, t) {
	return Array.isArray(e) ? e : (t === void 0 ? t = [e, e] : (t[0] = e, t[1] = e), t);
}
//#endregion
//#region node_modules/ol/structs/LRUCache.js
var ro = class {
	constructor(e) {
		this.highWaterMark = e === void 0 ? 2048 : e, this.count_ = 0, this.entries_ = {}, this.oldest_ = null, this.newest_ = null;
	}
	deleteOldest() {
		let e = this.pop();
		e instanceof c && e.dispose();
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
function io(e, t, n, r) {
	return r === void 0 ? [
		e,
		t,
		n
	] : (r[0] = e, r[1] = t, r[2] = n, r);
}
function ao(e, t, n) {
	return e + "/" + t + "/" + n;
}
function oo(e, t, n, r, i) {
	return `${O(e)},${t},${ao(n, r, i)}`;
}
function so(e) {
	return co(e[0], e[1], e[2]);
}
function co(e, t, n) {
	return (t << e) + n;
}
function lo(e, t) {
	let n = e[0], r = e[1], i = e[2];
	if (t.getMinZoom() > n || n > t.getMaxZoom()) return !1;
	let a = t.getFullTileRange(n);
	return !a || a.containsXY(r, i);
}
//#endregion
//#region node_modules/ol/color.js
var uo = [
	NaN,
	NaN,
	NaN,
	0
], fo;
function po() {
	return fo ||= I(1, 1, void 0, {
		willReadFrequently: !0,
		desynchronized: !0
	}), fo;
}
var mo = /^rgba?\(\s*(\d+%?)\s+(\d+%?)\s+(\d+%?)(?:\s*\/\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, ho = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, go = /^rgba?\(\s*(\d+%)\s*,\s*(\d+%)\s*,\s*(\d+%)(?:\s*,\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, _o = /^#([\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$/i;
function vo(e, t) {
	return e.endsWith("%") ? Number(e.substring(0, e.length - 1)) / t : Number(e);
}
function yo(e) {
	throw Error("failed to parse \"" + e + "\" as color");
}
function bo(e) {
	if (e.toLowerCase().startsWith("rgb")) {
		let t = e.match(ho) || e.match(mo) || e.match(go);
		if (t) {
			let e = t[4], n = 100 / 255;
			return [
				Dt(vo(t[1], n) + .5 | 0, 0, 255),
				Dt(vo(t[2], n) + .5 | 0, 0, 255),
				Dt(vo(t[3], n) + .5 | 0, 0, 255),
				e === void 0 ? 1 : Dt(vo(e, 100), 0, 1)
			];
		}
		yo(e);
	}
	if (e.startsWith("#")) {
		if (_o.test(e)) {
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
		yo(e);
	}
	let t = po();
	t.fillStyle = "#abcdef";
	let n = t.fillStyle;
	t.fillStyle = e, t.fillStyle === n && (t.fillStyle = "#fedcba", n = t.fillStyle, t.fillStyle = e, t.fillStyle === n && yo(e));
	let r = t.fillStyle;
	if (r.startsWith("#") || r.startsWith("rgba")) return bo(r);
	t.clearRect(0, 0, 1, 1), t.fillRect(0, 0, 1, 1);
	let i = Array.from(t.getImageData(0, 0, 1, 1).data);
	return i[3] = Ft(i[3] / 255, 3), i;
}
function xo(e) {
	return typeof e == "string" ? e : Po(e);
}
var So = 1024, Co = {}, wo = 0;
function To(e) {
	if (e.length === 4) return e;
	let t = e.slice();
	return t[3] = 1, t;
}
function Eo(e) {
	return e > .0031308 ? e ** (1 / 2.4) * 269.025 - 14.025 : e * 3294.6;
}
function Do(e) {
	return e > .2068965 ? e ** 3 : (e - 4 / 29) * (108 / 841);
}
function Oo(e) {
	return e > 10.314724 ? ((e + 14.025) / 269.025) ** 2.4 : e / 3294.6;
}
function ko(e) {
	return e > .0088564 ? e ** (1 / 3) : e / (108 / 841) + 4 / 29;
}
function Ao(e) {
	let t = Oo(e[0]), n = Oo(e[1]), r = Oo(e[2]), i = ko(t * .222488403 + n * .716873169 + r * .06060791), a = 500 * (ko(t * .452247074 + n * .399439023 + r * .148375274) - i), o = 200 * (i - ko(t * .016863605 + n * .117638439 + r * .865350722)), s = 180 / Math.PI * Math.atan2(o, a);
	return [
		116 * i - 16,
		Math.sqrt(a * a + o * o),
		s < 0 ? s + 360 : s,
		e[3]
	];
}
function jo(e) {
	let t = (e[0] + 16) / 116, n = e[1], r = e[2] * Math.PI / 180, i = Do(t), a = Do(t + n / 500 * Math.cos(r)), o = Do(t - n / 200 * Math.sin(r)), s = Eo(a * 3.021973625 - i * 1.617392459 - o * .404875592), c = Eo(a * -.943766287 + i * 1.916279586 + o * .027607165), l = Eo(a * .069407491 - i * .22898585 + o * 1.159737864);
	return [
		Dt(s + .5 | 0, 0, 255),
		Dt(c + .5 | 0, 0, 255),
		Dt(l + .5 | 0, 0, 255),
		e[3]
	];
}
function Mo(e) {
	if (e === "none") return uo;
	if (Co.hasOwnProperty(e)) return Co[e];
	if (wo >= So) {
		let e = 0;
		for (let t in Co) e++ & 3 || (delete Co[t], --wo);
	}
	let t = bo(e);
	t.length !== 4 && yo(e);
	for (let n of t) isNaN(n) && yo(e);
	return Co[e] = t, ++wo, t;
}
function No(e) {
	return Array.isArray(e) ? e : Mo(e);
}
function Po(e) {
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
//#region node_modules/ol/render/Event.js
var Fo = class extends S {
	constructor(e, t, n, r) {
		super(e), this.inversePixelTransform = t, this.frameState = n, this.context = r;
	}
}, Io = {
	PRERENDER: "prerender",
	POSTRENDER: "postrender",
	PRECOMPOSE: "precompose",
	POSTCOMPOSE: "postcompose",
	RENDERCOMPLETE: "rendercomplete"
}, Lo = class {
	constructor() {
		this.instructions_ = [], this.zIndex = 0, this.offset_ = 0, this.pendingMethod_, this.context_ = new Proxy(_e(), {
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
}, Ro = 5, zo = class extends w {
	constructor(e) {
		super(), this.ready = !0, this.boundHandleImageChange_ = this.handleImageChange_.bind(this), this.layer_ = e, this.staleKeys_ = [], this.maxStaleKeys = Ro, this.renderedSourceKey_;
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
		return E();
	}
	getData(e) {
		return null;
	}
	prepareFrame(e) {
		return E();
	}
	renderFrame(e, t) {
		return E();
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {}
	getLayer() {
		return this.layer_;
	}
	handleFontsChanged() {}
	handleImageChange_(e) {
		let t = e.target;
		(t.getState() === V.LOADED || t.getState() === V.ERROR) && this.renderIfReadyAndVisible();
	}
	loadImage(e) {
		let t = e.getState();
		return t != V.LOADED && t != V.ERROR && e.addEventListener(s.CHANGE, this.boundHandleImageChange_), t == V.IDLE && (e.load(), t = e.getState()), t == V.LOADED;
	}
	renderIfReadyAndVisible() {
		let e = this.getLayer();
		e && e.getVisible() && e.getSourceState() === "ready" && e.changed();
	}
	renderDeferred(e) {}
	disposeInternal() {
		delete this.layer_, super.disposeInternal();
	}
}, Bo = [], Vo = null;
function Ho() {
	Vo = I(1, 1, void 0, { willReadFrequently: !0 });
}
var Uo = class extends zo {
	constructor(e) {
		super(e), this.container = null, this.renderedResolution, this.tempTransform = Tr(), this.pixelTransform = Tr(), this.inversePixelTransform = Tr(), this.context = null, this.deferredContext_ = null, this.containerReused = !1, this.frameState = null;
	}
	getImageData(e, t, n) {
		Vo || Ho(), Vo.clearRect(0, 0, 1, 1);
		let r;
		try {
			Vo.drawImage(e, t, n, 1, 1, 0, 0, 1, 1), r = Vo.getImageData(0, 0, 1, 1).data;
		} catch {
			return Vo = null, null;
		}
		return r;
	}
	getBackground(e) {
		let t = this.getLayer().getBackground();
		return typeof t == "function" && (t = t(e.viewState.resolution)), t || void 0;
	}
	useContainer(e, t, n, r, i) {
		if (Te(e) && this.pixelTransform[1] === 0 && this.pixelTransform[2] === 0 && this.pixelTransform[4] === 0 && this.pixelTransform[5] === 0 && e.width === r && e.height === i) {
			let t = e, r = t.getContext("2d");
			if (r) {
				this.container = e, this.context = r, this.containerReused = !0, n && (r.fillStyle = n, r.fillRect(0, 0, t.width, t.height));
				return;
			}
		}
		let a = this.getLayer().getClassName(), o, s;
		if (e && e.className === a && (!n || e && e.style.backgroundColor && h(No(e.style.backgroundColor), No(n)))) {
			let t = e.firstElementChild;
			Te(t) && (s = t.getContext("2d"));
		}
		if (s && Rr(s.canvas.style.transform, t) ? (this.container = e, this.context = s, this.containerReused = !0) : this.containerReused ? (this.container = null, this.context = null, this.containerReused = !1) : this.container && (this.container.style.backgroundColor = null), !this.container) {
			o = pe ? we() : document.createElement("div"), o.className = a;
			let e = o.style;
			e.position = "absolute", e.width = "100%", e.height = "100%", s = I();
			let t = s.canvas;
			o.appendChild(t), e = t.style, e.position = "absolute", e.left = "0", e.transformOrigin = "top left", this.container = o, this.context = s;
		}
		!this.containerReused && n && !this.container.style.backgroundColor && (this.container.style.backgroundColor = n);
	}
	clipUnrotated(e, t, n) {
		let r = pt(n), i = mt(n), a = at(n), o = it(n);
		z(t.coordinateToPixelTransform, r), z(t.coordinateToPixelTransform, i), z(t.coordinateToPixelTransform, a), z(t.coordinateToPixelTransform, o);
		let s = this.inversePixelTransform;
		z(s, r), z(s, i), z(s, a), z(s, o), e.save(), e.beginPath(), e.moveTo(Math.round(r[0]), Math.round(r[1])), e.lineTo(Math.round(i[0]), Math.round(i[1])), e.lineTo(Math.round(a[0]), Math.round(a[1])), e.lineTo(Math.round(o[0]), Math.round(o[1])), e.clip();
	}
	prepareContainer(e, t) {
		let n = e.extent, r = e.viewState.resolution, i = e.viewState.rotation, a = e.pixelRatio, o = Math.round(L(n) / r * a), s = Math.round(ut(n) / r * a);
		Mr(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / a, 1 / a, i, -o / 2, -s / 2), Nr(this.inversePixelTransform, this.pixelTransform);
		let c = Ir(this.pixelTransform), l = this.getBackground(e);
		if (this.useContainer(t, c, l, o, s), !this.containerReused) {
			let e = this.context.canvas;
			e.width != o || e.height != s ? (e.width = o, e.height = s) : this.context.clearRect(0, 0, o, s), c !== e.style.transform && (e.style.transform = c);
		}
	}
	dispatchRenderEvent_(e, t, n) {
		let r = this.getLayer();
		if (r.hasListener(e)) {
			let i = new Fo(e, this.inversePixelTransform, n, t);
			r.dispatchEvent(i);
		}
	}
	preRender(e, t) {
		this.frameState = t, !t.declutter && this.dispatchRenderEvent_(Io.PRERENDER, e, t);
	}
	postRender(e, t) {
		t.declutter || this.dispatchRenderEvent_(Io.POSTRENDER, e, t);
	}
	renderDeferredInternal(e) {}
	getRenderContext(e) {
		return e.declutter && !this.deferredContext_ && (this.deferredContext_ = new Lo()), e.declutter ? this.deferredContext_.getContext() : this.context;
	}
	renderDeferred(e) {
		e.declutter && (this.dispatchRenderEvent_(Io.PRERENDER, this.context, e), e.declutter && this.deferredContext_ && (this.deferredContext_.draw(this.context), this.deferredContext_.clear()), this.renderDeferredInternal(e), this.dispatchRenderEvent_(Io.POSTRENDER, this.context, e));
	}
	getRenderTransform(e, t, n, r, i, a, o) {
		let s = i / 2, c = a / 2, l = r / t, u = -l, d = -e[0] + o, f = -e[1];
		return Mr(this.tempTransform, s, c, l, u, -n, d, f);
	}
	disposeInternal() {
		delete this.frameState, super.disposeInternal();
	}
};
//#endregion
//#region node_modules/ol/renderer/canvas/TileLayer.js
function Wo(e, t, n) {
	if (!(n in e)) return e[n] = /* @__PURE__ */ new Set([t]), !0;
	let r = e[n], i = r.has(t);
	return i || r.add(t), !i;
}
function Go(e, t, n) {
	let r = e[n];
	return r ? r.delete(t) : !1;
}
function Ko(e, t) {
	let n = e.layerStatesArray[e.layerIndex];
	n.extent && (t = dt(t, br(n.extent, e.viewState.projection)));
	let r = n.layer.getRenderSource();
	if (!r.getWrapX()) {
		let n = r.getTileGridForProjection(e.viewState.projection).getExtent();
		n && (t = dt(t, n));
	}
	return t;
}
var qo = class extends Uo {
	constructor(e, t) {
		super(e), t ||= {}, this.extentChanged = !0, this.renderComplete = !1, this.renderedExtent_ = null, this.renderedPixelRatio, this.renderedProjection = null, this.renderedTiles = [], this.renderedSourceRevision_, this.tempExtent = Ke(), this.tempTileRange_ = new za(0, 0, 0, 0), this.tempTileCoord_ = io(0, 0, 0);
		let n = t.cacheSize === void 0 ? 512 : t.cacheSize;
		this.tileCache_ = new ro(n), this.sourceTileCache_ = null, this.layerExtent = null, this.maxStaleKeys = n * .5;
	}
	getTileCache() {
		return this.tileCache_;
	}
	getSourceTileCache() {
		return this.sourceTileCache_ ||= new ro(512), this.sourceTileCache_;
	}
	getOrCreateTile(e, t, n, r) {
		let i = this.tileCache_, a = this.getLayer().getSource(), o = oo(a, a.getKey(), e, t, n), s;
		if (i.containsKey(o)) s = i.get(o);
		else {
			let c = r.viewState.projection, l = a.getProjection();
			if (s = a.getTile(e, t, n, r.pixelRatio, c, !l || lr(l, c) ? void 0 : this.getSourceTileCache()), !s) return null;
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
		if (i && !He(i, r)) return null;
		let a = t.viewState, o = n.getRenderSource(), s = o.getTileGridForProjection(a.projection), c = o.getTilePixelRatio(t.pixelRatio);
		for (let e = s.getZForResolution(a.resolution); e >= s.getMinZoom(); --e) {
			let n = s.getTileCoordForCoordAndZ(r, e), i = this.getTile(e, n[1], n[2], t);
			if (!i || i.getState() !== B.LOADED) continue;
			let l = s.getOrigin(e), u = no(s.getTileSize(e)), d = s.getResolution(e), f;
			if (i instanceof La || i instanceof $a) f = i.getImage();
			else if (i instanceof Na) {
				if (f = Aa(i.getData()), !f) continue;
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
		let a = e.viewState, o = this.getLayer(), s = o.getRenderSource(), c = s.getTileGridForProjection(a.projection), l = O(s);
		l in e.wantedTiles || (e.wantedTiles[l] = {});
		let u = e.wantedTiles[l], d = o.getMapInternal(), f = Math.max(n - i, c.getMinZoom(), c.getZForResolution(Math.min(o.getMaxResolution(), d ? d.getView().getResolutionForZoom(Math.max(o.getMinZoom(), 0)) : c.getResolution(0)), s.zDirection)), p = a.rotation, m = p ? lt(a.center, a.resolution, p, e.size) : void 0;
		for (let i = n; i >= f; --i) {
			let n = c.getTileRangeForExtentAndZ(t, i, this.tempTileRange_), a = c.getResolution(i);
			for (let t = n.minX; t <= n.maxX; ++t) for (let o = n.minY; o <= n.maxY; ++o) {
				if (p && !c.tileCoordIntersectsViewport([
					i,
					t,
					o
				], m)) continue;
				let n = this.getTile(i, t, o, e);
				if (!n || !Wo(r, n, i)) continue;
				let s = n.getKey();
				if (u[s] = !0, n.getState() === B.IDLE && !e.tileQueue.isKeyQueued(s)) {
					let r = io(i, t, o, this.tempTileCoord_);
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
			let s = oo(this.getLayer().getSource(), o[e], r, i, a);
			if (n.containsKey(s)) {
				let e = n.peek(s);
				if (e.getState() === B.LOADED) return e.endTransition(O(this)), Wo(t, e, r), !0;
			}
		}
		return !1;
	}
	findAltTiles_(e, t, n, r) {
		let i = e.getTileRangeForTileCoordAndZ(t, n, this.tempTileRange_);
		if (!i) return !1;
		let a = !0, o = this.tileCache_, s = this.getLayer().getRenderSource(), c = s.getKey();
		for (let e = i.minX; e <= i.maxX; ++e) for (let t = i.minY; t <= i.maxY; ++t) {
			let i = oo(s, c, n, e, t), l = !1;
			if (o.containsKey(i)) {
				let e = o.peek(i);
				e.getState() === B.LOADED && (Wo(r, e, n), l = !0);
			}
			l || (a = !1);
		}
		return a;
	}
	renderFrame(e, t) {
		this.renderComplete = !0;
		let n = e.layerStatesArray[e.layerIndex], r = e.viewState, i = r.projection, a = r.resolution, o = r.center, s = e.pixelRatio, c = this.getLayer(), l = c.getSource(), d = l.getTileGridForProjection(i), f = d.getZForResolution(a, l.zDirection), p = d.getResolution(f);
		this.updateStaleKeys(l.getKey());
		let m = e.extent, h = l.getTilePixelRatio(s);
		this.prepareContainer(e, t);
		let g = this.context.canvas.width, _ = this.context.canvas.height;
		this.layerExtent = n.extent ? br(n.extent, i) : null, this.layerExtent && (m = dt(m, this.layerExtent));
		let v = p * g / 2 / h, y = p * _ / 2 / h, b = [
			o[0] - v,
			o[1] - y,
			o[0] + v,
			o[1] + y
		], x = {};
		this.renderedTiles.length = 0;
		let S = c.getPreload();
		if (e.nextExtent && this.enqueueTilesForNextExtent()) {
			let t = d.getZForResolution(r.nextResolution, l.zDirection), n = Ko(e, e.nextExtent);
			this.enqueueTiles(e, n, t, x, S);
		}
		let C = Ko(e, m);
		if (this.enqueueTiles(e, C, f, x, 0), S > 0 && setTimeout(() => {
			this.enqueueTiles(e, C, f - 1, x, S - 1);
		}, 0), !(f in x)) return this.container;
		let w = O(this), T = e.time;
		for (let t of x[f]) {
			let n = t.getState();
			if (n === B.EMPTY) continue;
			let r = t.tileCoord;
			if (n === B.LOADED && t.getAlpha(w, T) === 1) {
				t.endTransition(w);
				continue;
			}
			if (n !== B.ERROR && (this.renderComplete = !1), this.findStaleTile_(r, x)) {
				Go(x, t, f), e.animate = !0;
				continue;
			}
			if (this.findAltTiles_(d, r, f + 1, x)) continue;
			let i = d.getMinZoom();
			for (let e = f - 1; e >= i && !this.findAltTiles_(d, r, e, x); --e);
		}
		let E = p / a * s / h, D = this.getRenderContext(e);
		Mr(this.tempTransform, g / 2, _ / 2, E, E, 0, -g / 2, -_ / 2), this.layerExtent && this.clipUnrotated(D, e, this.layerExtent), l.getInterpolate() || (D.imageSmoothingEnabled = !1), this.preRender(D, e);
		let k = Object.keys(x).map(Number);
		k.sort(u);
		let A = [], j = [], M = [];
		for (let t = k.length - 1; t >= 0; --t) {
			let n = k[t], r = l.getTilePixelSize(n, s, i), a = d.getResolution(n) / p, o = r[0] * a * E, c = r[1] * a * E, u = d.getTileCoordForCoordAndZ(pt(b), n), m = d.getTileCoordExtent(u), g = z(this.tempTransform, [h * (m[0] - b[0]) / p, h * (b[3] - m[3]) / p]), _ = h * l.getGutterForProjection(i);
			for (let t of x[n]) {
				if (t.getState() !== B.LOADED) continue;
				let r = t.tileCoord, i = u[1] - r[1], a = Math.round(g[0] - (i - 1) * o), s = u[2] - r[2], d = Math.round(g[1] - (s - 1) * c), p = Math.round(g[0] - i * o), m = Math.round(g[1] - s * c), h = a - p, v = d - m, y = n === f;
				if (y && t.inTransition(w)) {
					M.push({
						tile: t,
						x: p,
						y: m,
						w: h,
						h: v,
						gutter: _
					}), this.renderedTiles.unshift(t), this.updateUsedTiles(e.usedTiles, l, t);
					continue;
				}
				let b = [
					p,
					m,
					p + h,
					m + v
				], x = [];
				for (let e = 0, t = A.length; e < t; ++e) n < j[e] && ht(b, A[e]) && x.push(A[e]);
				let S;
				x.length > 0 && (S = Ct(b, x)), A.push(b), j.push(n), this.drawTile(t, e, p, m, h, v, _, y, S), this.renderedTiles.unshift(t), this.updateUsedTiles(e.usedTiles, l, t);
			}
		}
		for (let t = 0, n = M.length; t < n; ++t) {
			let { tile: n, x: r, y: i, w: a, h: o, gutter: s } = M[t];
			this.drawTile(n, e, r, i, a, o, s, !0, void 0);
		}
		return this.renderedResolution = p, this.extentChanged = !this.renderedExtent_ || !Ze(this.renderedExtent_, b), this.renderedExtent_ = b, this.renderedPixelRatio = s, this.postRender(this.context, e), this.layerExtent && D.restore(), D.imageSmoothingEnabled = !0, this.renderComplete && e.postRenderFunctions.push((e, t) => {
			let n = O(l), r = t.wantedTiles[n], i = r ? Object.keys(r).length : 0;
			this.updateCacheSize(i), this.tileCache_.expireCache(), this.sourceTileCache_?.expireCache();
		}), this.container;
	}
	updateCacheSize(e) {
		this.tileCache_.highWaterMark = Math.max(this.tileCache_.highWaterMark, e * 2);
	}
	drawTile(e, t, n, r, i, a, o, s, c) {
		let l;
		if (e instanceof Na) {
			if (l = Aa(e.getData()), !l) throw Error("Rendering array data is not yet supported");
		} else l = this.getTileImage(e);
		if (!l) return;
		let u = this.getRenderContext(t), d = O(this), f = t.layerStatesArray[t.layerIndex], p = f.opacity * (s ? e.getAlpha(d, t.time) : 1), m = p !== u.globalAlpha;
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
		let r = O(t);
		r in e || (e[r] = {}), e[r][n.getKey()] = !0;
	}
}, Jo = {
	ANIMATING: 0,
	INTERACTING: 1
}, Yo = {
	CENTER: "center",
	RESOLUTION: "resolution",
	ROTATION: "rotation"
};
//#endregion
//#region node_modules/ol/centerconstraint.js
function Xo(e, t, n) {
	return (function(r, i, a, o, s) {
		if (!r) return;
		if (!i && !t) return r;
		let c = t ? 0 : a[0] * i, l = t ? 0 : a[1] * i, u = s ? s[0] : 0, d = s ? s[1] : 0, f = e[0] + c / 2 + u, p = e[2] - c / 2 + u, m = e[1] + l / 2 + d, h = e[3] - l / 2 + d;
		f > p && (f = (p + f) / 2, p = f), m > h && (m = (h + m) / 2, h = m);
		let g = Dt(r[0], f, p), _ = Dt(r[1], m, h);
		if (o && n && i) {
			let e = 30 * i;
			g += -e * Math.log(1 + Math.max(0, f - r[0]) / e) + e * Math.log(1 + Math.max(0, r[0] - p) / e), _ += -e * Math.log(1 + Math.max(0, m - r[1]) / e) + e * Math.log(1 + Math.max(0, r[1] - h) / e);
		}
		return [g, _];
	});
}
function Zo(e) {
	return e;
}
//#endregion
//#region node_modules/ol/resolutionconstraint.js
function Qo(e, t, n, r) {
	let i = L(t) / n[0], a = ut(t) / n[1];
	return r ? Math.min(e, Math.max(i, a)) : Math.min(e, Math.min(i, a));
}
function $o(e, t, n) {
	let r = Math.min(e, t);
	return r *= Math.log(1 + 50 * Math.max(0, e / t - 1)) / 50 + 1, n && (r = Math.max(r, n), r /= Math.log(1 + 50 * Math.max(0, n / e - 1)) / 50 + 1), Dt(r, n / 2, t * 2);
}
function es(e, t, n, r) {
	return t = t === void 0 || t, (function(i, a, o, s) {
		if (i !== void 0) {
			let c = e[0], l = e[e.length - 1], u = n ? Qo(c, n, o, r) : c;
			if (s) return t ? $o(i, u, l) : Dt(i, l, u);
			let d = Math.floor(f(e, Math.min(u, i), a));
			return e[d] > u && d < e.length - 1 ? e[d + 1] : e[d];
		}
	});
}
function ts(e, t, n, r, i, a) {
	return r = r === void 0 || r, n = n === void 0 ? 0 : n, (function(o, s, c, l) {
		if (o !== void 0) {
			let u = i ? Qo(t, i, c, a) : t;
			if (l) return r ? $o(o, u, n) : Dt(o, n, u);
			let d = Math.ceil(Math.log(t / u) / Math.log(e) - 1e-9), f = -s * .499999999 + .5, p = Math.floor(Math.log(t / Math.min(u, o)) / Math.log(e) + f);
			return Dt(t / e ** +Math.max(d, p), n, u);
		}
	});
}
function ns(e, t, n, r, i) {
	return n = n === void 0 || n, (function(a, o, s, c) {
		if (a !== void 0) {
			let o = r ? Qo(e, r, s, i) : e;
			return !n || !c ? Dt(a, t, o) : $o(a, o, t);
		}
	});
}
//#endregion
//#region node_modules/ol/rotationconstraint.js
function rs(e) {
	if (e !== void 0) return 0;
}
function is(e) {
	if (e !== void 0) return e;
}
function as(e) {
	let t = 2 * Math.PI / e;
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return e = Math.floor(e / t + .5) * t, e;
	});
}
function os(e) {
	let t = e === void 0 ? Mt(5) : e;
	return (function(e, n) {
		return n || e === void 0 ? e : Math.abs(e) <= t ? 0 : e;
	});
}
//#endregion
//#region node_modules/ol/View.js
var ss = 0, cs = class extends A {
	constructor(e) {
		super(), this.on, this.once, this.un, e = Object.assign({}, e), this.hints_ = [0, 0], this.animations_ = [], this.updateAnimationKey_, this.projection_ = sr(e.projection, "EPSG:3857"), this.viewportSize_ = [100, 100], this.targetCenter_ = null, this.targetResolution_, this.targetRotation_, this.nextCenter_ = null, this.nextResolution_, this.nextRotation_, this.cancelAnchor_ = void 0, e.projection && Qn(), e.center && (e.center = vr(e.center, this.projection_)), e.extent && (e.extent = br(e.extent, this.projection_)), this.applyOptions_(e);
	}
	applyOptions_(e) {
		let t = Object.assign({}, e);
		for (let e in Yo) delete t[e];
		this.setProperties(t, !0);
		let n = ds(e);
		this.maxResolution_ = n.maxResolution, this.minResolution_ = n.minResolution, this.zoomFactor_ = n.zoomFactor, this.resolutions_ = e.resolutions, this.padding_ = e.padding, this.minZoom_ = n.minZoom;
		let r = us(e), i = n.constraint, a = fs(e);
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
			n.center && (n = Object.assign({}, n), n.center = vr(n.center, this.getProjection())), n.anchor && (n = Object.assign({}, n), n.anchor = vr(n.anchor, this.getProjection())), t[e] = n;
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
			n && ls(n, !0);
			return;
		}
		let i = Date.now(), a = this.targetCenter_.slice(), o = this.targetResolution_, s = this.targetRotation_, c = [];
		for (; r < t; ++r) {
			let e = arguments[r], t = {
				start: i,
				complete: !1,
				anchor: e.anchor,
				duration: e.duration === void 0 ? 1e3 : e.duration,
				easing: e.easing || je,
				callback: n
			};
			if (e.center && (t.sourceCenter = a, t.targetCenter = e.center.slice(), a = t.targetCenter), e.zoom === void 0 ? e.resolution && (t.sourceResolution = o, t.targetResolution = e.resolution, o = t.targetResolution) : (t.sourceResolution = o, t.targetResolution = this.getResolutionForZoom(e.zoom), o = t.targetResolution), e.rotation !== void 0) {
				t.sourceRotation = s;
				let n = Nt(e.rotation - s + Math.PI, 2 * Math.PI) - Math.PI;
				t.targetRotation = s + n, s = t.targetRotation;
			}
			ps(t) ? t.complete = !0 : i += t.duration, c.push(t);
		}
		this.animations_.push(c), this.setHint(Jo.ANIMATING, 1), this.updateAnimations_();
	}
	getAnimating() {
		return this.hints_[Jo.ANIMATING] > 0;
	}
	getInteracting() {
		return this.hints_[Jo.INTERACTING] > 0;
	}
	cancelAnimations() {
		this.setHint(Jo.ANIMATING, -this.hints_[Jo.ANIMATING]);
		let e;
		for (let t = 0, n = this.animations_.length; t < n; ++t) {
			let n = this.animations_[t];
			if (n[0].callback && ls(n[0].callback, !1), !e) for (let t = 0, r = n.length; t < r; ++t) {
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
					let e = c === 1 ? Nt(a.targetRotation + Math.PI, 2 * Math.PI) - Math.PI : a.sourceRotation + c * (a.targetRotation - a.sourceRotation);
					if (a.anchor) {
						let t = this.constraints_.rotation(e, !0);
						this.targetCenter_ = this.calculateCenterRotate(t, a.anchor);
					}
					this.nextRotation_ = a.targetRotation, this.targetRotation_ = e;
				}
				if (this.applyTargetState_(!0), t = !0, !a.complete) break;
			}
			if (i) {
				this.animations_[n] = null, this.setHint(Jo.ANIMATING, -1), this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
				let e = r[0].callback;
				e && ls(e, !0);
			}
		}
		this.animations_ = this.animations_.filter(Boolean), t && this.updateAnimationKey_ === void 0 && (this.updateAnimationKey_ = requestAnimationFrame(this.updateAnimations_.bind(this)));
	}
	calculateCenterRotate(e, t) {
		let n, r = this.getCenterInternal();
		return r !== void 0 && (n = [r[0] - t[0], r[1] - t[1]], Ht(n, e - this.getRotation()), zt(n, t)), n;
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
		return e && _r(e, this.getProjection());
	}
	getCenterInternal() {
		return this.get(Yo.CENTER);
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
		return yr(this.calculateExtentInternal(e), this.getProjection());
	}
	calculateExtentInternal(e) {
		e ||= this.getViewportSizeMinusPadding_();
		let t = this.getCenterInternal();
		R(t, "The view center is not defined");
		let n = this.getResolution();
		R(n !== void 0, "The view resolution is not defined");
		let r = this.getRotation();
		return R(r !== void 0, "The view rotation is not defined"), ct(t, n, r, e);
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
		return this.get(Yo.RESOLUTION);
	}
	getResolutions() {
		return this.resolutions_;
	}
	getResolutionForExtent(e, t) {
		return this.getResolutionForExtentInternal(br(e, this.getProjection()), t);
	}
	getResolutionForExtentInternal(e, t) {
		t ||= this.getViewportSizeMinusPadding_();
		let n = L(e) / t[0], r = ut(e) / t[1];
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
		return this.get(Yo.ROTATION);
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
			r = ms(r, this.getViewportSize_(), [e[0] / 2 + i[3], e[1] / 2 + i[0]], t, n);
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
			let i = f(this.resolutions_, e, 1);
			t = i, n = this.resolutions_[i], r = i == this.resolutions_.length - 1 ? 2 : n / this.resolutions_[i + 1];
		} else n = this.maxResolution_, r = this.zoomFactor_;
		return t + Math.log(n / e) / Math.log(r);
	}
	getResolutionForZoom(e) {
		if (this.resolutions_?.length) {
			if (this.resolutions_.length === 1) return this.resolutions_[0];
			let t = Dt(Math.floor(e), 0, this.resolutions_.length - 2), n = this.resolutions_[t] / this.resolutions_[t + 1];
			return this.resolutions_[t] / n ** +Dt(e - t, 0, 1);
		}
		return this.maxResolution_ / this.zoomFactor_ ** +(e - this.minZoom_);
	}
	fit(e, t) {
		let n;
		if (R(Array.isArray(e) || typeof e.getSimplifiedGeometry == "function", "Invalid extent or geometry provided as `geometry`"), Array.isArray(e)) R(!gt(e), "Cannot fit empty extent provided as `geometry`"), n = Xi(br(e, this.getProjection()));
		else if (e.getType() === "Circle") {
			let t = br(e.getExtent(), this.getProjection());
			n = Xi(t), n.rotate(this.getRotation(), ot(t));
		} else {
			let t = gr();
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
		let c = this.getRotation(), l = Math.sin(c), u = Math.cos(c), d = ot(o);
		d[0] += (r[1] - r[3]) / 2 * s, d[1] += (r[0] - r[2]) / 2 * s;
		let f = d[0] * u - d[1] * l, p = d[1] * u + d[0] * l, m = this.getConstrainedCenter([f, p], s), h = t.callback ? t.callback : y;
		t.duration === void 0 ? (this.targetResolution_ = s, this.targetCenter_ = m, this.applyTargetState_(!1, !0), ls(h, !0)) : this.animateInternal({
			resolution: s,
			center: m,
			duration: t.duration,
			easing: t.easing
		}, h);
	}
	centerOn(e, t, n) {
		this.centerOnInternal(vr(e, this.getProjection()), t, n);
	}
	centerOnInternal(e, t, n) {
		this.setCenterInternal(ms(e, t, n, this.getResolution(), this.getRotation()));
	}
	calculateCenterShift(e, t, n, r) {
		let i, a = this.padding_;
		if (a && e) {
			let o = this.getViewportSizeMinusPadding_(-n), s = ms(e, r, [o[0] / 2 + a[3], o[1] / 2 + a[0]], t, n);
			i = [e[0] - s[0], e[1] - s[1]];
		}
		return i;
	}
	isDef() {
		return !!this.getCenterInternal() && this.getResolution() !== void 0;
	}
	adjustCenter(e) {
		let t = _r(this.targetCenter_, this.getProjection());
		this.setCenter([t[0] + e[0], t[1] + e[1]]);
	}
	adjustCenterInternal(e) {
		let t = this.targetCenter_;
		this.setCenterInternal([t[0] + e[0], t[1] + e[1]]);
	}
	adjustResolution(e, t) {
		t &&= vr(t, this.getProjection()), this.adjustResolutionInternal(e, t);
	}
	adjustResolutionInternal(e, t) {
		let n = this.getAnimating() || this.getInteracting(), r = this.getViewportSize_(this.getRotation()), i = this.constraints_.resolution(this.targetResolution_ * e, 0, r, n);
		t && (this.targetCenter_ = this.calculateCenterZoom(i, t)), this.targetResolution_ *= e, this.applyTargetState_();
	}
	adjustZoom(e, t) {
		this.adjustResolution(this.zoomFactor_ ** +-e, t);
	}
	adjustRotation(e, t) {
		t &&= vr(t, this.getProjection()), this.adjustRotationInternal(e, t);
	}
	adjustRotationInternal(e, t) {
		let n = this.getAnimating() || this.getInteracting(), r = this.constraints_.rotation(this.targetRotation_ + e, n);
		t && (this.targetCenter_ = this.calculateCenterRotate(r, t)), this.targetRotation_ += e, this.applyTargetState_();
	}
	setCenter(e) {
		this.setCenterInternal(e && vr(e, this.getProjection()));
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
		this.get(Yo.ROTATION) !== r && this.set(Yo.ROTATION, r), this.get(Yo.RESOLUTION) !== a && (this.set(Yo.RESOLUTION, a), this.set("zoom", this.getZoom(), !0)), (!o || !this.get(Yo.CENTER) || !Vt(this.get(Yo.CENTER), o)) && this.set(Yo.CENTER, o), this.getAnimating() && !e && this.cancelAnimations(), this.cancelAnchor_ = void 0;
	}
	resolveConstraints(e, t, n) {
		e = e === void 0 ? 200 : e;
		let r = t || 0, i = this.constraints_.rotation(this.targetRotation_), a = this.getViewportSize_(i), o = this.constraints_.resolution(this.targetResolution_, r, a), s = this.constraints_.center(this.targetCenter_, o, a, !1, this.calculateCenterShift(this.targetCenter_, o, i, a));
		if (e === 0 && !this.cancelAnchor_) {
			this.targetResolution_ = o, this.targetRotation_ = i, this.targetCenter_ = s, this.applyTargetState_();
			return;
		}
		n ||= e === 0 ? this.cancelAnchor_ : void 0, this.cancelAnchor_ = void 0, (this.getResolution() !== o || this.getRotation() !== i || !this.getCenterInternal() || !Vt(this.getCenterInternal(), s)) && (this.getAnimating() && this.cancelAnimations(), this.animateInternal({
			rotation: i,
			center: s,
			resolution: o,
			duration: e,
			easing: Ae,
			anchor: n
		}));
	}
	beginInteraction() {
		this.resolveConstraints(0), this.setHint(Jo.INTERACTING, 1);
	}
	endInteraction(e, t, n) {
		n &&= vr(n, this.getProjection()), this.endInteractionInternal(e, t, n);
	}
	endInteractionInternal(e, t, n) {
		this.getInteracting() && (this.setHint(Jo.INTERACTING, -1), this.resolveConstraints(e, t, n));
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
function ls(e, t) {
	setTimeout(function() {
		e(t);
	}, 0);
}
function us(e) {
	if (e.extent !== void 0) {
		let t = e.smoothExtentConstraint === void 0 || e.smoothExtentConstraint;
		return Xo(e.extent, e.constrainOnlyCenter, t);
	}
	let t = sr(e.projection, "EPSG:3857");
	if (e.multiWorld !== !0 && t.isGlobal()) {
		let e = t.getExtent().slice();
		return e[0] = -Infinity, e[2] = Infinity, Xo(e, !1, !1);
	}
	return Zo;
}
function ds(e) {
	let t, n, r, i = e.minZoom === void 0 ? ss : e.minZoom, a = e.maxZoom === void 0 ? 28 : e.maxZoom, o = e.zoomFactor === void 0 ? 2 : e.zoomFactor, s = e.multiWorld !== void 0 && e.multiWorld, c = e.smoothResolutionConstraint === void 0 || e.smoothResolutionConstraint, l = e.showFullExtent !== void 0 && e.showFullExtent, u = sr(e.projection, "EPSG:3857"), d = u.getExtent(), f = e.constrainOnlyCenter, p = e.extent;
	if (!s && !p && u.isGlobal() && (f = !1, p = d), e.resolutions !== void 0) {
		let o = e.resolutions;
		n = o[i], r = o[a] === void 0 ? o[o.length - 1] : o[a], t = e.constrainResolution ? es(o, c, !f && p, l) : ns(n, r, c, !f && p, l);
	} else {
		let s = (d ? Math.max(L(d), ut(d)) : 360 * qt.degrees / u.getMetersPerUnit()) / 256 / 2 ** ss, m = s / 2 ** 28;
		n = e.maxResolution, n === void 0 ? n = s / o ** +i : i = 0, r = e.minResolution, r === void 0 && (r = e.maxZoom === void 0 ? m : e.maxResolution === void 0 ? s / o ** +a : n / o ** +a), a = i + Math.floor(Math.log(n / r) / Math.log(o)), r = n / o ** +(a - i), t = e.constrainResolution ? ts(o, n, r, c, !f && p, l) : ns(n, r, c, !f && p, l);
	}
	return {
		constraint: t,
		maxResolution: n,
		minResolution: r,
		minZoom: i,
		zoomFactor: o
	};
}
function fs(e) {
	if (e.enableRotation === void 0 || e.enableRotation) {
		let t = e.constrainRotation;
		return t === void 0 || t === !0 ? os() : t === !1 ? is : typeof t == "number" ? as(t) : is;
	}
	return rs;
}
function ps(e) {
	return !(e.sourceCenter && e.targetCenter && !Vt(e.sourceCenter, e.targetCenter) || e.sourceResolution !== e.targetResolution || e.sourceRotation !== e.targetRotation);
}
function ms(e, t, n, r, i) {
	let a = Math.cos(-i), o = Math.sin(-i), s = e[0] * a - e[1] * o, c = e[1] * a + e[0] * o;
	return s += (t[0] / 2 - n[0]) * r, c += (n[1] - t[1] / 2) * r, o = -o, [s * a - c * o, c * a + s * o];
}
//#endregion
//#region node_modules/ol/layer/Property.js
var H = {
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
}, hs = class extends A {
	constructor(e) {
		super(), this.on, this.once, this.un, this.background_ = e.background;
		let t = Object.assign({}, e);
		typeof e.properties == "object" && (delete t.properties, Object.assign(t, e.properties)), t[H.OPACITY] = e.opacity === void 0 ? 1 : e.opacity, R(typeof t[H.OPACITY] == "number", "Layer opacity must be a number"), t[H.VISIBLE] = e.visible === void 0 || e.visible, t[H.Z_INDEX] = e.zIndex, t[H.MAX_RESOLUTION] = e.maxResolution === void 0 ? Infinity : e.maxResolution, t[H.MIN_RESOLUTION] = e.minResolution === void 0 ? 0 : e.minResolution, t[H.MIN_ZOOM] = e.minZoom === void 0 ? -Infinity : e.minZoom, t[H.MAX_ZOOM] = e.maxZoom === void 0 ? Infinity : e.maxZoom, this.className_ = t.className === void 0 ? "ol-layer" : t.className, delete t.className, this.setProperties(t), this.state_ = null;
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
		return t.opacity = Dt(Math.round(this.getOpacity() * 100) / 100, 0, 1), t.visible = this.getVisible(), t.extent = this.getExtent(), t.zIndex = n === void 0 && !t.managed ? Infinity : n, t.maxResolution = this.getMaxResolution(), t.minResolution = Math.max(this.getMinResolution(), 0), t.minZoom = this.getMinZoom(), t.maxZoom = this.getMaxZoom(), this.state_ = t, t;
	}
	getLayersArray(e) {
		return E();
	}
	getLayerStatesArray(e) {
		return E();
	}
	getExtent() {
		return this.get(H.EXTENT);
	}
	getMaxResolution() {
		return this.get(H.MAX_RESOLUTION);
	}
	getMinResolution() {
		return this.get(H.MIN_RESOLUTION);
	}
	getMinZoom() {
		return this.get(H.MIN_ZOOM);
	}
	getMaxZoom() {
		return this.get(H.MAX_ZOOM);
	}
	getOpacity() {
		return this.get(H.OPACITY);
	}
	getSourceState() {
		return E();
	}
	getVisible() {
		return this.get(H.VISIBLE);
	}
	getZIndex() {
		return this.get(H.Z_INDEX);
	}
	setBackground(e) {
		this.background_ = e, this.changed();
	}
	setExtent(e) {
		this.set(H.EXTENT, e);
	}
	setMaxResolution(e) {
		this.set(H.MAX_RESOLUTION, e);
	}
	setMinResolution(e) {
		this.set(H.MIN_RESOLUTION, e);
	}
	setMaxZoom(e) {
		this.set(H.MAX_ZOOM, e);
	}
	setMinZoom(e) {
		this.set(H.MIN_ZOOM, e);
	}
	setOpacity(e) {
		R(typeof e == "number", "Layer opacity must be a number"), this.set(H.OPACITY, e);
	}
	setVisible(e) {
		this.set(H.VISIBLE, e);
	}
	setZIndex(e) {
		this.set(H.Z_INDEX, e);
	}
	disposeInternal() {
		this.state_ &&= (this.state_.layer = null, null), super.disposeInternal();
	}
}, gs = class extends hs {
	constructor(e) {
		let t = Object.assign({}, e);
		delete t.source, super(t), this.on, this.once, this.un, this.mapPrecomposeKey_ = null, this.mapRenderKey_ = null, this.sourceChangeKey_ = null, this.renderer_ = null, this.sourceReady_ = !1, this.rendered = !1, e.render && (this.render = e.render), e.map && this.setMap(e.map), this.addChangeListener(H.SOURCE, this.handleSourcePropertyChange_);
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
		return this.get(H.SOURCE) || null;
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
		this.sourceChangeKey_ &&= (o(this.sourceChangeKey_), null), this.sourceReady_ = !1;
		let e = this.getSource();
		e && (this.sourceChangeKey_ = i(e, s.CHANGE, this.handleSourceChange_, this), e.getState() === "ready" && (this.sourceReady_ = !0, setTimeout(() => {
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
		!e && n && (e = n.getView()), t = e instanceof cs ? {
			viewState: e.getState(),
			extent: e.calculateExtent()
		} : e, !t.layerStatesArray && n && (t.layerStatesArray = n.getLayerGroup().getLayerStatesArray());
		let r;
		if (t.layerStatesArray) {
			if (r = t.layerStatesArray.find((e) => e.layer === this), !r) return !1;
		} else r = this.getLayerState();
		let i = this.getExtent();
		return _s(r, t.viewState) && (!i || ht(i, t.extent));
	}
	getAttributions(e) {
		if (!this.isVisible(e)) return [];
		let t = this.getSource()?.getAttributions();
		if (!t) return [];
		let n = t(e instanceof cs ? e.getViewStateAndExtent() : e);
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
		e || this.unrender(), this.set(H.MAP, e);
	}
	getMapInternal() {
		return this.get(H.MAP);
	}
	setMap(e) {
		this.mapPrecomposeKey_ &&= (o(this.mapPrecomposeKey_), null), e || this.changed(), this.mapRenderKey_ &&= (o(this.mapRenderKey_), null), e && (this.mapPrecomposeKey_ = i(e, Io.PRECOMPOSE, this.handlePrecompose_, this), this.mapRenderKey_ = i(this, s.CHANGE, e.render, e), this.changed());
	}
	handlePrecompose_(e) {
		let t = e.frameState.layerStatesArray, n = this.getLayerState(!1);
		R(!t.some((e) => e.layer === n.layer), "A layer can only be added to the map once. Use either `layer.setMap()` or `map.addLayer()`, not both."), t.push(n);
	}
	setSource(e) {
		this.set(H.SOURCE, e);
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
function _s(e, t) {
	if (!e.visible) return !1;
	let n = t.resolution;
	if (n < e.minResolution || n >= e.maxResolution) return !1;
	let r = t.zoom;
	return r > e.minZoom && r <= e.maxZoom;
}
//#endregion
//#region node_modules/ol/layer/TileProperty.js
var vs = {
	PRELOAD: "preload",
	USE_INTERIM_TILES_ON_ERROR: "useInterimTilesOnError"
}, ys = class extends gs {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e), n = e.cacheSize;
		delete e.cacheSize, delete t.preload, delete t.useInterimTilesOnError, super(t), this.on, this.once, this.un, this.cacheSize_ = n, this.setPreload(e.preload === void 0 ? 0 : e.preload), this.setUseInterimTilesOnError(e.useInterimTilesOnError === void 0 || e.useInterimTilesOnError);
	}
	getCacheSize() {
		return this.cacheSize_;
	}
	getPreload() {
		return this.get(vs.PRELOAD);
	}
	setPreload(e) {
		this.set(vs.PRELOAD, e);
	}
	getUseInterimTilesOnError() {
		return this.get(vs.USE_INTERIM_TILES_ON_ERROR);
	}
	setUseInterimTilesOnError(e) {
		this.set(vs.USE_INTERIM_TILES_ON_ERROR, e);
	}
	getData(e) {
		return super.getData(e);
	}
}, bs = class extends ys {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new qo(this, { cacheSize: this.getCacheSize() });
	}
}, xs = class {
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
		let n = Ss(e, t);
		return n in this.cache_ ? this.cache_[n] : null;
	}
	getPattern(e, t) {
		let n = Ss(e, t);
		return n in this.patternCache_ ? this.patternCache_[n] : null;
	}
	set(e, t, n, r) {
		let i = Ss(e, t), a = i in this.cache_;
		this.cache_[i] = n, r && (n.getImageState() === V.IDLE && n.load(), n.getImageState() === V.LOADING ? n.ready().then(() => {
			this.patternCache_[i] = _e().createPattern(n.getImage(1), "repeat");
		}) : this.patternCache_[i] = _e().createPattern(n.getImage(1), "repeat")), a || ++this.cacheSize_;
	}
	setSize(e) {
		this.maxCacheSize_ = e, this.expire();
	}
};
function Ss(e, t) {
	let n = t ? No(t) : "null";
	return e + ":" + n;
}
var Cs = new xs(), ws = null, Ts = class extends C {
	constructor(e, t, n, r, i) {
		super(), this.hitDetectionImage_ = null, this.image_ = e, this.crossOrigin_ = n?.crossOrigin, this.referrerPolicy_ = n?.referrerPolicy, this.canvas_ = {}, this.color_ = i, this.imageState_ = r === void 0 ? V.IDLE : r, this.size_ = e && e.width && e.height ? [e.width, e.height] : null, this.src_ = t, this.tainted_, this.ready_ = null;
	}
	initializeImage_() {
		this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_);
	}
	isTainted_() {
		if (this.tainted_ === void 0 && this.imageState_ === V.LOADED) {
			ws ||= I(1, 1, void 0, { willReadFrequently: !0 }), ws.drawImage(this.image_, 0, 0);
			try {
				ws.getImageData(0, 0, 1, 1), this.tainted_ = !1;
			} catch {
				ws = null, this.tainted_ = !0;
			}
		}
		return this.tainted_ === !0;
	}
	dispatchChangeEvent_() {
		this.dispatchEvent(s.CHANGE);
	}
	handleImageError_() {
		this.imageState_ = V.ERROR, this.dispatchChangeEvent_();
	}
	handleImageLoad_() {
		this.imageState_ = V.LOADED, this.size_ = [this.image_.width, this.image_.height], this.dispatchChangeEvent_();
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
				let e = this.size_[0], t = this.size_[1], n = I(e, t);
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
		if (this.imageState_ === V.IDLE) {
			this.image_ || this.initializeImage_(), this.imageState_ = V.LOADING;
			try {
				this.src_ !== void 0 && (this.image_.src = this.src_);
			} catch {
				this.handleImageError_();
			}
			this.image_ instanceof HTMLImageElement && Ia(this.image_, this.src_).then((e) => {
				this.image_ = e, this.handleImageLoad_();
			}).catch(this.handleImageError_.bind(this));
		}
	}
	replaceColor_(e) {
		if (!this.color_ || this.canvas_[e] || this.imageState_ !== V.LOADED) return;
		let t = this.image_, n = I(Math.ceil(t.width * e), Math.ceil(t.height * e)), r = n.canvas;
		n.scale(e, e), n.drawImage(t, 0, 0), n.globalCompositeOperation = "multiply", n.fillStyle = xo(this.color_), n.fillRect(0, 0, r.width / e, r.height / e), n.globalCompositeOperation = "destination-in", n.drawImage(t, 0, 0), this.canvas_[e] = r;
	}
	ready() {
		return this.ready_ ||= new Promise((e) => {
			if (this.imageState_ === V.LOADED || this.imageState_ === V.ERROR) e();
			else {
				let t = () => {
					(this.imageState_ === V.LOADED || this.imageState_ === V.ERROR) && (this.removeEventListener(s.CHANGE, t), e());
				};
				this.addEventListener(s.CHANGE, t);
			}
		}), this.ready_;
	}
};
function Es(e, t, n, r, i, a) {
	let o = t === void 0 ? void 0 : Cs.get(t, i);
	return o || (o = new Ts(e, e && "src" in e ? e.src || void 0 : t, n, r, i), Cs.set(t, i, o, a)), a && o && !Cs.getPattern(t, i) && Cs.set(t, i, o, a), o;
}
//#endregion
//#region node_modules/ol/colorlike.js
function Ds(e) {
	return e ? Array.isArray(e) ? Po(e) : typeof e == "object" && "src" in e ? Os(e) : e : null;
}
function Os(e) {
	if (!e.offset || !e.size) return Cs.getPattern(e.src, e.color);
	let t = e.src + ":" + e.offset, n = Cs.getPattern(t, e.color);
	if (n) return n;
	let r = Cs.get(e.src, null);
	if (r.getImageState() !== V.LOADED) return null;
	let i = I(e.size[0], e.size[1]);
	return i.drawImage(r.getImage(1), e.offset[0], e.offset[1], e.size[0], e.size[1], 0, 0, e.size[0], e.size[1]), Es(i.canvas, t, void 0, V.LOADED, e.color, !0), Cs.getPattern(t, e.color);
}
//#endregion
//#region node_modules/ol/render/VectorContext.js
var ks = class {
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
}, As = "10px sans-serif", js = "#000", Ms = "round", Ns = [], Ps = "round", Fs = "#000", Is = "center", Ls = "middle", Rs = [
	0,
	0,
	0,
	0
], zs = new A(), Bs = null, Vs, Hs = {}, Us = /* @__PURE__ */ new Set([
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
function Ws(e, t, n) {
	return `${e} ${t} 16px "${n}"`;
}
var Gs = (function() {
	let e, t;
	async function r(e) {
		await t.ready;
		let n = se(e), r = n.families[0].toLowerCase(), i = n.weight, a = [];
		return t.forEach((e) => {
			let t = e.family.replace(/^['"]|['"]$/g, "").toLowerCase(), o = oe[e.weight] || e.weight;
			t === r && e.style === n.style && o == i && a.push(e);
		}), a.length !== 0 && (await Promise.all(a.map((e) => e.load().then(() => !0, () => !1)))).some((e) => e);
	}
	async function i() {
		await t.ready;
		let a = !0, o = zs.getProperties(), s = Object.keys(o).filter((e) => o[e] < 100);
		for (let e = s.length - 1; e >= 0; --e) {
			let t = s[e], i = o[t];
			i < 100 && (await r(t) ? (n(Hs), zs.set(t, 100)) : (i += 10, zs.set(t, i, !0), i < 100 && (a = !1)));
		}
		e = void 0, a || (e = setTimeout(i, 100));
	}
	return async function(n) {
		t ||= pe ? self.fonts : document.fonts;
		let r = se(n);
		if (!r) return;
		let a = r.families, o = !1;
		for (let e of a) {
			if (Us.has(e)) continue;
			let t = Ws(r.style, r.weight, e);
			zs.get(t) === void 0 && (zs.set(t, 0, !0), o = !0);
		}
		o && (clearTimeout(e), e = setTimeout(i, 100));
	};
})(), Ks = (function() {
	let e;
	return function(t) {
		let n = Hs[t];
		if (n == null) {
			if (pe) {
				let e = se(t), r = qs(t, "Žg");
				n = (isNaN(Number(e.lineHeight)) ? 1.2 : Number(e.lineHeight)) * (r.actualBoundingBoxAscent + r.actualBoundingBoxDescent);
			} else e || (e = document.createElement("div"), e.innerHTML = "M", e.style.minHeight = "0", e.style.maxHeight = "none", e.style.height = "auto", e.style.padding = "0", e.style.border = "none", e.style.position = "absolute", e.style.display = "block", e.style.left = "-99999px"), e.style.font = t, document.body.appendChild(e), n = e.offsetHeight, document.body.removeChild(e);
			Hs[t] = n;
		}
		return n;
	};
})();
function qs(e, t) {
	return Bs ||= I(1, 1), e != Vs && (Bs.font = e, Vs = Bs.font), Bs.measureText(t);
}
function Js(e, t) {
	return qs(e, t).width;
}
function Ys(e, t, n) {
	if (t in n) return n[t];
	let r = t.split("\n").reduce((t, n) => Math.max(t, Js(e, n)), 0);
	return n[t] = r, r;
}
function Xs(e, t) {
	let n = [], r = [], i = [], a = 0, o = 0, s = 0, c = 0;
	for (let l = 0, u = t.length; l <= u; l += 2) {
		let d = t[l];
		if (d === "\n" || l === u) {
			a = Math.max(a, o), i.push(o), o = 0, s += c, c = 0;
			continue;
		}
		let f = t[l + 1] || e.font, p = Js(f, d);
		n.push(p), o += p;
		let m = Ks(f);
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
function Zs(e, t, n, r, i, a, o, s, c, l, u) {
	e.save(), n !== 1 && (e.globalAlpha === void 0 ? e.globalAlpha = (e) => e.globalAlpha *= n : e.globalAlpha *= n), t && e.transform.apply(e, t), r.contextInstructions ? (e.translate(c, l), e.scale(u[0], u[1]), Qs(r, e)) : u[0] < 0 || u[1] < 0 ? (e.translate(c, l), e.scale(u[0], u[1]), e.drawImage(r, i, a, o, s, 0, 0, o, s)) : e.drawImage(r, i, a, o, s, c, l, o * u[0], s * u[1]), e.restore();
}
function Qs(e, t) {
	let n = e.contextInstructions;
	for (let e = 0, r = n.length; e < r; e += 2) Array.isArray(n[e + 1]) ? t[n[e]].apply(t, n[e + 1]) : t[n[e]] = n[e + 1];
}
//#endregion
//#region node_modules/ol/render/canvas/Instruction.js
var U = {
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
}, $s = [U.FILL], ec = [U.STROKE], tc = [U.BEGIN_PATH], nc = [U.CLOSE_PATH], rc = class extends ks {
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
		for (let o = 0, s = e.length; o < s; o += t) r[0] = e[o], r[1] = e[o + 1], He(n, r) && (i[a++] = r[0], i[a++] = r[1]);
		return a;
	}
	appendFlatLineCoordinates(e, t, n, r, i, a) {
		let o = this.coordinates, s = o.length, c = this.getBufferedMaxExtent();
		a && (t += r);
		let l = e[t], u = e[t + 1], d = this.tmpCoordinate_, f = !0, p, m, h;
		for (p = t + r; p < n; p += r) d[0] = e[p], d[1] = e[p + 1], h = Ge(c, d), h === m ? h === Ie.INTERSECTING ? (o[s++] = d[0], o[s++] = d[1], f = !1) : f = !0 : (f &&= (o[s++] = l, o[s++] = u, !1), o[s++] = d[0], o[s++] = d[1]), l = d[0], u = d[1], m = h;
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
					U.CUSTOM,
					s,
					d,
					e,
					n,
					di,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					d,
					e,
					r || n,
					di,
					i
				]);
				break;
			case "Polygon":
			case "MultiLineString":
				u = [], c = a == "Polygon" ? e.getOrientedFlatCoordinates() : e.getFlatCoordinates(), f = this.drawCustomCoordinates_(c, 0, e.getEnds(), o, u), this.instructions.push([
					U.CUSTOM,
					s,
					u,
					e,
					n,
					ui,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					u,
					e,
					r || n,
					ui,
					i
				]);
				break;
			case "LineString":
			case "Circle":
				c = e.getFlatCoordinates(), l = this.appendFlatLineCoordinates(c, 0, c.length, o, !1, !1), this.instructions.push([
					U.CUSTOM,
					s,
					l,
					e,
					n,
					li,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					l,
					e,
					r || n,
					li,
					i
				]);
				break;
			case "MultiPoint":
				c = e.getFlatCoordinates(), l = this.appendFlatPointCoordinates(c, o), l > s && (this.instructions.push([
					U.CUSTOM,
					s,
					l,
					e,
					n,
					li,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					l,
					e,
					r || n,
					li,
					i
				]));
				break;
			case "Point": c = e.getFlatCoordinates(), this.coordinates.push(c[0], c[1]), l = this.coordinates.length, this.instructions.push([
				U.CUSTOM,
				s,
				l,
				e,
				n,
				void 0,
				i
			]), this.hitDetectionInstructions.push([
				U.CUSTOM,
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
			U.BEGIN_GEOMETRY,
			t,
			0,
			e,
			n
		], this.instructions.push(this.beginGeometryInstruction1_), this.beginGeometryInstruction2_ = [
			U.BEGIN_GEOMETRY,
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
		for (t = 0; t < n; ++t) r = e[t], i = r[0], i == U.END_GEOMETRY ? a = t : i == U.BEGIN_GEOMETRY && (r[2] = t, p(this.hitDetectionInstructions, a, t), a = -1);
	}
	fillStyleToState(e, t = {}) {
		if (e) {
			let n = e.getColor();
			t.fillPatternScale = n && typeof n == "object" && "src" in n ? this.pixelRatio : 1, t.fillStyle = Ds(n || "#000") ?? void 0;
		} else t.fillStyle = void 0;
		return t;
	}
	strokeStyleToState(e, t = {}) {
		if (e) {
			t.strokeStyle = Ds(e.getColor() || Fs);
			let n = e.getLineCap();
			t.lineCap = n === void 0 ? Ms : n;
			let r = e.getLineDash();
			t.lineDash = r ? r.slice() : Ns, t.lineDashOffset = e.getLineDashOffset() || 0;
			let i = e.getLineJoin();
			t.lineJoin = i === void 0 ? Ps : i;
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
		let t = e.fillStyle, n = [U.SET_FILL_STYLE, t];
		return typeof t != "string" && n.push(e.fillPatternScale), n;
	}
	applyStroke(e) {
		this.instructions.push(this.createStroke(e));
	}
	createStroke(e) {
		return [
			U.SET_STROKE_STYLE,
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
		(e.currentStrokeStyle != n || e.currentLineCap != r || i != e.currentLineDash && !h(e.currentLineDash, i) || e.currentLineDashOffset != a || e.currentLineJoin != o || e.currentLineWidth != s || e.currentMiterLimit != c || e.currentStrokeOffset != l) && (t.call(this, e), e.currentStrokeStyle = n, e.currentLineCap = r, e.currentLineDash = i, e.currentLineDashOffset = a, e.currentLineJoin = o, e.currentLineWidth = s, e.currentMiterLimit = c, e.currentStrokeOffset = l);
	}
	endGeometry(e) {
		this.beginGeometryInstruction1_[2] = this.instructions.length, this.beginGeometryInstruction1_ = null, this.beginGeometryInstruction2_[2] = this.hitDetectionInstructions.length, this.beginGeometryInstruction2_ = null;
		let t = [U.END_GEOMETRY, e];
		this.instructions.push(t), this.hitDetectionInstructions.push(t);
	}
	getBufferedMaxExtent() {
		if (!this.bufferedMaxExtent_ && (this.bufferedMaxExtent_ = Be(this.maxExtent), this.maxLineWidth > 0)) {
			let e = this.resolution * (this.maxLineWidth + 1) / 2;
			ze(this.bufferedMaxExtent_, e, this.bufferedMaxExtent_);
		}
		return this.bufferedMaxExtent_;
	}
}, ic = class extends rc {
	constructor(e, t, n, r) {
		super(e, t, n, r), this.hitDetectionImage_ = null, this.image_ = null, this.imagePixelRatio_ = void 0, this.anchorX_ = void 0, this.anchorY_ = void 0, this.height_ = void 0, this.opacity_ = void 0, this.originX_ = void 0, this.originY_ = void 0, this.rotateWithView_ = void 0, this.rotation_ = void 0, this.scale_ = void 0, this.width_ = void 0, this.declutterMode_ = void 0, this.declutterImageWithText_ = void 0;
	}
	drawPoint(e, t, n) {
		if (!this.image_ || this.maxExtent && !He(this.maxExtent, e.getFlatCoordinates())) return;
		this.beginGeometry(e, t, n);
		let r = e.getFlatCoordinates(), i = e.getStride(), a = this.coordinates.length, o = this.appendFlatPointCoordinates(r, i);
		this.instructions.push([
			U.DRAW_IMAGE,
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
			U.DRAW_IMAGE,
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
		for (let t = 0, n = r.length; t < n; t += e.getStride()) (!this.maxExtent || He(this.maxExtent, r.slice(t, t + 2))) && i.push(r[t], r[t + 1]);
		let a = this.coordinates.length, o = this.appendFlatPointCoordinates(i, 2);
		this.instructions.push([
			U.DRAW_IMAGE,
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
			U.DRAW_IMAGE,
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
}, ac = class extends rc {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	drawFlatCoordinates_(e, t, n, r, i) {
		let a = this.coordinates.length, o = this.appendFlatLineCoordinates(e, t, n, r, !1, !1);
		return this.instructions.push([
			U.MOVE_TO_LINE_TO,
			a,
			o,
			i * this.pixelRatio
		]), this.hitDetectionInstructions.push([
			U.MOVE_TO_LINE_TO,
			a,
			o,
			i
		]), n;
	}
	drawLineString(e, t, n) {
		let r = this.state, i = r.strokeStyle, a = r.lineWidth, o = r.strokeOffset;
		if (i === void 0 || a === void 0) return;
		this.updateStrokeStyle(r, this.applyStroke), this.beginGeometry(e, t, n), this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Fs,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Ns,
			0
		], tc);
		let s = e.getFlatCoordinates(), c = e.getStride();
		this.drawFlatCoordinates_(s, 0, s.length, c, o), this.hitDetectionInstructions.push(ec), this.endGeometry(t);
	}
	drawMultiLineString(e, t, n) {
		let r = this.state, i = r.strokeStyle, a = r.lineWidth, o = r.strokeOffset;
		if (i === void 0 || a === void 0) return;
		this.updateStrokeStyle(r, this.applyStroke), this.beginGeometry(e, t, n), this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Fs,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Ns,
			0
		], tc);
		let s = e.getEnds(), c = e.getFlatCoordinates(), l = e.getStride(), u = 0;
		for (let e = 0, t = s.length; e < t; ++e) u = this.drawFlatCoordinates_(c, u, s[e], l, o);
		this.hitDetectionInstructions.push(ec), this.endGeometry(t);
	}
	finish() {
		let e = this.state;
		return e.lastStroke != null && e.lastStroke != this.coordinates.length && this.instructions.push(ec), this.reverseHitDetectionInstructions(), this.state = null, super.finish();
	}
	applyStroke(e) {
		e.lastStroke != null && e.lastStroke != this.coordinates.length && (this.instructions.push(ec), e.lastStroke = this.coordinates.length), e.lastStroke = 0, super.applyStroke(e), this.instructions.push(tc);
	}
}, oc = class extends rc {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	drawFlatCoordinatess_(e, t, n, r, i) {
		let a = this.state, o = a.fillStyle !== void 0, s = a.strokeStyle !== void 0, c = n.length;
		this.instructions.push(tc), this.hitDetectionInstructions.push(tc);
		for (let a = 0; a < c; ++a) {
			let o = n[a], c = this.coordinates.length, l = this.appendFlatLineCoordinates(e, t, o, r, !0, !s);
			this.instructions.push([
				U.MOVE_TO_LINE_TO,
				c,
				l,
				i * this.pixelRatio,
				!0
			]), this.hitDetectionInstructions.push([
				U.MOVE_TO_LINE_TO,
				c,
				l,
				i,
				!0
			]), s && (this.instructions.push(nc), this.hitDetectionInstructions.push(nc)), t = o;
		}
		return o && (this.instructions.push($s), this.hitDetectionInstructions.push($s)), s && (this.instructions.push(ec), this.hitDetectionInstructions.push(ec)), t;
	}
	drawCircle(e, t, n) {
		let r = this.state, i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawCircle(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([U.SET_FILL_STYLE, js]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Fs,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Ns,
			0
		]);
		let s = e.getFlatCoordinates(), c = e.getStride(), l = this.coordinates.length;
		this.appendFlatLineCoordinates(s, 0, s.length, c, !1, !1);
		let u = [
			U.CIRCLE,
			l,
			o
		];
		this.instructions.push(tc, u), this.hitDetectionInstructions.push(tc, u), r.fillStyle !== void 0 && (this.instructions.push($s), this.hitDetectionInstructions.push($s)), r.strokeStyle !== void 0 && (this.instructions.push(ec), this.hitDetectionInstructions.push(ec)), this.endGeometry(t);
	}
	drawPolygon(e, t, n) {
		let r = this.state, i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawPolygon(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([U.SET_FILL_STYLE, js]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Fs,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Ns,
			0
		]);
		let s = e.getEnds(), c = e.getOrientedFlatCoordinates(), l = e.getStride();
		this.drawFlatCoordinatess_(c, 0, s, l, o), this.endGeometry(t);
	}
	drawMultiPolygon(e, t, n) {
		let r = this.state, i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawMultiPolygon(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([U.SET_FILL_STYLE, js]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Fs,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Ns,
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
			for (let n = 0, r = t.length; n < r; ++n) t[n] = Oi(t[n], e);
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
}, sc = 0, cc = 1;
function lc(e, t, n, r, i, a, o, s) {
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
	return sc = u, cc = d, !0;
}
function uc(e, t, n, r) {
	let i = r[0], a = r[1], o = r[2], s = r[3], c = [], l = [], u = !1, d, f, p = 0;
	for (let r = 0, m = t.length; r < m; ++r) {
		let m = t[r], h = e[p], g = e[p + 1], _ = !1;
		for (let t = p + n; t < m; t += n) {
			let n = e[t], r = e[t + 1];
			if (lc(i, a, o, s, h, g, n, r)) {
				let e = n - h, t = r - g, i = h + sc * e, a = g + sc * t, o = h + cc * e, s = g + cc * t;
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
function dc(e, t, n, r, i) {
	let a = [], o = n, s = 0, c = t.slice(n, 2);
	for (; s < e && o + i < r;) {
		let [n, r] = c.slice(-2), l = t[o + i], u = t[o + i + 1], d = Math.sqrt((l - n) * (l - n) + (u - r) * (u - r));
		if (s += d, s >= e) {
			let t = (e - s + d) / d, f = Pt(n, l, t), p = Pt(r, u, t);
			c.push(f, p), a.push(c), c = [f, p], s == e && (o += i), s = 0;
		} else if (s < e) c.push(t[o + i], t[o + i + 1]), o += i;
		else {
			let e = d - s, t = Pt(n, l, e / d), f = Pt(r, u, e / d);
			c.push(t, f), a.push(c), c = [t, f], s = 0, o += i;
		}
	}
	return s > 0 && a.push(c), a;
}
//#endregion
//#region node_modules/ol/geom/flat/straightchunk.js
function fc(e, t, n, r, i) {
	let a = n, o = n, s = 0, c = 0, l = n, u, d, f, p, m, h, g, _, v, y;
	for (d = n; d < r; d += i) {
		let n = t[d], r = t[d + 1];
		m !== void 0 && (v = n - m, y = r - h, p = Math.sqrt(v * v + y * y), g !== void 0 && (c += f, u = Math.acos((g * v + _ * y) / (f * p)), u > e && (c > s && (s = c, a = l, o = d), c = 0, l = d - i)), f = p, g = v, _ = y), m = n, h = r;
	}
	return c += p, c > s ? [l, d] : [a, o];
}
//#endregion
//#region node_modules/ol/render/canvas/TextBuilder.js
var pc = {
	left: 0,
	center: .5,
	right: 1,
	top: 0,
	middle: .5,
	hanging: .2,
	alphabetic: .8,
	ideographic: .8,
	bottom: 1
}, mc = {
	Circle: oc,
	Default: rc,
	Image: ic,
	LineString: ac,
	Polygon: oc,
	Text: class extends rc {
		constructor(e, t, n, r) {
			super(e, t, n, r), this.labels_ = null, this.text_ = "", this.textOffsetX_ = 0, this.textOffsetY_ = 0, this.textRotateWithView_ = void 0, this.textKeepUpright_ = void 0, this.textRotation_ = 0, this.textFillState_ = null, this.fillStates = {}, this.fillStates[js] = { fillStyle: js }, this.textStrokeState_ = null, this.strokeStates = {}, this.textState_ = {}, this.textStates = {}, this.textKey_ = "", this.fillKey_ = "", this.strokeKey_ = "", this.declutterMode_ = void 0, this.declutterImageWithText_ = void 0;
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
				if (!ht(this.maxExtent, r)) return;
				let i;
				if (l = e.getFlatCoordinates(), c == "LineString") i = [l.length];
				else if (c == "MultiLineString") i = e.getEnds();
				else if (c == "Polygon") i = e.getEnds().slice(0, 1);
				else if (c == "MultiPolygon") {
					let t = e.getEndss();
					i = [];
					for (let e = 0, n = t.length; e < n; ++e) i.push(t[e][0]);
				}
				if ((c == "LineString" || c == "MultiLineString") && !Ue(this.getBufferedMaxExtent(), r)) {
					let e = uc(l, i, u, this.getBufferedMaxExtent());
					if (l = e.flatCoordinates, i = e.ends, u = 2, i.length === 0) return;
				}
				this.beginGeometry(e, t, n);
				let d = a.repeat, f = d ? void 0 : a.textAlign, p = 0;
				for (let e = 0, t = i.length; e < t; ++e) {
					let t;
					t = d ? dc(d * this.resolution, l, p, i[e], u) : [l.slice(p, i[e])];
					for (let n = 0, r = t.length; n < r; ++n) {
						let r = t[n], c = 0, l = r.length;
						if (f == null) {
							let e = fc(a.maxAngle, r, 0, r.length, 2);
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
				if (p != Rs && (a.scale[0] < 0 || a.scale[1] < 0)) {
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
					U.DRAW_IMAGE,
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
					p == Rs ? Rs : p.map(function(e) {
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
				g && (g[1] = js), this.hitDetectionInstructions.push([
					U.DRAW_IMAGE,
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
					this.fillKey_ ? js : this.fillKey_,
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
			let s = this.pixelRatio, c = pc[r.textBaseline], l = this.textOffsetX_ * s, u = this.textOffsetY_ * s, d = this.text_, f = n ? n.lineWidth * Math.abs(r.scale[0]) / 2 : 0;
			this.instructions.push([
				U.DRAW_CHARS,
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
				U.DRAW_CHARS,
				e,
				t,
				c,
				r.overflow,
				o && js,
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
				t ? (r = this.textFillState_, r || (r = {}, this.textFillState_ = r), r.fillStyle = Ds(t.getColor() || "#000")) : (r = null, this.textFillState_ = r);
				let a = e.getStroke();
				if (!a) i = null, this.textStrokeState_ = i;
				else {
					i = this.textStrokeState_, i || (i = {}, this.textStrokeState_ = i);
					let e = a.getLineDash(), t = a.getLineDashOffset(), n = a.getWidth(), r = a.getMiterLimit();
					i.lineCap = a.getLineCap() || "round", i.lineDash = e ? e.slice() : Ns, i.lineDashOffset = t === void 0 ? 0 : t, i.lineJoin = a.getLineJoin() || "round", i.lineWidth = n === void 0 ? 1 : n, i.miterLimit = r === void 0 ? 10 : r, i.strokeStyle = Ds(a.getColor() || "#000");
				}
				n = this.textState_;
				let o = e.getFont() || "10px sans-serif";
				Gs(o);
				let s = e.getScaleArray();
				n.overflow = e.getOverflow(), n.font = o, n.maxAngle = e.getMaxAngle(), n.placement = e.getPlacement(), n.textAlign = e.getTextAlign(), n.repeat = e.getRepeat(), n.justify = e.getJustify(), n.textBaseline = e.getTextBaseline() || "middle", n.backgroundFill = e.getBackgroundFill(), n.backgroundStroke = e.getBackgroundStroke(), n.padding = e.getPadding() || Rs, n.scale = s === void 0 ? [1, 1] : s;
				let c = e.getOffsetX(), l = e.getOffsetY(), u = e.getRotateWithView(), d = e.getKeepUpright(), f = e.getRotation();
				this.text_ = e.getText() || "", this.textOffsetX_ = c === void 0 ? 0 : c, this.textOffsetY_ = l === void 0 ? 0 : l, this.textRotateWithView_ = u !== void 0 && u, this.textKeepUpright_ = d === void 0 || d, this.textRotation_ = f === void 0 ? 0 : f, this.strokeKey_ = i ? (typeof i.strokeStyle == "string" ? i.strokeStyle : O(i.strokeStyle)) + i.lineCap + i.lineDashOffset + "|" + i.lineWidth + i.lineJoin + i.miterLimit + "[" + i.lineDash.join() + "]" : "", this.textKey_ = n.font + n.scale + (n.textAlign || "?") + (n.repeat || "?") + (n.justify || "?") + (n.textBaseline || "?"), this.fillKey_ = r && r.fillStyle ? typeof r.fillStyle == "string" ? r.fillStyle : "|" + O(r.fillStyle) : "";
			}
			this.declutterMode_ = e.getDeclutterMode(), this.declutterImageWithText_ = t;
		}
	}
}, hc = class {
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
			let e = mc[t];
			i = new e(this.tolerance_, this.maxExtent_, this.resolution_, this.pixelRatio_), r[t] = i;
		}
		return i;
	}
};
//#endregion
//#region node_modules/ol/geom/flat/lineoffset.js
function gc(e, t, n, r, i, a, o, s) {
	o ??= [], s ??= r;
	let c = e[t + r], l = e[t + r + 1], u = e[n - 2 * r], d = e[n - 2 * r + 1], f, p, m, h, g, _, v, y, b = 0;
	for (let x = t; x < n; x += r) {
		m = f, h = p, g = void 0, _ = void 0, x + r < n && (g = e[x + r], _ = e[x + r + 1]), a && x === t && (m = u, h = d), a && x === n - r && (g = c, _ = l), f = e[x], p = e[x + 1], [v, y] = _c(f, p, m, h, g, _, i), o[b++] = v, o[b++] = y;
		for (let t = 2; t < s; t++) o[b++] = e[x + t];
	}
	return o.length != b && (o.length = b), o;
}
function _c(e, t, n, r, i, a, o) {
	let s, c;
	n !== void 0 && r !== void 0 ? (s = e - n, c = t - r) : i !== void 0 && a !== void 0 ? (s = i - e, c = a - t) : (s = 1, c = 0);
	let l = Math.hypot(s, c), u = s / l, d = c / l;
	if (s = -d, c = u, n === void 0 || r === void 0 || i === void 0 || a === void 0) return [e + s * o, t + c * o];
	let f = Kt([e, t], [n, r], [i, a]);
	if (Math.cos(f) > .998) return [e + u * o, t + d * o];
	let p = Math.cos(f / 2), m = Math.sin(f / 2), h = m * s + p * c, g = -p * s + m * c, _ = 1 / m * h, v = 1 / m * g;
	return [e + _ * o, t + v * o];
}
function vc(e, t, n = !1) {
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
var yc;
function bc() {
	return yc ||= new Intl.Segmenter(void 0, { granularity: "grapheme" }), yc;
}
function xc(e, t, n, r, i, a, o, s, c, l, u, d, f = !0) {
	let p = e[t], m = e[t + 1], h = 0, g = 0, _ = 0, v = 0;
	function y() {
		h = p, g = m, t += r, p = e[t], m = e[t + 1], v += _, _ = Math.sqrt((p - h) * (p - h) + (m - g) * (m - g));
	}
	do
		y();
	while (t < n - r && v + _ < a);
	let b = _ === 0 ? 0 : (a - v) / _, x = Pt(h, p, b), S = Pt(g, m, b), C = t - r, w = v, T = a + s * c(l, i, u);
	for (; t < n - r && v + _ < T;) y();
	b = _ === 0 ? 0 : (T - v) / _;
	let E = Pt(h, p, b), D = Pt(g, m, b), O = !1;
	if (f) {
		if (d) {
			let e = [
				x,
				S,
				E,
				D
			];
			Br(e, 0, 4, 2, d, e, e), O = e[0] > e[2];
		} else O = x > E;
	}
	let k = Math.PI, A = [], j = C + r === t;
	t = C, _ = 0, v = w, p = e[t], m = e[t + 1];
	let M;
	if (j) return y(), M = Math.atan2(m - g, p - h), O && (M += M > 0 ? -k : k), A[0] = [
		(E + x) / 2,
		(D + S) / 2,
		(T - a) / 2,
		M,
		i
	], A;
	i = i.replace(/\n/g, " ");
	let N = Array.from(bc().segment(i), (e) => e.segment);
	for (let e = 0, i = N.length; e < i;) {
		y();
		let d = Math.atan2(m - g, p - h);
		if (O && (d += d > 0 ? -k : k), M !== void 0) {
			let e = d - M;
			if (e += e > k ? -2 * k : e < -k ? 2 * k : 0, Math.abs(e) > o) return null;
		}
		M = d;
		let f = e, x = 0;
		for (; e < i; ++e) {
			let o = s * c(l, N[O ? i - e - 1 : e], u);
			if (t + r < n && v + _ < a + x + o / 2) break;
			x += o;
		}
		if (e === f) continue;
		let S = (O ? N.slice(i - e, i - f) : N.slice(f, e)).join("");
		b = _ === 0 ? 0 : (a + x / 2 - v) / _;
		let C = Pt(h, p, b), w = Pt(g, m, b);
		A.push([
			C,
			w,
			x / 2,
			d,
			S
		]), a += x;
	}
	return A;
}
//#endregion
//#region node_modules/ol/render/canvas/Executor.js
var Sc = Ke(), Cc = [], wc = [], Tc = [], Ec = [];
function Dc(e) {
	return e[3].declutterBox;
}
var Oc = /* @__PURE__ */ RegExp("[֑-ࣿיִ-﷿ﹰ-ﻼࠀ-࿿-]");
function kc(e, t) {
	return t === "start" ? t = Oc.test(e) ? "right" : "left" : t === "end" && (t = Oc.test(e) ? "left" : "right"), pc[t];
}
function Ac(e, t, n) {
	return n > 0 && e.push("\n", ""), e.push(t, ""), e;
}
function jc(e, t, n) {
	return n % 2 == 0 && (e += t), e;
}
var Mc = class {
	constructor(e, t, n, r, i) {
		this.overlaps = n, this.pixelRatio = t, this.resolution = e, this.alignAndScaleFill_, this.instructions = r.instructions, this.coordinates = r.coordinates, this.coordinateCache_ = {}, this.renderedTransform_ = Tr(), this.hitDetectionInstructions = r.hitDetectionInstructions, this.pixelCoordinates_ = null, this.viewRotation_ = 0, this.fillStates = r.fillStates || {}, this.strokeStates = r.strokeStates || {}, this.textStates = r.textStates || {}, this.widths_ = {}, this.labels_ = {}, this.zIndexContext_ = i ? new Lo() : null;
	}
	getZIndexContext() {
		return this.zIndexContext_;
	}
	createLabel(e, t, n, r) {
		let i = e + t + n + r;
		if (this.labels_[i]) return this.labels_[i];
		let a = r ? this.strokeStates[r] : null, o = n ? this.fillStates[n] : null, s = this.textStates[t], c = this.pixelRatio, l = [s.scale[0] * c, s.scale[1] * c], u = s.justify ? pc[s.justify] : kc(Array.isArray(e) ? e[0] : e, s.textAlign || "center"), d = r && a.lineWidth ? a.lineWidth : 0, f = Array.isArray(e) ? e : String(e).split("\n").reduce(Ac, []), { width: p, height: m, widths: h, heights: g, lineWidths: _ } = Xs(s, f), v = p + d, y = [], b = (v + 2) * l[0], x = (m + d) * l[1], S = {
			width: b < 0 ? Math.floor(b) : Math.ceil(b),
			height: x < 0 ? Math.floor(x) : Math.ceil(x),
			contextInstructions: y
		};
		(l[0] != 1 || l[1] != 1) && y.push("scale", l), r && (y.push("strokeStyle", a.strokeStyle), y.push("lineWidth", d), y.push("lineCap", a.lineCap), y.push("lineJoin", a.lineJoin), y.push("miterLimit", a.miterLimit), y.push("setLineDash", [a.lineDash]), y.push("lineDashOffset", a.lineDashOffset)), n && y.push("fillStyle", o.fillStyle), y.push("textBaseline", "middle"), y.push("textAlign", "center");
		let C = .5 - u, w = u * v + C * d, T = [], E = [], D = 0, O = 0, k = 0, A = 0, j;
		for (let e = 0, t = f.length; e < t; e += 2) {
			let t = f[e];
			if (t === "\n") {
				O += D, D = 0, w = u * v + C * d, ++A;
				continue;
			}
			let i = f[e + 1] || s.font;
			i !== j && (r && T.push("font", i), n && E.push("font", i), j = i), D = Math.max(D, g[k]);
			let a = [
				t,
				w + C * h[k] + u * (h[k] - _[A]),
				.5 * (d + D) + O
			];
			w += h[k], r && T.push("strokeText", a), n && E.push("fillText", a), ++k;
		}
		return Array.prototype.push.apply(y, T), Array.prototype.push.apply(y, E), this.labels_[i] = S, S;
	}
	replayTextBackground_(e, t, n, r, i, a, o) {
		e.beginPath(), e.moveTo.apply(e, t), e.lineTo.apply(e, n), e.lineTo.apply(e, r), e.lineTo.apply(e, i), e.lineTo.apply(e, t), a && (this.alignAndScaleFill_ = a[2], e.fillStyle = a[1], this.fill_(e)), o && (this.setStrokeStyle_(e, o), e.stroke());
	}
	calculateImageOrLabelDimensions_(e, t, n, r, i, a, o, s, c, l, u, d, f, p, m, h) {
		o *= d[0], s *= d[1];
		let g = n - o, _ = r - s, v = i + c > e ? e - c : i, y = a + l > t ? t - l : a, b = p[3] + v * d[0] + p[1], x = p[0] + y * d[1] + p[2], S = g - p[3], C = _ - p[0];
		(m || u !== 0) && (Cc[0] = S, Ec[0] = S, Cc[1] = C, wc[1] = C, wc[0] = S + b, Tc[0] = wc[0], Tc[1] = C + x, Ec[1] = Tc[1]);
		let w;
		return u === 0 ? qe(Math.min(S, S + b), Math.min(C, C + x), Math.max(S, S + b), Math.max(C, C + x), Sc) : (w = Mr(Tr(), n, r, 1, 1, u, -n, -r), z(w, Cc), z(w, wc), z(w, Tc), z(w, Ec), qe(Math.min(Cc[0], wc[0], Tc[0], Ec[0]), Math.min(Cc[1], wc[1], Tc[1], Ec[1]), Math.max(Cc[0], wc[0], Tc[0], Ec[0]), Math.max(Cc[1], wc[1], Tc[1], Ec[1]), Sc)), f && (g = Math.round(g), _ = Math.round(_)), {
			drawImageX: g,
			drawImageY: _,
			drawImageW: v,
			drawImageH: y,
			originX: c,
			originY: l,
			declutterBox: {
				minX: Sc[0],
				minY: Sc[1],
				maxX: Sc[2],
				maxY: Sc[3],
				value: h
			},
			canvasTransform: w,
			scale: d
		};
	}
	replayImageOrLabel_(e, t, n, r, i, a, o) {
		let s = !!(a || o), c = r.declutterBox, l = o ? o[2] * r.scale[0] / 2 : 0;
		return c.minX - l <= t[0] && c.maxX + l >= 0 && c.minY - l <= t[1] && c.maxY + l >= 0 && (s && this.replayTextBackground_(e, Cc, wc, Tc, Ec, a, o), Zs(e, r.canvasTransform, i, n, r.originX, r.originY, r.drawImageW, r.drawImageH, r.drawImageX, r.drawImageY, r.scale)), !0;
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
		let i = this.textStates[t], a = this.createLabel(e, t, r, n), o = this.strokeStates[n], s = this.pixelRatio, c = kc(Array.isArray(e) ? e[0] : e, i.textAlign || "center"), l = pc[i.textBaseline || "middle"], u = o && o.lineWidth ? o.lineWidth : 0;
		return {
			label: a,
			anchorX: c * (a.width / s - 2 * i.scale[0]) + 2 * (.5 - c) * u,
			anchorY: l * a.height / s + 2 * (.5 - l) * u
		};
	}
	execute_(e, t, n, r, i, a, o, s) {
		let c = this.zIndexContext_, l;
		this.pixelCoordinates_ && h(n, this.renderedTransform_) ? l = this.pixelCoordinates_ : (this.pixelCoordinates_ ||= [], l = zr(this.coordinates, 0, this.coordinates.length, 2, n, this.pixelCoordinates_), kr(this.renderedTransform_, n));
		let u = 0, d = r.length, f = 0, p, m = [], g, _, v, y, b, x, S, C, w, T, E, D, O, k = 0, A = 0, j = this.coordinateCache_, M = this.viewRotation_, N = Math.round(Math.atan2(-n[1], n[0]) * 0xe8d4a51000) / 0xe8d4a51000, ee = {
			context: e,
			pixelRatio: this.pixelRatio,
			resolution: this.resolution,
			rotation: M
		}, P = this.instructions != r || this.overlaps ? 0 : 200, F, te, ne, re;
		for (; u < d;) {
			let n = r[u];
			switch (n[0]) {
				case U.BEGIN_GEOMETRY:
					F = n[1], re = n[3], F.getGeometry() ? o !== void 0 && !ht(o, re.getExtent()) ? u = n[2] + 1 : ++u : u = n[2], c && (c.zIndex = n[4]);
					break;
				case U.BEGIN_PATH:
					k > P && (this.fill_(e), k = 0), A > P && (e.stroke(), A = 0), !k && !A && (e.beginPath(), b = NaN, x = NaN), ++u;
					break;
				case U.CIRCLE:
					f = n[1], v = n[2] ?? 0;
					let r = l[f], d = l[f + 1], h = l[f + 2] - v, ie = l[f + 3] - v, ae = h - r, oe = ie - d, se = Math.sqrt(ae * ae + oe * oe);
					e.moveTo(r + se, d), e.arc(r, d, se, 0, 2 * Math.PI, !0), ++u;
					break;
				case U.CLOSE_PATH:
					e.closePath(), ++u;
					break;
				case U.CUSTOM:
					f = n[1], p = n[2];
					let ce = n[3], le = n[4], ue = n[5];
					ee.geometry = ce, ee.feature = F, u in j || (j[u] = []);
					let de = j[u];
					ue ? ue(l, f, p, 2, de) : (de[0] = l[f], de[1] = l[f + 1], de.length = 2), c && (c.zIndex = n[6]), le(de, ee), ++u;
					break;
				case U.DRAW_IMAGE:
					f = n[1], p = n[2], w = n[3], g = n[4], _ = n[5];
					let fe = n[6], pe = n[7], me = n[8], he = n[9], I = n[10], ge = n[11], _e = n[12], ve = n[13];
					y = n[14] || "declutter";
					let ye = n[15];
					if (!w && n.length >= 20) {
						T = n[19], E = n[20], D = n[21], O = n[22];
						let e = this.drawLabelWithPointPlacement_(T, E, D, O);
						w = e.label, n[3] = w;
						let t = n[23];
						g = (e.anchorX - t) * this.pixelRatio, n[4] = g;
						let r = n[24];
						_ = (e.anchorY - r) * this.pixelRatio, n[5] = _, fe = w.height, n[6] = fe, ve = w.width, n[13] = ve;
					}
					let be;
					n.length > 25 && (be = n[25]);
					let xe, Se, Ce;
					n.length > 17 ? (xe = n[16], Se = n[17], Ce = n[18]) : (xe = Rs, Se = null, Ce = null), I && N ? ge += M : !I && !N && (ge -= M);
					let we = 0;
					for (; f < p; f += 2) {
						if (be && be[we++] < ve / this.pixelRatio) continue;
						let n = this.calculateImageOrLabelDimensions_(w.width, w.height, l[f], l[f + 1], ve, fe, g, _, me, he, ge, _e, i, xe, !!Se || !!Ce, F), r = [
							e,
							t,
							w,
							n,
							pe,
							Se,
							Ce
						];
						if (s) {
							let e, t, i;
							if (ye) {
								let n = p - f;
								if (!ye[n]) {
									ye[n] = {
										args: r,
										declutterMode: y
									};
									continue;
								}
								let a = ye[n];
								e = a.args, t = a.declutterMode, delete ye[n], i = Dc(e);
							}
							let a, o;
							if (e && (t !== "declutter" || !s.collides(i)) && (a = !0), (y !== "declutter" || !s.collides(n.declutterBox)) && (o = !0), t === "declutter" && y === "declutter") {
								let e = a && o;
								a = e, o = e;
							}
							a && (t !== "none" && s.insert(i), this.replayImageOrLabel_.apply(this, e)), o && (y !== "none" && s.insert(n.declutterBox), this.replayImageOrLabel_.apply(this, r));
						} else this.replayImageOrLabel_.apply(this, r);
					}
					++u;
					break;
				case U.DRAW_CHARS:
					let Te = n[1], Ee = n[2], De = n[3], Oe = n[4];
					O = n[5];
					let ke = n[6], Ae = n[7], je = n[8];
					D = n[9];
					let Me = n[10];
					T = n[11], Array.isArray(T) && (T = T.reduce(jc, "")), E = n[12];
					let Ne = [n[13], n[13]];
					y = n[14] || "declutter";
					let Pe = n[15], Fe = n[16], Ie = this.textStates[E], Le = Ie.font, Re = [Ie.scale[0] * Ae, Ie.scale[1] * Ae], ze;
					Le in this.widths_ ? ze = this.widths_[Le] : (ze = {}, this.widths_[Le] = ze);
					let Be = Ti(l, Te, Ee, 2), Ve = Math.abs(Re[0]) * Ys(Le, T, ze);
					if (Oe || Ve <= Be) {
						let n = this.textStates[E].textAlign, r = (Be - Ve) * kc(T, n), i = xc(l, Te, Ee, 2, T, r, ke, Math.abs(Re[0]), Ys, Le, ze, N ? 0 : this.viewRotation_, Pe);
						drawChars: if (i) {
							let n = [], r, a, o, c, l;
							if (D) for (r = 0, a = i.length; r < a; ++r) {
								l = i[r], o = l[4], c = this.createLabel(o, E, "", D), g = l[2] + (Re[0] < 0 ? -Me : Me) - Fe, _ = De * c.height + (.5 - De) * 2 * Me * Re[1] / Re[0] - je;
								let a = this.calculateImageOrLabelDimensions_(c.width, c.height, l[0], l[1], c.width, c.height, g, _, 0, 0, l[3], Ne, !1, Rs, !1, F);
								if (s && y === "declutter" && s.collides(a.declutterBox)) break drawChars;
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
							if (O) for (r = 0, a = i.length; r < a; ++r) {
								l = i[r], o = l[4], c = this.createLabel(o, E, O, ""), g = l[2] - Fe, _ = De * c.height - je;
								let a = this.calculateImageOrLabelDimensions_(c.width, c.height, l[0], l[1], c.width, c.height, g, _, 0, 0, l[3], Ne, !1, Rs, !1, F);
								if (s && y === "declutter" && s.collides(a.declutterBox)) break drawChars;
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
							s && y !== "none" && s.load(n.map(Dc));
							for (let e = 0, t = n.length; e < t; ++e) this.replayImageOrLabel_.apply(this, n[e]);
						}
					}
					++u;
					break;
				case U.END_GEOMETRY:
					if (a !== void 0) {
						F = n[1];
						let e = a(F, re, y);
						if (e) return e;
					}
					++u;
					break;
				case U.FILL:
					P ? k++ : this.fill_(e), ++u;
					break;
				case U.MOVE_TO_LINE_TO:
					f = n[1], p = n[2], v = n[3];
					let He, Ue, We;
					if (v) {
						let e = (n[4] ?? !1) || Math.abs(l[f] - l[p - 2]) < 1e-6 && Math.abs(l[f + 1] - l[p - 1]) < 1e-6;
						gc(l, f, p, 2, v, e, m), vc(m, 2, e), He = m, Ue = 0, We = He.length;
					} else He = l, Ue = f, We = p;
					te = He[Ue], ne = He[Ue + 1], e.moveTo(te, ne), b = te + .5 | 0, x = ne + .5 | 0;
					for (let t = Ue + 2; t < We; t += 2) te = He[t], ne = He[t + 1], S = te + .5 | 0, C = ne + .5 | 0, (t == We - 2 || S !== b || C !== x) && (e.lineTo(te, ne), b = S, x = C);
					++u;
					break;
				case U.SET_FILL_STYLE:
					this.alignAndScaleFill_ = n[2], k ? (this.fill_(e), k = 0, A &&= (e.stroke(), 0)) : A && n[1] && (e.stroke(), A = 0), e.fillStyle = n[1], ++u;
					break;
				case U.SET_STROKE_STYLE:
					k && n[1] && (this.fill_(e), k = 0), A &&= (e.stroke(), 0), this.setStrokeStyle_(e, n), ++u;
					break;
				case U.STROKE:
					P ? A++ : e.stroke(), ++u;
					break;
				default: ++u;
			}
		}
		k && this.fill_(e), A && e.stroke();
	}
	execute(e, t, n, r, i, a) {
		this.viewRotation_ = r, this.execute_(e, t, n, this.instructions, i, void 0, void 0, a);
	}
	executeHitDetection(e, t, n, r, i) {
		return this.viewRotation_ = n, this.execute_(e, [e.canvas.width, e.canvas.height], t, this.hitDetectionInstructions, !0, r, i);
	}
}, Nc = [
	"Polygon",
	"Circle",
	"LineString",
	"Image",
	"Text",
	"Default"
], Pc = ["Image", "Text"], Fc = Nc.filter((e) => !Pc.includes(e)), Ic = !1, Lc = !1;
function Rc() {
	let e = 0, t = (t) => {
		let n = I(1, 1, null, { willReadFrequently: t }), r = 0, i = performance.now();
		for (; performance.now() - i < 50; ++r) n.fillStyle = `rgba(255,0,${r % 256},1)`, n.fillRect(0, 0, 1, 1), n.getImageData(0, 0, 1, 1);
		return e = r > e ? r : e, r;
	};
	Ic = {
		[t(!0)]: !0,
		[t(!1)]: !1,
		[t(void 0)]: void 0
	}[e], Lc = !0;
}
var zc = class {
	constructor(e, t, n, r, i, a, o) {
		this.maxExtent_ = e, this.overlaps_ = r, this.pixelRatio_ = n, this.resolution_ = t, this.renderBuffer_ = a, this.executorsByZIndex_ = {}, this.hitDetectionContext_ = null, this.hitDetectionTransform_ = Tr(), this.renderedContext_ = null, this.deferredZIndexContexts_ = {}, this.createExecutors_(i, o);
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
				r[e] = new Mc(this.resolution_, this.pixelRatio_, this.overlaps_, n, t);
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
		Lc === !1 && Rc(), r = Math.round(r);
		let o = r * 2 + 1, s = Mr(this.hitDetectionTransform_, r + .5, r + .5, 1 / t, -1 / t, -n, -e[0], -e[1]), c = !this.hitDetectionContext_;
		c && (this.hitDetectionContext_ = I(o, o, null, { willReadFrequently: Ic }));
		let l = this.hitDetectionContext_;
		l.canvas.width !== o || l.canvas.height !== o ? (l.canvas.width = o, l.canvas.height = o) : c || l.clearRect(0, 0, o, o);
		let d;
		this.renderBuffer_ !== void 0 && (d = Ke(), $e(d, e), ze(d, t * (this.renderBuffer_ + r), d));
		let f = Vc(r), p;
		function m(e, t, n) {
			let s = l.getImageData(0, 0, o, o).data;
			for (let c = 0, u = f.length; c < u; c++) if (s[f[c]] > 0) {
				if (!a || n === "none" || p !== "Image" && p !== "Text" || a.includes(e)) {
					let n = (f[c] - 3) / 4, a = r - n % o, s = r - (n / o | 0), l = i(e, t, a * a + s * s);
					if (l) return l;
				}
				l.clearRect(0, 0, o, o);
				break;
			}
		}
		let h = Object.keys(this.executorsByZIndex_).map(Number);
		h.sort(u);
		let g, _, v, y, b;
		for (g = h.length - 1; g >= 0; --g) {
			let e = h[g].toString();
			for (v = this.executorsByZIndex_[e], _ = Nc.length - 1; _ >= 0; --_) if (p = Nc[_], y = v[p], y !== void 0 && (b = y.executeHitDetection(l, s, n, m, d), b)) return b;
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
		return zr(o, 0, 8, 2, e, o), o;
	}
	isEmpty() {
		return r(this.executorsByZIndex_);
	}
	execute(e, t, n, r, i, a, o) {
		let s = Object.keys(this.executorsByZIndex_).map(Number);
		s.sort(o ? d : u), a ||= Nc;
		let c = Nc.length;
		for (let l = 0, u = s.length; l < u; ++l) {
			let u = s[l].toString(), d = this.executorsByZIndex_[u];
			for (let u = 0, f = a.length; u < f; ++u) {
				let f = a[u], p = d[f];
				if (p !== void 0) {
					let a = o === null ? void 0 : p.getZIndexContext(), u = a ? a.getContext() : e, d = this.maxExtent_ && f !== "Image" && f !== "Text";
					if (d && (u.save(), this.clip(u, n)), !a || f === "Text" || f === "Image" ? p.execute(u, t, n, r, i, o) : a.pushFunction((e) => p.execute(e, t, n, r, i, o)), d && u.restore(), a) {
						a.offset();
						let e = s[l] * c + Nc.indexOf(f);
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
		let e = this.deferredZIndexContexts_, t = Object.keys(e).map(Number).sort(u);
		for (let n = 0, r = t.length; n < r; ++n) e[t[n]].forEach((e) => {
			e.draw(this.renderedContext_), e.clear();
		}), e[t[n]].length = 0;
	}
}, Bc = {};
function Vc(e) {
	if (Bc[e] !== void 0) return Bc[e];
	let t = e * 2 + 1, n = e * e, r = Array(n + 1);
	for (let i = 0; i <= e; ++i) for (let a = 0; a <= e; ++a) {
		let o = i * i + a * a;
		if (o > n) break;
		let s = r[o];
		s || (s = [], r[o] = s), s.push(((e + i) * t + (e + a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e + a)) * 4 + 3), a > 0 && (s.push(((e + i) * t + (e - a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e - a)) * 4 + 3));
	}
	let i = [];
	for (let e = 0, t = r.length; e < t; ++e) r[e] && i.push(...r[e]);
	return Bc[e] = i, i;
}
//#endregion
//#region node_modules/ol/style/Image.js
var Hc = class e {
	constructor(e) {
		this.opacity_ = e.opacity, this.rotateWithView_ = e.rotateWithView, this.rotation_ = e.rotation, this.scale_ = e.scale, this.scaleArray_ = no(e.scale), this.displacement_ = e.displacement, this.declutterMode_ = e.declutterMode;
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
		return E();
	}
	getImage(e) {
		return E();
	}
	getHitDetectionImage() {
		return E();
	}
	getPixelRatio(e) {
		return 1;
	}
	getImageState() {
		return E();
	}
	getImageSize() {
		return E();
	}
	getOrigin() {
		return E();
	}
	getSize() {
		return E();
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
		this.scale_ = e, this.scaleArray_ = no(e);
	}
	listenImageChange(e) {
		E();
	}
	load() {
		E();
	}
	unlistenImageChange(e) {
		E();
	}
	ready() {
		return Promise.resolve();
	}
};
//#endregion
//#region node_modules/ol/style/Icon.js
function Uc(e, t, n, r) {
	return n !== void 0 && r !== void 0 ? [n / e, r / t] : n === void 0 ? r === void 0 ? 1 : r / t : n / e;
}
var Wc = class e extends Hc {
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
		R(!(o !== void 0 && a), "`image` and `src` cannot be provided at the same time"), (o === void 0 || o.length === 0) && a && (o = a.src || O(a)), R(o !== void 0 && o.length > 0, "A defined and non-empty `src` or `image` must be provided"), R(e.width === void 0 && e.height === void 0 || e.scale === void 0, "`width` or `height` cannot be provided together with `scale`");
		let s;
		if (e.src === void 0 ? a !== void 0 && (s = "complete" in a ? a.complete ? a.src ? V.LOADED : V.IDLE : V.LOADING : V.LOADED) : s = V.IDLE, this.color_ = e.color === void 0 ? null : No(e.color), this.iconImage_ = Es(a, o, {
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
						this.setScale(Uc(n[0], n[1], e.width, e.height));
					};
					this.listenImageChange(t);
					return;
				}
			}
			t !== void 0 && this.setScale(Uc(t, n, e.width, e.height));
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
		let t = e ? No(e) : null;
		if (this.color_ === t || this.color_ && t && this.color_.length === t.length && this.color_.every((e, n) => e === t[n])) return;
		this.color_ = t;
		let n = this.getSrc(), r = n === void 0 ? this.getHitDetectionImage() : null, i = n === void 0 ? this.iconImage_.getImageState() : V.IDLE;
		this.iconImage_ = Es(r, n, {
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
		this.iconImage_ = Es(null, e, {
			crossOrigin: this.crossOrigin_,
			referrerPolicy: this.referrerPolicy_
		}, V.IDLE, this.color_);
	}
	getSize() {
		return this.size_ ? this.size_ : this.iconImage_.getSize();
	}
	getWidth() {
		let e = this.getScaleArray();
		if (this.size_) return this.size_[0] * e[0];
		if (this.iconImage_.getImageState() == V.LOADED) return this.iconImage_.getSize()[0] * e[0];
	}
	getHeight() {
		let e = this.getScaleArray();
		if (this.size_) return this.size_[1] * e[1];
		if (this.iconImage_.getImageState() == V.LOADED) return this.iconImage_.getSize()[1] * e[1];
	}
	setScale(e) {
		delete this.initialOptions_, super.setScale(e);
	}
	listenImageChange(e) {
		this.iconImage_.addEventListener(s.CHANGE, e);
	}
	load() {
		this.iconImage_.load();
	}
	unlistenImageChange(e) {
		this.iconImage_.removeEventListener(s.CHANGE, e);
	}
	ready() {
		return this.iconImage_.ready();
	}
}, Gc = class extends ks {
	constructor(e, t, n, r, i, a, o) {
		super(), this.context_ = e, this.pixelRatio_ = t, this.extent_ = n, this.transform_ = r, this.transformRotation_ = r ? Ft(Math.atan2(r[1], r[0]), 10) : 0, this.viewRotation_ = i, this.squaredTolerance_ = a, this.userTransform_ = o, this.contextFillState_ = null, this.contextStrokeState_ = null, this.contextTextState_ = null, this.fillState_ = null, this.strokeState_ = null, this.image_ = null, this.imageAnchorX_ = 0, this.imageAnchorY_ = 0, this.imageHeight_ = 0, this.imageOpacity_ = 0, this.imageOriginX_ = 0, this.imageOriginY_ = 0, this.imageRotateWithView_ = !1, this.imageRotation_ = 0, this.imageScale_ = [0, 0], this.imageWidth_ = 0, this.text_ = "", this.textOffsetX_ = 0, this.textOffsetY_ = 0, this.textRotateWithView_ = !1, this.textRotation_ = 0, this.textScale_ = [0, 0], this.textFillState_ = null, this.textStrokeState_ = null, this.textState_ = null, this.pixelCoordinates_ = [], this.tmpLocalTransform_ = Tr();
	}
	drawImages_(e, t, n, r) {
		if (!this.image_) return;
		let i = zr(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.tmpLocalTransform_, s = a.globalAlpha;
		this.imageOpacity_ != 1 && (a.globalAlpha = s * this.imageOpacity_);
		let c = this.imageRotation_;
		this.transformRotation_ === 0 && (c -= this.viewRotation_), this.imageRotateWithView_ && (c += this.viewRotation_);
		for (let e = 0, t = i.length; e < t; e += 2) {
			let t = i[e] - this.imageAnchorX_, n = i[e + 1] - this.imageAnchorY_;
			if (c !== 0 || this.imageScale_[0] != 1 || this.imageScale_[1] != 1) {
				let e = t + this.imageAnchorX_, r = n + this.imageAnchorY_;
				Mr(o, e, r, 1, 1, c, -e, -r), a.save(), a.transform.apply(a, o), a.translate(e, r), a.scale(this.imageScale_[0], this.imageScale_[1]), a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, -this.imageAnchorX_, -this.imageAnchorY_, this.imageWidth_, this.imageHeight_), a.restore();
			} else a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, t, n, this.imageWidth_, this.imageHeight_);
		}
		this.imageOpacity_ != 1 && (a.globalAlpha = s);
	}
	drawText_(e, t, n, r) {
		if (!this.textState_ || this.text_ === "") return;
		this.textFillState_ && this.setContextFillState_(this.textFillState_), this.textStrokeState_ && this.setContextStrokeState_(this.textStrokeState_), this.setContextTextState_(this.textState_);
		let i = zr(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.textRotation_;
		for (this.transformRotation_ === 0 && (o -= this.viewRotation_), this.textRotateWithView_ && (o += this.viewRotation_); t < n; t += r) {
			let e = i[t] + this.textOffsetX_, n = i[t + 1] + this.textOffsetY_;
			o !== 0 || this.textScale_[0] != 1 || this.textScale_[1] != 1 ? (a.save(), a.translate(e - this.textOffsetX_, n - this.textOffsetY_), a.rotate(o), a.translate(this.textOffsetX_, this.textOffsetY_), a.scale(this.textScale_[0], this.textScale_[1]), this.textStrokeState_ && a.strokeText(this.text_, 0, 0), this.textFillState_ && a.fillText(this.text_, 0, 0), a.restore()) : (this.textStrokeState_ && a.strokeText(this.text_, e, n), this.textFillState_ && a.fillText(this.text_, e, n));
		}
	}
	moveToLineTo_(e, t, n, r, i, a) {
		let o = this.context_, s = zr(e, t, n, r, this.transform_, this.pixelCoordinates_);
		if (Math.abs(a) > 0) {
			let e = s.length, t = i || Math.abs(s[0] - s[e - 2]) < 1e-6 && Math.abs(s[1] - s[e - 1]) < 1e-6;
			s = gc(s, 0, e, 2, a, t, s), vc(s, 2, t);
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), ht(this.extent_, e.getExtent())) {
			if (this.fillState_ || this.strokeState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				let t = Yr(e, this.transform_, this.pixelCoordinates_), n = t[2] - t[0], r = t[3] - t[1], i = Math.sqrt(n * n + r * r), a = this.context_;
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), ht(this.extent_, e.getExtent())) {
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
		if (ht(this.extent_, t)) {
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), ht(this.extent_, e.getExtent())) {
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), ht(this.extent_, e.getExtent())) {
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
		n ? (n.lineCap != e.lineCap && (n.lineCap = e.lineCap, t.lineCap = e.lineCap), h(n.lineDash, e.lineDash) || t.setLineDash(n.lineDash = e.lineDash), n.lineDashOffset != e.lineDashOffset && (n.lineDashOffset = e.lineDashOffset, t.lineDashOffset = e.lineDashOffset), n.lineJoin != e.lineJoin && (n.lineJoin = e.lineJoin, t.lineJoin = e.lineJoin), n.lineWidth != e.lineWidth && (n.lineWidth = e.lineWidth, t.lineWidth = e.lineWidth), n.miterLimit != e.miterLimit && (n.miterLimit = e.miterLimit, t.miterLimit = e.miterLimit), n.strokeStyle != e.strokeStyle && (n.strokeStyle = e.strokeStyle, t.strokeStyle = e.strokeStyle)) : (t.lineCap = e.lineCap, t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset, t.lineJoin = e.lineJoin, t.lineWidth = e.lineWidth, t.miterLimit = e.miterLimit, t.strokeStyle = e.strokeStyle, this.contextStrokeState_ = {
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
		let t = this.context_, n = this.contextTextState_, r = e.textAlign ? e.textAlign : Is;
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
			this.fillState_ = { fillStyle: Ds(t || js) };
		}
		if (!t) this.strokeState_ = null;
		else {
			let e = t.getColor(), n = t.getLineCap(), r = t.getLineDash(), i = t.getLineDashOffset(), a = t.getLineJoin(), o = t.getWidth(), s = t.getMiterLimit(), c = r || Ns, l = t.getOffset();
			this.strokeState_ = {
				lineCap: n === void 0 ? Ms : n,
				lineDash: this.pixelRatio_ === 1 ? c : c.map((e) => e * this.pixelRatio_),
				lineDashOffset: (i || 0) * this.pixelRatio_,
				lineJoin: a === void 0 ? Ps : a,
				lineWidth: (o === void 0 ? 1 : o) * this.pixelRatio_,
				miterLimit: s === void 0 ? 10 : s,
				strokeStyle: Ds(e || Fs),
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
				this.textFillState_ = { fillStyle: Ds(e || js) };
			}
			let n = e.getStroke();
			if (!n) this.textStrokeState_ = null;
			else {
				let e = n.getColor(), t = n.getLineCap(), r = n.getLineDash(), i = n.getLineDashOffset(), a = n.getLineJoin(), o = n.getWidth(), s = n.getMiterLimit();
				this.textStrokeState_ = {
					lineCap: t === void 0 ? Ms : t,
					lineDash: r || Ns,
					lineDashOffset: i || 0,
					lineJoin: a === void 0 ? Ps : a,
					lineWidth: o === void 0 ? 1 : o,
					miterLimit: s === void 0 ? 10 : s,
					strokeStyle: Ds(e || Fs)
				};
			}
			let r = e.getFont(), i = e.getOffsetX(), a = e.getOffsetY(), o = e.getRotateWithView(), s = e.getRotation(), c = e.getScaleArray(), l = e.getText(), u = e.getTextAlign(), d = e.getTextBaseline();
			this.textState_ = {
				font: r === void 0 ? As : r,
				textAlign: u === void 0 ? Is : u,
				textBaseline: d === void 0 ? Ls : d
			}, this.text_ = l === void 0 ? "" : Array.isArray(l) ? l.reduce((e, t, n) => e += n % 2 ? " " : t, "") : l, this.textOffsetX_ = i === void 0 ? 0 : this.pixelRatio_ * i, this.textOffsetY_ = a === void 0 ? 0 : this.pixelRatio_ * a, this.textRotateWithView_ = o !== void 0 && o, this.textRotation_ = s === void 0 ? 0 : s, this.textScale_ = [this.pixelRatio_ * c[0], this.pixelRatio_ * c[1]];
		}
	}
}, Kc = .5;
function qc(e, t, n, r, i, a, o, s, c) {
	let l = c ? yr(i, c) : i, d = I(e[0] * Kc, e[1] * Kc);
	d.imageSmoothingEnabled = !1;
	let f = d.canvas, p = new Gc(d, Kc, i, null, o, s, c ? ur(gr(), c) : null), m = n.length, h = Math.floor(16777215 / m), g = {};
	for (let e = 1; e <= m; ++e) {
		let t = n[e - 1], i = t.getStyleFunction() || r;
		if (!i) continue;
		let o = i(t, a);
		if (!o) continue;
		Array.isArray(o) || (o = [o]);
		let s = (e * h).toString(16).padStart(7, "#00000");
		for (let e = 0, n = o.length; e < n; ++e) {
			let n = o[e], r = n.getGeometryFunction()(t);
			if (!r || !ht(l, r.getExtent())) continue;
			let i = n.clone(), a = i.getFill();
			a && a.setColor(s);
			let c = i.getStroke();
			c && (c.setColor(s), c.setLineDash(null)), i.setText(void 0);
			let u = n.getImage();
			if (u) {
				let e = u.getImageSize();
				if (!e) continue;
				let t = I(e[0], e[1], void 0, { alpha: !1 }), n = t.canvas;
				t.fillStyle = s, t.fillRect(0, 0, n.width, n.height), i.setImage(new Wc({
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
			let d = i.getZIndex() || 0, f = g[d];
			f || (f = {}, g[d] = f, f.Polygon = [], f.Circle = [], f.LineString = [], f.Point = []);
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
	let _ = Object.keys(g).map(Number).sort(u);
	for (let e = 0, n = _.length; e < n; ++e) {
		let n = g[_[e]];
		for (let e in n) {
			let r = n[e];
			for (let e = 0, n = r.length; e < n; e += 2) {
				p.setStyle(r[e + 1]);
				for (let n = 0, i = t.length; n < i; ++n) p.setTransform(t[n]), p.drawGeometry(r[e]);
			}
		}
	}
	return d.getImageData(0, 0, f.width, f.height);
}
function Jc(e, t, n) {
	let r = [];
	if (n) {
		let i = Math.floor(Math.round(e[0]) * Kc), a = Math.floor(Math.round(e[1]) * Kc), o = (Dt(i, 0, n.width - 1) + Dt(a, 0, n.height - 1) * n.width) * 4, s = n.data[o], c = n.data[o + 1], l = n.data[o + 2] + 256 * (c + 256 * s), u = Math.floor(16777215 / t.length);
		l && l % u === 0 && r.push(t[l / u - 1]);
	}
	return r;
}
//#endregion
//#region node_modules/ol/renderer/vector.js
var Yc = .5, Xc = {
	Point: cl,
	LineString: al,
	Polygon: ul,
	MultiPoint: ll,
	MultiLineString: ol,
	MultiPolygon: sl,
	GeometryCollection: il,
	Circle: el
};
function Zc(e, t) {
	return parseInt(O(e), 10) - parseInt(O(t), 10);
}
function Qc(e, t) {
	let n = $c(e, t);
	return n * n;
}
function $c(e, t) {
	return Yc * e / t;
}
function el(e, t, n, r, i) {
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
function tl(e, t, n, r, i, a, o, s) {
	let c = [], l = n.getImage();
	if (l) {
		let e = !0, t = l.getImageState();
		t == V.LOADED || t == V.ERROR ? e = !1 : t == V.IDLE && l.load(), e && c.push(l.ready());
	}
	let u = n.getFill();
	u && u.loading() && c.push(u.ready());
	let d = c.length > 0;
	return d && Promise.all(c).then(() => i(null)), nl(e, t, n, r, a, o, s), d;
}
function nl(e, t, n, r, i, a, o) {
	let s = n.getGeometryFunction()(t);
	if (!s) return;
	let c = s.simplifyTransformed(r, i);
	if (n.getRenderer()) rl(e, c, n, t, o);
	else {
		let r = Xc[c.getType()];
		r(e, c, n, t, o, a);
	}
}
function rl(e, t, n, r, i) {
	if (t.getType() == "GeometryCollection") {
		let a = t.getGeometries();
		for (let t = 0, o = a.length; t < o; ++t) rl(e, a[t], n, r, i);
		return;
	}
	e.getBuilder(n.getZIndex(), "Default").drawCustom(t, r, n.getRenderer(), n.getHitDetectionRenderer(), i);
}
function il(e, t, n, r, i, a) {
	let o = t.getGeometriesArray(), s, c;
	for (s = 0, c = o.length; s < c; ++s) {
		let t = Xc[o[s].getType()];
		t(e, o[s], n, r, i, a);
	}
}
function al(e, t, n, r, i) {
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
function ol(e, t, n, r, i) {
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
function sl(e, t, n, r, i) {
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
function cl(e, t, n, r, i, a) {
	let o = n.getImage(), s = n.getText(), c = s && s.getText(), l = a && o && c ? {} : void 0;
	if (o) {
		if (o.getImageState() != V.LOADED) return;
		let a = e.getBuilder(n.getZIndex(), "Image");
		a.setImageStyle(o, l), a.drawPoint(t, r, i);
	}
	if (c) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(s, l), a.drawText(t, r, i);
	}
}
function ll(e, t, n, r, i, a) {
	let o = n.getImage(), s = o && o.getOpacity() !== 0, c = n.getText(), l = c && c.getText(), u = a && s && l ? {} : void 0;
	if (s) {
		if (o.getImageState() != V.LOADED) return;
		let a = e.getBuilder(n.getZIndex(), "Image");
		a.setImageStyle(o, u), a.drawMultiPoint(t, r, i);
	}
	if (l) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(c, u), a.drawText(t, r, i);
	}
}
function ul(e, t, n, r, i) {
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
var dl = class extends Uo {
	constructor(e) {
		super(e), this.boundHandleStyleImageChange_ = this.handleStyleImageChange_.bind(this), this.animatingOrInteracting_, this.hitDetectionImageData_ = null, this.clipExtent_ = null, this.extendX_ = !1, this.renderedFeatures_ = null, this.renderedRevision_ = -1, this.renderedResolution_ = NaN, this.renderedExtent_ = Ke(), this.wrappedRenderedExtent_ = Ke(), this.renderedRotation_, this.renderedCenter_ = null, this.renderedProjection_ = null, this.renderedPixelRatio_ = 1, this.renderedRenderOrder_ = null, this.renderedFrameDeclutter_, this.replayGroup_ = null, this.replayGroupChanged = !0, this.clipping = !0, this.targetContext_ = null, this.opacity_ = 1;
	}
	renderWorlds(e, t, n) {
		let r = t.extent, i = t.viewState, a = i.center, o = i.resolution, s = i.projection, c = i.rotation, l = s.getExtent(), u = this.getLayer().getSource(), d = this.getLayer().getDeclutter(), f = t.pixelRatio, p = t.viewHints, m = !(p[Jo.ANIMATING] || p[Jo.INTERACTING]), h = this.context, g = Math.round(L(r) / o * f), _ = Math.round(ut(r) / o * f), v = u.getWrapX() && s.canWrapX(), y = v ? L(l) : null, b = v ? Math.ceil((r[2] - l[2]) / y) + (this.extendX_ ? 2 : 1) : 1, x = v ? Math.floor((r[0] - l[0]) / y) - +!!this.extendX_ : 0;
		do {
			let r = this.getRenderTransform(a, o, 0, f, g, _, x * y);
			t.declutter && (r = r.slice(0)), e.execute(h, [h.canvas.width, h.canvas.height], r, c, m, n === void 0 ? Nc : n ? Pc : Fc, n ? d && t.declutter[d] : void 0);
		} while (++x < b);
	}
	setDrawContext_() {
		this.opacity_ !== 1 && (this.targetContext_ = this.context, this.context = I(this.context.canvas.width, this.context.canvas.height, Bo));
	}
	resetDrawContext_() {
		if (this.opacity_ !== 1 && this.targetContext_) {
			let e = this.targetContext_.globalAlpha;
			this.targetContext_.globalAlpha = this.opacity_, this.targetContext_.drawImage(this.context.canvas, 0, 0), this.targetContext_.globalAlpha = e, ve(this.context), Bo.push(this.context.canvas), this.context = this.targetContext_, this.targetContext_ = null;
		}
	}
	renderDeclutter(e) {
		this.replayGroup_ && this.getLayer().getDeclutter() && this.renderWorlds(this.replayGroup_, e, !0);
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
		if (!o && !(this.getLayer().hasListener(Io.PRERENDER) || this.getLayer().hasListener(Io.POSTRENDER))) return this.container;
		this.setDrawContext_(), this.preRender(i, e);
		let s = r.projection;
		this.clipExtent_ = null;
		let c = !1;
		if (o && n.extent && this.clipping) {
			let t = br(n.extent, s);
			o = ht(t, e.extent), o && !Ue(t, e.extent) && (e.declutter ? this.clipExtent_ = t : (this.clipUnrotated(i, e, t), c = !0));
		}
		return o && this.renderWorlds(a, e, !this.getLayer().getDeclutter() && void 0), c && i.restore(), this.postRender(i, e), this.renderedRotation_ !== r.rotation && (this.renderedRotation_ = r.rotation, this.hitDetectionImageData_ = null), e.declutter || this.resetDrawContext_(), this.container;
	}
	getFeatures(e) {
		return new Promise((t) => {
			if (this.frameState && !this.hitDetectionImageData_ && !this.animatingOrInteracting_) {
				let e = this.frameState.size.slice(), t = this.renderedCenter_, n = this.renderedResolution_, r = this.renderedRotation_, i = this.renderedProjection_, a = this.wrappedRenderedExtent_, o = this.getLayer(), s = [], c = e[0] * Kc, l = e[1] * Kc;
				s.push(this.getRenderTransform(t, n, r, Kc, c, l, 0).slice());
				let u = o.getSource(), d = i.getExtent();
				if (u.getWrapX() && i.canWrapX() && !Ue(d, a)) {
					let e = a[0], i = L(d), o = 0, u;
					for (; e < d[0];) --o, u = i * o, s.push(this.getRenderTransform(t, n, r, Kc, c, l, u).slice()), e += i;
					for (o = 0, e = a[2]; e > d[2];) ++o, u = i * o, s.push(this.getRenderTransform(t, n, r, Kc, c, l, u).slice()), e -= i;
				}
				let f = gr();
				this.hitDetectionImageData_ = qc(e, s, this.renderedFeatures_, o.getStyleFunction(), a, n, r, Qc(n, this.renderedPixelRatio_), f ? i : null);
			}
			t(Jc(e, this.renderedFeatures_, this.hitDetectionImageData_));
		});
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		if (!this.replayGroup_) return;
		let a = t.viewState.resolution, o = t.viewState.rotation, s = this.getLayer(), c = {}, l = function(e, t, n) {
			let a = O(e), o = c[a];
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
		let r = e.viewHints[Jo.ANIMATING], i = e.viewHints[Jo.INTERACTING], a = t.getUpdateWhileAnimating(), o = t.getUpdateWhileInteracting();
		if (this.ready && !a && r || !o && i) return this.animatingOrInteracting_ = !0, !0;
		this.animatingOrInteracting_ = !1;
		let s = e.extent, c = e.viewState, l = c.projection, u = c.resolution, d = e.pixelRatio, f = t.getRevision(), p = t.getRenderBuffer(), m = t.getRenderOrder();
		m === void 0 && (m = Zc);
		let g = c.center.slice(), _ = ze(s, p * u), v = _.slice(), y = [_.slice()], b = l.getExtent(), x = n.getWrapX() && l.canWrapX();
		if (this.extendX_ = !1, x) {
			let e = n.getExtent();
			e && !gt(e) && (this.extendX_ = e[0] < b[0] || e[2] > b[2]);
		}
		if (x && (!Ue(b, e.extent) || this.extendX_)) {
			let e = L(b), t = Math.max(L(_) / 2, e), n = b[0], r = b[2];
			this.extendX_ && (n -= e, r += e), _[0] = n - t, _[2] = r + t, Wt(g, l);
			let i = xt(y[0], l);
			i[0] < b[0] && i[2] < b[2] ? y.push([
				i[0] + e,
				i[1],
				i[2] + e,
				i[3]
			]) : i[0] > b[0] && i[2] > b[2] && y.push([
				i[0] - e,
				i[1],
				i[2] - e,
				i[3]
			]);
		}
		if (this.ready && this.renderedResolution_ == u && this.renderedPixelRatio_ === d && this.renderedRevision_ == f && this.renderedRenderOrder_ == m && this.renderedFrameDeclutter_ === !!e.declutter && Ue(this.wrappedRenderedExtent_, _)) return h(this.renderedExtent_, v) || (this.hitDetectionImageData_ = null, this.renderedExtent_ = v), this.renderedCenter_ = g, this.replayGroupChanged = !1, !0;
		this.replayGroup_ = null;
		let S = new hc($c(u, d), _, u, d), C = gr(), w;
		if (C) {
			for (let e = 0, t = y.length; e < t; ++e) {
				let t = y[e], r = yr(t, l);
				n.loadFeatures(r, xr(u, l), C);
			}
			w = ur(C, l);
		} else for (let e = 0, t = y.length; e < t; ++e) n.loadFeatures(y[e], u, l);
		let T = Qc(u, d), E = !0, D = (e, n) => {
			let r, i = e.getStyleFunction() || t.getStyleFunction();
			if (i && (r = i(e, u)), r) {
				let t = this.renderFeature(e, T, r, S, w, this.getLayer().getDeclutter(), n);
				E &&= !t;
			}
		}, O = yr(_, l), k = n.getFeaturesInExtent(O);
		m && k.sort(m);
		for (let e = 0, t = k.length; e < t; ++e) D(k[e], e);
		this.renderedFeatures_ = k, this.ready = E;
		let A = S.finish(), j = new zc(_, u, d, n.getOverlaps(), A, t.getRenderBuffer(), !!e.declutter);
		return this.renderedResolution_ = u, this.renderedRevision_ = f, this.renderedRenderOrder_ = m, this.renderedFrameDeclutter_ = !!e.declutter, this.renderedExtent_ = v, this.wrappedRenderedExtent_ = _, this.renderedCenter_ = g, this.renderedProjection_ = l, this.renderedPixelRatio_ = d, this.replayGroup_ = j, this.hitDetectionImageData_ = null, this.replayGroupChanged = !0, !0;
	}
	renderFeature(e, t, n, r, i, a, o) {
		if (!n) return !1;
		let s = !1;
		if (Array.isArray(n)) for (let c = 0, l = n.length; c < l; ++c) s = tl(r, e, n[c], t, this.boundHandleStyleImageChange_, i, a, o) || s;
		else s = tl(r, e, n, t, this.boundHandleStyleImageChange_, i, a, o);
		return s;
	}
};
//#endregion
//#region node_modules/quickselect/index.js
function fl(e, t, n = 0, r = e.length - 1, i = ml) {
	for (; r > n;) {
		if (r - n > 600) {
			let a = r - n + 1, o = t - n + 1, s = Math.log(a), c = .5 * Math.exp(2 * s / 3), l = .5 * Math.sqrt(s * c * (a - c) / a) * (o - a / 2 < 0 ? -1 : 1);
			fl(e, t, Math.max(n, Math.floor(t - o * c / a + l)), Math.min(r, Math.floor(t + (a - o) * c / a + l)), i);
		}
		let a = e[t], o = n, s = r;
		for (pl(e, n, t), i(e[r], a) > 0 && pl(e, n, r); o < s;) {
			for (pl(e, o, s), o++, s--; i(e[o], a) < 0;) o++;
			for (; i(e[s], a) > 0;) s--;
		}
		i(e[n], a) === 0 ? pl(e, n, s) : (s++, pl(e, s, r)), s <= t && (n = s + 1), t <= s && (r = s - 1);
	}
}
function pl(e, t, n) {
	let r = e[t];
	e[t] = e[n], e[n] = r;
}
function ml(e, t) {
	return e < t ? -1 : +(e > t);
}
//#endregion
//#region node_modules/rbush/index.js
var hl = class {
	constructor(e = 9) {
		this._maxEntries = Math.max(4, e), this._minEntries = Math.max(2, Math.ceil(this._maxEntries * .4)), this.clear();
	}
	all() {
		return this._all(this.data, []);
	}
	search(e) {
		let t = this.data, n = [];
		if (!Dl(e, t)) return n;
		let r = this.toBBox, i = [];
		for (; t;) {
			for (let a = 0; a < t.children.length; a++) {
				let o = t.children[a], s = t.leaf ? r(o) : o;
				Dl(e, s) && (t.leaf ? n.push(o) : El(e, s) ? this._all(o, n) : i.push(o));
			}
			t = i.pop();
		}
		return n;
	}
	collides(e) {
		let t = this.data;
		if (!Dl(e, t)) return !1;
		let n = [];
		for (; t;) {
			for (let r = 0; r < t.children.length; r++) {
				let i = t.children[r], a = t.leaf ? this.toBBox(i) : i;
				if (Dl(e, a)) {
					if (t.leaf || El(e, a)) return !0;
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
		return this.data = Ol([]), this;
	}
	remove(e, t) {
		if (!e) return this;
		let n = this.data, r = this.toBBox(e), i = [], a = [], o, s, c;
		for (; n || i.length;) {
			if (n || (n = i.pop(), s = i[i.length - 1], o = a.pop(), c = !0), n.leaf) {
				let r = gl(e, n.children, t);
				if (r !== -1) return n.children.splice(r, 1), i.push(n), this._condense(i), this;
			}
			!c && !n.leaf && El(n, r) ? (i.push(n), a.push(o), o = 0, s = n, n = n.children[0]) : s ? (o++, n = s.children[o], c = !1) : n = null;
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
		if (i <= a) return o = Ol(e.slice(t, n + 1)), _l(o, this.toBBox), o;
		r || (r = Math.ceil(Math.log(i) / Math.log(a)), a = Math.ceil(i / a ** (r - 1))), o = Ol([]), o.leaf = !1, o.height = r;
		let s = Math.ceil(i / a), c = s * Math.ceil(Math.sqrt(a));
		kl(e, t, n, c, this.compareMinX);
		for (let i = t; i <= n; i += c) {
			let t = Math.min(i + c - 1, n);
			kl(e, i, t, s, this.compareMinY);
			for (let n = i; n <= t; n += s) {
				let i = Math.min(n + s - 1, t);
				o.children.push(this._build(e, n, i, r - 1));
			}
		}
		return _l(o, this.toBBox), o;
	}
	_chooseSubtree(e, t, n, r) {
		for (; r.push(t), !(t.leaf || r.length - 1 === n);) {
			let n = Infinity, r = Infinity, i;
			for (let a = 0; a < t.children.length; a++) {
				let o = t.children[a], s = Sl(o), c = wl(e, o) - s;
				c < r ? (r = c, n = s < n ? s : n, i = o) : c === r && s < n && (n = s, i = o);
			}
			t = i || t.children[0];
		}
		return t;
	}
	_insert(e, t, n) {
		let r = n ? e : this.toBBox(e), i = [], a = this._chooseSubtree(r, this.data, t, i);
		for (a.children.push(e), yl(a, r); t >= 0 && i[t].children.length > this._maxEntries;) this._split(i, t), t--;
		this._adjustParentBBoxes(r, i, t);
	}
	_split(e, t) {
		let n = e[t], r = n.children.length, i = this._minEntries;
		this._chooseSplitAxis(n, i, r);
		let a = this._chooseSplitIndex(n, i, r), o = Ol(n.children.splice(a, n.children.length - a));
		o.height = n.height, o.leaf = n.leaf, _l(n, this.toBBox), _l(o, this.toBBox), t ? e[t - 1].children.push(o) : this._splitRoot(n, o);
	}
	_splitRoot(e, t) {
		this.data = Ol([e, t]), this.data.height = e.height + 1, this.data.leaf = !1, _l(this.data, this.toBBox);
	}
	_chooseSplitIndex(e, t, n) {
		let r, i = Infinity, a = Infinity;
		for (let o = t; o <= n - t; o++) {
			let t = vl(e, 0, o, this.toBBox), s = vl(e, o, n, this.toBBox), c = Tl(t, s), l = Sl(t) + Sl(s);
			c < i ? (i = c, r = o, a = l < a ? l : a) : c === i && l < a && (a = l, r = o);
		}
		return r || n - t;
	}
	_chooseSplitAxis(e, t, n) {
		let r = e.leaf ? this.compareMinX : bl, i = e.leaf ? this.compareMinY : xl;
		this._allDistMargin(e, t, n, r) < this._allDistMargin(e, t, n, i) && e.children.sort(r);
	}
	_allDistMargin(e, t, n, r) {
		e.children.sort(r);
		let i = this.toBBox, a = vl(e, 0, t, i), o = vl(e, n - t, n, i), s = Cl(a) + Cl(o);
		for (let r = t; r < n - t; r++) {
			let t = e.children[r];
			yl(a, e.leaf ? i(t) : t), s += Cl(a);
		}
		for (let r = n - t - 1; r >= t; r--) {
			let t = e.children[r];
			yl(o, e.leaf ? i(t) : t), s += Cl(o);
		}
		return s;
	}
	_adjustParentBBoxes(e, t, n) {
		for (let r = n; r >= 0; r--) yl(t[r], e);
	}
	_condense(e) {
		for (let t = e.length - 1, n; t >= 0; t--) e[t].children.length === 0 ? t > 0 ? (n = e[t - 1].children, n.splice(n.indexOf(e[t]), 1)) : this.clear() : _l(e[t], this.toBBox);
	}
};
function gl(e, t, n) {
	if (!n) return t.indexOf(e);
	for (let r = 0; r < t.length; r++) if (n(e, t[r])) return r;
	return -1;
}
function _l(e, t) {
	vl(e, 0, e.children.length, t, e);
}
function vl(e, t, n, r, i) {
	i ||= Ol(null), i.minX = Infinity, i.minY = Infinity, i.maxX = -Infinity, i.maxY = -Infinity;
	for (let a = t; a < n; a++) {
		let t = e.children[a];
		yl(i, e.leaf ? r(t) : t);
	}
	return i;
}
function yl(e, t) {
	return e.minX = Math.min(e.minX, t.minX), e.minY = Math.min(e.minY, t.minY), e.maxX = Math.max(e.maxX, t.maxX), e.maxY = Math.max(e.maxY, t.maxY), e;
}
function bl(e, t) {
	return e.minX - t.minX;
}
function xl(e, t) {
	return e.minY - t.minY;
}
function Sl(e) {
	return (e.maxX - e.minX) * (e.maxY - e.minY);
}
function Cl(e) {
	return e.maxX - e.minX + (e.maxY - e.minY);
}
function wl(e, t) {
	return (Math.max(t.maxX, e.maxX) - Math.min(t.minX, e.minX)) * (Math.max(t.maxY, e.maxY) - Math.min(t.minY, e.minY));
}
function Tl(e, t) {
	let n = Math.max(e.minX, t.minX), r = Math.max(e.minY, t.minY), i = Math.min(e.maxX, t.maxX), a = Math.min(e.maxY, t.maxY);
	return Math.max(0, i - n) * Math.max(0, a - r);
}
function El(e, t) {
	return e.minX <= t.minX && e.minY <= t.minY && t.maxX <= e.maxX && t.maxY <= e.maxY;
}
function Dl(e, t) {
	return t.minX <= e.maxX && t.minY <= e.maxY && t.maxX >= e.minX && t.maxY >= e.minY;
}
function Ol(e) {
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
function kl(e, t, n, r, i) {
	let a = [t, n];
	for (; a.length;) {
		if (n = a.pop(), t = a.pop(), n - t <= r) continue;
		let o = t + Math.ceil((n - t) / r / 2) * r;
		fl(e, o, t, n, i), a.push(t, o, o, n);
	}
}
//#endregion
//#region node_modules/ol/expr/expression.js
var Al = 0, jl = 1 << Al++, W = 1 << Al++, G = 1 << Al++, K = 1 << Al++, Ml = 1 << Al++, Nl = 1 << Al++, Pl = 2 ** Al - 1, Fl = {
	[jl]: "boolean",
	[W]: "number",
	[G]: "string",
	[K]: "color",
	[Ml]: "number[]",
	[Nl]: "size"
}, Il = Object.keys(Fl).map(Number).sort(u);
function Ll(e) {
	return e in Fl;
}
function Rl(e) {
	let t = [];
	for (let n of Il) zl(e, n) && t.push(Fl[n]);
	return t.length === 0 ? "untyped" : t.length < 3 ? t.join(" or ") : t.slice(0, -1).join(", ") + ", or " + t[t.length - 1];
}
function zl(e, t) {
	return (e & t) === t;
}
function Bl(e, t) {
	return !!(e & t);
}
function Vl(e, t) {
	return e === t;
}
var Hl = class {
	constructor(e, t) {
		if (!Ll(e)) throw Error(`literal expressions must have a specific type, got ${Rl(e)}`);
		this.type = e, this.value = t;
	}
}, Ul = class {
	constructor(e, t, ...n) {
		this.type = e, this.operator = t, this.args = n;
	}
};
function Wl(e) {
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
function Gl(e, t, n) {
	switch (typeof e) {
		case "boolean":
			if (Vl(t, G)) return new Hl(G, e ? "true" : "false");
			if (!zl(t, jl)) throw Error(`got a boolean, but expected ${Rl(t)}`);
			return new Hl(jl, e);
		case "number":
			if (Vl(t, Nl)) return new Hl(Nl, no(e));
			if (Vl(t, jl)) return new Hl(jl, !!e);
			if (Vl(t, G)) return new Hl(G, e.toString());
			if (!zl(t, W)) throw Error(`got a number, but expected ${Rl(t)}`);
			return new Hl(W, e);
		case "string":
			if (Vl(t, K)) return new Hl(K, Mo(e));
			if (Vl(t, jl)) return new Hl(jl, !!e);
			if (!zl(t, G)) throw Error(`got a string, but expected ${Rl(t)}`);
			return new Hl(G, e);
	}
	if (!Array.isArray(e)) throw Error("expression must be an array or a primitive value");
	if (e.length === 0) throw Error("empty expression");
	if (typeof e[0] == "string") return lu(e, t, n);
	for (let t of e) if (typeof t != "number") throw Error("expected an array of numbers");
	if (Vl(t, Nl)) {
		if (e.length !== 2) throw Error(`expected an array of two values for a size, got ${e.length}`);
		return new Hl(Nl, e);
	}
	if (Vl(t, K)) {
		if (e.length === 3) return new Hl(K, [...e, 1]);
		if (e.length === 4) return new Hl(K, e);
		throw Error(`expected an array of 3 or 4 values for a color, got ${e.length}`);
	}
	if (!zl(t, Ml)) throw Error(`got an array of numbers, but expected ${Rl(t)}`);
	return new Hl(Ml, e);
}
var q = {
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
}, Kl = {
	[q.Get]: X(J(1, Infinity), ql),
	[q.Var]: Jl(),
	[q.Has]: X(J(1, Infinity), ql),
	[q.Id]: X(Yl, $l),
	[q.Concat]: X(J(2, Infinity), Y(G)),
	[q.GeometryType]: X(Xl, $l),
	[q.LineMetric]: X(Zl, $l),
	[q.Resolution]: X(Ql, $l),
	[q.Zoom]: X(Ql, $l),
	[q.Time]: X(Ql, $l),
	[q.Any]: X(J(2, Infinity), Y(jl)),
	[q.All]: X(J(2, Infinity), Y(jl)),
	[q.Not]: X(J(1, 1), Y(jl)),
	[q.Equal]: X(J(2, 2), tu()),
	[q.NotEqual]: X(J(2, 2), tu()),
	[q.GreaterThan]: X(J(2, 2), Y(W)),
	[q.GreaterThanOrEqualTo]: X(J(2, 2), Y(W)),
	[q.LessThan]: X(J(2, 2), Y(W)),
	[q.LessThanOrEqualTo]: X(J(2, 2), Y(W)),
	[q.Multiply]: X(J(2, Infinity), eu),
	[q.Coalesce]: X(J(2, Infinity), eu),
	[q.Divide]: X(J(2, 2), Y(W)),
	[q.Add]: X(J(2, Infinity), Y(W)),
	[q.Subtract]: X(J(2, 2), Y(W)),
	[q.Clamp]: X(J(3, 3), Y(W)),
	[q.Mod]: X(J(2, 2), Y(W)),
	[q.Pow]: X(J(2, 2), Y(W)),
	[q.Abs]: X(J(1, 1), Y(W)),
	[q.Floor]: X(J(1, 1), Y(W)),
	[q.Ceil]: X(J(1, 1), Y(W)),
	[q.Round]: X(J(1, 1), Y(W)),
	[q.Sin]: X(J(1, 1), Y(W)),
	[q.Cos]: X(J(1, 1), Y(W)),
	[q.Atan]: X(J(1, 2), Y(W)),
	[q.Sqrt]: X(J(1, 1), Y(W)),
	[q.Match]: X(J(4, Infinity), ru, iu),
	[q.Between]: X(J(3, 3), Y(W)),
	[q.Interpolate]: X(J(6, Infinity), ru, au),
	[q.Case]: X(J(3, Infinity), nu, ou),
	[q.In]: X(J(2, 2), su),
	[q.Number]: X(J(1, Infinity), Y(Pl)),
	[q.String]: X(J(1, Infinity), Y(Pl)),
	[q.Array]: X(J(1, Infinity), Y(W)),
	[q.Color]: X(J(1, 4), Y(W)),
	[q.Band]: X(J(1, 3), Y(W)),
	[q.Palette]: X(J(2, 2), cu),
	[q.ToString]: X(J(1, 1), Y(jl | W | G | K))
};
function ql(e, t, n) {
	let r = e.length - 1, i = Array(r);
	for (let a = 0; a < r; ++a) {
		let r = e[a + 1];
		switch (typeof r) {
			case "number":
				i[a] = new Hl(W, r);
				break;
			case "string":
				i[a] = new Hl(G, r);
				break;
			default: throw Error(`expected a string key or numeric array index for a get operation, got ${r}`);
		}
		a === 0 && n.properties.set(String(r), t);
	}
	return i;
}
function Jl() {
	return function(e, t, n) {
		let r = e[1];
		if (typeof r != "string") throw Error("expected a string argument for var operation");
		let i = t, a = n.inputVariables?.[r];
		if (a !== void 0) {
			let e = Gl(a, Pl, n);
			if (!(e instanceof Hl)) throw Error(`style variables should only be literal values (no expressions!), variable name: ${r}`);
			let o = e.type;
			if (typeof a == "string" && Bl(i, K) && !Bl(i, G) ? o = K : Array.isArray(a) && a.length === 2 && Bl(i, Nl) && !Bl(i, Ml) && (o = Nl), i &= o, i === 0) throw Error(`the type expected from the var operator (${Rl(t)}) did not have any overlap with the type of the corresponding style variables (${Rl(o)}), variable name: ${r}`);
		}
		if (n.variables.has(r)) {
			let e = n.variables.get(r);
			if (i &= e, i === 0) throw Error(`a new type expected from the var operator (${Rl(t)}) did not have any overlap with the previous type expected for it (${Rl(e)}), variable name: ${r}`);
		}
		return n.variables.set(r, i), new Ul(i, "var", new Hl(G, r));
	};
}
function Yl(e, t, n) {
	n.featureId = !0;
}
function Xl(e, t, n) {
	n.geometryType = !0;
}
function Zl(e, t, n) {
	n.mCoordinate = !0;
}
function Ql(e, t, n) {
	n.mapState = !0;
}
function $l(e, t, n) {
	let r = e[0];
	if (e.length !== 1) throw Error(`expected no arguments for ${r} operation`);
	return [];
}
function J(e, t) {
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
function eu(e, t, n) {
	let r = e.length - 1, i = Array(r);
	for (let a = 0; a < r; ++a) {
		let r = Gl(e[a + 1], t, n);
		i[a] = r;
	}
	return i;
}
function Y(e) {
	return function(t, n, r) {
		let i = t.length - 1, a = Array(i);
		for (let n = 0; n < i; ++n) {
			let i = Gl(t[n + 1], e, r);
			a[n] = i;
		}
		return a;
	};
}
function tu() {
	return function(e, t, n) {
		let r = e[0], i = e.length - 1, a = Array(i), o = Pl;
		for (let t = 0; t < i; ++t) {
			let r = Gl(e[t + 1], o, n);
			o &= r.type;
		}
		if (o === 0) throw Error(`no common type was found among the arguments of ${r}`);
		for (let t = 0; t < i; ++t) {
			let r = Gl(e[t + 1], o, n);
			a[t] = r;
		}
		return a;
	};
}
function nu(e, t, n) {
	let r = e[0], i = e.length - 1;
	if (i % 2 == 0) throw Error(`expected an odd number of arguments for ${r}, got ${i} instead`);
}
function ru(e, t, n) {
	let r = e[0], i = e.length - 1;
	if (i % 2 == 1) throw Error(`expected an even number of arguments for operation ${r}, got ${i} instead`);
}
function iu(e, t, n) {
	let r = e.length - 1, i = Gl(e[e.length - 1], t, n), a = G | W | jl, o = Array(r - 2);
	for (let t = 0; t < r - 2; t += 2) {
		try {
			let r = Gl(e[t + 2], a, n);
			a &= r.type;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of match expression: ${e.message}`);
		}
		if (a === 0) throw Error("no common type was found among the arguments of match expression");
	}
	for (let t = 0; t < r - 2; t += 2) {
		try {
			let r = Gl(e[t + 2], a, n);
			o[t] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of match expression: ${e.message}`);
		}
		try {
			let r = Gl(e[t + 3], i.type, n);
			o[t + 1] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 2} of match expression: ${e.message}`);
		}
	}
	return [
		Gl(e[1], a, n),
		...o,
		i
	];
}
function au(e, t, n) {
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
	let a = new Hl(W, i), o;
	try {
		o = Gl(e[2], W, n);
	} catch (e) {
		throw Error(`failed to parse argument 1 in interpolate expression: ${e.message}`);
	}
	let s = Array(e.length - 3);
	for (let r = 0; r < s.length; r += 2) {
		try {
			let t = Gl(e[r + 3], W, n);
			s[r] = t;
		} catch (e) {
			throw Error(`failed to parse argument ${r + 2} for interpolate expression: ${e.message}`);
		}
		try {
			let i = Gl(e[r + 4], t, n);
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
function ou(e, t, n) {
	let r = Gl(e[e.length - 1], t, n), i = Array(e.length - 1);
	for (let t = 0; t < i.length - 1; t += 2) {
		try {
			let r = Gl(e[t + 1], jl, n);
			i[t] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t} of case expression: ${e.message}`);
		}
		try {
			let a = Gl(e[t + 2], r.type, n);
			i[t + 1] = a;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of case expression: ${e.message}`);
		}
	}
	return i[i.length - 1] = r, i;
}
function su(e, t, n) {
	let r = e[2];
	if (!Array.isArray(r)) throw Error("the second argument for the \"in\" operator must be an array");
	let i;
	if (r[0] === "literal") {
		if (r = r[1], !Array.isArray(r)) throw Error("failed to parse \"in\" expression: the literal operator must be followed by an array");
	} else if (typeof r[0] == "string") throw Error("for the \"in\" operator, a string array should be wrapped in a \"literal\" operator to disambiguate from expressions");
	i = typeof r[0] == "string" ? G : W;
	let a = Array(r.length);
	for (let e = 0; e < a.length; e++) try {
		let t = Gl(r[e], i, n);
		a[e] = t;
	} catch (t) {
		throw Error(`failed to parse haystack item ${e} for "in" expression: ${t.message}`);
	}
	return [Gl(e[1], i, n), ...a];
}
function cu(e, t, n) {
	let r;
	try {
		r = Gl(e[1], W, n);
	} catch (e) {
		throw Error(`failed to parse first argument in palette expression: ${e.message}`);
	}
	let i = e[2];
	if (!Array.isArray(i)) throw Error("the second argument of palette must be an array");
	let a = Array(i.length);
	for (let e = 0; e < a.length; e++) {
		let t;
		try {
			t = Gl(i[e], K, n);
		} catch (t) {
			throw Error(`failed to parse color at index ${e} in palette expression: ${t.message}`);
		}
		if (!(t instanceof Hl)) throw Error(`the palette color at index ${e} must be a literal value`);
		a[e] = t;
	}
	return [r, ...a];
}
function X(...e) {
	return function(t, n, r) {
		let i = t[0], a;
		for (let i = 0; i < e.length; i++) {
			let o = e[i](t, n, r);
			if (i == e.length - 1) {
				if (!o) throw Error("expected last argument validator to return the parsed args");
				a = o;
			}
		}
		return new Ul(n, i, ...a);
	};
}
function lu(e, t, n) {
	let r = e[0], i = Kl[r];
	if (!i) throw Error(`unknown operator: ${r}`);
	return i(e, t, n);
}
function uu(e) {
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
		case "GeometryCollection": return uu(e.getGeometries()[0]);
		default: return "";
	}
}
//#endregion
//#region node_modules/ol/expr/cpu.js
function du() {
	return {
		variables: {},
		properties: {},
		resolution: NaN,
		featureId: null,
		geometryType: ""
	};
}
function fu(e, t, n) {
	return pu(Gl(e, t, n), n);
}
function pu(e, t) {
	if (e instanceof Hl) {
		if (e.type === K && typeof e.value == "string") {
			let t = Mo(e.value);
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
		case q.Number:
		case q.String:
		case q.Coalesce: return mu(e, t);
		case q.Get:
		case q.Var:
		case q.Has: return hu(e, t);
		case q.Id: return (e) => e.featureId;
		case q.GeometryType: return (e) => e.geometryType;
		case q.Concat: {
			let n = e.args.map((e) => pu(e, t));
			return (e) => "".concat(...n.map((t) => t(e).toString()));
		}
		case q.Resolution: return (e) => e.resolution;
		case q.Any:
		case q.All:
		case q.Between:
		case q.In:
		case q.Not: return _u(e, t);
		case q.Equal:
		case q.NotEqual:
		case q.LessThan:
		case q.LessThanOrEqualTo:
		case q.GreaterThan:
		case q.GreaterThanOrEqualTo: return gu(e, t);
		case q.Multiply:
		case q.Divide:
		case q.Add:
		case q.Subtract:
		case q.Clamp:
		case q.Mod:
		case q.Pow:
		case q.Abs:
		case q.Floor:
		case q.Ceil:
		case q.Round:
		case q.Sin:
		case q.Cos:
		case q.Atan:
		case q.Sqrt: return vu(e, t);
		case q.Case: return yu(e, t);
		case q.Match: return bu(e, t);
		case q.Interpolate: return xu(e, t);
		case q.ToString: return Su(e, t);
		default: throw Error(`Unsupported operator ${n}`);
	}
}
function mu(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = pu(e.args[n], t);
	switch (n) {
		case q.Coalesce: return (e) => {
			for (let t = 0; t < r; ++t) {
				let n = i[t](e);
				if (n != null) return n;
			}
			throw Error("Expected one of the values to be non-null");
		};
		case q.Number:
		case q.String: return (e) => {
			for (let t = 0; t < r; ++t) {
				let r = i[t](e);
				if (typeof r === n) return r;
			}
			throw Error(`Expected one of the values to be a ${n}`);
		};
		default: throw Error(`Unsupported assertion operator ${n}`);
	}
}
function hu(e, t) {
	let n = e.args[0].value;
	switch (e.operator) {
		case q.Get: return (t) => {
			let r = e.args, i = t.properties[n];
			for (let e = 1, t = r.length; e < t; ++e) {
				let t = r[e].value;
				i = i[t];
			}
			return i;
		};
		case q.Var: return (e) => e.variables[n];
		case q.Has: return (t) => {
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
function gu(e, t) {
	let n = e.operator, r = pu(e.args[0], t), i = pu(e.args[1], t);
	switch (n) {
		case q.Equal: return (e) => r(e) === i(e);
		case q.NotEqual: return (e) => r(e) !== i(e);
		case q.LessThan: return (e) => r(e) < i(e);
		case q.LessThanOrEqualTo: return (e) => r(e) <= i(e);
		case q.GreaterThan: return (e) => r(e) > i(e);
		case q.GreaterThanOrEqualTo: return (e) => r(e) >= i(e);
		default: throw Error(`Unsupported comparison operator ${n}`);
	}
}
function _u(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = pu(e.args[n], t);
	switch (n) {
		case q.Any: return (e) => {
			for (let t = 0; t < r; ++t) if (i[t](e)) return !0;
			return !1;
		};
		case q.All: return (e) => {
			for (let t = 0; t < r; ++t) if (!i[t](e)) return !1;
			return !0;
		};
		case q.Between: return (e) => {
			let t = i[0](e), n = i[1](e), r = i[2](e);
			return t >= n && t <= r;
		};
		case q.In: return (e) => {
			let t = i[0](e);
			for (let n = 1; n < r; ++n) if (t === i[n](e)) return !0;
			return !1;
		};
		case q.Not: return (e) => !i[0](e);
		default: throw Error(`Unsupported logical operator ${n}`);
	}
}
function vu(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = pu(e.args[n], t);
	switch (n) {
		case q.Multiply: return (e) => {
			let t = 1;
			for (let n = 0; n < r; ++n) t *= i[n](e);
			return t;
		};
		case q.Divide: return (e) => i[0](e) / i[1](e);
		case q.Add: return (e) => {
			let t = 0;
			for (let n = 0; n < r; ++n) t += i[n](e);
			return t;
		};
		case q.Subtract: return (e) => i[0](e) - i[1](e);
		case q.Clamp: return (e) => {
			let t = i[0](e), n = i[1](e);
			if (t < n) return n;
			let r = i[2](e);
			return t > r ? r : t;
		};
		case q.Mod: return (e) => i[0](e) % i[1](e);
		case q.Pow: return (e) => i[0](e) ** +i[1](e);
		case q.Abs: return (e) => Math.abs(i[0](e));
		case q.Floor: return (e) => Math.floor(i[0](e));
		case q.Ceil: return (e) => Math.ceil(i[0](e));
		case q.Round: return (e) => Math.round(i[0](e));
		case q.Sin: return (e) => Math.sin(i[0](e));
		case q.Cos: return (e) => Math.cos(i[0](e));
		case q.Atan: return r === 2 ? (e) => Math.atan2(i[0](e), i[1](e)) : (e) => Math.atan(i[0](e));
		case q.Sqrt: return (e) => Math.sqrt(i[0](e));
		default: throw Error(`Unsupported numeric operator ${n}`);
	}
}
function yu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = pu(e.args[i], t);
	return (e) => {
		for (let t = 0; t < n - 1; t += 2) if (r[t](e)) return r[t + 1](e);
		return r[n - 1](e);
	};
}
function bu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = pu(e.args[i], t);
	return (e) => {
		let t = r[0](e);
		for (let i = 1; i < n - 1; i += 2) if (t === r[i](e)) return r[i + 1](e);
		return r[n - 1](e);
	};
}
function xu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = pu(e.args[i], t);
	return (e) => {
		let t = r[0](e), i = r[1](e), a, o;
		for (let s = 2; s < n; s += 2) {
			let n = r[s](e), c = r[s + 1](e), l = Array.isArray(c);
			if (l && (c = To(c)), n >= i) return s === 2 ? c : l ? wu(t, i, a, o, n, c) : Cu(t, i, a, o, n, c);
			a = n, o = c;
		}
		return o;
	};
}
function Su(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = pu(e.args[n], t);
	switch (n) {
		case q.ToString: return (t) => {
			let n = i[0](t);
			return e.args[0].type === K ? Po(n) : n.toString();
		};
		default: throw Error(`Unsupported convert operator ${n}`);
	}
}
function Cu(e, t, n, r, i, a) {
	let o = i - n;
	if (o === 0) return r;
	let s = t - n;
	return r + (e === 1 ? s / o : (e ** +s - 1) / (e ** +o - 1)) * (a - r);
}
function wu(e, t, n, r, i, a) {
	if (i - n === 0) return r;
	let o = Ao(r), s = Ao(a), c = s[2] - o[2];
	return c > 180 ? c -= 360 : c < -180 && (c += 360), jo([
		Cu(e, t, n, o[0], i, s[0]),
		Cu(e, t, n, o[1], i, s[1]),
		o[2] + Cu(e, t, n, 0, i, c),
		Cu(e, t, n, r[3], i, a[3])
	]);
}
//#endregion
//#region node_modules/ol/style/RegularShape.js
var Tu = class e extends Hc {
	constructor(e) {
		super({
			opacity: 1,
			rotateWithView: e.rotateWithView !== void 0 && e.rotateWithView,
			rotation: e.rotation === void 0 ? 0 : e.rotation,
			scale: e.scale === void 0 ? 1 : e.scale,
			displacement: e.displacement === void 0 ? [0, 0] : e.displacement,
			declutterMode: e.declutterMode
		}), this.hitDetectionCanvas_ = null, this.fill_ = e.fill === void 0 ? null : e.fill, this.origin_ = [0, 0], this.points_ = e.points, this.radius = e.radius, this.radius2_ = e.radius2, this.angle_ = e.angle === void 0 ? 0 : e.angle, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.size_, this.renderOptions_, this.imageState_ = this.fill_ && this.fill_.loading() ? V.LOADING : V.LOADED, this.imageState_ === V.LOADING && this.ready().then(() => this.imageState_ = V.LOADED), this.render();
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
		let t = this.fill_?.getKey(), n = `${e},${this.angle_},${this.radius},${this.radius2_},${this.points_},${t}` + Object.values(this.renderOptions_).join(","), r = Cs.get(n, null)?.getImage(1);
		if (!r) {
			let t = this.renderOptions_, i = Math.ceil(t.size * e), a = I(i, i);
			this.draw_(t, a, e), r = a.canvas;
			let o = new Ts(r, void 0, null, V.LOADED, null);
			Cs.set(n, null, o), createImageBitmap(r).then((e) => {
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
		let e = Ms, t = Ps, n = 0, r = null, i = 0, a, o = 0;
		this.stroke_ && (a = Ds(this.stroke_.getColor() ?? "#000"), o = this.stroke_.getWidth() ?? 1, r = this.stroke_.getLineDash(), i = this.stroke_.getLineDashOffset() ?? 0, t = this.stroke_.getLineJoin() ?? "round", e = this.stroke_.getLineCap() ?? "round", n = this.stroke_.getMiterLimit() ?? 10);
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
			e === null && (e = js), t.fillStyle = Ds(e), t.fill();
		}
		e.strokeStyle && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineCap = e.lineCap, t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}
	createHitDetectionCanvas_(e) {
		let t;
		if (this.fill_) {
			let n = this.fill_.getColor(), r = 0;
			typeof n == "string" && (n = No(n)), n === null ? r = 1 : Array.isArray(n) && (r = n.length === 4 ? n[3] : 1), r === 0 && (t = I(e.size, e.size), this.drawHitDetectionCanvas_(e, t));
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
		t.translate(e.size / 2, e.size / 2), this.createPath_(t), t.fillStyle = js, t.fill(), e.strokeStyle && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}
	ready() {
		return this.fill_ ? this.fill_.ready() : Promise.resolve();
	}
}, Eu = class e extends Tu {
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
}, Du = class e {
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
			let t = Es(null, e.src, { crossOrigin: "anonymous" }, void 0, e.offset ? null : e.color ? e.color : null, !(e.offset && e.size));
			t.ready().then(() => {
				this.patternImage_ = null;
			}), t.getImageState() === V.IDLE && t.load(), t.getImageState() === V.LOADING && (this.patternImage_ = t);
		}
		this.color_ = e;
	}
	getKey() {
		let e = this.getColor();
		return e ? e instanceof CanvasPattern || e instanceof CanvasGradient ? O(e) : typeof e == "object" && "src" in e ? e.src + ":" + e.offset : No(e).toString() : "";
	}
	loading() {
		return !!this.patternImage_;
	}
	ready() {
		return this.patternImage_ ? this.patternImage_.ready() : Promise.resolve();
	}
}, Ou = class e {
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
}, ku = class e {
	constructor(e) {
		e ||= {}, this.geometry_ = null, this.geometryFunction_ = Nu, e.geometry !== void 0 && this.setGeometry(e.geometry), this.fill_ = e.fill === void 0 ? null : e.fill, this.image_ = e.image === void 0 ? null : e.image, this.renderer_ = e.renderer === void 0 ? null : e.renderer, this.hitDetectionRenderer_ = e.hitDetectionRenderer === void 0 ? null : e.hitDetectionRenderer, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.text_ = e.text === void 0 ? null : e.text, this.zIndex_ = e.zIndex;
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
		}) : this.geometryFunction_ = Nu, this.geometry_ = e;
	}
	setZIndex(e) {
		this.zIndex_ = e;
	}
};
function Au(e) {
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
var ju = null;
function Mu(e, t) {
	if (!ju) {
		let e = new Du({ color: "rgba(255,255,255,0.4)" }), t = new Ou({
			color: "#3399CC",
			width: 1.25
		});
		ju = [new ku({
			image: new Eu({
				fill: e,
				stroke: t,
				radius: 5
			}),
			fill: e,
			stroke: t
		})];
	}
	return ju;
}
function Nu(e) {
	return e.getGeometry();
}
//#endregion
//#region node_modules/ol/style/Text.js
var Pu = "#333", Fu = class e {
	constructor(e) {
		e ||= {}, this.font_ = e.font, this.rotation_ = e.rotation, this.rotateWithView_ = e.rotateWithView, this.keepUpright_ = e.keepUpright, this.scale_ = e.scale, this.scaleArray_ = no(e.scale === void 0 ? 1 : e.scale), this.text_ = e.text, this.textAlign_ = e.textAlign, this.justify_ = e.justify, this.repeat_ = e.repeat, this.textBaseline_ = e.textBaseline, this.fill_ = e.fill === void 0 ? new Du({ color: Pu }) : e.fill, this.maxAngle_ = e.maxAngle === void 0 ? Math.PI / 4 : e.maxAngle, this.placement_ = e.placement === void 0 ? "point" : e.placement, this.overflow_ = !!e.overflow, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.offsetX_ = e.offsetX === void 0 ? 0 : e.offsetX, this.offsetY_ = e.offsetY === void 0 ? 0 : e.offsetY, this.backgroundFill_ = e.backgroundFill ? e.backgroundFill : null, this.backgroundStroke_ = e.backgroundStroke ? e.backgroundStroke : null, this.padding_ = e.padding === void 0 ? null : e.padding, this.declutterMode_ = e.declutterMode;
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
			fill: this.getFill() instanceof Du ? this.getFill().clone() : this.getFill(),
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
		this.scale_ = e, this.scaleArray_ = no(e === void 0 ? 1 : e);
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
function Iu(e) {
	return !0;
}
function Lu(e, t) {
	t ??= Wl();
	let n = Bu(e, t), r = du();
	return function(e, i) {
		if (r.properties = e.getPropertiesInternal(), r.resolution = i, t.featureId) {
			let t = e.getId();
			t === void 0 ? r.featureId = null : r.featureId = t;
		}
		return t.geometryType && (r.geometryType = uu(e.getGeometry())), n(r);
	};
}
function Ru(e, t) {
	t ??= Wl();
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Vu(e[i], t);
	let i = du(), a = Array(n);
	return function(e, o) {
		if (i.properties = e.getPropertiesInternal(), i.resolution = o, t.featureId) {
			let t = e.getId();
			t === void 0 ? i.featureId = null : i.featureId = t;
		}
		t.geometryType && (i.geometryType = uu(e.getGeometry()));
		let s = 0;
		for (let e = 0; e < n; ++e) {
			let t = r[e](i);
			t && (a[s] = t, s += 1);
		}
		return a.length = s, a;
	};
}
function zu(e, t) {
	if (t ??= Wl(), !Array.isArray(e)) return Ru([e], t);
	let n = e.length;
	if ("style" in e[0]) {
		let r = Array(n);
		for (let t = 0; t < n; ++t) {
			let n = e[t];
			if (!("style" in n)) throw Error("Expected a list of rules with a style property");
			r[t] = n;
		}
		return Lu(r, t);
	}
	return Ru(e, t);
}
function Bu(e, t) {
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) {
		let n = e[i], a = "filter" in n ? fu(n.filter, jl, t) : Iu, o;
		if (Array.isArray(n.style)) {
			let e = n.style.length;
			o = Array(e);
			for (let r = 0; r < e; ++r) o[r] = Vu(n.style[r], t);
		} else o = [Vu(n.style, t)];
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
function Vu(e, t) {
	let n = Hu(e, "", t), i = Uu(e, "", t), a = Wu(e, t), o = Gu(e, t), s = Xu(e, "z-index", t);
	if (!n && !i && !a && !o && !r(e)) throw Error("No fill, stroke, point, or text symbolizer properties in style: " + JSON.stringify(e));
	let c = new ku();
	return function(e) {
		let t = !0;
		if (n) {
			let r = n(e);
			r && (t = !1), c.setFill(r);
		}
		if (i) {
			let n = i(e);
			n && (t = !1), c.setStroke(n);
		}
		if (a) {
			let n = a(e);
			n && (t = !1), c.setText(n);
		}
		if (o) {
			let n = o(e);
			n && (t = !1), c.setImage(n);
		}
		return s && c.setZIndex(s(e)), t ? null : c;
	};
}
function Hu(e, t, n) {
	let r;
	if (t + "fill-pattern-src" in e) r = Qu(e, t + "fill-", n);
	else {
		if (e[t + "fill-color"] === "none") return (e) => null;
		r = ed(e, t + "fill-color", n);
	}
	if (!r) return null;
	let i = new Du();
	return function(e) {
		let t = r(e);
		return t === uo ? null : (i.setColor(t), i);
	};
}
function Uu(e, t, n) {
	let r = Xu(e, t + "stroke-width", n), i = ed(e, t + "stroke-color", n);
	if (!r && !i) return null;
	let a = Zu(e, t + "stroke-line-cap", n), o = Zu(e, t + "stroke-line-join", n), s = td(e, t + "stroke-line-dash", n), c = Xu(e, t + "stroke-line-dash-offset", n), l = Xu(e, t + "stroke-miter-limit", n), u = Xu(e, t + "stroke-offset", n), d = new Ou();
	return function(e) {
		if (i) {
			let t = i(e);
			if (t === uo) return null;
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
function Wu(e, t) {
	let n = "text-", r = Zu(e, "text-value", t);
	if (!r) return null;
	let i = Hu(e, n, t), a = Hu(e, "text-background-", t), o = Uu(e, n, t), s = Uu(e, "text-background-", t), c = Zu(e, "text-font", t), l = Xu(e, "text-max-angle", t), u = Xu(e, "text-offset-x", t), d = Xu(e, "text-offset-y", t), f = $u(e, "text-overflow", t), p = Zu(e, "text-placement", t), m = Xu(e, "text-repeat", t), h = id(e, "text-scale", t), g = $u(e, "text-rotate-with-view", t), _ = Xu(e, "text-rotation", t), v = Zu(e, "text-align", t), y = Zu(e, "text-justify", t), b = Zu(e, "text-baseline", t), x = $u(e, "text-keep-upright", t), S = td(e, "text-padding", t), C = new Fu({ declutterMode: dd(e, "text-declutter-mode") });
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
function Gu(e, t) {
	return "icon-src" in e ? Ku(e, t) : "shape-points" in e ? qu(e, t) : "circle-radius" in e ? Ju(e, t) : null;
}
function Ku(e, t) {
	let n = "icon-src", r = pd(e[n], n), i = nd(e, "icon-anchor", t), a = id(e, "icon-scale", t), o = Xu(e, "icon-opacity", t), s = nd(e, "icon-displacement", t), c = Xu(e, "icon-rotation", t), l = $u(e, "icon-rotate-with-view", t), u = cd(e, "icon-anchor-origin"), d = ld(e, "icon-anchor-x-units"), f = ld(e, "icon-anchor-y-units"), p = Yu(e, "icon-color"), m, h = null;
	p !== void 0 && (Array.isArray(p) && p.length > 0 && typeof p[0] == "string" ? h = ed(e, "icon-color", t) : m = hd(p, "icon-color"));
	let g = sd(e, "icon-cross-origin"), _ = ud(e, "icon-offset"), v = cd(e, "icon-offset-origin"), y = ad(e, "icon-width"), b = {
		src: r,
		anchorOrigin: u,
		anchorXUnits: d,
		anchorYUnits: f,
		crossOrigin: g,
		offset: _,
		offsetOrigin: v,
		height: ad(e, "icon-height"),
		width: y,
		size: od(e, "icon-size"),
		declutterMode: dd(e, "icon-declutter-mode")
	}, x = null;
	return function(e) {
		if (x) h && x.setColor(h(e));
		else {
			let t = h ? h(e) : m;
			x = new Wc(t === void 0 ? Object.assign({}, b) : Object.assign({}, b, { color: t }));
		}
		return o && x.setOpacity(o(e)), s && x.setDisplacement(s(e)), c && x.setRotation(c(e)), l && x.setRotateWithView(l(e)), a && x.setScale(a(e)), i && x.setAnchor(i(e)), x;
	};
}
function qu(e, t) {
	let n = "shape-", r = "shape-points", i = "shape-radius", a = md(e[r], r);
	if (!(i in e)) throw Error(`Expected a number for ${i}`);
	let o = Xu(e, i, t), s = typeof e[i] == "number" ? e[i] : 5, c = "shape-radius2", l = Xu(e, c, t), u = typeof e[c] == "number" ? e[c] : void 0, d = Hu(e, n, t), f = Uu(e, n, t), p = id(e, "shape-scale", t), m = nd(e, "shape-displacement", t), h = Xu(e, "shape-rotation", t), g = $u(e, "shape-rotate-with-view", t), _ = new Tu({
		points: a,
		radius: s,
		radius2: u,
		angle: ad(e, "shape-angle"),
		declutterMode: dd(e, "shape-declutter-mode")
	});
	return function(e) {
		return o && _.setRadius(o(e)), l && _.setRadius2(l(e)), d && _.setFill(d(e)), f && _.setStroke(f(e)), m && _.setDisplacement(m(e)), h && _.setRotation(h(e)), g && _.setRotateWithView(g(e)), p && _.setScale(p(e)), _;
	};
}
function Ju(e, t) {
	let n = "circle-", r = Hu(e, n, t), i = Uu(e, n, t), a = Xu(e, "circle-radius", t), o = id(e, "circle-scale", t), s = nd(e, "circle-displacement", t), c = Xu(e, "circle-rotation", t), l = $u(e, "circle-rotate-with-view", t), u = new Eu({
		radius: 5,
		declutterMode: dd(e, "circle-declutter-mode")
	});
	return function(e) {
		return a && u.setRadius(a(e)), r && u.setFill(r(e)), i && u.setStroke(i(e)), s && u.setDisplacement(s(e)), c && u.setRotation(c(e)), l && u.setRotateWithView(l(e)), o && u.setScale(o(e)), u;
	};
}
function Yu(e, t) {
	if (!(t in e)) return;
	let n = e[t];
	return n === void 0 ? void 0 : n;
}
function Xu(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return;
	let i = fu(r, W, n);
	return function(e) {
		return md(i(e), t);
	};
}
function Zu(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	let i = fu(r, G, n);
	return function(e) {
		return pd(i(e), t);
	};
}
function Qu(e, t, n) {
	let r = Zu(e, t + "pattern-src", n), i = rd(e, t + "pattern-offset", n), a = rd(e, t + "pattern-size", n), o = ed(e, t + "color", n);
	return function(e) {
		return {
			src: r(e),
			offset: i && i(e),
			size: a && a(e),
			color: o && o(e)
		};
	};
}
function $u(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	let i = fu(r, jl, n);
	return function(e) {
		let n = i(e);
		if (typeof n != "boolean") throw Error(`Expected a boolean for ${t}`);
		return n;
	};
}
function ed(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	let i = fu(r, K, n);
	return function(e) {
		return hd(i(e), t);
	};
}
function td(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	if (Array.isArray(r) && (r.length === 0 || typeof r[0] != "string")) {
		let e = r.map((e, r) => {
			if (typeof e == "number") return () => e;
			let i = fu(e, W, n);
			return function(e) {
				return md(i(e), `${t}[${r}]`);
			};
		});
		return function(t) {
			let n = Array(e.length);
			for (let r = 0; r < e.length; ++r) n[r] = e[r](t);
			return n;
		};
	}
	let i = fu(r, Ml, n);
	return function(e) {
		return fd(i(e), t);
	};
}
function nd(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	let i = fu(r, Ml, n);
	return function(e) {
		let n = fd(i(e), t);
		if (n.length !== 2) throw Error(`Expected two numbers for ${t}`);
		return n;
	};
}
function rd(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	let i = fu(r, Ml, n);
	return function(e) {
		return gd(i(e), t);
	};
}
function id(e, t, n) {
	let r = Yu(e, t);
	if (r === void 0) return null;
	let i = fu(r, Ml | W, n);
	return function(e) {
		return _d(i(e), t);
	};
}
function ad(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n != "number") throw Error(`Expected a number for ${t}`);
		return n;
	}
}
function od(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n == "number") return no(n);
		if (!Array.isArray(n) || n.length !== 2 || typeof n[0] != "number" || typeof n[1] != "number") throw Error(`Expected a number or size array for ${t}`);
		return n;
	}
}
function sd(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n != "string") throw Error(`Expected a string for ${t}`);
		return n;
	}
}
function cd(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (n !== "bottom-left" && n !== "bottom-right" && n !== "top-left" && n !== "top-right") throw Error(`Expected bottom-left, bottom-right, top-left, or top-right for ${t}`);
		return n;
	}
}
function ld(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (n !== "pixels" && n !== "fraction") throw Error(`Expected pixels or fraction for ${t}`);
		return n;
	}
}
function ud(e, t) {
	let n = e[t];
	if (n !== void 0) return fd(n, t);
}
function dd(e, t) {
	let n = e[t];
	if (n !== void 0) {
		if (typeof n != "string") throw Error(`Expected a string for ${t}`);
		if (n !== "declutter" && n !== "obstacle" && n !== "none") throw Error(`Expected declutter, obstacle, or none for ${t}`);
		return n;
	}
}
function fd(e, t) {
	if (!Array.isArray(e)) throw Error(`Expected an array for ${t}`);
	let n = e.length;
	for (let r = 0; r < n; ++r) if (typeof e[r] != "number") throw Error(`Expected an array of numbers for ${t}`);
	return e;
}
function pd(e, t) {
	if (typeof e != "string") throw Error(`Expected a string for ${t}`);
	return e;
}
function md(e, t) {
	if (typeof e != "number") throw Error(`Expected a number for ${t}`);
	return e;
}
function hd(e, t) {
	if (typeof e == "string") return e;
	let n = fd(e, t), r = n.length;
	if (r < 3 || r > 4) throw Error(`Expected a color with 3 or 4 values for ${t}`);
	return n;
}
function gd(e, t) {
	let n = fd(e, t);
	if (n.length !== 2) throw Error(`Expected an array of two numbers for ${t}`);
	return n;
}
function _d(e, t) {
	return typeof e == "number" ? e : gd(e, t);
}
//#endregion
//#region node_modules/ol/layer/BaseVector.js
var vd = { RENDER_ORDER: "renderOrder" }, yd = class extends gs {
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
		return this.get(vd.RENDER_ORDER);
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
		n in e.declutter || (e.declutter[n] = new hl(9)), this.getRenderer().renderDeclutter(e, t);
	}
	setRenderOrder(e) {
		this.set(vd.RENDER_ORDER, e);
	}
	setStyle(e) {
		this.style_ = e === void 0 ? Mu : e;
		let t = bd(e);
		this.styleFunction_ = e === null ? void 0 : Au(t), this.changed();
	}
	setDeclutter(e) {
		this.declutter_ = e ? String(e) : void 0, this.changed();
	}
};
function bd(e) {
	if (e === void 0) return Mu;
	if (!e) return null;
	if (typeof e == "function" || e instanceof ku) return e;
	if (Array.isArray(e) && e.length === 0) return [];
	if (Array.isArray(e) && e[0] instanceof ku) {
		let t = e.length, n = Array(t);
		for (let r = 0; r < t; ++r) {
			let t = e[r];
			if (!(t instanceof ku)) throw Error("Expected a list of style instances");
			n[r] = t;
		}
		return n;
	}
	return zu(e);
}
//#endregion
//#region node_modules/ol/layer/Vector.js
var xd = class extends yd {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new dl(this);
	}
}, Sd = class extends S {
	constructor(e, t, n) {
		super(e), this.map = t, this.frameState = n === void 0 ? null : n;
	}
}, Cd = class extends Sd {
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
}, wd = {
	SINGLECLICK: "singleclick",
	CLICK: s.CLICK,
	DBLCLICK: s.DBLCLICK,
	POINTERDRAG: "pointerdrag",
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, Td = {
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, Ed = class extends C {
	constructor(e, t) {
		super(e), this.map_ = e, this.clickTimeoutId_, this.emulateClicks_ = !1, this.dragging_ = !1, this.dragListenerKeys_ = [], this.moveTolerance_ = t === void 0 ? 1 : t, this.down_ = null;
		let n = this.map_.getViewport();
		this.activePointers_ = [], this.trackedTouches_ = {}, this.element_ = n, this.pointerdownListenerKey_ = i(n, Td.POINTERDOWN, this.handlePointerDown_, this), this.originalPointerMoveEvent_, this.relayedListenerKey_ = i(n, Td.POINTERMOVE, this.relayMoveEvent_, this), this.boundHandleTouchMove_ = this.handleTouchMove_.bind(this), this.element_.addEventListener(s.TOUCHMOVE, this.boundHandleTouchMove_, he ? { passive: !1 } : !1);
	}
	emulateClick_(e) {
		let t = new Cd(wd.CLICK, this.map_, e);
		this.dispatchEvent(t), this.clickTimeoutId_ === void 0 ? this.clickTimeoutId_ = setTimeout(() => {
			this.clickTimeoutId_ = void 0;
			let t = new Cd(wd.SINGLECLICK, this.map_, e);
			this.dispatchEvent(t);
		}, 250) : (clearTimeout(this.clickTimeoutId_), this.clickTimeoutId_ = void 0, t = new Cd(wd.DBLCLICK, this.map_, e), this.dispatchEvent(t));
	}
	updateActivePointers_(e) {
		let t = e, n = t.pointerId;
		if (t.type == wd.POINTERUP || t.type == wd.POINTERCANCEL) {
			delete this.trackedTouches_[n];
			for (let e in this.trackedTouches_) if (this.trackedTouches_[e].target !== t.target) {
				delete this.trackedTouches_[e];
				break;
			}
		} else (t.type == wd.POINTERDOWN || t.type == wd.POINTERMOVE) && (this.trackedTouches_[n] = t);
		this.activePointers_ = Object.values(this.trackedTouches_);
	}
	handlePointerUp_(e) {
		this.updateActivePointers_(e);
		let t = new Cd(wd.POINTERUP, this.map_, e, void 0, void 0, this.activePointers_);
		this.dispatchEvent(t), this.emulateClicks_ && !t.defaultPrevented && !this.dragging_ && this.isMouseActionButton_(e) && this.emulateClick_(this.down_), this.activePointers_.length === 0 && (this.dragListenerKeys_.forEach(o), this.dragListenerKeys_.length = 0, this.dragging_ = !1, this.down_ = null);
	}
	isMouseActionButton_(e) {
		return e.button === 0;
	}
	handlePointerDown_(e) {
		this.emulateClicks_ = this.activePointers_.length === 0, this.updateActivePointers_(e);
		let t = new Cd(wd.POINTERDOWN, this.map_, e, void 0, void 0, this.activePointers_);
		if (this.dispatchEvent(t), this.down_ = new PointerEvent(e.type, e), Object.defineProperty(this.down_, "target", {
			writable: !1,
			value: e.target
		}), this.dragListenerKeys_.length === 0) {
			let e = this.map_.getOwnerDocument();
			this.dragListenerKeys_.push(i(e, wd.POINTERMOVE, this.handlePointerMove_, this), i(e, wd.POINTERUP, this.handlePointerUp_, this), i(this.element_, wd.POINTERCANCEL, this.handlePointerUp_, this)), this.element_.getRootNode && this.element_.getRootNode() !== e && this.dragListenerKeys_.push(i(this.element_.getRootNode(), wd.POINTERUP, this.handlePointerUp_, this));
		}
	}
	handlePointerMove_(e) {
		if (this.isMoving_(e)) {
			this.updateActivePointers_(e), this.dragging_ = !0;
			let t = new Cd(wd.POINTERDRAG, this.map_, e, this.dragging_, void 0, this.activePointers_);
			this.dispatchEvent(t);
		}
	}
	relayMoveEvent_(e) {
		this.originalPointerMoveEvent_ = e;
		let t = !!(this.down_ && this.isMoving_(e));
		this.dispatchEvent(new Cd(wd.POINTERMOVE, this.map_, e, t));
	}
	handleTouchMove_(e) {
		let t = this.originalPointerMoveEvent_;
		(!t || t.defaultPrevented) && (typeof e.cancelable != "boolean" || e.cancelable === !0) && e.preventDefault();
	}
	isMoving_(e) {
		return this.dragging_ || Math.abs(e.clientX - this.down_.clientX) > this.moveTolerance_ || Math.abs(e.clientY - this.down_.clientY) > this.moveTolerance_;
	}
	disposeInternal() {
		this.relayedListenerKey_ &&= (o(this.relayedListenerKey_), null), this.element_.removeEventListener(s.TOUCHMOVE, this.boundHandleTouchMove_), this.pointerdownListenerKey_ &&= (o(this.pointerdownListenerKey_), null), this.dragListenerKeys_.forEach(o), this.dragListenerKeys_.length = 0, this.element_ = null, super.disposeInternal();
	}
}, Dd = {
	LAYERGROUP: "layergroup",
	SIZE: "size",
	TARGET: "target",
	VIEW: "view"
}, Od = Infinity, kd = class {
	constructor(e, t) {
		this.priorityFunction_ = e, this.keyFunction_ = t, this.elements_ = [], this.priorities_ = [], this.queuedElements_ = {};
	}
	clear() {
		this.elements_.length = 0, this.priorities_.length = 0, n(this.queuedElements_);
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
}, Ad = class extends kd {
	constructor(e, t) {
		super((t) => e.apply(null, t), (e) => e[0].getKey()), this.boundHandleTileChange_ = this.handleTileChange.bind(this), this.tileChangeCallback_ = t, this.tilesLoading_ = 0, this.tilesLoadingKeys_ = {};
	}
	enqueue(e) {
		let t = super.enqueue(e);
		return t && e[0].addEventListener(s.CHANGE, this.boundHandleTileChange_), t;
	}
	getTilesLoading() {
		return this.tilesLoading_;
	}
	handleTileChange(e) {
		let t = e.target, n = t.getState();
		if (n === B.LOADED || n === B.ERROR || n === B.EMPTY) {
			n !== B.ERROR && t.removeEventListener(s.CHANGE, this.boundHandleTileChange_);
			let e = t.getKey();
			e in this.tilesLoadingKeys_ && (delete this.tilesLoadingKeys_[e], --this.tilesLoading_), this.tileChangeCallback_();
		}
	}
	loadMoreTiles(e, t) {
		let n = 0;
		for (; this.tilesLoading_ < e && n < t && this.getCount() > 0;) {
			let e = this.dequeue()[0], t = e.getKey();
			e.getState() === B.IDLE && !(t in this.tilesLoadingKeys_) && (this.tilesLoadingKeys_[t] = !0, ++this.tilesLoading_, ++n, e.load());
		}
	}
};
function jd(e, t, n, r, i) {
	if (!e || !(n in e.wantedTiles) || !e.wantedTiles[n][t.getKey()]) return Od;
	let a = e.viewState.center, o = r[0] - a[0], s = r[1] - a[1];
	return 65536 * Math.log(i) + Math.sqrt(o * o + s * s) / i;
}
//#endregion
//#region node_modules/ol/Kinetic.js
var Md = class {
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
}, Nd = { ACTIVE: "active" }, Pd = class extends A {
	constructor(e) {
		super(), this.on, this.once, this.un, e && e.handleEvent && (this.handleEvent = e.handleEvent), this.map_ = null, this.setActive(!0);
	}
	getActive() {
		return this.get(Nd.ACTIVE);
	}
	getMap() {
		return this.map_;
	}
	handleEvent(e) {
		return !0;
	}
	setActive(e) {
		this.set(Nd.ACTIVE, e);
	}
	setMap(e) {
		this.map_ = e;
	}
};
function Fd(e, t, n) {
	let r = e.getCenterInternal();
	if (r) {
		let i = [r[0] + t[0], r[1] + t[1]];
		e.animateInternal({
			duration: n === void 0 ? 250 : n,
			easing: Me,
			center: e.getConstrainedCenter(i)
		});
	}
}
function Id(e, t, n, r) {
	let i = e.getZoom();
	if (i === void 0) return;
	let a = e.getConstrainedZoom(i + t), o = e.getResolutionForZoom(a);
	e.getAnimating() && e.cancelAnimations(), e.animate({
		resolution: o,
		anchor: n,
		duration: r === void 0 ? 250 : r,
		easing: Ae
	});
}
//#endregion
//#region node_modules/ol/interaction/DoubleClickZoom.js
var Ld = class extends Pd {
	constructor(e) {
		super(), e ||= {}, this.delta_ = e.delta ? e.delta : 1, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == wd.DBLCLICK) {
			let n = e.originalEvent, r = e.map, i = e.coordinate, a = n.shiftKey ? -this.delta_ : this.delta_;
			Id(r.getView(), a, i, this.duration_), n.preventDefault(), t = !0;
		}
		return !t;
	}
};
//#endregion
//#region node_modules/ol/events/condition.js
function Rd(e) {
	let t = arguments;
	return function(e) {
		let n = !0;
		for (let r = 0, i = t.length; r < i && (n &&= t[r](e), n); ++r);
		return n;
	};
}
var zd = function(e) {
	let t = e.originalEvent;
	return t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, Bd = function(e) {
	let t = e.map.getTargetElement(), n = t.getRootNode(), r = e.map.getOwnerDocument().activeElement;
	return n instanceof ShadowRoot ? n.host.contains(r) : t.contains(r);
}, Vd = function(e) {
	let t = e.map.getTargetElement(), n = t.getRootNode();
	return !(n instanceof ShadowRoot ? n.host : t).hasAttribute("tabindex") || Bd(e);
}, Hd = _, Ud = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.button == 0 && !(ue && de && t.ctrlKey);
}, Wd = function(e) {
	let t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && !t.shiftKey;
}, Gd = function(e) {
	let t = e.originalEvent;
	return de ? t.metaKey : t.ctrlKey;
}, Kd = function(e) {
	let t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, qd = function(e) {
	let t = e.originalEvent, n = t.target.tagName;
	return n !== "INPUT" && n !== "SELECT" && n !== "TEXTAREA" && !t.target.isContentEditable;
}, Jd = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.pointerType == "mouse";
}, Yd = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.isPrimary && t.button === 0;
}, Xd = class extends Pd {
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
			if (e.type == wd.POINTERDRAG) this.handleDragEvent(e), e.originalEvent.preventDefault();
			else if (e.type == wd.POINTERUP) {
				let t = this.handleUpEvent(e);
				this.handlingDownUpSequence = t && this.targetPointers.length > 0;
			}
		} else if (e.type == wd.POINTERDOWN) {
			let n = this.handleDownEvent(e);
			this.handlingDownUpSequence = n, t = this.stopDown(n);
		} else e.type == wd.POINTERMOVE && this.handleMoveEvent(e);
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
function Zd(e) {
	let t = e.length, n = 0, r = 0;
	for (let i = 0; i < t; i++) n += e[i].clientX, r += e[i].clientY;
	return {
		clientX: n / t,
		clientY: r / t
	};
}
//#endregion
//#region node_modules/ol/interaction/DragPan.js
var Qd = class extends Xd {
	constructor(e) {
		super({ stopDown: v }), e ||= {}, this.kinetic_ = e.kinetic, this.lastCentroid = null, this.lastPointersCount_, this.panning_ = !1;
		let t = e.condition ? e.condition : Rd(Wd, Yd);
		this.condition_ = e.onFocusOnly ? Rd(Vd, t) : t, this.noKinetic_ = !1;
	}
	handleDragEvent(e) {
		let t = e.map;
		this.panning_ || (this.panning_ = !0, t.getView().beginInteraction());
		let n = this.targetPointers, r = t.getEventPixel(Zd(n));
		if (n.length == this.lastPointersCount_) {
			if (this.kinetic_ && this.kinetic_.update(r[0], r[1]), this.lastCentroid) {
				let t = [this.lastCentroid[0] - r[0], r[1] - this.lastCentroid[1]], n = e.map.getView();
				Ut(t, n.getResolution()), Ht(t, n.getRotation()), n.adjustCenterInternal(t);
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
					easing: Ae
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
}, $d = class extends Xd {
	constructor(e) {
		e ||= {}, super({ stopDown: v }), this.condition_ = e.condition ? e.condition : zd, this.lastAngle_ = void 0, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleDragEvent(e) {
		if (!Jd(e)) return;
		let t = e.map, n = t.getView();
		if (n.getConstraints().rotation === rs) return;
		let r = t.getSize(), i = e.pixel, a = Math.atan2(r[1] / 2 - i[1], i[0] - r[0] / 2);
		if (this.lastAngle_ !== void 0) {
			let e = a - this.lastAngle_;
			n.adjustRotationInternal(-e);
		}
		this.lastAngle_ = a;
	}
	handleUpEvent(e) {
		return !Jd(e) || (e.map.getView().endInteraction(this.duration_), !1);
	}
	handleDownEvent(e) {
		return Jd(e) && Ud(e) && this.condition_(e) ? (e.map.getView().beginInteraction(), this.lastAngle_ = void 0, !0) : !1;
	}
}, ef = class extends c {
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
		n[4] = n[0].slice(), this.geometry_ ? this.geometry_.setCoordinates([n]) : this.geometry_ = new Yi([n]);
	}
	getGeometry() {
		return this.geometry_;
	}
}, tf = {
	BOXSTART: "boxstart",
	BOXDRAG: "boxdrag",
	BOXEND: "boxend",
	BOXCANCEL: "boxcancel"
}, nf = class extends S {
	constructor(e, t, n) {
		super(e), this.coordinate = t, this.mapBrowserEvent = n;
	}
}, rf = class extends Xd {
	constructor(e) {
		super(), this.on, this.once, this.un, e ??= {}, this.box_ = new ef(e.className || "ol-dragbox"), this.minArea_ = e.minArea ?? 64, e.onBoxEnd && (this.onBoxEnd = e.onBoxEnd), this.startPixel_ = null, this.condition_ = e.condition ?? Ud, this.boxEndCondition_ = e.boxEndCondition ?? this.defaultBoxEndCondition;
	}
	defaultBoxEndCondition(e, t, n) {
		let r = n[0] - t[0], i = n[1] - t[1];
		return r * r + i * i >= this.minArea_;
	}
	getGeometry() {
		return this.box_.getGeometry();
	}
	handleDragEvent(e) {
		this.startPixel_ && (this.box_.setPixels(this.startPixel_, e.pixel), this.dispatchEvent(new nf(tf.BOXDRAG, e.coordinate, e)));
	}
	handleUpEvent(e) {
		if (!this.startPixel_) return !1;
		let t = this.boxEndCondition_(e, this.startPixel_, e.pixel);
		return t && this.onBoxEnd(e), this.dispatchEvent(new nf(t ? tf.BOXEND : tf.BOXCANCEL, e.coordinate, e)), this.box_.setMap(null), this.startPixel_ = null, !1;
	}
	handleDownEvent(e) {
		return this.condition_(e) ? (this.startPixel_ = e.pixel, this.box_.setMap(e.map), this.box_.setPixels(this.startPixel_, this.startPixel_), this.dispatchEvent(new nf(tf.BOXSTART, e.coordinate, e)), !0) : !1;
	}
	onBoxEnd(e) {}
	setActive(e) {
		e || (this.box_.setMap(null), this.startPixel_ &&= (this.dispatchEvent(new nf(tf.BOXCANCEL, this.startPixel_, null)), null)), super.setActive(e);
	}
	setMap(e) {
		this.getMap() && (this.box_.setMap(null), this.startPixel_ &&= (this.dispatchEvent(new nf(tf.BOXCANCEL, this.startPixel_, null)), null)), super.setMap(e);
	}
}, af = class extends rf {
	constructor(e) {
		e ||= {};
		let t = e.condition ? e.condition : Kd;
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
			easing: Ae
		});
	}
}, of = {
	LEFT: "ArrowLeft",
	UP: "ArrowUp",
	RIGHT: "ArrowRight",
	DOWN: "ArrowDown"
}, sf = class extends Pd {
	constructor(e) {
		super(), e ||= {}, this.defaultCondition_ = function(e) {
			return Wd(e) && qd(e);
		}, this.condition_ = e.condition === void 0 ? this.defaultCondition_ : e.condition, this.duration_ = e.duration === void 0 ? 100 : e.duration, this.pixelDelta_ = e.pixelDelta === void 0 ? 128 : e.pixelDelta;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == s.KEYDOWN) {
			let n = e.originalEvent, r = n.key;
			if (this.condition_(e) && (r == of.DOWN || r == of.LEFT || r == of.RIGHT || r == of.UP)) {
				let i = e.map.getView(), a = i.getResolution() * this.pixelDelta_, o = 0, s = 0;
				r == of.DOWN ? s = -a : r == of.LEFT ? o = -a : r == of.RIGHT ? o = a : s = a;
				let c = [o, s];
				Ht(c, i.getRotation()), Fd(i, c, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}
}, cf = class extends Pd {
	constructor(e) {
		super(), e ||= {}, this.condition_ = e.condition ? e.condition : function(e) {
			return !Gd(e) && qd(e);
		}, this.delta_ = e.delta ? e.delta : 1, this.duration_ = e.duration === void 0 ? 100 : e.duration;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == s.KEYDOWN || e.type == s.KEYPRESS) {
			let n = e.originalEvent, r = n.key;
			if (this.condition_(e) && (r === "+" || r === "-")) {
				let i = e.map, a = r === "+" ? this.delta_ : -this.delta_;
				Id(i.getView(), a, void 0, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}
}, lf = 40, uf = 300, df = 3, ff = class extends Pd {
	constructor(e) {
		e ||= {}, super(e), this.totalDelta_ = 0, this.lastDelta_ = 0, this.maxDelta_ = e.maxDelta === void 0 ? 1 : e.maxDelta, this.duration_ = e.duration === void 0 ? 250 : e.duration, this.timeout_ = e.timeout === void 0 ? 80 : e.timeout, this.useAnchor_ = e.useAnchor === void 0 || e.useAnchor, this.constrainResolution_ = e.constrainResolution !== void 0 && e.constrainResolution;
		let t = e.condition ? e.condition : Hd;
		this.condition_ = e.onFocusOnly ? Rd(Vd, t) : t, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_, this.mode_ = void 0, this.trackpadEventGap_ = 400, this.trackpadTimeoutId_, this.deltaPerZoom_ = 300, this.ctrlKeyPressed_ = !1, this.ctrlKeyListenerKeys_ = [];
	}
	setMap(e) {
		if (this.ctrlKeyListenerKeys_.forEach(o), this.ctrlKeyListenerKeys_.length = 0, this.ctrlKeyPressed_ = !1, super.setMap(e), e) {
			let t = e.getOwnerDocument();
			this.ctrlKeyListenerKeys_.push(i(t, "keydown", (e) => {
				e.key === "Control" && (this.ctrlKeyPressed_ = !0);
			}), i(t, "keyup", (e) => {
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
		if (!this.condition_(e) || e.type !== s.WHEEL) return !0;
		let t = e.map, n = e.originalEvent;
		n.preventDefault();
		let r = n.ctrlKey && !this.ctrlKeyPressed_;
		n.ctrlKey || (this.ctrlKeyPressed_ = !1), this.useAnchor_ && (this.lastAnchor_ = e.pixel);
		let i = n.deltaY;
		switch (n.deltaMode) {
			case WheelEvent.DOM_DELTA_LINE:
				i *= lf;
				break;
			case WheelEvent.DOM_DELTA_PAGE: i *= uf;
		}
		if (i === 0) return !1;
		this.lastDelta_ = i;
		let a = Date.now();
		this.startTime_ === void 0 && (this.startTime_ = a), (!this.mode_ || a - this.startTime_ > this.trackpadEventGap_) && (this.mode_ = Math.abs(i) < 4 ? "trackpad" : "wheel");
		let o = t.getView();
		if (this.mode_ === "trackpad") return this.trackpadTimeoutId_ ? clearTimeout(this.trackpadTimeoutId_) : (o.getAnimating() && o.cancelAnimations(), o.beginInteraction()), this.trackpadTimeoutId_ = setTimeout(this.endInteraction_.bind(this), this.timeout_), r && (i *= df), o.adjustZoom(-i / this.deltaPerZoom_, this.lastAnchor_ ? t.getCoordinateFromPixel(this.lastAnchor_) : null), this.startTime_ = a, !1;
		this.totalDelta_ += i;
		let c = Math.max(this.timeout_ - (a - this.startTime_), 0);
		return clearTimeout(this.timeoutId_), this.timeoutId_ = setTimeout(this.handleWheelZoom_.bind(this, t), c), !1;
	}
	handleWheelZoom_(e) {
		let t = e.getView();
		t.getAnimating() && t.cancelAnimations();
		let n = -Dt(this.totalDelta_, -this.maxDelta_ * this.deltaPerZoom_, this.maxDelta_ * this.deltaPerZoom_) / this.deltaPerZoom_;
		(t.getConstrainResolution() || this.constrainResolution_) && (n = n ? n > 0 ? 1 : -1 : 0), Id(t, n, this.lastAnchor_ ? e.getCoordinateFromPixel(this.lastAnchor_) : null, this.duration_), this.mode_ = void 0, this.totalDelta_ = 0, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_ = void 0;
	}
	setMouseAnchor(e) {
		this.useAnchor_ = e, e || (this.lastAnchor_ = null);
	}
}, pf = class extends Xd {
	constructor(e) {
		e ||= {};
		let t = e;
		t.stopDown ||= v, super(t), this.anchor_ = null, this.lastAngle_ = void 0, this.rotating_ = !1, this.rotationDelta_ = 0, this.threshold_ = e.threshold === void 0 ? .3 : e.threshold, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleDragEvent(e) {
		let t = 0, n = this.targetPointers[0], r = this.targetPointers[1], i = Math.atan2(r.clientY - n.clientY, r.clientX - n.clientX);
		if (this.lastAngle_ !== void 0) {
			let e = i - this.lastAngle_;
			this.rotationDelta_ += e, !this.rotating_ && Math.abs(this.rotationDelta_) > this.threshold_ && (this.rotating_ = !0), t = e;
		}
		this.lastAngle_ = i;
		let a = e.map, o = a.getView();
		o.getConstraints().rotation !== rs && (this.anchor_ = a.getCoordinateFromPixelInternal(a.getEventPixel(Zd(this.targetPointers))), this.rotating_ && (a.render(), o.adjustRotationInternal(t, this.anchor_)));
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
}, mf = class extends Xd {
	constructor(e) {
		e ||= {};
		let t = e;
		t.stopDown ||= v, super(t), this.anchor_ = null, this.duration_ = e.duration === void 0 ? 400 : e.duration, this.lastDistance_ = void 0, this.lastScaleDelta_ = 1;
	}
	handleDragEvent(e) {
		let t = 1, n = this.targetPointers[0], r = this.targetPointers[1], i = n.clientX - r.clientX, a = n.clientY - r.clientY, o = Math.sqrt(i * i + a * a);
		this.lastDistance_ !== void 0 && (t = this.lastDistance_ / o), this.lastDistance_ = o;
		let s = e.map, c = s.getView();
		t != 1 && (this.lastScaleDelta_ = t), this.anchor_ = s.getCoordinateFromPixelInternal(s.getEventPixel(Zd(this.targetPointers))), s.render(), c.adjustResolutionInternal(t, this.anchor_);
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
function hf(e) {
	e ||= {};
	let t = new N(), n = new Md(-.005, .05, 100);
	return (e.altShiftDragRotate === void 0 || e.altShiftDragRotate) && t.push(new $d()), (e.doubleClickZoom === void 0 || e.doubleClickZoom) && t.push(new Ld({
		delta: e.zoomDelta,
		duration: e.zoomDuration
	})), (e.dragPan === void 0 || e.dragPan) && t.push(new Qd({
		onFocusOnly: e.onFocusOnly,
		kinetic: n
	})), (e.pinchRotate === void 0 || e.pinchRotate) && t.push(new pf()), (e.pinchZoom === void 0 || e.pinchZoom) && t.push(new mf({ duration: e.zoomDuration })), (e.keyboard === void 0 || e.keyboard) && (t.push(new sf()), t.push(new cf({
		delta: e.zoomDelta,
		duration: e.zoomDuration
	}))), (e.mouseWheelZoom === void 0 || e.mouseWheelZoom) && t.push(new ff({
		onFocusOnly: e.onFocusOnly,
		duration: e.zoomDuration
	})), (e.shiftDragZoom === void 0 || e.shiftDragZoom) && t.push(new af({ duration: e.zoomDuration })), t;
}
//#endregion
//#region node_modules/ol/layer/Group.js
var gf = {
	ADDLAYER: "addlayer",
	REMOVELAYER: "removelayer"
}, _f = class extends S {
	constructor(e, t) {
		super(e), this.layer = t;
	}
}, vf = { LAYERS: "layers" }, yf = class r extends hs {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.layers;
		let n = e.layers;
		super(t), this.on, this.once, this.un, this.layersListenerKeys_ = [], this.listenerKeys_ = {}, this.addChangeListener(vf.LAYERS, this.handleLayersChanged_), n ? Array.isArray(n) ? n = new N(n.slice(), { unique: !0 }) : R(typeof n.getArray == "function", "Expected `layers` to be an array or a `Collection`") : n = new N(void 0, { unique: !0 }), this.setLayers(n);
	}
	handleLayerChange_() {
		this.changed();
	}
	handleLayersChanged_() {
		this.layersListenerKeys_.forEach(o), this.layersListenerKeys_.length = 0;
		let t = this.getLayers();
		this.layersListenerKeys_.push(i(t, e.ADD, this.handleLayersAdd_, this), i(t, e.REMOVE, this.handleLayersRemove_, this));
		for (let e in this.listenerKeys_) this.listenerKeys_[e].forEach(o);
		n(this.listenerKeys_);
		let r = t.getArray();
		for (let e = 0, t = r.length; e < t; e++) {
			let t = r[e];
			this.registerLayerListeners_(t), this.dispatchEvent(new _f(gf.ADDLAYER, t));
		}
		this.changed();
	}
	registerLayerListeners_(e) {
		let n = [i(e, t.PROPERTYCHANGE, this.handleLayerChange_, this), i(e, s.CHANGE, this.handleLayerChange_, this)];
		e instanceof r && n.push(i(e, gf.ADDLAYER, this.handleLayerGroupAdd_, this), i(e, gf.REMOVELAYER, this.handleLayerGroupRemove_, this)), this.listenerKeys_[O(e)] = n;
	}
	handleLayerGroupAdd_(e) {
		this.dispatchEvent(new _f(gf.ADDLAYER, e.layer));
	}
	handleLayerGroupRemove_(e) {
		this.dispatchEvent(new _f(gf.REMOVELAYER, e.layer));
	}
	handleLayersAdd_(e) {
		let t = e.element;
		this.registerLayerListeners_(t), this.dispatchEvent(new _f(gf.ADDLAYER, t)), this.changed();
	}
	handleLayersRemove_(e) {
		let t = e.element, n = O(t);
		this.listenerKeys_[n].forEach(o), delete this.listenerKeys_[n], this.dispatchEvent(new _f(gf.REMOVELAYER, t)), this.changed();
	}
	getLayers() {
		return this.get(vf.LAYERS);
	}
	setLayers(e) {
		let t = this.getLayers();
		if (t) {
			let e = t.getArray();
			for (let t = 0, n = e.length; t < n; ++t) this.dispatchEvent(new _f(gf.REMOVELAYER, e[t]));
		}
		this.set(vf.LAYERS, e);
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
			n.opacity *= r.opacity, n.visible = n.visible && r.visible, n.maxResolution = Math.min(n.maxResolution, r.maxResolution), n.minResolution = Math.max(n.minResolution, r.minResolution), n.minZoom = Math.max(n.minZoom, r.minZoom), n.maxZoom = Math.min(n.maxZoom, r.maxZoom), r.extent !== void 0 && (n.extent = n.extent === void 0 ? r.extent : dt(n.extent, r.extent)), n.zIndex === void 0 && (n.zIndex = i);
		}
		return t;
	}
	getSourceState() {
		return "ready";
	}
}, bf = class extends c {
	constructor(e) {
		super(), this.map_ = e;
	}
	dispatchRenderEvent(e, t) {
		E();
	}
	calculateMatrices2D(e) {
		let t = e.viewState, n = e.coordinateToPixelTransform, r = e.pixelToCoordinateTransform;
		Mr(n, e.size[0] / 2, e.size[1] / 2, 1 / t.resolution, -1 / t.resolution, -t.rotation, -t.center[0], -t.center[1]), Nr(r, n);
	}
	forEachFeatureAtCoordinate(e, t, n, r, i, a, o, s) {
		let c, l = t.viewState;
		function u(e, t, n, r) {
			return i.call(a, t, e ? n : null, r);
		}
		let d = l.projection, f = Wt(e.slice(), d), p = [[0, 0]];
		if (d.canWrapX() && r) {
			let e = L(d.getExtent());
			p.push([-e, 0], [e, 0]);
		}
		let m = t.layerStatesArray, h = m.length, g = [], _ = [];
		for (let r = 0; r < p.length; r++) for (let i = h - 1; i >= 0; --i) {
			let a = m[i], d = a.layer;
			if (d.hasRenderer() && _s(a, l) && o.call(s, d)) {
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
		return this.forEachFeatureAtCoordinate(e, t, n, r, _, this, i, a) !== void 0;
	}
	getMap() {
		return this.map_;
	}
	renderFrame(e) {
		E();
	}
	scheduleExpireIconCache(e) {
		Cs.canExpireCache() && e.postRenderFunctions.push(xf);
	}
};
function xf(e, t) {
	Cs.expire();
}
//#endregion
//#region node_modules/ol/renderer/Composite.js
var Sf = class extends bf {
	constructor(e) {
		super(e), this.fontChangeListenerKey_ = i(zs, t.PROPERTYCHANGE, e.redrawText, e), this.element_ = pe ? we() : document.createElement("div");
		let n = this.element_.style;
		n.position = "absolute", n.width = "100%", n.height = "100%", n.zIndex = "0", this.element_.className = F + " ol-layers";
		let r = e.getViewport();
		r && r.insertBefore(this.element_, r.firstChild || null), this.children_ = [], this.renderedVisible_ = !0;
	}
	dispatchRenderEvent(e, t) {
		let n = this.getMap();
		if (n.hasListener(e)) {
			let r = new Fo(e, void 0, t);
			n.dispatchEvent(r);
		}
	}
	disposeInternal() {
		o(this.fontChangeListenerKey_), this.element_.remove(), super.disposeInternal();
	}
	renderFrame(e) {
		if (!e) {
			this.renderedVisible_ &&= (this.element_.style.display = "none", !1);
			return;
		}
		this.calculateMatrices2D(e), this.dispatchRenderEvent(Io.PRECOMPOSE, e);
		let t = e.layerStatesArray.sort((e, t) => e.zIndex - t.zIndex);
		t.some((e) => e.layer instanceof yd && e.layer.getDeclutter()) && (e.declutter = {});
		let n = e.viewState;
		this.children_.length = 0;
		let r = this.getMap().getTargetElement(), i;
		Te(r) && (i = r.getContext("2d"), i.setTransform(1, 0, 0, 1, 0, 0), i.clearRect(0, 0, r.width, r.height));
		let a = [], o = i ? r : null;
		for (let r = 0, i = t.length; r < i; ++r) {
			let i = t[r];
			e.layerIndex = r;
			let s = i.layer, c = s.getSourceState();
			if (!_s(i, n) || c != "ready" && c != "undefined") {
				s.unrender();
				continue;
			}
			let l = s.render(e, o);
			l && (l !== o && (this.children_.push(l), o = l), a.push(i));
		}
		this.declutter(e, a), Ce(this.element_, this.children_);
		for (let e of i ? this.children_ : []) {
			let t = e.firstElementChild || e, n = e.style.backgroundColor;
			if (n && (!Te(t) || t.width > 0) && (i.fillStyle = n, i.fillRect(0, 0, i.canvas.width, i.canvas.height)), !Te(t) || t.width === 0) continue;
			i.save();
			let r = e.style.opacity || t.style.opacity;
			i.globalAlpha = r === "" ? 1 : Number(r);
			let a = t.style.transform;
			if (a) i.transform(...Lr(a));
			else {
				let e = parseFloat(t.style.width) / t.width, n = parseFloat(t.style.height) / t.height;
				i.transform(e, 0, 0, n, 0, 0);
			}
			i.drawImage(t, 0, 0), i.restore();
		}
		this.dispatchRenderEvent(Io.POSTCOMPOSE, e), this.renderedVisible_ ||= (this.element_.style.display = "", !0), this.scheduleExpireIconCache(e);
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
function Cf(e) {
	if (e instanceof gs) {
		e.setMapInternal(null);
		return;
	}
	e instanceof yf && e.getLayers().forEach(Cf);
}
function wf(e, t) {
	if (e instanceof gs) {
		e.setMapInternal(t);
		return;
	}
	if (e instanceof yf) {
		let n = e.getLayers().getArray();
		for (let e = 0, r = n.length; e < r; ++e) wf(n[e], t);
	}
}
var Tf = class extends A {
	constructor(t) {
		super(), t ||= {}, this.on, this.once, this.un;
		let n = Ef(t);
		this.renderComplete_ = !1, this.loaded_ = !0, this.boundHandleBrowserEvent_ = this.handleBrowserEvent.bind(this), this.maxTilesLoading_ = t.maxTilesLoading === void 0 ? 16 : t.maxTilesLoading, this.pixelRatio_ = t.pixelRatio === void 0 ? fe : t.pixelRatio, this.postRenderTimeoutHandle_, this.animationDelayKey_, this.animationDelay_ = this.animationDelay_.bind(this), this.coordinateToPixelTransform_ = Tr(), this.pixelToCoordinateTransform_ = Tr(), this.frameIndex_ = 0, this.frameState_ = null, this.previousExtent_ = null, this.viewPropertyListenerKey_ = null, this.viewChangeListenerKey_ = null, this.layerGroupPropertyListenerKeys_ = null, pe || (this.viewport_ = document.createElement("div"), this.viewport_.className = "ol-viewport" + ("ontouchstart" in window ? " ol-touch" : ""), this.viewport_.style.position = "relative", this.viewport_.style.overflow = "hidden", this.viewport_.style.width = "100%", this.viewport_.style.height = "100%", this.overlayContainer_ = document.createElement("div"), this.overlayContainer_.style.position = "absolute", this.overlayContainer_.style.zIndex = "0", this.overlayContainer_.style.width = "100%", this.overlayContainer_.style.height = "100%", this.overlayContainer_.style.pointerEvents = "none", this.overlayContainer_.className = "ol-overlaycontainer", this.viewport_.appendChild(this.overlayContainer_), this.overlayContainerStopEvent_ = document.createElement("div"), this.overlayContainerStopEvent_.style.position = "absolute", this.overlayContainerStopEvent_.style.zIndex = "0", this.overlayContainerStopEvent_.style.width = "100%", this.overlayContainerStopEvent_.style.height = "100%", this.overlayContainerStopEvent_.style.pointerEvents = "none", this.overlayContainerStopEvent_.className = "ol-overlaycontainer-stopevent", this.viewport_.appendChild(this.overlayContainerStopEvent_)), this.mapBrowserEventHandler_ = null, this.moveTolerance_ = t.moveTolerance, this.keyboardEventTarget_ = n.keyboardEventTarget, this.targetChangeHandlerKeys_ = null, this.targetElement_ = null, pe || (this.resizeObserver_ = new ResizeObserver(() => this.updateSize())), this.controls = n.controls || (pe ? new N() : Fe()), this.interactions = n.interactions || (pe ? new N() : hf({ onFocusOnly: !0 })), this.overlays_ = n.overlays, this.overlayIdIndex_ = {}, this.renderer_ = null, this.postRenderFunctions_ = [], this.tileQueue_ = new Ad(this.getTilePriority.bind(this), this.handleTileChange_.bind(this)), this.addChangeListener(Dd.LAYERGROUP, this.handleLayerGroupChanged_), this.addChangeListener(Dd.VIEW, this.handleViewChanged_), this.addChangeListener(Dd.SIZE, this.handleSizeChanged_), this.addChangeListener(Dd.TARGET, this.handleTargetChanged_), this.setProperties(n.values);
		let r = this;
		t.view && !(t.view instanceof cs) && t.view.then(function(e) {
			r.setView(new cs(e));
		}), this.controls.addEventListener(e.ADD, (e) => {
			e.element.setMap(this);
		}), this.controls.addEventListener(e.REMOVE, (e) => {
			e.element.setMap(null);
		}), this.interactions.addEventListener(e.ADD, (e) => {
			e.element.setMap(this);
		}), this.interactions.addEventListener(e.REMOVE, (e) => {
			e.element.setMap(null);
		}), this.overlays_.addEventListener(e.ADD, (e) => {
			this.addOverlayInternal_(e.element);
		}), this.overlays_.addEventListener(e.REMOVE, (e) => {
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
		wf(e.layer, this);
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
		let i = n.hitTolerance === void 0 ? 0 : n.hitTolerance, a = n.layerFilter === void 0 ? _ : n.layerFilter, o = n.checkWrapped !== !1;
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
				n instanceof yf ? t(n.getLayers()) : e.push(n);
			});
		}
		return t(this.getLayers()), e;
	}
	hasFeatureAtPixel(e, t) {
		if (!this.frameState_ || !this.renderer_) return !1;
		let n = this.getCoordinateFromPixelInternal(e);
		t = t === void 0 ? {} : t;
		let r = t.layerFilter === void 0 ? _ : t.layerFilter, i = t.hitTolerance === void 0 ? 0 : t.hitTolerance, a = t.checkWrapped !== !1;
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
		return this.get(Dd.TARGET);
	}
	getTargetElement() {
		return this.targetElement_;
	}
	getCoordinateFromPixel(e) {
		return _r(this.getCoordinateFromPixelInternal(e), this.getView().getProjection());
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
		return this.get(Dd.LAYERGROUP);
	}
	setLayers(e) {
		let t = this.getLayerGroup();
		if (e instanceof N) {
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
		let t = vr(e, this.getView().getProjection());
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
		return this.get(Dd.SIZE);
	}
	getView() {
		return this.get(Dd.VIEW);
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
		return jd(this.frameState_, e, t, n, r);
	}
	handleBrowserEvent(e, t) {
		t ||= e.type;
		let n = new Cd(t, this, e);
		this.handleMapBrowserEvent(n);
	}
	handleMapBrowserEvent(e) {
		if (!this.frameState_) return;
		let t = e.originalEvent, n = t.type;
		if (n === Td.POINTERDOWN || n === s.WHEEL || n === s.KEYDOWN) {
			let e = this.getOwnerDocument(), n = this.viewport_.getRootNode ? this.viewport_.getRootNode() : e, r = t.target, i = n instanceof ShadowRoot ? n.host === r ? n.host.ownerDocument : n : n === e ? e.documentElement : n;
			if (this.overlayContainerStopEvent_.contains(r) || !i.contains(r)) return;
		}
		if (e.frameState = this.frameState_, this.dispatchEvent(e) !== !1) {
			let t = this.getInteractions().getArray().slice();
			for (let n = t.length - 1; n >= 0; n--) {
				let r = t[n];
				if (r.getMap() === this && r.getActive() && this.getTargetElement() && (!r.handleEvent(e) || e.propagationStopped)) break;
			}
		}
	}
	handlePostRender() {
		let e = this.frameState_, t = this.tileQueue_;
		if (!t.isEmpty()) {
			let n = this.maxTilesLoading_, r = n, i = e ? e.viewHints : void 0, a = i ? i[Jo.ANIMATING] || i[Jo.INTERACTING] : !1;
			if (a) {
				let t = Date.now() - e.time > 8;
				n = t ? 0 : 8, r = t ? 0 : 2;
			}
			t.getTilesLoading() < n && (a && t.reprioritize(), t.loadMoreTiles(n, r));
		}
		e && this.renderer_ && !e.animate && (this.renderComplete_ ? (this.hasListener(Io.RENDERCOMPLETE) && this.renderer_.dispatchRenderEvent(Io.RENDERCOMPLETE, e), this.loaded_ === !1 && (this.loaded_ = !0, this.dispatchEvent(new Sd(Ee.LOADEND, this, e)))) : this.loaded_ === !0 && (this.loaded_ = !1, this.dispatchEvent(new Sd(Ee.LOADSTART, this, e))));
		let n = this.postRenderFunctions_;
		if (e) for (let t = 0, r = n.length; t < r; ++t) n[t](this, e);
		n.length = 0;
	}
	handleSizeChanged_() {
		this.getView() && !this.getView().getAnimating() && this.getView().resolveConstraints(0), this.render();
	}
	handleTargetChanged_() {
		if (this.mapBrowserEventHandler_) {
			for (let e = 0, t = this.targetChangeHandlerKeys_.length; e < t; ++e) o(this.targetChangeHandlerKeys_[e]);
			this.targetChangeHandlerKeys_ = null, this.viewport_.removeEventListener(s.CONTEXTMENU, this.boundHandleBrowserEvent_), this.viewport_.removeEventListener(s.WHEEL, this.boundHandleBrowserEvent_), this.mapBrowserEventHandler_.dispose(), this.mapBrowserEventHandler_ = null, this.viewport_.remove();
		}
		if (this.targetElement_ && !Te(this.targetElement_)) {
			this.resizeObserver_?.unobserve(this.targetElement_);
			let e = this.targetElement_.getRootNode();
			e instanceof ShadowRoot && this.resizeObserver_.unobserve(e.host), this.setSize(void 0);
		}
		let e = this.getTarget(), t = typeof e == "string" ? document.getElementById(e) : e;
		if (this.targetElement_ = t, !t) this.renderer_ &&= (clearTimeout(this.postRenderTimeoutHandle_), this.postRenderTimeoutHandle_ = void 0, this.postRenderFunctions_.length = 0, this.renderer_.dispose(), null), this.animationDelayKey_ &&= (cancelAnimationFrame(this.animationDelayKey_), void 0);
		else {
			if (Te(t) || t.appendChild(this.viewport_), this.renderer_ ||= new Sf(this), !Te(t)) {
				this.mapBrowserEventHandler_ = new Ed(this, this.moveTolerance_);
				for (let e in wd) this.mapBrowserEventHandler_.addEventListener(wd[e], this.handleMapBrowserEvent.bind(this));
				this.viewport_.addEventListener(s.CONTEXTMENU, this.boundHandleBrowserEvent_, !1), this.viewport_.addEventListener(s.WHEEL, this.boundHandleBrowserEvent_, he ? { passive: !1 } : !1);
				let e;
				if (this.keyboardEventTarget_) e = this.keyboardEventTarget_;
				else {
					let n = t.getRootNode();
					e = n instanceof ShadowRoot ? n.host : t;
				}
				if (this.targetChangeHandlerKeys_ = [i(e, s.KEYDOWN, this.handleBrowserEvent, this), i(e, s.KEYPRESS, this.handleBrowserEvent, this)], !Te(t)) {
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
		this.viewPropertyListenerKey_ &&= (o(this.viewPropertyListenerKey_), null), this.viewChangeListenerKey_ &&= (o(this.viewChangeListenerKey_), null);
		let e = this.getView();
		e && (this.updateViewportSize_(this.getSize()), this.viewPropertyListenerKey_ = i(e, t.PROPERTYCHANGE, this.handleViewPropertyChanged_, this), this.viewChangeListenerKey_ = i(e, s.CHANGE, this.handleViewPropertyChanged_, this), e.resolveConstraints(0)), this.render();
	}
	handleLayerGroupChanged_() {
		this.layerGroupPropertyListenerKeys_ &&= (this.layerGroupPropertyListenerKeys_.forEach(o), null);
		let e = this.getLayerGroup();
		e && (this.handleLayerAdd_(new _f("addlayer", e)), this.layerGroupPropertyListenerKeys_ = [
			i(e, t.PROPERTYCHANGE, this.render, this),
			i(e, s.CHANGE, this.render, this),
			i(e, "addlayer", this.handleLayerAdd_, this),
			i(e, "removelayer", this.handleLayerRemove_, this)
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
		Cf(e.layer);
	}
	removeOverlay(e) {
		return this.getOverlays().remove(e);
	}
	renderFrame_(e) {
		let t = this.getSize(), n = this.getView(), r = this.frameState_, i = null;
		if (t !== void 0 && eo(t) && n && n.isDef()) {
			let r = n.getHints(this.frameState_ ? this.frameState_.viewHints : void 0), a = n.getState();
			if (i = {
				animate: !1,
				coordinateToPixelTransform: this.coordinateToPixelTransform_,
				declutter: null,
				extent: ct(a.center, a.resolution, a.rotation, t),
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
				mapId: O(this),
				renderTargets: {}
			}, a.nextCenter && a.nextResolution) {
				let e = isNaN(a.nextRotation) ? a.rotation : a.nextRotation;
				i.nextExtent = ct(a.nextCenter, a.nextResolution, e, t);
			}
		}
		this.frameState_ = i, this.renderer_.renderFrame(i), i && (i.animate && this.render(), Array.prototype.push.apply(this.postRenderFunctions_, i.postRenderFunctions), r && (!this.previousExtent_ || !gt(this.previousExtent_) && !Ze(i.extent, this.previousExtent_)) && (this.dispatchEvent(new Sd(Ee.MOVESTART, this, r)), this.previousExtent_ = Je(this.previousExtent_)), this.previousExtent_ && !i.viewHints[Jo.ANIMATING] && !i.viewHints[Jo.INTERACTING] && !Ze(i.extent, this.previousExtent_) && (this.dispatchEvent(new Sd(Ee.MOVEEND, this, i)), Be(i.extent, this.previousExtent_))), this.dispatchEvent(new Sd(Ee.POSTRENDER, this, i)), this.renderComplete_ = (this.hasListener(Ee.LOADSTART) || this.hasListener(Ee.LOADEND) || this.hasListener(Io.RENDERCOMPLETE)) && !this.tileQueue_.getTilesLoading() && !this.tileQueue_.getCount() && !this.getLoadingOrNotReady(), this.postRenderTimeoutHandle_ ||= setTimeout(() => {
			this.postRenderTimeoutHandle_ = void 0, this.handlePostRender();
		}, 0);
	}
	setLayerGroup(e) {
		let t = this.getLayerGroup();
		t && this.handleLayerRemove_(new _f("removelayer", t)), this.set(Dd.LAYERGROUP, e);
	}
	setSize(e) {
		this.set(Dd.SIZE, e);
	}
	setTarget(e) {
		this.set(Dd.TARGET, e);
	}
	setView(e) {
		if (!e || e instanceof cs) {
			this.set(Dd.VIEW, e);
			return;
		}
		this.set(Dd.VIEW, new cs());
		let t = this;
		e.then(function(e) {
			t.setView(new cs(e));
		});
	}
	updateSize() {
		let e = this.getTargetElement(), t;
		if (e) {
			let n, r;
			if (Te(e)) {
				let t = e.getContext("2d").getTransform();
				n = e.width / t.a, r = e.height / t.d;
			} else {
				let t = getComputedStyle(e);
				n = e.offsetWidth - parseFloat(t.borderLeftWidth) - parseFloat(t.paddingLeft) - parseFloat(t.paddingRight) - parseFloat(t.borderRightWidth), r = e.offsetHeight - parseFloat(t.borderTopWidth) - parseFloat(t.paddingTop) - parseFloat(t.paddingBottom) - parseFloat(t.borderBottomWidth);
			}
			!isNaN(n) && !isNaN(r) && (t = [Math.max(0, n), Math.max(0, r)], !eo(t) && (e.offsetWidth || e.offsetHeight || e.getClientRects().length) && Et("No map visible because the map container's width or height are 0."));
		}
		let n = this.getSize();
		t && (!n || !h(t, n)) && (this.updateViewportSize_(t), this.setSize(t));
	}
	updateViewportSize_(e) {
		let t = this.getView();
		t && t.setViewportSize(e);
	}
};
function Ef(e) {
	let t = null;
	e.keyboardEventTarget !== void 0 && (t = typeof e.keyboardEventTarget == "string" ? document.getElementById(e.keyboardEventTarget) : e.keyboardEventTarget);
	let n = {}, r = e.layers && typeof e.layers.getLayers == "function" ? e.layers : new yf({ layers: e.layers });
	n[Dd.LAYERGROUP] = r, n[Dd.TARGET] = e.target, n[Dd.VIEW] = e.view instanceof cs ? e.view : new cs();
	let i;
	e.controls !== void 0 && (Array.isArray(e.controls) ? i = new N(e.controls.slice()) : (R(typeof e.controls.getArray == "function", "Expected `controls` to be an array or an `ol/Collection.js`"), i = e.controls));
	let a;
	e.interactions !== void 0 && (Array.isArray(e.interactions) ? a = new N(e.interactions.slice()) : (R(typeof e.interactions.getArray == "function", "Expected `interactions` to be an array or an `ol/Collection.js`"), a = e.interactions));
	let o;
	return e.overlays === void 0 ? o = new N() : Array.isArray(e.overlays) ? o = new N(e.overlays.slice()) : (R(typeof e.overlays.getArray == "function", "Expected `overlays` to be an array or an `ol/Collection.js`"), o = e.overlays), {
		controls: i,
		interactions: a,
		keyboardEventTarget: t,
		overlays: o,
		values: n
	};
}
//#endregion
//#region node_modules/ol/featureloader.js
var Df = !1;
function Of(e, t, n, r, i, a, o) {
	let s = new XMLHttpRequest();
	s.open("GET", typeof e == "function" ? e(n, r, i) : e, !0), t.getType() == "arraybuffer" && (s.responseType = "arraybuffer"), s.withCredentials = Df, s.onload = function(e) {
		if (!s.status || s.status >= 200 && s.status < 300) {
			let e = t.getType();
			try {
				let r;
				e == "text" || e == "json" ? r = s.responseText : e == "xml" ? r = s.responseXML || s.responseText : e == "arraybuffer" && (r = s.response), r ? a(t.readFeatures(r, {
					extent: n,
					featureProjection: i
				}), t.readProjection(r)) : o();
			} catch {
				o();
			}
		} else o();
	}, s.onerror = o, s.send();
}
function kf(e, t) {
	return function(n, r, i, a, o) {
		Of(e, t, n, r, i, (e, t) => {
			this.addFeatures(e), a !== void 0 && a(e);
		}, () => {
			this.changed(), o !== void 0 && o();
		});
	};
}
//#endregion
//#region node_modules/ol/loadingstrategy.js
function Af(e, t) {
	return [[
		-Infinity,
		-Infinity,
		Infinity,
		Infinity
	]];
}
//#endregion
//#region node_modules/ol/structs/RBush.js
var jf = class {
	constructor(e) {
		this.rbush_ = new hl(e), this.items_ = {};
	}
	insert(e, t) {
		let n = {
			minX: e[0],
			minY: e[1],
			maxX: e[2],
			maxY: e[3],
			value: t
		};
		this.rbush_.insert(n), this.items_[O(t)] = n;
	}
	load(e, t) {
		let n = Array(t.length);
		for (let r = 0, i = t.length; r < i; r++) {
			let i = e[r], a = t[r], o = {
				minX: i[0],
				minY: i[1],
				maxX: i[2],
				maxY: i[3],
				value: a
			};
			n[r] = o, this.items_[O(a)] = o;
		}
		this.rbush_.load(n);
	}
	remove(e) {
		let t = O(e), n = this.items_[t];
		return delete this.items_[t], this.rbush_.remove(n) !== null;
	}
	update(e, t) {
		let n = this.items_[O(t)];
		Ze([
			n.minX,
			n.minY,
			n.maxX,
			n.maxY
		], e) || (this.remove(t), this.insert(e, t));
	}
	getAll() {
		return this.rbush_.all().map(function(e) {
			return e.value;
		});
	}
	getInExtent(e) {
		let t = {
			minX: e[0],
			minY: e[1],
			maxX: e[2],
			maxY: e[3]
		};
		return this.rbush_.search(t).map(function(e) {
			return e.value;
		});
	}
	forEach(e) {
		return this.forEach_(this.getAll(), e);
	}
	forEachInExtent(e, t) {
		return this.forEach_(this.getInExtent(e), t);
	}
	forEach_(e, t) {
		let n;
		for (let r = 0, i = e.length; r < i; r++) if (n = t(e[r]), n) return n;
		return n;
	}
	isEmpty() {
		return r(this.items_);
	}
	clear() {
		this.rbush_.clear(), this.items_ = {};
	}
	getExtent(e) {
		let t = this.rbush_.toJSON();
		return qe(t.minX, t.minY, t.maxX, t.maxY, e);
	}
	concat(e) {
		this.rbush_.load(e.rbush_.all());
		for (let t in e.items_) this.items_[t] = e.items_[t];
	}
}, Mf = class extends A {
	constructor(e) {
		super(), this.projection = rr(e.projection), this.attributions_ = Nf(e.attributions), this.attributionsCollapsible_ = e.attributionsCollapsible ?? !0, this.loading = !1, this.state_ = e.state === void 0 ? "ready" : e.state, this.wrapX_ = e.wrapX !== void 0 && e.wrapX, this.interpolate_ = !!e.interpolate, this.viewResolver = null, this.viewRejector = null;
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
		this.attributions_ = Nf(e), this.changed();
	}
	setState(e) {
		this.state_ = e, this.changed();
	}
};
function Nf(e) {
	return e ? typeof e == "function" ? e : (Array.isArray(e) || (e = [e]), (t) => e) : null;
}
//#endregion
//#region node_modules/ol/source/VectorEventType.js
var Pf = {
	ADDFEATURE: "addfeature",
	CHANGEFEATURE: "changefeature",
	CLEAR: "clear",
	REMOVEFEATURE: "removefeature",
	FEATURESLOADSTART: "featuresloadstart",
	FEATURESLOADEND: "featuresloadend",
	FEATURESLOADERROR: "featuresloaderror"
}, Ff = class extends S {
	constructor(e, t, n) {
		super(e), this.feature = t, this.features = n;
	}
}, If = class extends Mf {
	constructor(e) {
		e ||= {}, super({
			attributions: e.attributions,
			interpolate: !0,
			projection: void 0,
			state: "ready",
			wrapX: e.wrapX === void 0 || e.wrapX
		}), this.on, this.once, this.un, this.loader_ = y, this.format_ = e.format || null, this.overlaps_ = e.overlaps === void 0 || e.overlaps, this.url_ = e.url, e.loader === void 0 ? this.url_ !== void 0 && (R(this.format_, "`format` must be set when `url` is set"), this.loader_ = kf(this.url_, this.format_)) : this.loader_ = e.loader, this.strategy_ = e.strategy === void 0 ? Af : e.strategy;
		let t = e.useSpatialIndex === void 0 || e.useSpatialIndex;
		this.featuresRtree_ = t ? new jf() : null, this.loadedExtentsRtree_ = new jf(), this.nullGeometryFeatures_ = {}, this.idIndex_ = {}, this.uidIndex_ = {}, this.featureChangeKeys_ = {}, this.featuresCollection_ = null;
		let n, r;
		Array.isArray(e.features) ? r = e.features : e.features && (n = e.features, r = n.getArray()), !t && n === void 0 && (n = new N(r)), r !== void 0 && this.addFeaturesInternal(r), n !== void 0 && this.bindFeaturesCollection_(n);
	}
	addFeature(e) {
		this.addFeatureInternal(e), this.changed();
	}
	addFeatureInternal(e) {
		let t = O(e);
		if (!this.addToIndex_(t, e)) {
			this.featuresCollection_ && this.featuresCollection_.remove(e);
			return;
		}
		this.setupChangeEvents_(t, e);
		let n = e.getGeometry();
		if (n) {
			let t = n.getExtent();
			this.featuresRtree_ && this.featuresRtree_.insert(t, e);
		} else this.nullGeometryFeatures_[t] = e;
		this.dispatchEvent(new Ff(Pf.ADDFEATURE, e));
	}
	setupChangeEvents_(e, n) {
		n instanceof ea || (this.featureChangeKeys_[e] = [i(n, s.CHANGE, this.handleFeatureChange_, this), i(n, t.PROPERTYCHANGE, this.handleFeatureChange_, this)]);
	}
	addToIndex_(e, t) {
		let n = !0;
		if (t.getId() !== void 0) {
			let e = String(t.getId());
			if (!(e in this.idIndex_)) this.idIndex_[e] = t;
			else if (t instanceof ea) {
				let r = this.idIndex_[e];
				r instanceof ea ? Array.isArray(r) ? r.push(t) : this.idIndex_[e] = [r, t] : n = !1;
			} else n = !1;
		}
		return n && (R(!(e in this.uidIndex_), "The passed `feature` was already added to the source"), this.uidIndex_[e] = t), n;
	}
	addFeatures(e) {
		this.addFeaturesInternal(e), this.changed();
	}
	addFeaturesInternal(e) {
		let t = [], n = [], r = [];
		for (let t = 0, r = e.length; t < r; t++) {
			let r = e[t], i = O(r);
			this.addToIndex_(i, r) && n.push(r);
		}
		for (let e = 0, i = n.length; e < i; e++) {
			let i = n[e], a = O(i);
			this.setupChangeEvents_(a, i);
			let o = i.getGeometry();
			if (o) {
				let e = o.getExtent();
				t.push(e), r.push(i);
			} else this.nullGeometryFeatures_[a] = i;
		}
		if (this.featuresRtree_ && this.featuresRtree_.load(t, r), this.hasListener(Pf.ADDFEATURE)) for (let e = 0, t = n.length; e < t; e++) this.dispatchEvent(new Ff(Pf.ADDFEATURE, n[e]));
	}
	bindFeaturesCollection_(t) {
		let n = !1;
		this.addEventListener(Pf.ADDFEATURE, function(e) {
			n ||= (n = !0, t.push(e.feature), !1);
		}), this.addEventListener(Pf.REMOVEFEATURE, function(e) {
			n ||= (n = !0, t.remove(e.feature), !1);
		}), t.addEventListener(e.ADD, (e) => {
			n ||= (n = !0, this.addFeature(e.element), !1);
		}), t.addEventListener(e.REMOVE, (e) => {
			n ||= (n = !0, this.removeFeature(e.element), !1);
		}), this.featuresCollection_ = t;
	}
	clear(e) {
		if (e) {
			for (let e in this.featureChangeKeys_) this.featureChangeKeys_[e].forEach(o);
			this.featuresCollection_ || (this.featureChangeKeys_ = {}, this.idIndex_ = {}, this.uidIndex_ = {});
		} else if (this.featuresRtree_) {
			this.featuresRtree_.forEach((e) => {
				this.removeFeatureInternal(e);
			});
			for (let e in this.nullGeometryFeatures_) this.removeFeatureInternal(this.nullGeometryFeatures_[e]);
		}
		this.featuresCollection_ && this.featuresCollection_.clear(), this.featuresRtree_ && this.featuresRtree_.clear(), this.nullGeometryFeatures_ = {};
		let t = new Ff(Pf.CLEAR);
		this.dispatchEvent(t), this.changed();
	}
	forEachFeature(e) {
		if (this.featuresRtree_) return this.featuresRtree_.forEach(e);
		this.featuresCollection_ && this.featuresCollection_.forEach(e);
	}
	forEachFeatureAtCoordinateDirect(e, t) {
		let n = [
			e[0],
			e[1],
			e[0],
			e[1]
		];
		return this.forEachFeatureInExtent(n, function(n) {
			let r = n.getGeometry();
			if (r instanceof ea || r.intersectsCoordinate(e)) return t(n);
		});
	}
	forEachFeatureInExtent(e, t) {
		if (this.featuresRtree_) return this.featuresRtree_.forEachInExtent(e, t);
		this.featuresCollection_ && this.featuresCollection_.forEach(t);
	}
	forEachFeatureIntersectingExtent(e, t) {
		return this.forEachFeatureInExtent(e, function(n) {
			let r = n.getGeometry();
			if (r instanceof ea || r.intersectsExtent(e)) {
				let e = t(n);
				if (e) return e;
			}
		});
	}
	getFeaturesCollection() {
		return this.featuresCollection_;
	}
	getFeatures() {
		let e;
		return this.featuresCollection_ ? e = this.featuresCollection_.getArray().slice(0) : this.featuresRtree_ && (e = this.featuresRtree_.getAll(), r(this.nullGeometryFeatures_) || m(e, Object.values(this.nullGeometryFeatures_))), e;
	}
	getFeaturesAtCoordinate(e) {
		let t = [];
		return this.forEachFeatureAtCoordinateDirect(e, function(e) {
			t.push(e);
		}), t;
	}
	getFeaturesInExtent(e, t) {
		if (this.featuresRtree_) {
			if (!(t && t.canWrapX() && this.getWrapX())) return this.featuresRtree_.getInExtent(e);
			let n = St(e, t);
			return [].concat(...n.map((e) => this.featuresRtree_.getInExtent(e)));
		}
		return this.featuresCollection_ ? this.featuresCollection_.getArray().slice(0) : [];
	}
	getClosestFeatureToCoordinate(e, t) {
		let n = e[0], r = e[1], i = null, a = [NaN, NaN], o = Infinity, s = [
			-Infinity,
			-Infinity,
			Infinity,
			Infinity
		];
		return t ||= _, this.featuresRtree_.forEachInExtent(s, function(e) {
			if (t(e)) {
				let t = e.getGeometry(), c = o;
				if (o = t instanceof ea ? 0 : t.closestPointXY(n, r, a, o), o < c) {
					i = e;
					let t = Math.sqrt(o);
					s[0] = n - t, s[1] = r - t, s[2] = n + t, s[3] = r + t;
				}
			}
		}), i;
	}
	getExtent(e) {
		return this.featuresRtree_?.getExtent(e) ?? null;
	}
	getFeatureById(e) {
		let t = this.idIndex_[e.toString()];
		return t === void 0 ? null : t;
	}
	getFeatureByUid(e) {
		let t = this.uidIndex_[e];
		return t === void 0 ? null : t;
	}
	getFormat() {
		return this.format_;
	}
	getOverlaps() {
		return this.overlaps_;
	}
	getUrl() {
		return this.url_;
	}
	handleFeatureChange_(e) {
		let t = e.target, n = O(t), r = t.getGeometry();
		if (!r) n in this.nullGeometryFeatures_ || (this.featuresRtree_ && this.featuresRtree_.remove(t), this.nullGeometryFeatures_[n] = t);
		else {
			let e = r.getExtent();
			n in this.nullGeometryFeatures_ ? (delete this.nullGeometryFeatures_[n], this.featuresRtree_ && this.featuresRtree_.insert(e, t)) : this.featuresRtree_ && this.featuresRtree_.update(e, t);
		}
		let i = t.getId();
		if (i !== void 0) {
			let e = i.toString();
			this.idIndex_[e] !== t && (this.removeFromIdIndex_(t), this.idIndex_[e] = t);
		} else this.removeFromIdIndex_(t), this.uidIndex_[n] = t;
		this.changed(), this.dispatchEvent(new Ff(Pf.CHANGEFEATURE, t));
	}
	hasFeature(e) {
		let t = e.getId();
		if (t !== void 0) {
			let n = this.idIndex_[String(t)];
			return Array.isArray(n) ? n.includes(e) : n === e;
		}
		return O(e) in this.uidIndex_;
	}
	isEmpty() {
		return this.featuresRtree_ ? this.featuresRtree_.isEmpty() && r(this.nullGeometryFeatures_) : !this.featuresCollection_ || this.featuresCollection_.getLength() === 0;
	}
	loadFeatures(e, t, n) {
		let r = this.loadedExtentsRtree_, i = this.strategy_(e, t, n);
		for (let e = 0, a = i.length; e < a; ++e) {
			let a = i[e];
			if (!r.forEachInExtent(a, function(e) {
				return Ue(e.extent, a);
			})) {
				this.loading = Number(this.loading) + 1, this.dispatchEvent(new Ff(Pf.FEATURESLOADSTART));
				let e = (e) => {
					this.loading = Number(this.loading) - 1, this.dispatchEvent(new Ff(Pf.FEATURESLOADEND, void 0, e));
				}, i = () => {
					this.changed(), this.loading = Number(this.loading) - 1, this.dispatchEvent(new Ff(Pf.FEATURESLOADERROR));
				}, o = !1, s = this.loader_.call(this, a, t, n, (t) => o || e(t), () => o || i());
				s instanceof Promise ? (o = !0, s.then((t) => {
					this.addFeatures(t), e(t);
				}).catch(i)) : this.loader_.length < 4 && (this.loading = !1), r.insert(a, { extent: a.slice() });
			}
		}
	}
	refresh() {
		this.clear(!0), this.loadedExtentsRtree_.clear(), super.refresh();
	}
	removeLoadedExtent(e) {
		let t = this.loadedExtentsRtree_, n = [];
		t.forEachInExtent(e, function(e) {
			n.push(e);
		}), n.forEach((n) => {
			t.remove(n);
			let r = ft(n.extent, e);
			for (let e of r) t.insert(e, { extent: e });
		});
	}
	removeFeatures(e) {
		let t = !1;
		for (let n = 0, r = e.length; n < r; ++n) t = this.removeFeatureInternal(e[n]) || t;
		t && this.changed();
	}
	removeFeature(e) {
		e && this.removeFeatureInternal(e) && this.changed();
	}
	removeFeatureInternal(e) {
		let t = O(e);
		if (!(t in this.uidIndex_)) return !1;
		t in this.nullGeometryFeatures_ ? delete this.nullGeometryFeatures_[t] : this.featuresRtree_ && this.featuresRtree_.remove(e), this.featureChangeKeys_[t]?.forEach(o), delete this.featureChangeKeys_[t];
		let n = e.getId();
		if (n !== void 0) {
			let t = n.toString(), r = this.idIndex_[t];
			r === e ? delete this.idIndex_[t] : Array.isArray(r) && (r.splice(r.indexOf(e), 1), r.length === 1 && (this.idIndex_[t] = r[0]));
		}
		return delete this.uidIndex_[t], this.hasListener(Pf.REMOVEFEATURE) && this.dispatchEvent(new Ff(Pf.REMOVEFEATURE, e)), !0;
	}
	removeFromIdIndex_(e) {
		for (let t in this.idIndex_) if (this.idIndex_[t] === e) {
			delete this.idIndex_[t];
			break;
		}
	}
	setLoader(e) {
		this.loader_ = e;
	}
	setUrl(e) {
		R(this.format_, "`format` must be set when `url` is set"), this.url_ = e, this.setLoader(kf(e, this.format_));
	}
	setOverlaps(e) {
		this.overlaps_ = e, this.changed();
	}
}, Lf = [
	0,
	0,
	0
], Rf = 5, zf = class {
	constructor(e) {
		let t = e.minZoom, n = e.resolutions;
		t === void 0 && n && (t = n.findIndex((e) => e !== void 0)), this.minZoom = t === void 0 ? 0 : t, this.resolutions_ = n, R(g(this.resolutions_, (e, t) => t - e, !0), "`resolutions` must be sorted in descending order");
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
		i !== void 0 && !this.origin_ && !this.origins_ && (this.origin_ = pt(i)), R(!this.origin_ && this.origins_ || this.origin_ && !this.origins_, "Either `origin` or `origins` must be configured, never both"), this.tileSizes_ = null, e.tileSizes !== void 0 && (this.tileSizes_ = e.tileSizes, R(this.tileSizes_.length == this.resolutions_.length, "Number of `tileSizes` and `resolutions` must be equal")), this.tileSize_ = e.tileSize === void 0 ? this.tileSizes_ ? null : 256 : e.tileSize, R(!this.tileSize_ && this.tileSizes_ || this.tileSize_ && !this.tileSizes_, "Either `tileSize` or `tileSizes` must be configured, never both"), this.extent_ = i === void 0 ? null : i, this.fullTileRanges_ = null, this.tmpSize_ = [0, 0], this.tmpExtent_ = [
			0,
			0,
			0,
			0
		], e.tileRanges === void 0 ? e.sizes === void 0 ? i && this.calculateTileRanges_(i) : this.fullTileRanges_ = e.sizes.map((e, t) => {
			let n = new za(Math.min(0, e[0]), Math.max(e[0] - 1, -1), Math.min(0, e[1]), Math.max(e[1] - 1, -1));
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
			if (a !== void 0 && o !== void 0 ? (a = Math.floor(a / 2), o = Math.floor(o / 2), i = Ba(a, a, o, o, n)) : i = this.getTileRangeForExtentAndZ(s, c, n), t(c, i)) return !0;
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
				return Ba(n, n + 1, r, r + 1, t);
			}
			let r = this.getTileCoordExtent(e, n || this.tmpExtent_);
			return this.getTileRangeForExtentAndZ(r, e[0] + 1, t);
		}
		return null;
	}
	getTileRangeForTileCoordAndZ(e, t, n) {
		if (t > this.maxZoom || t < this.minZoom) return null;
		let r = e[0], i = e[1], a = e[2];
		if (t === r) return Ba(i, a, i, a, n);
		if (this.zoomFactor_) {
			let e = this.zoomFactor_ ** +(t - r), o = Math.floor(i * e), s = Math.floor(a * e);
			return t < r ? Ba(o, o, s, s, n) : Ba(o, Math.floor(e * (i + 1)) - 1, s, Math.floor(e * (a + 1)) - 1, n);
		}
		let o = this.getTileCoordExtent(e, this.tmpExtent_);
		return this.getTileRangeForExtentAndZ(o, t, n);
	}
	getTileRangeForExtentAndZ(e, t, n) {
		this.getTileCoordForXYAndZ_(e[0], e[3], t, !1, Lf);
		let r = Lf[1], i = Lf[2];
		this.getTileCoordForXYAndZ_(e[2], e[1], t, !0, Lf);
		let a = Lf[1], o = Lf[2];
		return Ba(r, a, i, o, n);
	}
	getTileCoordCenter(e) {
		let t = this.getOrigin(e[0]), n = this.getResolution(e[0]), r = no(this.getTileSize(e[0]), this.tmpSize_);
		return [t[0] + (e[1] + .5) * r[0] * n, t[1] - (e[2] + .5) * r[1] * n];
	}
	getTileCoordExtent(e, t) {
		let n = this.getOrigin(e[0]), r = this.getResolution(e[0]), i = no(this.getTileSize(e[0]), this.tmpSize_), a = n[0] + e[1] * i[0] * r, o = n[1] - (e[2] + 1) * i[1] * r;
		return qe(a, o, a + i[0] * r, o + i[1] * r, t);
	}
	getTileCoordForCoordAndResolution(e, t, n) {
		return this.getTileCoordForXYAndResolution_(e[0], e[1], t, !1, n);
	}
	getTileCoordForXYAndResolution_(e, t, n, r, i) {
		let a = this.getZForResolution(n), o = n / this.getResolution(a), s = this.getOrigin(a), c = no(this.getTileSize(a), this.tmpSize_), l = o * (e - s[0]) / n / c[0], u = o * (s[1] - t) / n / c[1];
		return r ? (l = Lt(l, Rf) - 1, u = Lt(u, Rf) - 1) : (l = It(l, Rf), u = It(u, Rf)), io(a, l, u, i);
	}
	getTileCoordForXYAndZ_(e, t, n, r, i) {
		let a = this.getOrigin(n), o = this.getResolution(n), s = no(this.getTileSize(n), this.tmpSize_), c = (e - a[0]) / o / s[0], l = (a[1] - t) / o / s[1];
		return r ? (c = Lt(c, Rf) - 1, l = Lt(l, Rf) - 1) : (c = It(c, Rf), l = It(l, Rf)), io(n, c, l, i);
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
		return Dt(f(this.resolutions_, e, t || 0), this.minZoom, this.maxZoom);
	}
	tileCoordIntersectsViewport(e, t) {
		return Si(t, 0, t.length, 2, this.getTileCoordExtent(e));
	}
	calculateTileRanges_(e) {
		let t = this.resolutions_.length, n = Array(t);
		for (let r = this.minZoom; r < t; ++r) n[r] = this.getTileRangeForExtentAndZ(e, r);
		this.fullTileRanges_ = n;
	}
}, Bf = /\{z\}/g, Vf = /\{x\}/g, Hf = /\{y\}/g, Uf = /\{-y\}/g;
function Wf(e, t, n, r, i) {
	return e.replace(Bf, t.toString()).replace(Vf, n.toString()).replace(Hf, r.toString()).replace(Uf, function() {
		if (i === void 0) throw Error("If the URL template has a {-y} placeholder, the grid extent must be known");
		return (i - r).toString();
	});
}
function Gf(e) {
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
function Kf(e, t) {
	return (function(n, r, i) {
		if (!n) return;
		let a, o = n[0];
		if (t) {
			let e = t.getFullTileRange(o);
			e && (a = e.getHeight() - 1);
		}
		return Wf(e, o, n[1], n[2], a);
	});
}
function qf(e, t) {
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Kf(e[i], t);
	return Jf(r);
}
function Jf(e) {
	return e.length === 1 ? e[0] : (function(t, n, r) {
		return t ? e[Nt(so(t), e.length)](t, n, r) : void 0;
	});
}
//#endregion
//#region node_modules/ol/tilegrid.js
function Yf(e) {
	let t = e.getDefaultTileGrid();
	return t || (t = $f(e), e.setDefaultTileGrid(t)), t;
}
function Xf(e, t, n) {
	let r = t[0], i = e.getTileCoordCenter(t), a = ep(n);
	if (!He(a, i)) {
		let t = L(a), n = Math.ceil((a[0] - i[0]) / t);
		return i[0] += t * n, e.getTileCoordForCoordAndZ(i, r);
	}
	return t;
}
function Zf(e, t, n, r) {
	r = r === void 0 ? "top-left" : r;
	let i = Qf(e, t, n);
	return new zf({
		extent: e,
		origin: st(e, r),
		resolutions: i,
		tileSize: n
	});
}
function Qf(e, t, n, r) {
	t = t === void 0 ? 42 : t, n = no(n === void 0 ? 256 : n);
	let i = ut(e), a = L(e);
	r = r > 0 ? r : Math.max(a / n[0], i / n[1]);
	let o = t + 1, s = Array(o);
	for (let e = 0; e < o; ++e) s[e] = r / 2 ** e;
	return s;
}
function $f(e, t, n, r) {
	return Zf(ep(e), t, n, r);
}
function ep(e) {
	e = rr(e);
	let t = e.getExtent();
	if (!t) {
		let n = 180 * qt.degrees / e.getMetersPerUnit();
		t = qe(-n, -n, n, n);
	}
	return t;
}
//#endregion
//#region node_modules/ol/source/Tile.js
var tp = class extends Mf {
	constructor(e) {
		super({
			attributions: e.attributions,
			attributionsCollapsible: e.attributionsCollapsible,
			projection: e.projection,
			state: e.state,
			wrapX: e.wrapX,
			interpolate: e.interpolate
		}), this.on, this.once, this.un, this.tilePixelRatio_ = e.tilePixelRatio === void 0 ? 1 : e.tilePixelRatio, this.tileGrid = e.tileGrid === void 0 ? null : e.tileGrid, this.tileGrid && no(this.tileGrid.getTileSize(this.tileGrid.getMinZoom()), [256, 256]), this.tmpSize = [0, 0], this.key_ = e.key || O(this), this.tileOptions = {
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
		return E();
	}
	getTileGrid() {
		return this.tileGrid;
	}
	getTileGridForProjection(e) {
		return this.tileGrid ? this.tileGrid : Yf(e);
	}
	getTilePixelRatio(e) {
		return this.tilePixelRatio_;
	}
	getTilePixelSize(e, t, n) {
		let r = this.getTileGridForProjection(n), i = this.getTilePixelRatio(t), a = no(r.getTileSize(e), this.tmpSize);
		return i == 1 ? a : to(a, i, this.tmpSize);
	}
	getTileCoordForTileUrlFunction(e, t) {
		let n = t === void 0 ? this.getProjection() : t, r = t === void 0 && this.tileGrid || this.getTileGridForProjection(n);
		return this.getWrapX() && n.isGlobal() && (e = Xf(r, e, n)), lo(e, r) ? e : null;
	}
	clear() {}
	refresh() {
		this.clear(), super.refresh();
	}
}, np = class extends S {
	constructor(e, t) {
		super(e), this.tile = t;
	}
}, rp = {
	TILELOADSTART: "tileloadstart",
	TILELOADEND: "tileloadend",
	TILELOADERROR: "tileloaderror"
}, ip = class e extends tp {
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
		let t = e.target, n = O(t), r = t.getState(), i;
		r == B.LOADING ? (this.tileLoadingKeys_[n] = !0, i = rp.TILELOADSTART) : n in this.tileLoadingKeys_ && (delete this.tileLoadingKeys_[n], i = r == B.ERROR ? rp.TILELOADERROR : r == B.LOADED ? rp.TILELOADEND : void 0), i != null && this.dispatchEvent(new np(i, t));
	}
	setTileLoadFunction(e) {
		this.tileLoadFunction = e, this.changed();
	}
	setTileUrlFunction(e, t) {
		this.tileUrlFunction = e, t === void 0 ? this.changed() : this.setKey(t);
	}
	setUrl(e) {
		let t = Gf(e);
		this.urls = t, this.setUrls(t);
	}
	setUrls(e) {
		this.urls = e;
		let t = e.join("\n");
		this.generateTileUrlFunction_ ? this.setTileUrlFunction(qf(e, this.tileGrid), t) : this.setKey(t);
	}
	tileUrlFunction(e, t, n) {}
}, ap = class extends ip {
	constructor(e) {
		super({
			attributions: e.attributions,
			cacheSize: e.cacheSize,
			projection: e.projection,
			state: e.state,
			tileGrid: e.tileGrid,
			tileLoadFunction: e.tileLoadFunction ? e.tileLoadFunction : op,
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
		}), this.crossOrigin = e.crossOrigin === void 0 ? null : e.crossOrigin, this.referrerPolicy = e.referrerPolicy, this.tileClass = e.tileClass === void 0 ? La : e.tileClass, this.tileGridForProjection = {}, this.reprojectionErrorThreshold_ = e.reprojectionErrorThreshold, this.renderReprojectionEdges_ = !1;
	}
	getGutterForProjection(e) {
		return this.getProjection() && e && !lr(this.getProjection(), e) ? 0 : this.getGutter();
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
		if (this.tileGrid && (!t || lr(t, e))) return this.tileGrid;
		let n = O(e);
		return n in this.tileGridForProjection || (this.tileGridForProjection[n] = Yf(e)), this.tileGridForProjection[n];
	}
	createTile_(e, t, n, r, i, a) {
		let o = [
			e,
			t,
			n
		], c = this.getTileCoordForTileUrlFunction(o, i), l = c ? this.tileUrlFunction(c, r, i) : void 0, u = new this.tileClass(o, l === void 0 ? B.EMPTY : B.IDLE, l === void 0 ? "" : l, {
			crossOrigin: this.crossOrigin,
			referrerPolicy: this.referrerPolicy
		}, this.tileLoadFunction, this.tileOptions);
		return u.key = a, u.addEventListener(s.CHANGE, this.handleTileChange.bind(this)), u;
	}
	getTile(e, t, n, r, i, a) {
		let o = this.getProjection();
		if (!o || !i || lr(o, i)) return this.getTileInternal(e, t, n, r, o || i);
		let s = [
			e,
			t,
			n
		], c = this.getKey(), l = new $a(o, this.getTileGridForProjection(o), i, this.getTileGridForProjection(i), s, this.getTileCoordForTileUrlFunction(s, i), this.getTilePixelRatio(r), this.getGutter(), (e, t, n, r) => this.getTileInternal(e, t, n, r, o, a), this.reprojectionErrorThreshold_, this.renderReprojectionEdges_, this.tileOptions);
		return l.key = c, l;
	}
	getTileInternal(e, t, n, r, i, a) {
		let o = this.getKey(), s = oo(this, o, e, t, n);
		if (a && a.containsKey(s)) return a.get(s);
		let c = this.createTile_(e, t, n, r, i, o);
		return a?.set(s, c), c;
	}
	setRenderReprojectionEdges(e) {
		this.renderReprojectionEdges_ != e && (this.renderReprojectionEdges_ = e, this.changed());
	}
	setTileGridForProjection(e, t) {
		let n = rr(e);
		if (n) {
			let e = O(n);
			e in this.tileGridForProjection || (this.tileGridForProjection[e] = t);
		}
	}
};
function op(e, t) {
	if (pe) {
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
var sp = class extends La {
	constructor(e, t, n, r, i, a, o) {
		super(t, n, r, i, a, o), this.zoomifyImage_ = null, this.tileSize_ = e;
	}
	getImage() {
		if (this.zoomifyImage_) return this.zoomifyImage_;
		let e = super.getImage();
		if (this.state == B.LOADED) {
			let t = this.tileSize_;
			if (e.width == t[0] && e.height == t[1]) return this.zoomifyImage_ = e, e;
			let n = I(t[0], t[1]);
			return n.drawImage(e, 0, 0), this.zoomifyImage_ = n.canvas, n.canvas;
		}
		return e;
	}
}, cp = class extends ap {
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
		let d = new zf({
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
		let p = Gf(f), m = s * r;
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
		let g = Jf(p.map(h)), _ = sp.bind(null, no(s * r));
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
		let v = g(d.getTileCoordForCoordAndResolution(ot(d.getExtent()), l[l.length - 1]), 1, null), y = new Image();
		y.addEventListener("error", () => {
			m = s, this.changed();
		}), y.src = v;
	}
}, lp = {
	image: [
		"Polygon",
		"Circle",
		"LineString",
		"Image",
		"Text"
	],
	hybrid: ["Polygon", "LineString"],
	vector: []
}, up = {
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
}, dp = class extends qo {
	constructor(e, t) {
		super(e, t), this.boundHandleStyleImageChange_ = this.handleStyleImageChange_.bind(this), this.renderedLayerRevision_, this.renderedPixelToCoordinateTransform_ = null, this.renderedRotation_, this.renderedOpacity_ = 1, this.tmpTransform_ = Tr(), this.tileClipContexts_ = null;
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
		let a = r.viewState, o = a.resolution, s = r.viewHints, c = this.getLayer().getSource(), l = c.getTileGridForProjection(a.projection), u = !(s[Jo.ANIMATING] || s[Jo.INTERACTING]), d = l.getZForResolution(o, c.zDirection) === e;
		return u && d ? i.wantedResolution = o : i.wantedResolution ||= l.getResolution(e), i;
	}
	prepareFrame(e) {
		let t = this.getLayer().getRevision();
		return this.renderedLayerRevision_ !== t && (this.renderedLayerRevision_ = t, this.renderedTiles.length = 0), super.prepareFrame(e);
	}
	updateExecutorGroup_(e, t, n) {
		let r = this.getLayer(), i = r.getRevision(), a = r.getRenderOrder() || null, o = e.wantedResolution, s = e.getReplayState(r);
		if (!s.dirty && s.renderedResolution === o && s.renderedRevision == i && s.renderedPixelRatio === t && s.renderedRenderOrder == a) return;
		let c = r.getSource(), l = !!r.getDeclutter(), u = c.getTileGrid(), d = c.getTileGridForProjection(n).getTileCoordExtent(e.wrappedTileCoord), f = c.getSourceTiles(t, n, e), p = O(r);
		delete e.hitDetectionImageData[p], e.executorGroups[p] = [], s.dirty = !1;
		for (let i = 0, m = f.length; i < m; ++i) {
			let m = f[i];
			if (m.getState() != B.LOADED) continue;
			let h = c.getProjection(), g = m.tileCoord, _ = u.getTileCoordExtent(g);
			n && h && !lr(n, h) && (_ = mr(_, h, n, 32));
			let v = dt(d, _), y = ze(v, r.getRenderBuffer() * o, this.tempExtent), b = Ze(_, v) ? null : y, x = new hc(0, v, o, t), S = Qc(o, t), C = function(e, t) {
				let n, i = e.getStyleFunction() || r.getStyleFunction();
				if (i && (n = i(e, o)), n) {
					let r = this.renderFeature(e, S, n, x, l, t);
					s.dirty = s.dirty || r;
				}
			}, w = m.getFeatures();
			a && a !== s.renderedRenderOrder && w.sort(a);
			for (let e = 0, t = w.length; e < t; ++e) {
				let t = w[e];
				n && m.projection && !lr(n, m.projection) && (t = t.clone(), t.getGeometry().applyTransform(fr(m.projection, n))), (!b || ht(b, t.getGeometry().getExtent())) && C.call(this, t, e);
			}
			let T = x.finish(), E = new zc(r.getRenderMode() !== "vector" && l && f.length === 1 ? null : v, o, t, c.getOverlaps(), T, r.getRenderBuffer(), !0);
			e.executorGroups[p].push(E);
		}
		s.renderedRevision = i, s.renderedPixelRatio = t, s.renderedRenderOrder = a, s.renderedResolution = o;
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		let a = t.viewState.resolution, o = t.viewState.rotation;
		n ??= 0;
		let s = this.getLayer(), c = s.getSource().getTileGridForProjection(t.viewState.projection), l = s.getRenderBuffer(), u = Le([e]);
		ze(u, a * (l + n), u);
		let d = {}, f = function(e, t, n) {
			let a = e.getId();
			a === void 0 && (a = O(e));
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
		}, p = this.renderedTiles, m = O(s), h = s.getDeclutter(), g = h ? t.declutter?.[h]?.all().map((e) => e.value) : null, _;
		foundFeature: for (let t = p.length - 1; t >= 0; --t) {
			let r = p[t];
			if (!ht(c.getTileCoordExtent(r.wrappedTileCoord), u)) continue;
			let i = r.executorGroups[m];
			for (let t = 0, r = i.length; t < r; ++t) if (_ = i[t].forEachFeatureAtCoordinate(e, a, o, n, f, g), _) break foundFeature;
		}
		return _;
	}
	getFeatures(e) {
		return this.renderedTiles.length === 0 ? Promise.resolve([]) : new Promise((t, n) => {
			let r = this.getLayer(), i = r.getSource(), a = this.renderedProjection, o = a.getExtent(), s = this.renderedResolution, c = i.getTileGridForProjection(a), l = z(this.renderedPixelToCoordinateTransform_, e.slice()), u = c.getTileCoordForCoordAndResolution(l, s).toString(), d = this.renderedTiles.find((e) => e.tileCoord.toString() === u && e.getState() === B.LOADED);
			if (!d || d.loadingSourceTiles > 0) {
				t([]);
				return;
			}
			i.getWrapX() && a.canWrapX() && !Ue(o, c.getTileCoordExtent(d.tileCoord)) && Wt(l, a);
			let f = O(r), p = pt(c.getTileCoordExtent(d.wrappedTileCoord)), m = [(l[0] - p[0]) / s, (p[1] - l[1]) / s], h = d.getSourceTiles().reduce((e, t) => e.concat(t.getFeatures()), []), g = d.hitDetectionImageData[f];
			if (!g) {
				let e = no(c.getTileSize(c.getZForResolution(s, i.zDirection))), t = this.renderedRotation_;
				g = qc(e, [this.getRenderTransform(c.getTileCoordCenter(d.wrappedTileCoord), s, 0, Kc, e[0] * Kc, e[1] * Kc, 0)], h, r.getStyleFunction(), c.getTileCoordExtent(d.wrappedTileCoord), d.getReplayState(r).renderedResolution, t), d.hitDetectionImageData[f] = g;
			}
			t(Jc(m, h, g));
		});
	}
	getFeaturesInExtent(e) {
		let t = [], n = this.getTileCache();
		if (n.getCount() === 0) return t;
		let r = this.getLayer().getSource().getTileGridForProjection(this.frameState.viewState.projection), i = r.getZForResolution(this.renderedResolution), a = {};
		return n.forEach((n) => {
			if (n.tileCoord[0] !== i || n.getState() !== B.LOADED) return;
			let o = n.getSourceTiles();
			for (let n = 0, i = o.length; n < i; ++n) {
				let i = o[n], s = i.getKey();
				if (s in a) continue;
				a[s] = !0;
				let c = i.tileCoord;
				if (ht(e, r.getTileCoordExtent(c))) {
					let n = i.getFeatures();
					if (n) for (let r = 0, i = n.length; r < i; ++r) {
						let i = n[r];
						ht(e, i.getGeometry().getExtent()) && t.push(i);
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
		let i = e.viewHints, a = !(i[Jo.ANIMATING] || i[Jo.INTERACTING]), o = [this.context.canvas.width, this.context.canvas.height], s = this.getLayer().getDeclutter(), c = s ? e.declutter?.[s] : void 0, l = O(this.getLayer()), u = this.renderedTiles;
		for (let t = 0, n = u.length; t < n; ++t) {
			let n = u[t], r = n.executorGroups[l];
			if (r) for (let t = r.length - 1; t >= 0; --t) r[t].execute(this.context, o, this.getTileRenderTransform(n, e), e.viewState.rotation, a, Pc, c);
		}
		n.globalAlpha = r;
	}
	renderDeferredInternal(e) {
		let t = this.renderedTiles, n = O(this.getLayer()), r = t.reduce((e, t, r) => (t.executorGroups[n].forEach((t) => e.push({
			executorGroup: t,
			index: r
		})), e), []), i = r.map(({ executorGroup: e }) => e.getDeferredZIndexContexts()), a = {};
		for (let e = 0, t = r.length; e < t; ++e) {
			let t = r[e].executorGroup.getDeferredZIndexContexts();
			for (let e in t) a[e] = !0;
		}
		let o = Object.keys(a).map(Number).sort(u);
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
		return Dr(Ar(this.inversePixelTransform.slice(), 1 / n, 1 / n), this.getRenderTransform(i, a, o, n, c, l, p));
	}
	clipTileContext_(e, t, n, r, i, a) {
		let o = [];
		for (let e = 0, a = n.length; e < a; ++e) i < r[e] && ht(t, n[e]) && o.push(n[e]);
		if (o.length === 0) return !1;
		let s = Ct(t, o);
		e.save(), e.beginPath();
		for (let t = 0, n = s.length; t < n; ++t) {
			let n = s[t], r = z(a, [n[0], n[1]]), i = z(a, [n[0], n[3]]), o = z(a, [n[2], n[3]]), c = z(a, [n[2], n[1]]);
			e.moveTo(r[0], r[1]), e.lineTo(i[0], i[1]), e.lineTo(o[0], o[1]), e.lineTo(c[0], c[1]), e.closePath();
		}
		return e.clip(), !0;
	}
	postRender(e, t) {
		let n = t.viewHints, r = !(n[Jo.ANIMATING] || n[Jo.INTERACTING]);
		this.renderedPixelToCoordinateTransform_ = t.pixelToCoordinateTransform.slice(), this.renderedRotation_ = t.viewState.rotation, this.renderedOpacity_ = t.layerStatesArray[t.layerIndex].opacity;
		let i = this.getLayer(), a = i.getRenderMode(), o = e.globalAlpha;
		e.globalAlpha = this.renderedOpacity_;
		let s = i.getDeclutter(), c = s ? up[a].filter((e) => !Pc.includes(e)) : up[a], l = t.viewState, u = l.rotation;
		this.layerExtent && this.clipUnrotated(e, t, this.layerExtent);
		let d = i.getSource(), f = d.getTileGridForProjection(l.projection).getZForResolution(l.resolution, d.zDirection), p = this.renderedTiles, m = [], h = [], g = [], _ = O(i), v = !0;
		for (let n = p.length - 1; n >= 0; --n) {
			let a = p[n];
			v &&= !a.getReplayState(i).dirty;
			let o = a.executorGroups[_].filter((e) => e.hasExecutors(c));
			if (o.length === 0) continue;
			let l = this.getTileRenderTransform(a, t), d = a.tileCoord[0], y = !1, b = o[0].getClipCoords(Cr), x = e, S;
			if (b) {
				let e = [
					b[0],
					b[1],
					b[4],
					b[5]
				];
				S = new Lo(), x = S.getContext(), f !== d && (y = this.clipTileContext_(x, e, m, h, d, l)), m.push(e), h.push(d);
			}
			for (let n = 0, i = o.length; n < i; ++n) o[n].execute(e, [e.canvas.width, e.canvas.height], l, u, r, c, t.declutter?.[s]);
			y && (x === e ? x.restore() : g[n] = S);
		}
		this.layerExtent && e.restore(), e.globalAlpha = o, this.ready = v, this.tileClipContexts_ = g, t.declutter || this.renderDeferredInternal(t), super.postRender(e, t);
	}
	renderFeature(e, t, n, r, i, a) {
		if (!n) return !1;
		let o = !1;
		if (Array.isArray(n)) for (let s = 0, c = n.length; s < c; ++s) o = tl(r, e, n[s], t, this.boundHandleStyleImageChange_, void 0, i, a) || o;
		else o = tl(r, e, n, t, this.boundHandleStyleImageChange_, void 0, i, a);
		return o;
	}
	tileImageNeedsRender_(e) {
		let t = this.getLayer();
		if (t.getRenderMode() === "vector") return !1;
		let n = e.getReplayState(t), r = t.getRevision(), i = e.wantedResolution;
		return n.renderedTileResolution !== i || n.renderedTileRevision !== r;
	}
	renderTileImage_(e, t) {
		let n = this.getLayer(), r = e.getReplayState(n), i = n.getRevision(), a = e.executorGroups[O(n)];
		r.renderedTileRevision = i;
		let o = e.wrappedTileCoord, s = o[0], c = n.getSource(), l = t.pixelRatio, u = t.viewState.projection, d = c.getTileGridForProjection(u), f = d.getResolution(e.tileCoord[0]), p = t.pixelRatio / e.wantedResolution * f, m = d.getResolution(s), h = e.getContext();
		l = Math.round(Math.max(l, p / l));
		let g = c.getTilePixelSize(s, l, u);
		h.canvas.width = g[0], h.canvas.height = g[1];
		let _ = l / p;
		if (_ !== 1) {
			let e = Er(this.tmpTransform_);
			Ar(e, _, _), h.setTransform.apply(h, e);
		}
		let v = d.getTileCoordExtent(o, this.tempExtent), y = p / m, b = Er(this.tmpTransform_);
		Ar(b, y, -y), jr(b, -v[0], -v[3]);
		for (let e = 0, t = a.length; e < t; ++e) a[e].execute(h, [h.canvas.width * _, h.canvas.height * _], b, 0, !0, lp[n.getRenderMode()], null);
		r.renderedTileResolution = e.wantedResolution;
	}
}, fp = class extends yd {
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
		return new dp(this, { cacheSize: this.cacheSize_ });
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
		return this.get(vs.PRELOAD);
	}
	getUseInterimTilesOnError() {
		return this.get(vs.USE_INTERIM_TILES_ON_ERROR);
	}
	setPreload(e) {
		this.set(vs.PRELOAD, e);
	}
	setUseInterimTilesOnError(e) {
		this.set(vs.USE_INTERIM_TILES_ON_ERROR, e);
	}
}, pp = class extends Uo {
	constructor(e) {
		super(e), this.image = null, this.renderedSourceRevision_ = 0;
	}
	getImage() {
		return this.image ? this.image.getImage() : null;
	}
	prepareFrame(e) {
		let t = e.layerStatesArray[e.layerIndex], n = e.pixelRatio, r = e.viewState, i = r.resolution, a = this.getLayer().getSource(), o = e.viewHints, s = e.extent;
		if (t.extent !== void 0 && (s = dt(s, br(t.extent, r.projection))), !o[Jo.ANIMATING] && !o[Jo.INTERACTING] && !gt(s)) {
			if (a) {
				!this.getLayer().rendered && this.renderedSourceRevision_ !== a.getRevision() && (this.image = null), this.renderedSourceRevision_ = a.getRevision();
				let e = r.projection, t = a.getImage(s, i, n, e);
				t && (this.loadImage(t) ? this.image = t : t.getState() === V.EMPTY && (this.image = null));
			} else this.image = null;
		}
		return !!this.image;
	}
	getData(e) {
		let t = this.frameState;
		if (!t) return null;
		let n = this.getLayer(), r = z(t.pixelToCoordinateTransform, e.slice()), i = n.getExtent();
		if (i && !He(i, r)) return null;
		let a = this.image.getExtent(), o = this.image.getImage(), s = L(a), c = Math.floor(o.width * ((r[0] - a[0]) / s));
		if (c < 0 || c >= o.width) return null;
		let l = ut(a), u = Math.floor(o.height * ((a[3] - r[1]) / l));
		return u < 0 || u >= o.height ? null : this.getImageData(o, c, u);
	}
	renderFrame(e, t) {
		let n = this.image, r = n.getExtent(), i = n.getResolution(), [a, o] = Array.isArray(i) ? i : [i, i], s = n.getPixelRatio(), c = e.layerStatesArray[e.layerIndex], l = e.pixelRatio, u = e.viewState, d = u.center, f = u.resolution, p = l * a / (f * s), m = l * o / (f * s);
		this.prepareContainer(e, t);
		let h = this.context.canvas.width, g = this.context.canvas.height, _ = this.getRenderContext(e), v = !1, y = !0;
		if (c.extent) {
			let t = br(c.extent, u.projection);
			y = ht(t, e.extent), v = y && !Ue(t, e.extent), v && this.clipUnrotated(_, e, t);
		}
		let b = n.getImage(), x = Mr(this.tempTransform, h / 2, g / 2, p, m, 0, s * (r[0] - d[0]) / a, s * (d[1] - r[3]) / o);
		this.renderedResolution = o * l / s;
		let S = b.width * x[0], C = b.height * x[3];
		if (this.getLayer().getSource().getInterpolate() || (_.imageSmoothingEnabled = !1), this.preRender(_, e), y && S >= .5 && C >= .5) {
			let e = x[4], t = x[5], n = c.opacity;
			n !== 1 && (_.save(), _.globalAlpha = n), _.drawImage(b, 0, 0, +b.width, +b.height, e, t, S, C), n !== 1 && _.restore();
		}
		return this.postRender(this.context, e), v && _.restore(), _.imageSmoothingEnabled = !0, this.container;
	}
}, mp = class extends gs {
	constructor(e) {
		e ||= {}, super(e);
	}
}, hp = class extends mp {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new pp(this);
	}
	getData(e) {
		return super.getData(e);
	}
};
//#endregion
//#region node_modules/ol/vec/mat4.js
function gp() {
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
function _p(e) {
	return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = 1, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = 1, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, e;
}
function vp(e, t) {
	return e[0] = t[0], e[1] = t[1], e[4] = t[2], e[5] = t[3], e[12] = t[4], e[13] = t[5], e;
}
function yp(e, t, n, r, i) {
	return i ??= gp(), i[0] = e[0] * t, i[1] = e[1] * t, i[2] = e[2] * t, i[3] = e[3] * t, i[4] = e[4] * n, i[5] = e[5] * n, i[6] = e[6] * n, i[7] = e[7] * n, i[8] = e[8] * r, i[9] = e[9] * r, i[10] = e[10] * r, i[11] = e[11] * r, i[12] = e[12], i[13] = e[13], i[14] = e[14], i[15] = e[15], i;
}
function bp(e, t, n, r, i) {
	i ??= gp();
	let a, o, s, c, l, u, d, f, p, m, h, g;
	return e === i ? (i[12] = e[0] * t + e[4] * n + e[8] * r + e[12], i[13] = e[1] * t + e[5] * n + e[9] * r + e[13], i[14] = e[2] * t + e[6] * n + e[10] * r + e[14], i[15] = e[3] * t + e[7] * n + e[11] * r + e[15]) : (a = e[0], o = e[1], s = e[2], c = e[3], l = e[4], u = e[5], d = e[6], f = e[7], p = e[8], m = e[9], h = e[10], g = e[11], i[0] = a, i[1] = o, i[2] = s, i[3] = c, i[4] = l, i[5] = u, i[6] = d, i[7] = f, i[8] = p, i[9] = m, i[10] = h, i[11] = g, i[12] = a * t + l * n + p * r + e[12], i[13] = o * t + u * n + m * r + e[13], i[14] = s * t + d * n + h * r + e[14], i[15] = c * t + f * n + g * r + e[15]), i;
}
function xp(e, t, n) {
	n ??= gp();
	let r = Math.cos(t), i = Math.sin(t), a = r, o = -i, s = i, c = r, l = e[0], u = e[1], d = e[2], f = e[3], p = e[4], m = e[5], h = e[6], g = e[7];
	return n[0] = a * l + o * p, n[1] = a * u + o * m, n[2] = a * d + o * h, n[3] = a * f + o * g, n[4] = s * l + c * p, n[5] = s * u + c * m, n[6] = s * d + c * h, n[7] = s * f + c * g, n !== e && (n[8] = e[8], n[9] = e[9], n[10] = e[10], n[11] = e[11], n[12] = e[12], n[13] = e[13], n[14] = e[14], n[15] = e[15]), n;
}
//#endregion
//#region node_modules/ol/webgl.js
var Sp = 34962, Cp = 34963, wp = 35040, Tp = 35044, Ep = 35048, Dp = 5121, Op = 5123, kp = 5125, Ap = 5126, jp = [
	"experimental-webgl",
	"webgl",
	"webkit-3d",
	"moz-webgl"
];
function Mp(e, t) {
	t = Object.assign({
		preserveDrawingBuffer: !0,
		antialias: !le
	}, t);
	let n = jp.length;
	for (let r = 0; r < n; ++r) try {
		let n = e.getContext(jp[r], t);
		if (n) return n;
	} catch {}
	return null;
}
//#endregion
//#region node_modules/ol/webgl/Buffer.js
var Np = {
	STATIC_DRAW: Tp,
	STREAM_DRAW: wp,
	DYNAMIC_DRAW: Ep
}, Pp = class {
	constructor(e, t) {
		this.array_ = null, this.type_ = e, R(e === 34962 || e === 34963, "A `WebGLArrayBuffer` must either be of type `ELEMENT_ARRAY_BUFFER` or `ARRAY_BUFFER`"), this.usage_ = t === void 0 ? Np.STATIC_DRAW : t;
	}
	ofSize(e) {
		return this.array_ = new (Fp(this.type_))(e), this;
	}
	fromArray(e) {
		return this.array_ = Fp(this.type_).from(e), this;
	}
	fromArrayBuffer(e) {
		return this.array_ = new (Fp(this.type_))(e), this;
	}
	getType() {
		return this.type_;
	}
	getArray() {
		return this.array_;
	}
	setArray(e) {
		let t = Fp(this.type_);
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
function Fp(e) {
	switch (e) {
		case Sp: return Float32Array;
		case Cp: return Uint32Array;
		default: return Float32Array;
	}
}
//#endregion
//#region node_modules/ol/webgl/ContextEventType.js
var Ip = {
	LOST: "webglcontextlost",
	RESTORED: "webglcontextrestored"
}, Lp = "\n  precision mediump float;\n\n  attribute vec2 a_position;\n  varying vec2 v_texCoord;\n  varying vec2 v_screenCoord;\n\n  uniform vec2 u_screenSize;\n\n  void main() {\n    v_texCoord = a_position * 0.5 + 0.5;\n    v_screenCoord = v_texCoord * u_screenSize;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n  }\n", Rp = "\n  precision mediump float;\n\n  uniform sampler2D u_image;\n  uniform float u_opacity;\n\n  varying vec2 v_texCoord;\n\n  void main() {\n    gl_FragColor = texture2D(u_image, v_texCoord) * u_opacity;\n  }\n", zp = class {
	constructor(e) {
		this.gl_ = e.webGlContext;
		let t = this.gl_;
		this.scaleRatio_ = e.scaleRatio || 1, this.renderTargetTexture_ = t.createTexture(), this.renderTargetTextureSize_ = null, this.frameBuffer_ = t.createFramebuffer(), this.depthBuffer_ = t.createRenderbuffer();
		let n = t.createShader(t.VERTEX_SHADER);
		t.shaderSource(n, e.vertexShader || Lp), t.compileShader(n);
		let r = t.createShader(t.FRAGMENT_SHADER);
		if (t.shaderSource(r, e.fragmentShader || Rp), t.compileShader(r), !t.getShaderParameter(r, t.COMPILE_STATUS)) {
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
			let t = O(i.canvas);
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
}, Bp = {
	PROJECTION_MATRIX: "u_projectionMatrix",
	INVERT_PROJECTION_MATRIX: "u_invertProjectionMatrix",
	TIME: "u_time",
	ZOOM: "u_zoom",
	RESOLUTION: "u_resolution",
	ROTATION: "u_rotation",
	VIEWPORT_SIZE_PX: "u_viewportSizePx",
	PIXEL_RATIO: "u_pixelRatio",
	HIT_DETECTION: "u_hitDetection"
}, Vp = {
	UNSIGNED_BYTE: Dp,
	UNSIGNED_SHORT: Op,
	UNSIGNED_INT: kp,
	FLOAT: Ap
}, Hp = {};
function Up(e) {
	return "shared/" + e;
}
var Wp = 0;
function Gp() {
	let e = "unique/" + Wp;
	return Wp += 1, e;
}
function Kp(e) {
	let t = Hp[e];
	if (!t) {
		let n = document.createElement("canvas");
		n.width = 1, n.height = 1, n.style.position = "absolute", n.style.left = "0", t = {
			users: 0,
			context: Mp(n)
		}, Hp[e] = t;
	}
	return t.users += 1, t.context;
}
function qp(e) {
	let t = Hp[e];
	if (!t || (--t.users, t.users > 0)) return;
	let n = t.context, r = n.getExtension("WEBGL_lose_context");
	r && r.loseContext();
	let i = n.canvas;
	i.width = 1, i.height = 1, delete Hp[e];
}
var Jp = class extends c {
	constructor(e) {
		super(), e ||= {}, this.boundHandleWebGLContextLost_ = this.handleWebGLContextLost.bind(this), this.boundHandleWebGLContextRestored_ = this.handleWebGLContextRestored.bind(this), this.canvasCacheKey_ = e.canvasCacheKey ? Up(e.canvasCacheKey) : Gp(), this.gl_ = Kp(this.canvasCacheKey_), this.bufferCache_ = {}, this.extensionCache_ = {}, this.currentProgram_ = null, this.needsToBeRecreated_ = !1;
		let t = this.gl_.canvas;
		t.addEventListener(Ip.LOST, this.boundHandleWebGLContextLost_), t.addEventListener(Ip.RESTORED, this.boundHandleWebGLContextRestored_), this.tmpMat4_ = gp(), this.uniformLocationsByProgram_ = {}, this.attribLocationsByProgram_ = {}, this.uniforms_ = [], e.uniforms && this.setUniforms(e.uniforms), this.postProcessPasses_ = e.postProcesses?.length ? e.postProcesses.map((e) => new zp({
			webGlContext: this.gl_,
			scaleRatio: e.scaleRatio,
			vertexShader: e.vertexShader,
			fragmentShader: e.fragmentShader,
			uniforms: e.uniforms
		})) : [new zp({ webGlContext: this.gl_ })], this.shaderCompileErrors_ = null, this.startTime_ = Date.now(), this.maxAttributeCount_ = this.gl_.getParameter(this.gl_.MAX_VERTEX_ATTRIBS);
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
		return this.canvasCacheKey_ === Up(e);
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
		let t = this.gl_, n = O(e), r = this.bufferCache_[n];
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
		let t = O(e);
		delete this.bufferCache_[t];
	}
	disposeInternal() {
		let e = this.gl_.canvas;
		e.removeEventListener(Ip.LOST, this.boundHandleWebGLContextLost_), e.removeEventListener(Ip.RESTORED, this.boundHandleWebGLContextRestored_), qp(this.canvasCacheKey_), delete this.gl_;
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
		this.setUniformFloatValue(Bp.TIME, (Date.now() - this.startTime_) * .001), this.setUniformFloatValue(Bp.ZOOM, e.viewState.zoom), this.setUniformFloatValue(Bp.RESOLUTION, e.viewState.resolution), this.setUniformFloatValue(Bp.PIXEL_RATIO, r), this.setUniformFloatVec2(Bp.VIEWPORT_SIZE_PX, [t[0], t[1]]), this.setUniformFloatValue(Bp.ROTATION, n);
	}
	applyHitDetectionUniform(e) {
		let t = this.getUniformLocation(Bp.HIT_DETECTION);
		this.getGL().uniform1i(t, +!!e), e && this.setUniformFloatValue(Bp.PIXEL_RATIO, .5);
	}
	applyUniforms(e) {
		let t = this.gl_, n, r = 0;
		this.uniforms_.forEach((i) => {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof HTMLImageElement || n instanceof ImageData || n instanceof WebGLTexture) {
				n instanceof WebGLTexture && !i.texture ? (i.prevValue = void 0, i.texture = n) : i.texture ||= (i.prevValue = void 0, t.createTexture()), this.bindTexture(i.texture, r, i.name), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE);
				let e = !(n instanceof HTMLImageElement) || n.complete;
				!(n instanceof WebGLTexture) && e && i.prevValue !== n && (i.prevValue = n, t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n)), r++;
			} else if (Array.isArray(n) && n.length === 6) this.setUniformMatrixValue(i.name, vp(this.tmpMat4_, n));
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
		let t = O(this.currentProgram_);
		return this.uniformLocationsByProgram_[t] === void 0 && (this.uniformLocationsByProgram_[t] = {}), this.uniformLocationsByProgram_[t][e] === void 0 && (this.uniformLocationsByProgram_[t][e] = this.gl_.getUniformLocation(this.currentProgram_, e)), this.uniformLocationsByProgram_[t][e];
	}
	getAttributeLocation(e) {
		let t = O(this.currentProgram_);
		return this.attribLocationsByProgram_[t] === void 0 && (this.attribLocationsByProgram_[t] = {}), this.attribLocationsByProgram_[t][e] === void 0 && (this.attribLocationsByProgram_[t][e] = this.gl_.getAttribLocation(this.currentProgram_, e)), this.attribLocationsByProgram_[t][e];
	}
	makeProjectionTransform(e, t, n) {
		let r = e.size, i = n ? 0 : e.viewState.rotation, a = e.viewState.resolution, o = e.viewState.center;
		return Mr(t, 0, 0, 2 / (a * r[0]), 2 / (a * r[1]), -i, -o[0], -o[1]), t;
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
		let n = Yp(e), r = 0;
		for (let i = 0; i < e.length; i++) {
			let a = e[i];
			a.name && this.enableAttributeArray_(a.name, a.size, a.type || 5126, n, r, t), r += a.size * Xp(a.type);
		}
	}
	enableAttributes(e) {
		this.enableAttributes_(e, !1);
	}
	enableAttributesInstanced(e) {
		this.enableAttributes_(e, !0);
	}
	handleWebGLContextLost(e) {
		n(this.bufferCache_), this.currentProgram_ = null, e.preventDefault();
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
function Yp(e) {
	let t = 0;
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		t += r.size * Xp(r.type);
	}
	return t;
}
function Xp(e) {
	switch (e) {
		case Vp.UNSIGNED_BYTE: return Uint8Array.BYTES_PER_ELEMENT;
		case Vp.UNSIGNED_SHORT: return Uint16Array.BYTES_PER_ELEMENT;
		case Vp.UNSIGNED_INT: return Uint32Array.BYTES_PER_ELEMENT;
		case Vp.FLOAT:
		default: return Float32Array.BYTES_PER_ELEMENT;
	}
}
//#endregion
//#region node_modules/ol/renderer/webgl/Layer.js
var Zp = class e extends zo {
	constructor(e, t) {
		super(e), t ||= {}, this.inversePixelTransform_ = Tr(), this.postProcesses_ = t.postProcesses, this.uniforms_ = t.uniforms, this.helper, this.onMapChanged_ = () => {
			this.clearCache(), this.removeHelper();
		}, e.addChangeListener(H.MAP, this.onMapChanged_), this.dispatchPreComposeEvent = this.dispatchPreComposeEvent.bind(this), this.dispatchPostComposeEvent = this.dispatchPostComposeEvent.bind(this);
	}
	dispatchPreComposeEvent(e, t) {
		let n = this.getLayer();
		if (n.hasListener(Io.PRECOMPOSE)) {
			let r = new Fo(Io.PRECOMPOSE, void 0, t, e);
			n.dispatchEvent(r);
		}
	}
	dispatchPostComposeEvent(e, t) {
		let n = this.getLayer();
		if (n.hasListener(Io.POSTCOMPOSE)) {
			let r = new Fo(Io.POSTCOMPOSE, void 0, t, e);
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
			(!this.helper || !this.helper.canvasCacheKeyMatches(a) || this.helper.needsToBeRecreated()) && (this.removeHelper(), this.helper = new Jp({
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
		this.clearCache(), this.removeHelper(), this.getLayer()?.removeChangeListener(H.MAP, this.onMapChanged_), super.disposeInternal();
	}
	dispatchRenderEvent_(e, t, n) {
		let r = this.getLayer();
		if (r.hasListener(e)) {
			Mr(this.inversePixelTransform_, 0, 0, n.pixelRatio, -n.pixelRatio, 0, 0, -n.size[1]);
			let i = new Fo(e, this.inversePixelTransform_, n, t);
			r.dispatchEvent(i);
		}
	}
	preRender(e, t) {
		this.dispatchRenderEvent_(Io.PRERENDER, e, t);
	}
	postRender(e, t) {
		this.dispatchRenderEvent_(Io.POSTRENDER, e, t);
	}
}, Qp = {
	...Bp,
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
({ TEXTURE_COORD: "a_textureCoord" }).TEXTURE_COORD, Vp.FLOAT;
//#endregion
//#region node_modules/ol/webgl/PaletteTexture.js
var $p = class {
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
function em(e, t) {
	return `operator_${e}_${Object.keys(t.functions).length}`;
}
function tm(e) {
	let t = e.toString();
	return t.includes(".") ? t : t + ".0";
}
function nm(e) {
	if (e.length < 2 || e.length > 4) throw Error("`formatArray` can only output `vec2`, `vec3` or `vec4` arrays.");
	return `vec${e.length}(${e.map(tm).join(", ")})`;
}
function rm(e) {
	let t = No(e), n = t.length > 3 ? t[3] : 1;
	return nm([
		t[0] / 255,
		t[1] / 255,
		t[2] / 255,
		n
	]);
}
function im(e) {
	return nm(no(e));
}
var am = {}, om = 0;
function sm(e) {
	return e in am || (am[e] = om++), am[e];
}
function cm(e) {
	return tm(sm(e));
}
function lm(e) {
	return "u_var_" + e;
}
function um(e) {
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
var dm = "getBandValue", fm = "u_paletteTextures", pm = "featureId", mm = "geometryType", hm = -9999999;
function gm(e, t, n, r) {
	let i = Gl(e, t, n);
	return r.properties = new Map([...r.properties, ...n.properties]), r.variables = new Map([...r.variables, ...n.variables]), vm(i, t, r);
}
function Z(e) {
	return (t, n, r) => {
		let i = n.args.length, a = Array(i);
		for (let e = 0; e < i; ++e) a[e] = vm(n.args[e], r, t);
		return e(a, t);
	};
}
var _m = {
	[q.Get]: (e, t) => {
		let n = "a_prop_" + t.args[0].value;
		return Vl(t.type, jl) && (n = `(${n} > 0.0)`), n;
	},
	[q.Id]: (e) => (e.featureId = !0, "a_featureId"),
	[q.GeometryType]: (e) => (e.geometryType = !0, "a_geometryType"),
	[q.LineMetric]: () => "currentLineMetric",
	[q.Var]: (e, t) => {
		let n = t.args[0].value, r = lm(n);
		return Vl(t.type, jl) && (r = `(${r} > 0.0)`), r;
	},
	[q.Has]: (e, t) => `(a_prop_${t.args[0].value} != ${tm(hm)})`,
	[q.Resolution]: () => "u_resolution",
	[q.Zoom]: () => "u_zoom",
	[q.Time]: () => "u_time",
	[q.Any]: Z((e) => `(${e.join(" || ")})`),
	[q.All]: Z((e) => `(${e.join(" && ")})`),
	[q.Not]: Z(([e]) => `(!${e})`),
	[q.Equal]: Z(([e, t]) => `(${e} == ${t})`),
	[q.NotEqual]: Z(([e, t]) => `(${e} != ${t})`),
	[q.GreaterThan]: Z(([e, t]) => `(${e} > ${t})`),
	[q.GreaterThanOrEqualTo]: Z(([e, t]) => `(${e} >= ${t})`),
	[q.LessThan]: Z(([e, t]) => `(${e} < ${t})`),
	[q.LessThanOrEqualTo]: Z(([e, t]) => `(${e} <= ${t})`),
	[q.Multiply]: Z((e) => `(${e.join(" * ")})`),
	[q.Divide]: Z(([e, t]) => `(${e} / ${t})`),
	[q.Add]: Z((e) => `(${e.join(" + ")})`),
	[q.Subtract]: Z(([e, t]) => `(${e} - ${t})`),
	[q.Clamp]: Z(([e, t, n]) => `clamp(${e}, ${t}, ${n})`),
	[q.Mod]: Z(([e, t]) => `mod(${e}, ${t})`),
	[q.Pow]: Z(([e, t]) => `pow(${e}, ${t})`),
	[q.Abs]: Z(([e]) => `abs(${e})`),
	[q.Floor]: Z(([e]) => `floor(${e})`),
	[q.Ceil]: Z(([e]) => `ceil(${e})`),
	[q.Round]: Z(([e]) => `floor(${e} + 0.5)`),
	[q.Sin]: Z(([e]) => `sin(${e})`),
	[q.Cos]: Z(([e]) => `cos(${e})`),
	[q.Atan]: Z(([e, t]) => t === void 0 ? `atan(${e})` : `atan(${e}, ${t})`),
	[q.Sqrt]: Z(([e]) => `sqrt(${e})`),
	[q.Match]: Z((e) => {
		let t = e[0], n = e[e.length - 1], r = null;
		for (let i = e.length - 3; i >= 1; i -= 2) r = `(${t} == ${e[i]} ? ${e[i + 1]} : ${r || n})`;
		return r;
	}),
	[q.Between]: Z(([e, t, n]) => `(${e} >= ${t} && ${e} <= ${n})`),
	[q.Interpolate]: Z(([e, t, ...n]) => {
		let r = "";
		for (let i = 0; i < n.length - 2; i += 2) {
			let a = n[i], o = r || n[i + 1], s = n[i + 2], c = n[i + 3], l;
			l = e === tm(1) ? `(${t} - ${a}) / (${s} - ${a})` : `(pow(${e}, (${t} - ${a})) - 1.0) / (pow(${e}, (${s} - ${a})) - 1.0)`, r = `mix(${o}, ${c}, clamp(${l}, 0.0, 1.0))`;
		}
		return r;
	}),
	[q.Case]: Z((e) => {
		let t = e[e.length - 1], n = null;
		for (let r = e.length - 3; r >= 0; r -= 2) n = `(${e[r]} ? ${e[r + 1]} : ${n || t})`;
		return n;
	}),
	[q.In]: Z(([e, ...t], n) => {
		let r = em("in", n), i = [];
		for (let e = 0; e < t.length; e += 1) i.push(`  if (inputValue == ${t[e]}) { return true; }`);
		return n.functions[r] = `bool ${r}(float inputValue) {
${i.join("\n")}
  return false;
}`, `${r}(${e})`;
	}),
	[q.Array]: Z((e) => `vec${e.length}(${e.join(", ")})`),
	[q.Color]: Z((e) => {
		if (e.length === 1) return `vec4(vec3(${e[0]} / 255.0), 1.0)`;
		if (e.length === 2) return `vec4(vec3(${e[0]} / 255.0), ${e[1]})`;
		let t = e.slice(0, 3).map((e) => `${e} / 255.0`);
		if (e.length === 3) return `vec4(${t.join(", ")}, 1.0)`;
		let n = e[3];
		return `vec4(${t.join(", ")}, ${n})`;
	}),
	[q.Band]: Z(([e, t, n], r) => {
		if (!(dm in r.functions)) {
			let e = "", t = r.bandCount || 1;
			for (let n = 0; n < t; n++) {
				let r = Math.floor(n / 4), i = n % 4;
				n === t - 1 && i === 1 && (i = 3);
				let a = `${Qp.TILE_TEXTURE_ARRAY}[${r}]`;
				e += `  if (band == ${n + 1}.0) {
    return texture2D(${a}, v_textureCoord + vec2(dx, dy))[${i}];
  }
`;
			}
			r.functions[dm] = `float getBandValue(float band, float xOffset, float yOffset) {
  float dx = xOffset / ${Qp.TEXTURE_PIXEL_WIDTH};
  float dy = yOffset / ${Qp.TEXTURE_PIXEL_HEIGHT};
${e}
}`;
		}
		return `${dm}(${e}, ${t ?? "0.0"}, ${n ?? "0.0"})`;
	}),
	[q.Palette]: (e, t) => {
		let [n, ...r] = t.args, i = r.length, a = new Uint8Array(i * 4);
		for (let e = 0; e < r.length; e++) {
			let t = r[e].value, n = No(t), i = e * 4;
			a[i] = n[0], a[i + 1] = n[1], a[i + 2] = n[2], a[i + 3] = n[3] * 255;
		}
		e.paletteTextures ||= [];
		let o = `${fm}[${e.paletteTextures.length}]`, s = new $p(o, a);
		return e.paletteTextures.push(s), `texture2D(${o}, vec2((${vm(n, W, e)} + 0.5) / ${i}.0, 0.5))`;
	}
};
function vm(e, t, n) {
	if (e instanceof Ul) {
		let r = _m[e.operator];
		if (r === void 0) throw Error(`No compiler defined for this operator: ${JSON.stringify(e.operator)}`);
		return r(n, e, t);
	}
	if ((e.type & W) > 0) return tm(e.value);
	if ((e.type & jl) > 0) return e.value.toString();
	if ((e.type & G) > 0) return cm(e.value.toString());
	if ((e.type & K) > 0) return rm(e.value);
	if ((e.type & Ml) > 0) return nm(e.value);
	if ((e.type & Nl) > 0) return im(e.value);
	throw Error(`Unexpected expression ${e.value} (expected type ${Rl(t)})`);
}
//#endregion
//#region node_modules/ol/style/flat.js
function ym() {
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
var bm = .985;
//#endregion
//#region node_modules/ol/render/webgl/compileUtil.js
function Q(e, t, n, r) {
	return gm(t, n, r ?? Wl(e.inputVariables), e);
}
function xm(e) {
	let t = No(e), n = t[0] * 256, r = t[1], i = t[2] * 256, a = Math.round(t[3] * 255);
	return [n + r, i + a];
}
var Sm = "vec4 unpackColor(vec2 packedColor) {\n  return vec4(\n    min(floor(packedColor[0] / 256.0) / 255.0, 1.0),\n    min(mod(packedColor[0], 256.0) / 255.0, 1.0),\n    min(floor(packedColor[1] / 256.0) / 255.0, 1.0),\n    min(mod(packedColor[1], 256.0) / 255.0, 1.0)\n  );\n}";
function Cm(e) {
	return e === K || e === Nl ? 2 : e === Ml ? 4 : e === G ? 3 : 1;
}
function wm(e) {
	if (e === G) return "float";
	let t = Cm(e);
	return t > 1 ? `vec${t}` : "float";
}
function Tm(e, t) {
	for (let n of t.variables.entries()) {
		let [t, r] = n, i = lm(t), a = wm(r);
		r === K && (a = "vec4"), e.addUniform(i, a);
	}
	for (let n of t.properties.entries()) {
		let [t, r] = n, i = wm(r), a = `a_prop_${t}`;
		r === K ? e.addAttribute(a, i, `unpackColor(${a})`, "vec4") : e.addAttribute(a, i);
	}
	for (let n in t.functions) e.addVertexShaderFunction(t.functions[n]), e.addFragmentShaderFunction(t.functions[n]);
}
function Em(e, t) {
	let n = {};
	for (let r of e.variables.entries()) {
		let [e, i] = r, a = lm(e);
		n[a] = () => {
			let n = t[e];
			if (i === jl) return +!!n;
			if (i === K) {
				let e = [...No(n || "#eee")];
				return e[0] /= 255, e[1] /= 255, e[2] /= 255, e[3] ??= 1, e;
			}
			return i === G ? sm(n) : n;
		};
	}
	return n;
}
function Dm(e) {
	let t = {};
	for (let n of e.properties.entries()) {
		let [e, r] = n, i = (t) => {
			let n = t.get(e);
			return r === K ? xm([...No(n || "#eee")]) : r === jl ? +!!n : n;
		};
		t[`prop_${e}`] = {
			size: Cm(r),
			callback: i
		};
	}
	return t;
}
//#endregion
//#region node_modules/ol/render/webgl/float64Util.js
function Om(e) {
	return e - km(e);
}
function km(e) {
	return Math.fround(e);
}
//#endregion
//#region node_modules/ol/render/webgl/ShaderBuilder.js
var Am = `#ifdef GL_FRAGMENT_PRECISION_HIGH
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
${Sm}

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

`, jm = ym(), Mm = class {
	constructor() {
		this.uniforms_ = [], this.attributes_ = [], this.hasSymbol_ = !1, this.symbolSizeExpression_ = `vec2(${tm(jm["circle-radius"])} + ${tm(jm["circle-stroke-width"] * .5)})`, this.symbolRotationExpression_ = "0.0", this.symbolOffsetExpression_ = "vec2(0.0)", this.symbolColorExpression_ = rm(jm["circle-fill-color"]), this.texCoordExpression_ = "vec4(0.0, 0.0, 1.0, 1.0)", this.fragmentDiscardExpression_ = null, this.shapeDiscardExpression_ = null, this.symbolRotateWithView_ = !1, this.hasStroke_ = !1, this.strokeWidthExpression_ = tm(jm["stroke-width"]), this.strokeColorExpression_ = rm(jm["stroke-color"]), this.strokeOffsetExpression_ = "0.", this.strokeCapExpression_ = cm("round"), this.strokeJoinExpression_ = cm("round"), this.strokeMiterLimitExpression_ = "10.", this.strokeDistanceFieldExpression_ = "-1000.", this.strokePatternLengthExpression_ = null, this.hasFill_ = !1, this.fillColorExpression_ = rm(jm["fill-color"]), this.fillPatternSizeExpression_ = null, this.vertexShaderFunctions_ = [], this.fragmentShaderFunctions_ = [];
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
		return this.hasSymbol_ ? `${Am}
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
		return this.hasSymbol_ ? `${Am}
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
		return this.hasStroke_ ? `${Am}
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
  if (cos(angle) > ${bm} || isCap(angle)) {
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
		return this.hasStroke_ ? `${Am}
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
  if (cos(joinAngle) > ${bm}) { // avoid risking a division by zero
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
   if (capType == ${cm("butt")}) {
    return buttCapDistanceField(point, start, end);
  } else if (capType == ${cm("square")}) {
    return squareCapDistanceField(point, start, end, width);
  }
  return roundCapDistanceField(point, start, end, width);
}

float joinDistanceField(vec2 point, vec2 start, vec2 end, float width, float joinAngle, float joinType) {
  if (joinType == ${cm("bevel")}) {
    return bevelJoinField(point, start, end, width, joinAngle);
  } else if (joinType == ${cm("miter")}) {
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
		return this.hasFill_ ? `${Am}
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
		return this.hasFill_ ? `${Am}
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
}, Nm = class e {
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
		let t = O(e), n = this.pointBatch.entries[t];
		if (n) return this.pointBatch.geometriesCount -= n.flatCoordss.length, delete this.pointBatch.entries[t], n;
	}
	clearFeatureEntryInLineStringBatch_(e) {
		let t = O(e), n = this.lineStringBatch.entries[t];
		if (n) return this.lineStringBatch.verticesCount -= n.verticesCount, this.lineStringBatch.geometriesCount -= n.flatCoordss.length, delete this.lineStringBatch.entries[t], n;
	}
	clearFeatureEntryInPolygonBatch_(e) {
		let t = O(e), n = this.polygonBatch.entries[t];
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
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEndss(), t, O(t), r.getStride());
				break;
			}
			case "MultiLineString": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEnds(), t, O(t), r.getStride());
				break;
			}
			case "MultiPoint": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, O(t), r.getStride());
				break;
			}
			case "Polygon": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEnds(), t, O(t), r.getStride());
				break;
			}
			case "Point": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, O(t), r.getStride());
				break;
			}
			case "LineString":
			case "LinearRing": {
				let r = e, i = r.getStride();
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, O(t), i, r.getLayout?.());
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
				if (r instanceof ea) {
					let n = Ji(t, e);
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
				this.polygonBatch.verticesCount += s, this.polygonBatch.ringsCount += c, this.polygonBatch.geometriesCount++, this.polygonBatch.entries[i].flatCoordss.push(Pm(t, a)), this.polygonBatch.entries[i].ringsVerticesCounts.push(l), this.polygonBatch.entries[i].verticesCount += s, this.polygonBatch.entries[i].ringsCount += c;
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
			})), s = t.length / a, this.lineStringBatch.verticesCount += s, this.lineStringBatch.geometriesCount++, this.lineStringBatch.entries[i].flatCoordss.push(Fm(t, a, o)), this.lineStringBatch.entries[i].verticesCount += s;
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
		if (!this.uidToRef_.get(O(e))) return;
		this.removeFeature(e);
		let n = e.getGeometry();
		n && (t && (n = n.clone(), n.applyTransform(t)), this.addGeometry_(n, e));
	}
	removeFeature(e) {
		let t = this.clearFeatureEntryInPointBatch_(e);
		t = this.clearFeatureEntryInPolygonBatch_(e) || t, t = this.clearFeatureEntryInLineStringBatch_(e) || t, t && this.removeRef_(t.ref, O(t.feature));
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
function Pm(e, t) {
	return t === 2 ? e : e.filter((e, n) => n % t < 2);
}
function Fm(e, t, n) {
	return t === 3 && n === "XYM" ? e : t === 4 ? e.filter((e, n) => n % t !== 2) : t === 3 ? e.map((e, n) => n % t === 2 ? 0 : e) : Array(e.length * 1.5).fill(0).map((t, n) => n % 3 == 2 ? 0 : e[Math.round(n / 1.5)]);
}
//#endregion
//#region node_modules/ol/webgl/LabelsArray.js
var Im = new TextEncoder(), Lm = 1e5, Rm = class {
	constructor() {
		this.array_ = new Uint8Array(Lm), this.actualSize_ = 0, this.labelPositionMap_ = /* @__PURE__ */ new Map();
	}
	push(e) {
		if (e === "") return [0, 0];
		if (this.labelPositionMap_.has(e)) return this.labelPositionMap_.get(e);
		let t = Im.encode(e);
		if (this.actualSize_ + t.length > this.array_.length) {
			let e = new Uint8Array(this.array_.length + Lm);
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
function zm() {
	let e = "function t(t,e){return t>e?1:t<e?-1:0}function e(t,e,i){for(;e<i;){const n=t[e];t[e]=t[i],t[i]=n,++e,--i}}function i(t,e){const i=Array.isArray(e)?e:[e],n=i.length;for(let e=0;e<n;e++)t[t.length]=i[e]}function n(t,e){const i=t.length;if(i!==e.length)return!1;for(let n=0;n<i;n++)if(t[n]!==e[n])return!1;return!0}const r=\"undefined\"!=typeof navigator&&void 0!==navigator.userAgent?navigator.userAgent.toLowerCase():\"\";r.includes(\"safari\")&&!r.includes(\"chrom\")&&(r.includes(\"version/15.4\")||/cpu (os|iphone os) 15_4 like mac os x/.test(r)),r.includes(\"webkit\")&&r.includes(\"edge\"),r.includes(\"macintosh\");const s=\"undefined\"!=typeof WorkerGlobalScope&&\"undefined\"!=typeof OffscreenCanvas&&self instanceof WorkerGlobalScope,o=\"undefined\"!=typeof Image&&Image.prototype.decode;function a(t,e,i,n){let r;return r=s?new class extends OffscreenCanvas{style={}}(t??300,e??150):document.createElement(\"canvas\"),t&&(r.width=t),e&&(r.height=e),r.getContext(\"2d\",n)}let l;function h(){return l||(l=a(1,1)),l}function c(t,e,i){return Math.min(Math.max(t,e),i)}function u(t,e,i,n,r,s){const o=r-i,a=s-n;if(0!==o||0!==a){const l=((t-i)*o+(e-n)*a)/(o*o+a*a);l>1?(i=r,n=s):l>0&&(i+=o*l,n+=a*l)}return f(t,e,i,n)}function f(t,e,i,n){const r=i-t,s=n-e;return r*r+s*s}function d(t){return 180*t/Math.PI}function g(t){return t*Math.PI/180}function p(t,e,i){return t+i*(e-t)}function _(t,e,i){if(t>=e&&t<i)return t;const n=i-e;return((t-e)%n+n)%n+e}!function(){let t=!1;try{const e=Object.defineProperty({},\"passive\",{get:function(){t=!0}});window.addEventListener(\"_\",null,e),window.removeEventListener(\"_\",null,e)}catch{}}();const m=[NaN,NaN,NaN,0];let y;const w=/^rgba?\\(\\s*(\\d+%?)\\s+(\\d+%?)\\s+(\\d+%?)(?:\\s*\\/\\s*(\\d+%|\\d*\\.\\d+|[01]))?\\s*\\)$/i,x=/^rgba?\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)(?:\\s*,\\s*(\\d+%|\\d*\\.\\d+|[01]))?\\s*\\)$/i,v=/^rgba?\\(\\s*(\\d+%)\\s*,\\s*(\\d+%)\\s*,\\s*(\\d+%)(?:\\s*,\\s*(\\d+%|\\d*\\.\\d+|[01]))?\\s*\\)$/i,S=/^#([\\da-f]{3,4}|[\\da-f]{6}|[\\da-f]{8})$/i;function C(t,e){return t.endsWith(\"%\")?Number(t.substring(0,t.length-1))/e:Number(t)}function b(t){throw new Error('failed to parse \"'+t+'\" as color')}function M(t){if(t.toLowerCase().startsWith(\"rgb\")){const e=t.match(x)||t.match(w)||t.match(v);if(e){const t=e[4],i=100/255;return[c(C(e[1],i)+.5|0,0,255),c(C(e[2],i)+.5|0,0,255),c(C(e[3],i)+.5|0,0,255),void 0!==t?c(C(t,100),0,1):1]}b(t)}if(t.startsWith(\"#\")){if(S.test(t)){const e=t.substring(1),i=e.length<=4?1:2,n=[0,0,0,255];for(let t=0,r=e.length;t<r;t+=i){let r=parseInt(e.substring(t,t+i),16);1===i&&(r+=r<<4),n[t/i]=r}return n[3]=n[3]/255,n}b(t)}const e=(y||(y=a(1,1,0,{willReadFrequently:!0,desynchronized:!0})),y);e.fillStyle=\"#abcdef\";let i=e.fillStyle;e.fillStyle=t,e.fillStyle===i&&(e.fillStyle=\"#fedcba\",i=e.fillStyle,e.fillStyle=t,e.fillStyle===i&&b(t));const n=e.fillStyle;if(n.startsWith(\"#\")||n.startsWith(\"rgba\"))return M(n);e.clearRect(0,0,1,1),e.fillRect(0,0,1,1);const r=Array.from(e.getImageData(0,0,1,1).data);return r[3]=function(t,e){const i=Math.pow(10,e);return Math.round(t*i)/i}(r[3]/255,3),r}const I={};let E=0;function k(t){if(4===t.length)return t;const e=t.slice();return e[3]=1,e}function A(t){return t>.0031308?269.025*Math.pow(t,1/2.4)-14.025:3294.6*t}function P(t){return t>.2068965?Math.pow(t,3):108/841*(t-4/29)}function O(t){return t>10.314724?Math.pow((t+14.025)/269.025,2.4):t/3294.6}function R(t){return t>.0088564?Math.pow(t,1/3):t/(108/841)+4/29}function L(t){const e=O(t[0]),i=O(t[1]),n=O(t[2]),r=R(.222488403*e+.716873169*i+.06060791*n),s=500*(R(.452247074*e+.399439023*i+.148375274*n)-r),o=200*(r-R(.016863605*e+.117638439*i+.865350722*n)),a=Math.atan2(o,s)*(180/Math.PI);return[116*r-16,Math.sqrt(s*s+o*o),a<0?a+360:a,t[3]]}function D(t){if(\"none\"===t)return m;if(I.hasOwnProperty(t))return I[t];if(E>=1024){let t=0;for(const e in I)3&t++||(delete I[e],--E)}const e=M(t);4!==e.length&&b(t);for(const i of e)isNaN(i)&&b(t);return I[t]=e,++E,e}function F(t){return Array.isArray(t)?t:D(t)}function T(t){let e=t[0];e!=(0|e)&&(e=e+.5|0);let i=t[1];i!=(0|i)&&(i=i+.5|0);let n=t[2];n!=(0|n)&&(n=n+.5|0);return\"rgba(\"+e+\",\"+i+\",\"+n+\",\"+(void 0===t[3]?1:Math.round(1e3*t[3])/1e3)+\")\"}function z(t,e){return Array.isArray(t)?t:(void 0===e?e=[t,t]:(e[0]=t,e[1]=t),e)}let $=0;const W=1<<$++,G=1<<$++,N=1<<$++,X=1<<$++,Y=1<<$++,B=1<<$++,U=Math.pow(2,6)-1,j={[W]:\"boolean\",[G]:\"number\",[N]:\"string\",[X]:\"color\",[Y]:\"number[]\",[B]:\"size\"},V=Object.keys(j).map(Number).sort(t);function q(t){const e=[];for(const i of V)J(t,i)&&e.push(j[i]);return 0===e.length?\"untyped\":e.length<3?e.join(\" or \"):e.slice(0,-1).join(\", \")+\", or \"+e[e.length-1]}function J(t,e){return(t&e)===e}function K(t,e){return!!(t&e)}function H(t,e){return t===e}class Z{constructor(t,e){if(!function(t){return t in j}(t))throw new Error(`literal expressions must have a specific type, got ${q(t)}`);this.type=t,this.value=e}}class Q{constructor(t,e,...i){this.type=t,this.operator=e,this.args=i}}function tt(t){return{variables:new Map,properties:new Map,featureId:!1,geometryType:!1,mCoordinate:!1,mapState:!1,inputVariables:t}}function et(t,e,i){switch(typeof t){case\"boolean\":if(H(e,N))return new Z(N,t?\"true\":\"false\");if(!J(e,W))throw new Error(`got a boolean, but expected ${q(e)}`);return new Z(W,t);case\"number\":if(H(e,B))return new Z(B,z(t));if(H(e,W))return new Z(W,!!t);if(H(e,N))return new Z(N,t.toString());if(!J(e,G))throw new Error(`got a number, but expected ${q(e)}`);return new Z(G,t);case\"string\":if(H(e,X))return new Z(X,D(t));if(H(e,W))return new Z(W,!!t);if(!J(e,N))throw new Error(`got a string, but expected ${q(e)}`);return new Z(N,t)}if(!Array.isArray(t))throw new Error(\"expression must be an array or a primitive value\");if(0===t.length)throw new Error(\"empty expression\");if(\"string\"==typeof t[0])return function(t,e,i){const n=t[0],r=Jt[n];if(!r)throw new Error(`unknown operator: ${n}`);return r(t,e,i)}(t,e,i);for(const e of t)if(\"number\"!=typeof e)throw new Error(\"expected an array of numbers\");if(H(e,B)){if(2!==t.length)throw new Error(`expected an array of two values for a size, got ${t.length}`);return new Z(B,t)}if(H(e,X)){if(3===t.length)return new Z(X,[...t,1]);if(4===t.length)return new Z(X,t);throw new Error(`expected an array of 3 or 4 values for a color, got ${t.length}`)}if(!J(e,Y))throw new Error(`got an array of numbers, but expected ${q(e)}`);return new Z(Y,t)}const it=\"get\",nt=\"var\",rt=\"concat\",st=\"geometry-type\",ot=\"line-metric\",at=\"any\",lt=\"all\",ht=\"!\",ct=\"resolution\",ut=\"zoom\",ft=\"time\",dt=\"==\",gt=\"!=\",pt=\">\",_t=\">=\",mt=\"<\",yt=\"<=\",wt=\"*\",xt=\"/\",vt=\"+\",St=\"-\",Ct=\"clamp\",bt=\"%\",Mt=\"^\",It=\"abs\",Et=\"floor\",kt=\"ceil\",At=\"round\",Pt=\"sin\",Ot=\"cos\",Rt=\"atan\",Lt=\"sqrt\",Dt=\"match\",Ft=\"between\",Tt=\"interpolate\",zt=\"coalesce\",$t=\"case\",Wt=\"in\",Gt=\"number\",Nt=\"string\",Xt=\"array\",Yt=\"color\",Bt=\"id\",Ut=\"band\",jt=\"palette\",Vt=\"to-string\",qt=\"has\",Jt={[it]:re(Qt(1,1/0),Kt),[nt]:function(t,e,i){const n=t[1];if(\"string\"!=typeof n)throw new Error(\"expected a string argument for var operation\");let r=e;const s=i.inputVariables?.[n];if(void 0!==s){const t=et(s,U,i);if(!(t instanceof Z))throw new Error(`style variables should only be literal values (no expressions!), variable name: ${n}`);let o=t.type;if(\"string\"==typeof s&&K(r,X)&&!K(r,N)?o=X:Array.isArray(s)&&2===s.length&&K(r,B)&&!K(r,Y)&&(o=B),r&=o,0===r)throw new Error(`the type expected from the var operator (${q(e)}) did not have any overlap with the type of the corresponding style variables (${q(o)}), variable name: ${n}`)}if(i.variables.has(n)){const t=i.variables.get(n);if(r&=t,0===r)throw new Error(`a new type expected from the var operator (${q(e)}) did not have any overlap with the previous type expected for it (${q(t)}), variable name: ${n}`)}return i.variables.set(n,r),new Q(r,\"var\",new Z(N,n))},[qt]:re(Qt(1,1/0),Kt),[Bt]:re(function(t,e,i){i.featureId=!0},Zt),[rt]:re(Qt(2,1/0),ee(N)),[st]:re(function(t,e,i){i.geometryType=!0},Zt),[ot]:re(function(t,e,i){i.mCoordinate=!0},Zt),[ct]:re(Ht,Zt),[ut]:re(Ht,Zt),[ft]:re(Ht,Zt),[at]:re(Qt(2,1/0),ee(W)),[lt]:re(Qt(2,1/0),ee(W)),[ht]:re(Qt(1,1),ee(W)),[dt]:re(Qt(2,2),ie()),[gt]:re(Qt(2,2),ie()),[pt]:re(Qt(2,2),ee(G)),[_t]:re(Qt(2,2),ee(G)),[mt]:re(Qt(2,2),ee(G)),[yt]:re(Qt(2,2),ee(G)),[wt]:re(Qt(2,1/0),te),[zt]:re(Qt(2,1/0),te),[xt]:re(Qt(2,2),ee(G)),[vt]:re(Qt(2,1/0),ee(G)),[St]:re(Qt(2,2),ee(G)),[Ct]:re(Qt(3,3),ee(G)),[bt]:re(Qt(2,2),ee(G)),[Mt]:re(Qt(2,2),ee(G)),[It]:re(Qt(1,1),ee(G)),[Et]:re(Qt(1,1),ee(G)),[kt]:re(Qt(1,1),ee(G)),[At]:re(Qt(1,1),ee(G)),[Pt]:re(Qt(1,1),ee(G)),[Ot]:re(Qt(1,1),ee(G)),[Rt]:re(Qt(1,2),ee(G)),[Lt]:re(Qt(1,1),ee(G)),[Dt]:re(Qt(4,1/0),ne,function(t,e,i){const n=t.length-1,r=et(t[t.length-1],e,i);let s=N|G|W;const o=new Array(n-2);for(let e=0;e<n-2;e+=2){try{s&=et(t[e+2],s,i).type}catch(t){throw new Error(`failed to parse argument ${e+1} of match expression: ${t.message}`)}if(0===s)throw new Error(\"no common type was found among the arguments of match expression\")}for(let e=0;e<n-2;e+=2){try{const n=et(t[e+2],s,i);o[e]=n}catch(t){throw new Error(`failed to parse argument ${e+1} of match expression: ${t.message}`)}try{const n=et(t[e+3],r.type,i);o[e+1]=n}catch(t){throw new Error(`failed to parse argument ${e+2} of match expression: ${t.message}`)}}const a=et(t[1],s,i);return[a,...o,r]}),[Ft]:re(Qt(3,3),ee(G)),[Tt]:re(Qt(6,1/0),ne,function(t,e,i){const n=t[1];let r;switch(n[0]){case\"linear\":r=1;break;case\"exponential\":const t=n[1];if(\"number\"!=typeof t||t<=0)throw new Error(`expected a number base for exponential interpolation, got ${JSON.stringify(t)} instead`);r=t;break;default:throw new Error(`invalid interpolation type: ${JSON.stringify(n)}`)}const s=new Z(G,r);let o;try{o=et(t[2],G,i)}catch(t){throw new Error(`failed to parse argument 1 in interpolate expression: ${t.message}`)}const a=new Array(t.length-3);for(let n=0;n<a.length;n+=2){try{const e=et(t[n+3],G,i);a[n]=e}catch(t){throw new Error(`failed to parse argument ${n+2} for interpolate expression: ${t.message}`)}try{const r=et(t[n+4],e,i);a[n+1]=r}catch(t){throw new Error(`failed to parse argument ${n+3} for interpolate expression: ${t.message}`)}}return[s,o,...a]}),[$t]:re(Qt(3,1/0),function(t,e,i){const n=t[0],r=t.length-1;if(r%2==0)throw new Error(`expected an odd number of arguments for ${n}, got ${r} instead`)},function(t,e,i){const n=et(t[t.length-1],e,i),r=new Array(t.length-1);for(let e=0;e<r.length-1;e+=2){try{const n=et(t[e+1],W,i);r[e]=n}catch(t){throw new Error(`failed to parse argument ${e} of case expression: ${t.message}`)}try{const s=et(t[e+2],n.type,i);r[e+1]=s}catch(t){throw new Error(`failed to parse argument ${e+1} of case expression: ${t.message}`)}}return r[r.length-1]=n,r}),[Wt]:re(Qt(2,2),function(t,e,i){let n,r=t[2];if(!Array.isArray(r))throw new Error('the second argument for the \"in\" operator must be an array');if(\"literal\"===r[0]){if(r=r[1],!Array.isArray(r))throw new Error('failed to parse \"in\" expression: the literal operator must be followed by an array')}else if(\"string\"==typeof r[0])throw new Error('for the \"in\" operator, a string array should be wrapped in a \"literal\" operator to disambiguate from expressions');n=\"string\"==typeof r[0]?N:G;const s=new Array(r.length);for(let t=0;t<s.length;t++)try{const e=et(r[t],n,i);s[t]=e}catch(e){throw new Error(`failed to parse haystack item ${t} for \"in\" expression: ${e.message}`)}const o=et(t[1],n,i);return[o,...s]}),[Gt]:re(Qt(1,1/0),ee(U)),[Nt]:re(Qt(1,1/0),ee(U)),[Xt]:re(Qt(1,1/0),ee(G)),[Yt]:re(Qt(1,4),ee(G)),[Ut]:re(Qt(1,3),ee(G)),[jt]:re(Qt(2,2),function(t,e,i){let n;try{n=et(t[1],G,i)}catch(t){throw new Error(`failed to parse first argument in palette expression: ${t.message}`)}const r=t[2];if(!Array.isArray(r))throw new Error(\"the second argument of palette must be an array\");const s=new Array(r.length);for(let t=0;t<s.length;t++){let e;try{e=et(r[t],X,i)}catch(e){throw new Error(`failed to parse color at index ${t} in palette expression: ${e.message}`)}if(!(e instanceof Z))throw new Error(`the palette color at index ${t} must be a literal value`);s[t]=e}return[n,...s]}),[Vt]:re(Qt(1,1),ee(W|G|N|X))};function Kt(t,e,i){const n=t.length-1,r=new Array(n);for(let s=0;s<n;++s){const n=t[s+1];switch(typeof n){case\"number\":r[s]=new Z(G,n);break;case\"string\":r[s]=new Z(N,n);break;default:throw new Error(`expected a string key or numeric array index for a get operation, got ${n}`)}0===s&&i.properties.set(String(n),e)}return r}function Ht(t,e,i){i.mapState=!0}function Zt(t,e,i){const n=t[0];if(1!==t.length)throw new Error(`expected no arguments for ${n} operation`);return[]}function Qt(t,e){return function(i,n,r){const s=i[0],o=i.length-1;if(t===e){if(o!==t){throw new Error(`expected ${t} argument${1===t?\"\":\"s\"} for ${s}, got ${o}`)}}else if(o<t||o>e){throw new Error(`expected ${e===1/0?`${t} or more`:`${t} to ${e}`} arguments for ${s}, got ${o}`)}}}function te(t,e,i){const n=t.length-1,r=new Array(n);for(let s=0;s<n;++s){const n=et(t[s+1],e,i);r[s]=n}return r}function ee(t){return function(e,i,n){const r=e.length-1,s=new Array(r);for(let i=0;i<r;++i){const r=et(e[i+1],t,n);s[i]=r}return s}}function ie(){return function(t,e,i){const n=t[0],r=t.length-1,s=new Array(r);let o=U;for(let e=0;e<r;++e){o&=et(t[e+1],o,i).type}if(0===o)throw new Error(`no common type was found among the arguments of ${n}`);for(let e=0;e<r;++e){const n=et(t[e+1],o,i);s[e]=n}return s}}function ne(t,e,i){const n=t[0],r=t.length-1;if(r%2==1)throw new Error(`expected an even number of arguments for operation ${n}, got ${r} instead`)}function re(...t){return function(e,i,n){const r=e[0];let s;for(let r=0;r<t.length;r++){const o=t[r](e,i,n);if(r==t.length-1){if(!o)throw new Error(\"expected last argument validator to return the parsed args\");s=o}}return new Q(i,r,...s)}}function se(t){if(!t)return\"\";const e=t.getType();switch(e){case\"Point\":case\"LineString\":case\"Polygon\":return e;case\"MultiPoint\":case\"MultiLineString\":case\"MultiPolygon\":return e.substring(5);case\"Circle\":return\"Polygon\";case\"GeometryCollection\":return se(t.getGeometries()[0]);default:return\"\"}}var oe=0,ae=1,le=2,he=4,ce=8,ue=16;function fe(t,e,i){let n,r;return n=e<t[0]?t[0]-e:t[2]<e?e-t[2]:0,r=i<t[1]?t[1]-i:t[3]<i?i-t[3]:0,n*n+r*r}function de(t,e){return ge(t,e[0],e[1])}function ge(t,e,i){return t[0]<=e&&e<=t[2]&&t[1]<=i&&i<=t[3]}function pe(t,e){const i=t[0],n=t[1],r=t[2],s=t[3],o=e[0],a=e[1];let l=oe;return o<i?l|=ue:o>r&&(l|=he),a<n?l|=ce:a>s&&(l|=le),l===oe&&(l=ae),l}function _e(t,e,i,n,r){return r?(r[0]=t,r[1]=e,r[2]=i,r[3]=n,r):[t,e,i,n]}function me(t){return _e(1/0,1/0,-1/0,-1/0,t)}function ye(t,e){const i=t[0],n=t[1];return _e(i,n,i,n,e)}function we(t,e,i,n,r){return xe(me(r),t,e,i,n)}function xe(t,e,i,n,r){for(;i<n;i+=r)ve(t,e[i],e[i+1]);return t}function ve(t,e,i){t[0]=Math.min(t[0],e),t[1]=Math.min(t[1],i),t[2]=Math.max(t[2],e),t[3]=Math.max(t[3],i)}function Se(t){return[(t[0]+t[2])/2,(t[1]+t[3])/2]}function Ce(t){return t[3]-t[1]}function be(t,e){return t[0]<=e[2]&&t[2]>=e[0]&&t[1]<=e[3]&&t[3]>=e[1]}function Me(t,e,i,n){let r=t[e],s=t[e+1],o=0;for(let a=e+n;a<i;a+=n){const e=t[a],i=t[a+1];o+=Math.sqrt((e-r)*(e-r)+(i-s)*(i-s)),r=e,s=i}return o}function Ie(t,e,i,n,r,s,o,a){o=o??[],a=a??n;const l=t[e+n],h=t[e+n+1],c=t[i-2*n],u=t[i-2*n+1];let f,d,g,p,_,m,y,w,x=0;for(let v=e;v<i;v+=n){g=f,p=d,_=void 0,m=void 0,v+n<i&&(_=t[v+n],m=t[v+n+1]),s&&v===e&&(g=c,p=u),s&&v===i-n&&(_=l,m=h),f=t[v],d=t[v+1],[y,w]=Ee(f,d,g,p,_,m,r),o[x++]=y,o[x++]=w;for(let e=2;e<a;e++)o[x++]=t[v+e]}return o.length!=x&&(o.length=x),o}function Ee(t,e,i,n,r,s,o){let a,l;void 0!==i&&void 0!==n?(a=t-i,l=e-n):void 0!==r&&void 0!==s?(a=r-t,l=s-e):(a=1,l=0);const h=Math.hypot(a,l),u=a/h,f=l/h;if(a=-f,l=u,void 0===i||void 0===n)return[t+a*o,e+l*o];if(void 0===r||void 0===s)return[t+a*o,e+l*o];const d=function(t,e,i){const n=Math.sqrt((e[0]-t[0])*(e[0]-t[0])+(e[1]-t[1])*(e[1]-t[1])),r=[(e[0]-t[0])/n,(e[1]-t[1])/n],s=[-r[1],r[0]],o=Math.sqrt((i[0]-t[0])*(i[0]-t[0])+(i[1]-t[1])*(i[1]-t[1])),a=[(i[0]-t[0])/o,(i[1]-t[1])/o];let l=0===n||0===o?0:Math.acos(c(a[0]*r[0]+a[1]*r[1],-1,1));return l=Math.max(l,1e-5),a[0]*s[0]+a[1]*s[1]>0?l:2*Math.PI-l}([t,e],[i,n],[r,s]);if(Math.cos(d)>.998)return[t+u*o,e+f*o];const g=Math.cos(d/2),p=Math.sin(d/2);return[t+(p*a+g*l)*(1/p)*o,e+(-g*a+p*l)*(1/p)*o]}function ke(t,e,i=!1){for(let n=0,r=t.length-2;n<r;n+=e){for(let r=i&&0===n?t.length-3*e:t.length-2*e;r>n+e;r-=e){const i=t[n],s=t[n+1],o=t[n+e],a=t[n+e+1],l=t[r],h=t[r+1],c=t[r+e],u=t[r+e+1],f=(u-h)*(o-i)-(c-l)*(a-s);if(0===f)continue;const d=((c-l)*(s-h)-(u-h)*(i-l))/f,g=((o-i)*(s-h)-(a-s)*(i-l))/f;if(d>0&&d<1&&g>0&&g<1){const l=i+d*(o-i),h=s+d*(a-s);t[n+e]=l,t[n+e+1]=h,t.splice(n+2*e,r-n-e);break}}}return t}function Ae(t,e,i,n,r,s,o){s=s||[],o=o||2;let a=0;for(let l=e;l<i;l+=n){const e=t[l],i=t[l+1];s[a++]=r[0]*e+r[2]*i+r[4],s[a++]=r[1]*e+r[3]*i+r[5];for(let e=2;e<o;e++)s[a++]=t[l+e]}return s&&s.length!=a&&(s.length=a),s}function Pe(t,e,i,n,r,s,o){o=o||[];const a=Math.cos(r),l=Math.sin(r),h=s[0],c=s[1];let u=0;for(let r=e;r<i;r+=n){const e=t[r]-h,i=t[r+1]-c;o[u++]=h+e*a-i*l,o[u++]=c+e*l+i*a;for(let e=r+2;e<r+n;++e)o[u++]=t[e]}return o&&o.length!=u&&(o.length=u),o}let Oe;function Re(t,e,i,n,r,s,o,a,l,h,c,u,f=!0){let d=t[e],g=t[e+1],_=0,m=0,y=0,w=0;function x(){_=d,m=g,d=t[e+=n],g=t[e+1],w+=y,y=Math.sqrt((d-_)*(d-_)+(g-m)*(g-m))}do{x()}while(e<i-n&&w+y<s);let v=0===y?0:(s-w)/y;const S=p(_,d,v),C=p(m,g,v),b=e-n,M=w,I=s+a*l(h,r,c);for(;e<i-n&&w+y<I;)x();v=0===y?0:(I-w)/y;const E=p(_,d,v),k=p(m,g,v);let A=!1;if(f)if(u){const t=[S,C,E,k];Pe(t,0,4,2,u,t,t),A=t[0]>t[2]}else A=S>E;const P=Math.PI,O=[],R=b+n===e;let L;if(y=0,w=M,d=t[e=b],g=t[e+1],R){x(),L=Math.atan2(g-m,d-_),A&&(L+=L>0?-P:P);const t=(E+S)/2,e=(k+C)/2;return O[0]=[t,e,(I-s)/2,L,r],O}r=r.replace(/\\n/g,\" \");const D=Array.from((Oe||(Oe=new Intl.Segmenter(void 0,{granularity:\"grapheme\"})),Oe).segment(r),t=>t.segment);for(let t=0,r=D.length;t<r;){x();let u=Math.atan2(g-m,d-_);if(A&&(u+=u>0?-P:P),void 0!==L){let t=u-L;if(t+=t>P?-2*P:t<-P?2*P:0,Math.abs(t)>o)return null}L=u;const f=t;let S=0;for(;t<r;++t){const o=a*l(h,D[A?r-t-1:t],c);if(e+n<i&&w+y<s+S+o/2)break;S+=o}if(t===f)continue;const C=(A?D.slice(r-t,r-f):D.slice(f,t)).join(\"\");v=0===y?0:(s+S/2-w)/y;const b=p(_,d,v),M=p(m,g,v);O.push([b,M,S/2,u,C]),s+=S}return O}function Le(t,e){if(!t)throw new Error(e)}const De=[1,0,0,1,0,0];function Fe(){return De.slice(0)}function Te(t,e){const i=t[0],n=t[1],r=t[2],s=t[3],o=t[4],a=t[5],l=e[0],h=e[1],c=e[2],u=e[3],f=e[4],d=e[5];return t[0]=i*l+r*h,t[1]=n*l+s*h,t[2]=i*c+r*u,t[3]=n*c+s*u,t[4]=i*f+r*d+o,t[5]=n*f+s*d+a,t}function ze(t,e){const i=e[0],n=e[1];return e[0]=t[0]*i+t[2]*n+t[4],e[1]=t[1]*i+t[3]*n+t[5],e}function $e(t,e,i,n,r,s,o,a){const l=Math.sin(s),h=Math.cos(s);return t[0]=n*h,t[1]=r*l,t[2]=-n*l,t[3]=r*h,t[4]=o*n*h-a*n*l+e,t[5]=o*r*l+a*r*h+i,t}function We(t){return function(t,e){const i=(n=e,n[0]*n[3]-n[1]*n[2]);var n;Le(0!==i,\"Transformation matrix cannot be inverted\");const r=e[0],s=e[1],o=e[2],a=e[3],l=e[4],h=e[5];return t[0]=a/i,t[1]=-s/i,t[2]=-o/i,t[3]=r/i,t[4]=(o*h-a*l)/i,t[5]=-(r*h-s*l)/i,t}(t,t)}new Array(6);var Ge=\"propertychange\";function Ne(t){for(const e in t)delete t[e]}function Xe(t){let e;for(e in t)return!1;return!e}function Ye(t,e,i,n,r){if(r){const n=i;i=function(r){return t.removeEventListener(e,i),n.call(this,r)}}const s={target:t,type:e,listener:i};return t.addEventListener(e,i),s}function Be(t,e,i,n){return Ye(t,e,i,0,!0)}function Ue(t){t&&t.target&&(t.target.removeEventListener(t.type,t.listener),Ne(t))}var je=\"change\";class Ve{constructor(){this.disposed=!1}dispose(){this.disposed||(this.disposed=!0,this.disposeInternal())}disposeInternal(){}}function qe(){}function Je(t){let e,i,r;return function(){const s=Array.prototype.slice.call(arguments);return i&&this===r&&n(s,i)||(r=this,i=s,e=t.apply(this,arguments)),e}}class Ke{constructor(t){this.propagationStopped,this.defaultPrevented,this.type=t,this.target=null}preventDefault(){this.defaultPrevented=!0}stopPropagation(){this.propagationStopped=!0}}class He extends Ve{constructor(t){super(),this.eventTarget_=t,this.pendingRemovals_=null,this.dispatching_=null,this.listeners_=null}addEventListener(t,e){if(!t||!e)return;const i=this.listeners_||(this.listeners_={}),n=i[t]||(i[t]=[]);n.includes(e)||n.push(e)}dispatchEvent(t){const e=\"string\"==typeof t,i=e?t:t.type,n=this.listeners_&&this.listeners_[i];if(!n)return;const r=e?new Ke(t):t;r.target||(r.target=this.eventTarget_||this);const s=this.dispatching_||(this.dispatching_={}),o=this.pendingRemovals_||(this.pendingRemovals_={});let a;i in s||(s[i]=0,o[i]=0),++s[i];for(let t=0,e=n.length;t<e;++t)if(a=\"handleEvent\"in n[t]?n[t].handleEvent(r):n[t].call(this,r),!1===a||r.propagationStopped){a=!1;break}if(0===--s[i]){let t=o[i];for(delete o[i];t--;)this.removeEventListener(i,qe);delete s[i]}return a}disposeInternal(){this.listeners_&&Ne(this.listeners_)}getListeners(t){return this.listeners_&&this.listeners_[t]||void 0}hasListener(t){return!!this.listeners_&&(t?t in this.listeners_:Object.keys(this.listeners_).length>0)}removeEventListener(t,e){if(!this.listeners_)return;const i=this.listeners_[t];if(!i)return;const n=i.indexOf(e);-1!==n&&(this.pendingRemovals_&&t in this.pendingRemovals_?(i[n]=qe,++this.pendingRemovals_[t]):(i.splice(n,1),0===i.length&&delete this.listeners_[t]))}}class Ze extends He{constructor(){super(),this.on=this.onInternal,this.once=this.onceInternal,this.un=this.unInternal,this.revision_=0}changed(){++this.revision_,this.dispatchEvent(je)}getRevision(){return this.revision_}onInternal(t,e){if(Array.isArray(t)){const i=t.length,n=new Array(i);for(let r=0;r<i;++r)n[r]=Ye(this,t[r],e);return n}return Ye(this,t,e)}onceInternal(t,e){let i;if(Array.isArray(t)){const n=t.length;i=new Array(n);for(let r=0;r<n;++r)i[r]=Be(this,t[r],e)}else i=Be(this,t,e);return e.ol_key=i,i}unInternal(t,e){const i=e.ol_key;if(i)!function(t){if(Array.isArray(t))for(let e=0,i=t.length;e<i;++e)Ue(t[e]);else Ue(t)}(i);else if(Array.isArray(t))for(let i=0,n=t.length;i<n;++i)this.removeEventListener(t[i],e);else this.removeEventListener(t,e)}}function Qe(){throw new Error(\"Unimplemented abstract method.\")}let ti=0;function ei(t){return t.ol_uid||(t.ol_uid=String(++ti))}class ii extends Ke{constructor(t,e,i){super(t),this.key=e,this.oldValue=i}}class ni extends Ze{constructor(t){super(),this.on,this.once,this.un,ei(this),this.values_=null,void 0!==t&&this.setProperties(t)}get(t){let e;return this.values_&&this.values_.hasOwnProperty(t)&&(e=this.values_[t]),e}getKeys(){return this.values_&&Object.keys(this.values_)||[]}getProperties(){return this.values_&&Object.assign({},this.values_)||{}}getPropertiesInternal(){return this.values_}hasProperties(){return!!this.values_}notify(t,e){let i;i=`change:${t}`,this.hasListener(i)&&this.dispatchEvent(new ii(i,t,e)),i=Ge,this.hasListener(i)&&this.dispatchEvent(new ii(i,t,e))}addChangeListener(t,e){this.addEventListener(`change:${t}`,e)}removeChangeListener(t,e){this.removeEventListener(`change:${t}`,e)}set(t,e,i){const n=this.values_||(this.values_={});if(i)n[t]=e;else{const i=n[t];n[t]=e,i!==e&&this.notify(t,i)}}setProperties(t,e){for(const i in t)this.set(i,t[i],e)}applyProperties(t){t.values_&&Object.assign(this.values_||(this.values_={}),t.values_)}unset(t,e){if(this.values_&&t in this.values_){const i=this.values_[t];delete this.values_[t],Xe(this.values_)&&(this.values_=null),e||this.notify(t,i)}}}const ri=new RegExp([\"^\\\\s*(?=(?:(?:[-a-z]+\\\\s*){0,2}(italic|oblique))?)\",\"(?=(?:(?:[-a-z]+\\\\s*){0,2}(small-caps))?)\",\"(?=(?:(?:[-a-z]+\\\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)\",\"(?:(?:normal|\\\\1|\\\\2|\\\\3)\\\\s*){0,3}((?:xx?-)?\",\"(?:small|large)|medium|smaller|larger|[\\\\.\\\\d]+(?:\\\\%|in|[cem]m|ex|p[ctx]))\",\"(?:\\\\s*\\\\/\\\\s*(normal|[\\\\.\\\\d]+(?:\\\\%|in|[cem]m|ex|p[ctx])?))\",\"?\\\\s*([-,\\\\\\\"\\\\'\\\\sa-z0-9]+?)\\\\s*$\"].join(\"\"),\"i\"),si=[\"style\",\"variant\",\"weight\",\"size\",\"lineHeight\",\"family\"],oi={normal:400,bold:700},ai=function(t){const e=t.match(ri);if(!e)return null;const i={lineHeight:\"normal\",size:\"1.2em\",style:\"normal\",weight:\"400\",variant:\"normal\"};for(let t=0,n=si.length;t<n;++t){const n=e[t+1];void 0!==n&&(i[si[t]]=\"string\"==typeof n?n.trim():n)}return isNaN(Number(i.weight))&&i.weight in oi&&(i.weight=oi[i.weight]),i.families=i.family.split(/,\\s?/).map(t=>t.trim().replace(/^['\"]|['\"]$/g,\"\")),i},li=\"#000\",hi=\"round\",ci=[],ui=\"round\",fi=\"#000\",di=\"center\",gi=\"middle\",pi=[0,0,0,0],_i=new ni;let mi,yi=null;const wi={},xi=new Set([\"serif\",\"sans-serif\",\"monospace\",\"cursive\",\"fantasy\",\"system-ui\",\"ui-serif\",\"ui-sans-serif\",\"ui-monospace\",\"ui-rounded\",\"emoji\",\"math\",\"fangsong\"]);function vi(t,e,i){return`${t} ${e} 16px \"${i}\"`}const Si=function(){const t=100;let e,i;async function n(t){await i.ready;const e=ai(t),n=e.families[0].toLowerCase(),r=e.weight,s=[];if(i.forEach(t=>{const i=t.family.replace(/^['\"]|['\"]$/g,\"\").toLowerCase(),o=oi[t.weight]||t.weight;i===n&&t.style===e.style&&o==r&&s.push(t)}),0===s.length)return!1;return(await Promise.all(s.map(t=>t.load().then(()=>!0,()=>!1)))).some(t=>t)}async function r(){await i.ready;let s=!0;const o=_i.getProperties(),a=Object.keys(o).filter(e=>o[e]<t);for(let e=a.length-1;e>=0;--e){const i=a[e];let r=o[i];r<t&&(await n(i)?(Ne(wi),_i.set(i,t)):(r+=10,_i.set(i,r,!0),r<t&&(s=!1)))}e=void 0,s||(e=setTimeout(r,100))}return async function(t){i||(i=s?self.fonts:document.fonts);const n=ai(t);if(!n)return;const o=n.families;let a=!1;for(const t of o){if(xi.has(t))continue;const e=vi(n.style,n.weight,t);void 0===_i.get(e)&&(_i.set(e,0,!0),a=!0)}a&&(clearTimeout(e),e=setTimeout(r,100))}}(),Ci=function(){let t;return function(e){let i=wi[e];if(null==i){if(s){const t=ai(e),n=bi(e,\"Žg\");i=(isNaN(Number(t.lineHeight))?1.2:Number(t.lineHeight))*(n.actualBoundingBoxAscent+n.actualBoundingBoxDescent)}else t||(t=document.createElement(\"div\"),t.innerHTML=\"M\",t.style.minHeight=\"0\",t.style.maxHeight=\"none\",t.style.height=\"auto\",t.style.padding=\"0\",t.style.border=\"none\",t.style.position=\"absolute\",t.style.display=\"block\",t.style.left=\"-99999px\"),t.style.font=e,document.body.appendChild(t),i=t.offsetHeight,document.body.removeChild(t);wi[e]=i}return i}}();function bi(t,e){return yi||(yi=a(1,1)),t!=mi&&(yi.font=t,mi=yi.font),yi.measureText(e)}function Mi(t,e){return bi(t,e).width}function Ii(t,e,i){if(e in i)return i[e];const n=e.split(\"\\n\").reduce((e,i)=>Math.max(e,Mi(t,i)),0);return i[e]=n,n}function Ei(t,e,i,n,r,s,o,a,l,h,c){t.save(),1!==i&&(void 0===t.globalAlpha?t.globalAlpha=t=>t.globalAlpha*=i:t.globalAlpha*=i),e&&t.transform.apply(t,e),n.contextInstructions?(t.translate(l,h),t.scale(c[0],c[1]),function(t,e){const i=t.contextInstructions;for(let t=0,n=i.length;t<n;t+=2)Array.isArray(i[t+1])?e[i[t]].apply(e,i[t+1]):e[i[t]]=i[t+1]}(n,t)):c[0]<0||c[1]<0?(t.translate(l,h),t.scale(c[0],c[1]),t.drawImage(n,r,s,o,a,0,0,o,a)):t.drawImage(n,r,s,o,a,l,h,o*c[0],a*c[1]),t.restore()}class ki{constructor(){this.instructions_=[],this.zIndex=0,this.offset_=0,this.pendingMethod_,this.context_=new Proxy(h(),{get:(t,e)=>{if(\"function\"==typeof t[e])return this.pendingMethod_=e,this.pushMethodArgs_},set:(t,e,i)=>(this.push_(e,i),!0)})}push_(...t){const e=this.instructions_,i=this.zIndex+this.offset_;e[i]||(e[i]=[]),e[i].push(...t)}pushMethodArgs_=(...t)=>{this.push_(this.pendingMethod_,t)};pushFunction(t){this.push_(t)}getContext(){return this.context_}draw(t){this.instructions_.forEach(e=>{for(let i=0,n=e.length;i<n;++i){const n=e[i];if(\"function\"==typeof n){n(t);continue}const r=e[++i];\"function\"==typeof t[n]?t[n](...r):t[n]=\"function\"==typeof r?r(t):r}})}clear(){this.instructions_.length=0,this.zIndex=0,this.offset_=0}offset(){this.offset_=this.instructions_.length,this.zIndex=0}}const Ai=0,Pi=1,Oi=2,Ri=3,Li=4,Di=5,Fi=6,Ti=7,zi=8,$i=9,Wi=10,Gi=11,Ni=12;var Xi=0,Yi=1,Bi=2,Ui=3;function ji(t,e){return e&&(t.src=e),t.src&&o?new Promise((e,i)=>t.decode().then(()=>e(t)).catch(n=>t.complete&&t.width?e(t):i(n))):function(t){return new Promise((e,i)=>{function n(){s(),e(t)}function r(){s(),i(new Error(\"Image load error\"))}function s(){t.removeEventListener(\"load\",n),t.removeEventListener(\"error\",r)}t.addEventListener(\"load\",n),t.addEventListener(\"error\",r)})}(t)}function Vi(t,e){return t+\":\"+(e?F(e):\"null\")}const qi=new class{constructor(){this.cache_={},this.patternCache_={},this.cacheSize_=0,this.maxCacheSize_=1024}clear(){this.cache_={},this.patternCache_={},this.cacheSize_=0}canExpireCache(){return this.cacheSize_>this.maxCacheSize_}expire(){if(this.canExpireCache()){let t=0;for(const e in this.cache_){const i=this.cache_[e];3&t++||i.hasListener()||(delete this.cache_[e],delete this.patternCache_[e],--this.cacheSize_)}}}get(t,e){const i=Vi(t,e);return i in this.cache_?this.cache_[i]:null}getPattern(t,e){const i=Vi(t,e);return i in this.patternCache_?this.patternCache_[i]:null}set(t,e,i,n){const r=Vi(t,e),s=r in this.cache_;this.cache_[r]=i,n&&(i.getImageState()===Xi&&i.load(),i.getImageState()===Yi?i.ready().then(()=>{this.patternCache_[r]=h().createPattern(i.getImage(1),\"repeat\")}):this.patternCache_[r]=h().createPattern(i.getImage(1),\"repeat\")),s||++this.cacheSize_}setSize(t){this.maxCacheSize_=t,this.expire()}};let Ji=null;class Ki extends He{constructor(t,e,i,n,r){super(),this.hitDetectionImage_=null,this.image_=t,this.crossOrigin_=i?.crossOrigin,this.referrerPolicy_=i?.referrerPolicy,this.canvas_={},this.color_=r,this.imageState_=void 0===n?Xi:n,this.size_=t&&t.width&&t.height?[t.width,t.height]:null,this.src_=e,this.tainted_,this.ready_=null}initializeImage_(){this.image_=new Image,null!==this.crossOrigin_&&(this.image_.crossOrigin=this.crossOrigin_),void 0!==this.referrerPolicy_&&(this.image_.referrerPolicy=this.referrerPolicy_)}isTainted_(){if(void 0===this.tainted_&&this.imageState_===Bi){Ji||(Ji=a(1,1,0,{willReadFrequently:!0})),Ji.drawImage(this.image_,0,0);try{Ji.getImageData(0,0,1,1),this.tainted_=!1}catch{Ji=null,this.tainted_=!0}}return!0===this.tainted_}dispatchChangeEvent_(){this.dispatchEvent(je)}handleImageError_(){this.imageState_=Ui,this.dispatchChangeEvent_()}handleImageLoad_(){this.imageState_=Bi,this.size_=[this.image_.width,this.image_.height],this.dispatchChangeEvent_()}getImage(t){return this.image_||this.initializeImage_(),this.replaceColor_(t),this.canvas_[t]?this.canvas_[t]:this.image_}setImage(t){this.image_=t}getPixelRatio(t){return this.replaceColor_(t),this.canvas_[t]?t:1}getImageState(){return this.imageState_}getHitDetectionImage(){if(this.image_||this.initializeImage_(),!this.hitDetectionImage_)if(this.isTainted_()){const t=this.size_[0],e=this.size_[1],i=a(t,e);i.fillRect(0,0,t,e),this.hitDetectionImage_=i.canvas}else this.hitDetectionImage_=this.image_;return this.hitDetectionImage_}getSize(){return this.size_}getSrc(){return this.src_}load(){if(this.imageState_===Xi){this.image_||this.initializeImage_(),this.imageState_=Yi;try{void 0!==this.src_&&(this.image_.src=this.src_)}catch{this.handleImageError_()}this.image_ instanceof HTMLImageElement&&ji(this.image_,this.src_).then(t=>{this.image_=t,this.handleImageLoad_()}).catch(this.handleImageError_.bind(this))}}replaceColor_(t){if(!this.color_||this.canvas_[t]||this.imageState_!==Bi)return;const e=this.image_,i=a(Math.ceil(e.width*t),Math.ceil(e.height*t)),n=i.canvas;var r;i.scale(t,t),i.drawImage(e,0,0),i.globalCompositeOperation=\"multiply\",i.fillStyle=\"string\"==typeof(r=this.color_)?r:T(r),i.fillRect(0,0,n.width/t,n.height/t),i.globalCompositeOperation=\"destination-in\",i.drawImage(e,0,0),this.canvas_[t]=n}ready(){return this.ready_||(this.ready_=new Promise(t=>{if(this.imageState_===Bi||this.imageState_===Ui)t();else{const e=()=>{this.imageState_!==Bi&&this.imageState_!==Ui||(this.removeEventListener(je,e),t())};this.addEventListener(je,e)}})),this.ready_}}function Hi(t,e,i,n,r,s){let o=void 0===e?void 0:qi.get(e,r);return o||(o=new Ki(t,t&&\"src\"in t?t.src||void 0:e,i,n,r),qi.set(e,r,o,s)),s&&o&&!qi.getPattern(e,r)&&qi.set(e,r,o,s),o}function Zi(t){return t?Array.isArray(t)?T(t):\"object\"==typeof t&&\"src\"in t?function(t){if(!t.offset||!t.size)return qi.getPattern(t.src,t.color);const e=t.src+\":\"+t.offset,i=qi.getPattern(e,t.color);if(i)return i;const n=qi.get(t.src,null);if(n.getImageState()!==Bi)return null;const r=a(t.size[0],t.size[1]);return r.drawImage(n.getImage(1),t.offset[0],t.offset[1],t.size[0],t.size[1],0,0,t.size[0],t.size[1]),Hi(r.canvas,e,void 0,Bi,t.color,!0),qi.getPattern(e,t.color)}(t):t:null}let Qi=0,tn=1;function en(t,e,i,n,r,s,o,a){const l=o-r,h=a-s;let c=0,u=1;if(0===l){if(r<t||r>i)return!1}else{let e=(t-r)/l,n=(i-r)/l;if(e>n){const t=e;e=n,n=t}if(e>c&&(c=e),n<u&&(u=n),c>u)return!1}if(0===h){if(s<e||s>n)return!1}else{let t=(e-s)/h,i=(n-s)/h;if(t>i){const e=t;t=i,i=e}if(t>c&&(c=t),i<u&&(u=i),c>u)return!1}return Qi=c,tn=u,!0}function nn(t,e,i,n,r){const s=[];let o=i,a=0,l=e.slice(i,2);for(;a<t&&o+r<n;){const[i,n]=l.slice(-2),h=e[o+r],c=e[o+r+1],u=Math.sqrt((h-i)*(h-i)+(c-n)*(c-n));if(a+=u,a>=t){const e=(t-a+u)/u,f=p(i,h,e),d=p(n,c,e);l.push(f,d),s.push(l),l=[f,d],a==t&&(o+=r),a=0}else if(a<t)l.push(e[o+r],e[o+r+1]),o+=r;else{const t=u-a,e=p(i,h,t/u),f=p(n,c,t/u);l.push(e,f),s.push(l),l=[e,f],a=0,o+=r}}return a>0&&s.push(l),s}function rn(t,e,i,n,r){let s,o,a,l,h,c,u,f,d,g,p=i,_=i,m=0,y=0,w=i;for(o=i;o<n;o+=r){const i=e[o],n=e[o+1];void 0!==h&&(d=i-h,g=n-c,l=Math.sqrt(d*d+g*g),void 0!==u&&(y+=a,s=Math.acos((u*d+f*g)/(a*l)),s>t&&(y>m&&(m=y,p=w,_=o),y=0,w=o-r)),a=l,u=d,f=g),h=i,c=n}return y+=l,y>m?[w,o]:[p,_]}function sn(t,e,i,n,r){r=void 0!==r?r:[];let s=0;for(let o=e;o<i;o+=n)r[s++]=t.slice(o,o+n);return r.length=s,r}function on(t,e,i,n,r){r=void 0!==r?r:[];let s=0;for(let o=0,a=i.length;o<a;++o){const a=i[o];r[s++]=sn(t,e,a,n,r[s]),e=a}return r.length=s,r}function an(t,e,i,n,r){r=void 0!==r?r:[];let s=0;for(let o=0,a=i.length;o<a;++o){const a=i[o];r[s++]=1===a.length&&a[0]===e?[]:on(t,e,a,n,r[s]),e=a[a.length-1]}return r.length=s,r}class ln{drawCustom(t,e,i,n,r){}drawGeometry(t){}setStyle(t){}drawCircle(t,e,i){}drawFeature(t,e,i){}drawGeometryCollection(t,e,i){}drawLineString(t,e,i){}drawMultiLineString(t,e,i){}drawMultiPoint(t,e,i){}drawMultiPolygon(t,e,i){}drawPoint(t,e,i){}drawPolygon(t,e,i){}drawText(t,e,i){}setFillStrokeStyle(t,e){}setImageStyle(t,e){}setTextStyle(t,e){}}class hn extends ln{constructor(t,e,i,n){super(),this.tolerance=t,this.maxExtent=e,this.pixelRatio=n,this.maxLineWidth=0,this.resolution=i,this.beginGeometryInstruction1_=null,this.beginGeometryInstruction2_=null,this.bufferedMaxExtent_=null,this.instructions=[],this.coordinates=[],this.tmpCoordinate_=[],this.hitDetectionInstructions=[],this.state={}}applyPixelRatio(t){const e=this.pixelRatio;return 1==e?t:t.map(function(t){return t*e})}appendFlatPointCoordinates(t,e){const i=this.getBufferedMaxExtent(),n=this.tmpCoordinate_,r=this.coordinates;let s=r.length;for(let o=0,a=t.length;o<a;o+=e)n[0]=t[o],n[1]=t[o+1],de(i,n)&&(r[s++]=n[0],r[s++]=n[1]);return s}appendFlatLineCoordinates(t,e,i,n,r,s){const o=this.coordinates;let a=o.length;const l=this.getBufferedMaxExtent();s&&(e+=n);let h=t[e],c=t[e+1];const u=this.tmpCoordinate_;let f,d,g,p=!0;for(f=e+n;f<i;f+=n)u[0]=t[f],u[1]=t[f+1],g=pe(l,u),g!==d?(p&&(o[a++]=h,o[a++]=c,p=!1),o[a++]=u[0],o[a++]=u[1]):g===ae?(o[a++]=u[0],o[a++]=u[1],p=!1):p=!0,h=u[0],c=u[1],d=g;return(r&&p||f===e+n)&&(o[a++]=h,o[a++]=c),a}drawCustomCoordinates_(t,e,i,n,r){for(let s=0,o=i.length;s<o;++s){const o=i[s],a=this.appendFlatLineCoordinates(t,e,o,n,!1,!1);r.push(a),e=o}return e}drawCustom(t,e,i,n,r){this.beginGeometry(t,e,r);const s=t.getType(),o=t.getStride(),a=this.coordinates.length;let l,h,c,u,f;switch(s){case\"MultiPolygon\":l=t.getOrientedFlatCoordinates(),u=[];const e=t.getEndss();f=0;for(let t=0,i=e.length;t<i;++t){const i=[];f=this.drawCustomCoordinates_(l,f,e[t],o,i),u.push(i)}this.instructions.push([Li,a,u,t,i,an,r]),this.hitDetectionInstructions.push([Li,a,u,t,n||i,an,r]);break;case\"Polygon\":case\"MultiLineString\":c=[],l=\"Polygon\"==s?t.getOrientedFlatCoordinates():t.getFlatCoordinates(),f=this.drawCustomCoordinates_(l,0,t.getEnds(),o,c),this.instructions.push([Li,a,c,t,i,on,r]),this.hitDetectionInstructions.push([Li,a,c,t,n||i,on,r]);break;case\"LineString\":case\"Circle\":l=t.getFlatCoordinates(),h=this.appendFlatLineCoordinates(l,0,l.length,o,!1,!1),this.instructions.push([Li,a,h,t,i,sn,r]),this.hitDetectionInstructions.push([Li,a,h,t,n||i,sn,r]);break;case\"MultiPoint\":l=t.getFlatCoordinates(),h=this.appendFlatPointCoordinates(l,o),h>a&&(this.instructions.push([Li,a,h,t,i,sn,r]),this.hitDetectionInstructions.push([Li,a,h,t,n||i,sn,r]));break;case\"Point\":l=t.getFlatCoordinates(),this.coordinates.push(l[0],l[1]),h=this.coordinates.length,this.instructions.push([Li,a,h,t,i,void 0,r]),this.hitDetectionInstructions.push([Li,a,h,t,n||i,void 0,r])}this.endGeometry(e)}beginGeometry(t,e,i){this.beginGeometryInstruction1_=[Ai,e,0,t,i],this.instructions.push(this.beginGeometryInstruction1_),this.beginGeometryInstruction2_=[Ai,e,0,t,i],this.hitDetectionInstructions.push(this.beginGeometryInstruction2_)}finish(){return{instructions:this.instructions,hitDetectionInstructions:this.hitDetectionInstructions,coordinates:this.coordinates}}reverseHitDetectionInstructions(){const t=this.hitDetectionInstructions;let i;t.reverse();const n=t.length;let r,s,o=-1;for(i=0;i<n;++i)r=t[i],s=r[0],s==Ti?o=i:s==Ai&&(r[2]=i,e(this.hitDetectionInstructions,o,i),o=-1)}fillStyleToState(t,e={}){if(t){const i=t.getColor();e.fillPatternScale=i&&\"object\"==typeof i&&\"src\"in i?this.pixelRatio:1,e.fillStyle=Zi(i||li)??void 0}else e.fillStyle=void 0;return e}strokeStyleToState(t,e={}){if(t){const i=t.getColor();e.strokeStyle=Zi(i||fi);const n=t.getLineCap();e.lineCap=void 0!==n?n:hi;const r=t.getLineDash();e.lineDash=r?r.slice():ci;const s=t.getLineDashOffset();e.lineDashOffset=s||0;const o=t.getLineJoin();e.lineJoin=void 0!==o?o:ui;const a=t.getWidth();e.lineWidth=void 0!==a?a:1;const l=t.getMiterLimit();e.miterLimit=void 0!==l?l:10;const h=t.getOffset();e.strokeOffset=h??0,e.lineWidth>this.maxLineWidth&&(this.maxLineWidth=e.lineWidth,this.bufferedMaxExtent_=null)}else e.strokeStyle=void 0,e.lineCap=void 0,e.lineDash=null,e.lineDashOffset=void 0,e.lineJoin=void 0,e.lineWidth=void 0,e.miterLimit=void 0,e.strokeOffset=void 0;return e}setFillStrokeStyle(t,e){const i=this.state;this.fillStyleToState(t,i),this.strokeStyleToState(e,i)}createFill(t){const e=t.fillStyle,i=[Wi,e];return\"string\"!=typeof e&&i.push(t.fillPatternScale),i}applyStroke(t){this.instructions.push(this.createStroke(t))}createStroke(t){return[Gi,t.strokeStyle,t.lineWidth*this.pixelRatio,t.lineCap,t.lineJoin,t.miterLimit,t.lineDash?this.applyPixelRatio(t.lineDash):null,t.lineDashOffset*this.pixelRatio]}updateFillStyle(t,e){const i=t.fillStyle;(void 0!==i&&\"string\"!=typeof i||t.currentFillStyle!=i)&&(this.instructions.push(e.call(this,t)),t.currentFillStyle=i)}updateStrokeStyle(t,e){const i=t.strokeStyle,r=t.lineCap,s=t.lineDash,o=t.lineDashOffset,a=t.lineJoin,l=t.lineWidth,h=t.miterLimit,c=t.strokeOffset;(t.currentStrokeStyle!=i||t.currentLineCap!=r||s!=t.currentLineDash&&!n(t.currentLineDash,s)||t.currentLineDashOffset!=o||t.currentLineJoin!=a||t.currentLineWidth!=l||t.currentMiterLimit!=h||t.currentStrokeOffset!=c)&&(e.call(this,t),t.currentStrokeStyle=i,t.currentLineCap=r,t.currentLineDash=s,t.currentLineDashOffset=o,t.currentLineJoin=a,t.currentLineWidth=l,t.currentMiterLimit=h,t.currentStrokeOffset=c)}endGeometry(t){this.beginGeometryInstruction1_[2]=this.instructions.length,this.beginGeometryInstruction1_=null,this.beginGeometryInstruction2_[2]=this.hitDetectionInstructions.length,this.beginGeometryInstruction2_=null;const e=[Ti,t];this.instructions.push(e),this.hitDetectionInstructions.push(e)}getBufferedMaxExtent(){if(!this.bufferedMaxExtent_&&(this.bufferedMaxExtent_=this.maxExtent.slice(),this.maxLineWidth>0)){const t=this.resolution*(this.maxLineWidth+1)/2;!function(t,e,i){i?(i[0]=t[0]-e,i[1]=t[1]-e,i[2]=t[2]+e,i[3]=t[3]+e):(t[0],t[1],t[2],t[3])}(this.bufferedMaxExtent_,t,this.bufferedMaxExtent_)}return this.bufferedMaxExtent_}}const cn={left:0,center:.5,right:1,top:0,middle:.5,hanging:.2,alphabetic:.8,ideographic:.8,bottom:1};class un extends hn{constructor(t,e,i,n){super(t,e,i,n),this.labels_=null,this.text_=\"\",this.textOffsetX_=0,this.textOffsetY_=0,this.textRotateWithView_=void 0,this.textKeepUpright_=void 0,this.textRotation_=0,this.textFillState_=null,this.fillStates={},this.fillStates[li]={fillStyle:li},this.textStrokeState_=null,this.strokeStates={},this.textState_={},this.textStates={},this.textKey_=\"\",this.fillKey_=\"\",this.strokeKey_=\"\",this.declutterMode_=void 0,this.declutterImageWithText_=void 0}finish(){const t=super.finish();return t.textStates=this.textStates,t.fillStates=this.fillStates,t.strokeStates=this.strokeStates,t}drawText(t,e,i){const n=this.textFillState_,r=this.textStrokeState_,s=this.textState_;if(\"\"===this.text_||!s||!n&&!r)return;const o=this.coordinates;let a=o.length;const l=t.getType();let h=null,c=t.getStride();if(\"line\"!==s.placement||\"LineString\"!=l&&\"MultiLineString\"!=l&&\"Polygon\"!=l&&\"MultiPolygon\"!=l){let n=s.overflow?null:[];switch(l){case\"Point\":case\"MultiPoint\":h=t.getFlatCoordinates();break;case\"LineString\":h=t.getFlatMidpoint();break;case\"Circle\":h=t.getCenter();break;case\"MultiLineString\":h=t.getFlatMidpoints(),c=2;break;case\"Polygon\":h=t.getFlatInteriorPoint(),s.overflow||n.push(h[2]/this.resolution),c=3;break;case\"MultiPolygon\":const e=t.getFlatInteriorPoints();h=[];for(let t=0,i=e.length;t<i;t+=3)s.overflow||n.push(e[t+2]/this.resolution),h.push(e[t],e[t+1]);if(0===h.length)return;c=2}const r=this.appendFlatPointCoordinates(h,c);if(r===a)return;if(n&&(r-a)/2!==h.length/c){let t=a/2;n=n.filter((e,i)=>{const n=o[2*(t+i)]===h[i*c]&&o[2*(t+i)+1]===h[i*c+1];return n||--t,n})}this.saveTextStates_();const u=s.backgroundFill?this.createFill(this.fillStyleToState(s.backgroundFill)):null,f=s.backgroundStroke?this.createStroke(this.strokeStyleToState(s.backgroundStroke)):null;this.beginGeometry(t,e,i);let d=s.padding;if(d!=pi&&(s.scale[0]<0||s.scale[1]<0)){let t=s.padding[0],e=s.padding[1],i=s.padding[2],n=s.padding[3];s.scale[0]<0&&(e=-e,n=-n),s.scale[1]<0&&(t=-t,i=-i),d=[t,e,i,n]}const g=this.pixelRatio;this.instructions.push([Fi,a,r,null,NaN,NaN,NaN,1,0,0,this.textRotateWithView_,this.textRotation_,[1,1],NaN,this.declutterMode_,this.declutterImageWithText_,d==pi?pi:d.map(function(t){return t*g}),u,f,this.text_,this.textKey_,this.strokeKey_,this.fillKey_,this.textOffsetX_,this.textOffsetY_,n]);const p=1/g,_=u?u.slice(0):null;_&&(_[1]=li),this.hitDetectionInstructions.push([Fi,a,r,null,NaN,NaN,NaN,1,0,0,this.textRotateWithView_,this.textRotation_,[p,p],NaN,this.declutterMode_,this.declutterImageWithText_,d,_,f,this.text_,this.textKey_,this.strokeKey_,this.fillKey_?li:this.fillKey_,this.textOffsetX_,this.textOffsetY_,n]),this.endGeometry(e)}else{const n=t.getExtent();if(!be(this.maxExtent,n))return;let r;if(h=t.getFlatCoordinates(),\"LineString\"==l)r=[h.length];else if(\"MultiLineString\"==l)r=t.getEnds();else if(\"Polygon\"==l)r=t.getEnds().slice(0,1);else if(\"MultiPolygon\"==l){const e=t.getEndss();r=[];for(let t=0,i=e.length;t<i;++t)r.push(e[t][0])}if(!(\"LineString\"!=l&&\"MultiLineString\"!=l||(u=this.getBufferedMaxExtent(),f=n,u[0]<=f[0]&&f[2]<=u[2]&&u[1]<=f[1]&&f[3]<=u[3]))){const t=function(t,e,i,n){const r=n[0],s=n[1],o=n[2],a=n[3],l=[],h=[];let c,u,f=!1,d=0;for(let n=0,g=e.length;n<g;++n){const g=e[n];let p=t[d],_=t[d+1],m=!1;for(let e=d+i;e<g;e+=i){const i=t[e],n=t[e+1];if(en(r,s,o,a,p,_,i,n)){const t=i-p,e=n-_,r=p+Qi*t,s=_+Qi*e,o=p+tn*t,a=_+tn*e;f&&m&&r===c&&s===u?l.push(o,a):(f&&h.push(l.length),l.push(r,s,o,a),f=!0),c=o,u=a,m=!0}p=i,_=n}d=g}return f&&h.push(l.length),{flatCoordinates:l,ends:h}}(h,r,c,this.getBufferedMaxExtent());if(h=t.flatCoordinates,r=t.ends,c=2,0===r.length)return}this.beginGeometry(t,e,i);const d=s.repeat,g=d?void 0:s.textAlign;let p=0;for(let t=0,e=r.length;t<e;++t){let e;e=d?nn(d*this.resolution,h,p,r[t],c):[h.slice(p,r[t])];for(let i=0,n=e.length;i<n;++i){const n=e[i];let l=0,h=n.length;if(null==g){const t=rn(s.maxAngle,n,0,n.length,2);l=t[0],h=t[1]}for(let t=l;t<h;t+=c)o.push(n[t],n[t+1]);const u=o.length;p=r[t],this.drawChars_(a,u),a=u}}this.endGeometry(e)}var u,f}saveTextStates_(){const t=this.textStrokeState_,e=this.textState_,i=this.textFillState_,n=this.strokeKey_;t&&(n in this.strokeStates||(this.strokeStates[n]={strokeStyle:t.strokeStyle,lineCap:t.lineCap,lineDashOffset:t.lineDashOffset,lineWidth:t.lineWidth,lineJoin:t.lineJoin,miterLimit:t.miterLimit,lineDash:t.lineDash}));const r=this.textKey_;r in this.textStates||(this.textStates[r]={font:e.font,textAlign:e.textAlign||di,justify:e.justify,textBaseline:e.textBaseline||gi,scale:e.scale});const s=this.fillKey_;i&&(s in this.fillStates||(this.fillStates[s]={fillStyle:i.fillStyle}))}drawChars_(t,e){const i=this.textStrokeState_,n=this.textState_,r=this.strokeKey_,s=this.textKey_,o=this.fillKey_;this.saveTextStates_();const a=this.pixelRatio,l=cn[n.textBaseline],h=this.textOffsetX_*a,c=this.textOffsetY_*a,u=this.text_,f=i?i.lineWidth*Math.abs(n.scale[0])/2:0;this.instructions.push([Di,t,e,l,n.overflow,o,n.maxAngle,a,c,r,f*a,u,s,1,this.declutterMode_,this.textKeepUpright_,h]),this.hitDetectionInstructions.push([Di,t,e,l,n.overflow,o?li:o,n.maxAngle,a,c,r,f*a,u,s,1/a,this.declutterMode_,this.textKeepUpright_,h])}setTextStyle(t,e){let i,n,r;if(t){const e=t.getFill();e?(n=this.textFillState_,n||(n={},this.textFillState_=n),n.fillStyle=Zi(e.getColor()||li)):(n=null,this.textFillState_=n);const s=t.getStroke();if(s){r=this.textStrokeState_,r||(r={},this.textStrokeState_=r);const t=s.getLineDash(),e=s.getLineDashOffset(),i=s.getWidth(),n=s.getMiterLimit();r.lineCap=s.getLineCap()||hi,r.lineDash=t?t.slice():ci,r.lineDashOffset=void 0===e?0:e,r.lineJoin=s.getLineJoin()||ui,r.lineWidth=void 0===i?1:i,r.miterLimit=void 0===n?10:n,r.strokeStyle=Zi(s.getColor()||fi)}else r=null,this.textStrokeState_=r;i=this.textState_;const o=t.getFont()||\"10px sans-serif\";Si(o);const a=t.getScaleArray();i.overflow=t.getOverflow(),i.font=o,i.maxAngle=t.getMaxAngle(),i.placement=t.getPlacement(),i.textAlign=t.getTextAlign(),i.repeat=t.getRepeat(),i.justify=t.getJustify(),i.textBaseline=t.getTextBaseline()||gi,i.backgroundFill=t.getBackgroundFill(),i.backgroundStroke=t.getBackgroundStroke(),i.padding=t.getPadding()||pi,i.scale=void 0===a?[1,1]:a;const l=t.getOffsetX(),h=t.getOffsetY(),c=t.getRotateWithView(),u=t.getKeepUpright(),f=t.getRotation();this.text_=t.getText()||\"\",this.textOffsetX_=void 0===l?0:l,this.textOffsetY_=void 0===h?0:h,this.textRotateWithView_=void 0!==c&&c,this.textKeepUpright_=void 0===u||u,this.textRotation_=void 0===f?0:f,this.strokeKey_=r?(\"string\"==typeof r.strokeStyle?r.strokeStyle:ei(r.strokeStyle))+r.lineCap+r.lineDashOffset+\"|\"+r.lineWidth+r.lineJoin+r.miterLimit+\"[\"+r.lineDash.join()+\"]\":\"\",this.textKey_=i.font+i.scale+(i.textAlign||\"?\")+(i.repeat||\"?\")+(i.justify||\"?\")+(i.textBaseline||\"?\"),this.fillKey_=n&&n.fillStyle?\"string\"==typeof n.fillStyle?n.fillStyle:\"|\"+ei(n.fillStyle):\"\"}else this.text_=\"\";this.declutterMode_=t.getDeclutterMode(),this.declutterImageWithText_=e}}const fn=[1/0,1/0,-1/0,-1/0],dn=[],gn=[],pn=[],_n=[];function mn(t){return t[3].declutterBox}const yn=new RegExp(\"[\"+String.fromCharCode(1425)+\"-\"+String.fromCharCode(2303)+String.fromCharCode(64285)+\"-\"+String.fromCharCode(65023)+String.fromCharCode(65136)+\"-\"+String.fromCharCode(65276)+String.fromCharCode(67584)+\"-\"+String.fromCharCode(69631)+String.fromCharCode(124928)+\"-\"+String.fromCharCode(126975)+\"]\");function wn(t,e){return\"start\"===e?e=yn.test(t)?\"right\":\"left\":\"end\"===e&&(e=yn.test(t)?\"left\":\"right\"),cn[e]}function xn(t,e,i){return i>0&&t.push(\"\\n\",\"\"),t.push(e,\"\"),t}function vn(t,e,i){return i%2==0&&(t+=e),t}class Sn{constructor(t,e,i,n,r){this.overlaps=i,this.pixelRatio=e,this.resolution=t,this.alignAndScaleFill_,this.instructions=n.instructions,this.coordinates=n.coordinates,this.coordinateCache_={},this.renderedTransform_=Fe(),this.hitDetectionInstructions=n.hitDetectionInstructions,this.pixelCoordinates_=null,this.viewRotation_=0,this.fillStates=n.fillStates||{},this.strokeStates=n.strokeStates||{},this.textStates=n.textStates||{},this.widths_={},this.labels_={},this.zIndexContext_=r?new ki:null}getZIndexContext(){return this.zIndexContext_}createLabel(t,e,i,n){const r=t+e+i+n;if(this.labels_[r])return this.labels_[r];const s=n?this.strokeStates[n]:null,o=i?this.fillStates[i]:null,a=this.textStates[e],l=this.pixelRatio,h=[a.scale[0]*l,a.scale[1]*l],c=a.justify?cn[a.justify]:wn(Array.isArray(t)?t[0]:t,a.textAlign||di),u=n&&s.lineWidth?s.lineWidth:0,f=Array.isArray(t)?t:String(t).split(\"\\n\").reduce(xn,[]),{width:d,height:g,widths:p,heights:_,lineWidths:m}=function(t,e){const i=[],n=[],r=[];let s=0,o=0,a=0,l=0;for(let h=0,c=e.length;h<=c;h+=2){const u=e[h];if(\"\\n\"===u||h===c){s=Math.max(s,o),r.push(o),o=0,a+=l,l=0;continue}const f=e[h+1]||t.font,d=Mi(f,u);i.push(d),o+=d;const g=Ci(f);n.push(g),l=Math.max(l,g)}return{width:s,height:a,widths:i,heights:n,lineWidths:r}}(a,f),y=d+u,w=[],x=(y+2)*h[0],v=(g+u)*h[1],S={width:x<0?Math.floor(x):Math.ceil(x),height:v<0?Math.floor(v):Math.ceil(v),contextInstructions:w};1==h[0]&&1==h[1]||w.push(\"scale\",h),n&&(w.push(\"strokeStyle\",s.strokeStyle),w.push(\"lineWidth\",u),w.push(\"lineCap\",s.lineCap),w.push(\"lineJoin\",s.lineJoin),w.push(\"miterLimit\",s.miterLimit),w.push(\"setLineDash\",[s.lineDash]),w.push(\"lineDashOffset\",s.lineDashOffset)),i&&w.push(\"fillStyle\",o.fillStyle),w.push(\"textBaseline\",\"middle\"),w.push(\"textAlign\",\"center\");const C=.5-c;let b=c*y+C*u;const M=[],I=[];let E,k=0,A=0,P=0,O=0;for(let t=0,e=f.length;t<e;t+=2){const e=f[t];if(\"\\n\"===e){A+=k,k=0,b=c*y+C*u,++O;continue}const r=f[t+1]||a.font;r!==E&&(n&&M.push(\"font\",r),i&&I.push(\"font\",r),E=r),k=Math.max(k,_[P]);const s=[e,b+C*p[P]+c*(p[P]-m[O]),.5*(u+k)+A];b+=p[P],n&&M.push(\"strokeText\",s),i&&I.push(\"fillText\",s),++P}return Array.prototype.push.apply(w,M),Array.prototype.push.apply(w,I),this.labels_[r]=S,S}replayTextBackground_(t,e,i,n,r,s,o){t.beginPath(),t.moveTo.apply(t,e),t.lineTo.apply(t,i),t.lineTo.apply(t,n),t.lineTo.apply(t,r),t.lineTo.apply(t,e),s&&(this.alignAndScaleFill_=s[2],t.fillStyle=s[1],this.fill_(t)),o&&(this.setStrokeStyle_(t,o),t.stroke())}calculateImageOrLabelDimensions_(t,e,i,n,r,s,o,a,l,h,c,u,f,d,g,p){let _=i-(o*=u[0]),m=n-(a*=u[1]);const y=r+l>t?t-l:r,w=s+h>e?e-h:s,x=d[3]+y*u[0]+d[1],v=d[0]+w*u[1]+d[2],S=_-d[3],C=m-d[0];let b;return(g||0!==c)&&(dn[0]=S,_n[0]=S,dn[1]=C,gn[1]=C,gn[0]=S+x,pn[0]=gn[0],pn[1]=C+v,_n[1]=pn[1]),0!==c?(b=$e(Fe(),i,n,1,1,c,-i,-n),ze(b,dn),ze(b,gn),ze(b,pn),ze(b,_n),_e(Math.min(dn[0],gn[0],pn[0],_n[0]),Math.min(dn[1],gn[1],pn[1],_n[1]),Math.max(dn[0],gn[0],pn[0],_n[0]),Math.max(dn[1],gn[1],pn[1],_n[1]),fn)):_e(Math.min(S,S+x),Math.min(C,C+v),Math.max(S,S+x),Math.max(C,C+v),fn),f&&(_=Math.round(_),m=Math.round(m)),{drawImageX:_,drawImageY:m,drawImageW:y,drawImageH:w,originX:l,originY:h,declutterBox:{minX:fn[0],minY:fn[1],maxX:fn[2],maxY:fn[3],value:p},canvasTransform:b,scale:u}}replayImageOrLabel_(t,e,i,n,r,s,o){const a=!(!s&&!o),l=n.declutterBox,h=o?o[2]*n.scale[0]/2:0;return l.minX-h<=e[0]&&l.maxX+h>=0&&l.minY-h<=e[1]&&l.maxY+h>=0&&(a&&this.replayTextBackground_(t,dn,gn,pn,_n,s,o),Ei(t,n.canvasTransform,r,i,n.originX,n.originY,n.drawImageW,n.drawImageH,n.drawImageX,n.drawImageY,n.scale)),!0}fill_(t){const e=this.alignAndScaleFill_;if(e){const i=ze(this.renderedTransform_,[0,0]),n=512*this.pixelRatio;t.save(),t.translate(i[0]%n,i[1]%n),1!==e&&t.scale(e,e)}t.fill(),e&&t.restore()}setStrokeStyle_(t,e){t.strokeStyle=e[1],e[1]&&(t.lineWidth=e[2],t.lineCap=e[3],t.lineJoin=e[4],t.miterLimit=e[5],t.lineDashOffset=e[7],t.setLineDash(e[6]))}drawLabelWithPointPlacement_(t,e,i,n){const r=this.textStates[e],s=this.createLabel(t,e,n,i),o=this.strokeStates[i],a=this.pixelRatio,l=wn(Array.isArray(t)?t[0]:t,r.textAlign||di),h=cn[r.textBaseline||gi],c=o&&o.lineWidth?o.lineWidth:0;return{label:s,anchorX:l*(s.width/a-2*r.scale[0])+2*(.5-l)*c,anchorY:h*s.height/a+2*(.5-h)*c}}execute_(t,e,i,r,s,o,a,l){const h=this.zIndexContext_;let c;var u,f;this.pixelCoordinates_&&n(i,this.renderedTransform_)?c=this.pixelCoordinates_:(this.pixelCoordinates_||(this.pixelCoordinates_=[]),c=Ae(this.coordinates,0,this.coordinates.length,2,i,this.pixelCoordinates_),u=this.renderedTransform_,f=i,u[0]=f[0],u[1]=f[1],u[2]=f[2],u[3]=f[3],u[4]=f[4],u[5]=f[5]);let d=0;const g=r.length;let p,_=0;const m=[];let y,w,x,v,S,C,b,M,I,E,k,A,P,O=0,R=0;const L=this.coordinateCache_,D=this.viewRotation_,F=Math.round(1e12*Math.atan2(-i[1],i[0]))/1e12,T={context:t,pixelRatio:this.pixelRatio,resolution:this.resolution,rotation:D},z=this.instructions!=r||this.overlaps?0:200;let $,W,G,N;for(;d<g;){const i=r[d];switch(i[0]){case Ai:$=i[1],N=i[3],$.getGeometry()?void 0===a||be(a,N.getExtent())?++d:d=i[2]+1:d=i[2],h&&(h.zIndex=i[4]);break;case Pi:O>z&&(this.fill_(t),O=0),R>z&&(t.stroke(),R=0),O||R||(t.beginPath(),S=NaN,C=NaN),++d;break;case Oi:_=i[1],x=i[2]??0;const n=c[_],r=c[_+1],u=c[_+2]-x-n,f=c[_+3]-x-r,g=Math.sqrt(u*u+f*f);t.moveTo(n+g,r),t.arc(n,r,g,0,2*Math.PI,!0),++d;break;case Ri:t.closePath(),++d;break;case Li:_=i[1],p=i[2];const X=i[3],Y=i[4],B=i[5];T.geometry=X,T.feature=$,d in L||(L[d]=[]);const U=L[d];B?B(c,_,p,2,U):(U[0]=c[_],U[1]=c[_+1],U.length=2),h&&(h.zIndex=i[6]),Y(U,T),++d;break;case Fi:_=i[1],p=i[2],I=i[3],y=i[4],w=i[5];let j=i[6];const V=i[7],q=i[8],J=i[9],K=i[10];let H=i[11];const Z=i[12];let Q=i[13];v=i[14]||\"declutter\";const tt=i[15];if(!I&&i.length>=20){E=i[19],k=i[20],A=i[21],P=i[22];const t=this.drawLabelWithPointPlacement_(E,k,A,P);I=t.label,i[3]=I;const e=i[23];y=(t.anchorX-e)*this.pixelRatio,i[4]=y;const n=i[24];w=(t.anchorY-n)*this.pixelRatio,i[5]=w,j=I.height,i[6]=j,Q=I.width,i[13]=Q}let et,it,nt,rt;i.length>25&&(et=i[25]),i.length>17?(it=i[16],nt=i[17],rt=i[18]):(it=pi,nt=null,rt=null),K&&F?H+=D:K||F||(H-=D);let st=0;for(;_<p;_+=2){if(et&&et[st++]<Q/this.pixelRatio)continue;const i=this.calculateImageOrLabelDimensions_(I.width,I.height,c[_],c[_+1],Q,j,y,w,q,J,H,Z,s,it,!!nt||!!rt,$),n=[t,e,I,i,V,nt,rt];if(l){let t,e,r,s,o;if(tt){const i=p-_;if(!tt[i]){tt[i]={args:n,declutterMode:v};continue}const s=tt[i];t=s.args,e=s.declutterMode,delete tt[i],r=mn(t)}if(!t||\"declutter\"===e&&l.collides(r)||(s=!0),\"declutter\"===v&&l.collides(i.declutterBox)||(o=!0),\"declutter\"===e&&\"declutter\"===v){const t=s&&o;s=t,o=t}s&&(\"none\"!==e&&l.insert(r),this.replayImageOrLabel_.apply(this,t)),o&&(\"none\"!==v&&l.insert(i.declutterBox),this.replayImageOrLabel_.apply(this,n))}else this.replayImageOrLabel_.apply(this,n)}++d;break;case Di:const ot=i[1],at=i[2],lt=i[3],ht=i[4];P=i[5];const ct=i[6],ut=i[7],ft=i[8];A=i[9];const dt=i[10];E=i[11],Array.isArray(E)&&(E=E.reduce(vn,\"\")),k=i[12];const gt=[i[13],i[13]];v=i[14]||\"declutter\";const pt=i[15],_t=i[16],mt=this.textStates[k],yt=mt.font,wt=[mt.scale[0]*ut,mt.scale[1]*ut];let xt;yt in this.widths_?xt=this.widths_[yt]:(xt={},this.widths_[yt]=xt);const vt=Me(c,ot,at,2),St=Math.abs(wt[0])*Ii(yt,E,xt);if(ht||St<=vt){const i=Re(c,ot,at,2,E,(vt-St)*wn(E,this.textStates[k].textAlign),ct,Math.abs(wt[0]),Ii,yt,xt,F?0:this.viewRotation_,pt);t:if(i){const n=[];let r,s,o,a,h;if(A)for(r=0,s=i.length;r<s;++r){h=i[r],o=h[4],a=this.createLabel(o,k,\"\",A),y=h[2]+(wt[0]<0?-dt:dt)-_t,w=lt*a.height+2*(.5-lt)*dt*wt[1]/wt[0]-ft;const s=this.calculateImageOrLabelDimensions_(a.width,a.height,h[0],h[1],a.width,a.height,y,w,0,0,h[3],gt,!1,pi,!1,$);if(l&&\"declutter\"===v&&l.collides(s.declutterBox))break t;n.push([t,e,a,s,1,null,null])}if(P)for(r=0,s=i.length;r<s;++r){h=i[r],o=h[4],a=this.createLabel(o,k,P,\"\"),y=h[2]-_t,w=lt*a.height-ft;const s=this.calculateImageOrLabelDimensions_(a.width,a.height,h[0],h[1],a.width,a.height,y,w,0,0,h[3],gt,!1,pi,!1,$);if(l&&\"declutter\"===v&&l.collides(s.declutterBox))break t;n.push([t,e,a,s,1,null,null])}l&&\"none\"!==v&&l.load(n.map(mn));for(let t=0,e=n.length;t<e;++t)this.replayImageOrLabel_.apply(this,n[t])}}++d;break;case Ti:if(void 0!==o){$=i[1];const t=o($,N,v);if(t)return t}++d;break;case zi:z?O++:this.fill_(t),++d;break;case $i:let Ct,bt,Mt;if(_=i[1],p=i[2],x=i[3],x){const t=(i[4]??!1)||Math.abs(c[_]-c[p-2])<1e-6&&Math.abs(c[_+1]-c[p-1])<1e-6;Ie(c,_,p,2,x,t,m),ke(m,2,t),Ct=m,bt=0,Mt=Ct.length}else Ct=c,bt=_,Mt=p;W=Ct[bt],G=Ct[bt+1],t.moveTo(W,G),S=W+.5|0,C=G+.5|0;for(let e=bt+2;e<Mt;e+=2)W=Ct[e],G=Ct[e+1],b=W+.5|0,M=G+.5|0,e!=Mt-2&&b===S&&M===C||(t.lineTo(W,G),S=b,C=M);++d;break;case Wi:this.alignAndScaleFill_=i[2],O?(this.fill_(t),O=0,R&&(t.stroke(),R=0)):R&&i[1]&&(t.stroke(),R=0),t.fillStyle=i[1],++d;break;case Gi:O&&i[1]&&(this.fill_(t),O=0),R&&(t.stroke(),R=0),this.setStrokeStyle_(t,i),++d;break;case Ni:z?R++:t.stroke(),++d;break;default:++d}}O&&this.fill_(t),R&&t.stroke()}execute(t,e,i,n,r,s){this.viewRotation_=n,this.execute_(t,e,i,this.instructions,r,void 0,void 0,s)}executeHitDetection(t,e,i,n,r){return this.viewRotation_=i,this.execute_(t,[t.canvas.width,t.canvas.height],e,this.hitDetectionInstructions,!0,n,r)}}function Cn(t,e,i){return bn(et(t,e,i))}function bn(t,e){if(t instanceof Z){if(t.type===X&&\"string\"==typeof t.value){const e=D(t.value);return function(){return e}}return function(){return t.value}}const i=t.operator;switch(i){case Gt:case Nt:case zt:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);switch(e){case zt:return t=>{for(let e=0;e<i;++e){const i=n[e](t);if(null!=i)return i}throw new Error(\"Expected one of the values to be non-null\")};case Gt:case Nt:return t=>{for(let r=0;r<i;++r){const i=n[r](t);if(typeof i===e)return i}throw new Error(`Expected one of the values to be a ${e}`)};default:throw new Error(`Unsupported assertion operator ${e}`)}}(t);case it:case nt:case qt:return function(t){const e=t.args[0],i=e.value;switch(t.operator){case it:return e=>{const n=t.args;let r=e.properties[i];for(let t=1,e=n.length;t<e;++t){r=r[n[t].value]}return r};case nt:return t=>t.variables[i];case qt:return e=>{const n=t.args;if(!(i in e.properties))return!1;let r=e.properties[i];for(let t=1,e=n.length;t<e;++t){const e=n[t].value;if(!r||!Object.hasOwn(r,e))return!1;r=r[e]}return!0};default:throw new Error(`Unsupported accessor operator ${t.operator}`)}}(t);case Bt:return t=>t.featureId;case st:return t=>t.geometryType;case rt:{const e=t.args.map(t=>bn(t));return t=>\"\".concat(...e.map(e=>e(t).toString()))}case ct:return t=>t.resolution;case at:case lt:case Ft:case Wt:case ht:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);switch(e){case at:return t=>{for(let e=0;e<i;++e)if(n[e](t))return!0;return!1};case lt:return t=>{for(let e=0;e<i;++e)if(!n[e](t))return!1;return!0};case Ft:return t=>{const e=n[0](t),i=n[1](t),r=n[2](t);return e>=i&&e<=r};case Wt:return t=>{const e=n[0](t);for(let r=1;r<i;++r)if(e===n[r](t))return!0;return!1};case ht:return t=>!n[0](t);default:throw new Error(`Unsupported logical operator ${e}`)}}(t);case dt:case gt:case mt:case yt:case pt:case _t:return function(t){const e=t.operator,i=bn(t.args[0]),n=bn(t.args[1]);switch(e){case dt:return t=>i(t)===n(t);case gt:return t=>i(t)!==n(t);case mt:return t=>i(t)<n(t);case yt:return t=>i(t)<=n(t);case pt:return t=>i(t)>n(t);case _t:return t=>i(t)>=n(t);default:throw new Error(`Unsupported comparison operator ${e}`)}}(t);case wt:case xt:case vt:case St:case Ct:case bt:case Mt:case It:case Et:case kt:case At:case Pt:case Ot:case Rt:case Lt:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);switch(e){case wt:return t=>{let e=1;for(let r=0;r<i;++r)e*=n[r](t);return e};case xt:return t=>n[0](t)/n[1](t);case vt:return t=>{let e=0;for(let r=0;r<i;++r)e+=n[r](t);return e};case St:return t=>n[0](t)-n[1](t);case Ct:return t=>{const e=n[0](t),i=n[1](t);if(e<i)return i;const r=n[2](t);return e>r?r:e};case bt:return t=>n[0](t)%n[1](t);case Mt:return t=>Math.pow(n[0](t),n[1](t));case It:return t=>Math.abs(n[0](t));case Et:return t=>Math.floor(n[0](t));case kt:return t=>Math.ceil(n[0](t));case At:return t=>Math.round(n[0](t));case Pt:return t=>Math.sin(n[0](t));case Ot:return t=>Math.cos(n[0](t));case Rt:return 2===i?t=>Math.atan2(n[0](t),n[1](t)):t=>Math.atan(n[0](t));case Lt:return t=>Math.sqrt(n[0](t));default:throw new Error(`Unsupported numeric operator ${e}`)}}(t);case $t:return function(t){const e=t.args.length,i=new Array(e);for(let n=0;n<e;++n)i[n]=bn(t.args[n]);return t=>{for(let n=0;n<e-1;n+=2){if(i[n](t))return i[n+1](t)}return i[e-1](t)}}(t);case Dt:return function(t){const e=t.args.length,i=new Array(e);for(let n=0;n<e;++n)i[n]=bn(t.args[n]);return t=>{const n=i[0](t);for(let r=1;r<e-1;r+=2)if(n===i[r](t))return i[r+1](t);return i[e-1](t)}}(t);case Tt:return function(t){const e=t.args.length,i=new Array(e);for(let n=0;n<e;++n)i[n]=bn(t.args[n]);return t=>{const n=i[0](t),r=i[1](t);let s,o;for(let a=2;a<e;a+=2){const e=i[a](t);let l=i[a+1](t);const h=Array.isArray(l);if(h&&(l=k(l)),e>=r)return 2===a?l:h?In(n,r,s,o,e,l):Mn(n,r,s,o,e,l);s=e,o=l}return o}}(t);case Vt:return function(t){const e=t.operator,i=t.args.length,n=new Array(i);for(let e=0;e<i;++e)n[e]=bn(t.args[e]);if(e===Vt)return e=>{const i=n[0](e);return t.args[0].type===X?T(i):i.toString()};throw new Error(`Unsupported convert operator ${e}`)}(t);default:throw new Error(`Unsupported operator ${i}`)}}function Mn(t,e,i,n,r,s){const o=r-i;if(0===o)return n;const a=e-i;return n+(1===t?a/o:(Math.pow(t,a)-1)/(Math.pow(t,o)-1))*(s-n)}function In(t,e,i,n,r,s){if(0===r-i)return n;const o=L(n),a=L(s);let l=a[2]-o[2];l>180?l-=360:l<-180&&(l+=360);return function(t){const e=(t[0]+16)/116,i=t[1],n=t[2]*Math.PI/180,r=P(e),s=P(e+i/500*Math.cos(n)),o=P(e-i/200*Math.sin(n)),a=A(3.021973625*s-1.617392459*r-.404875592*o),l=A(-.943766287*s+1.916279586*r+.027607165*o),h=A(.069407491*s-.22898585*r+1.159737864*o);return[c(a+.5|0,0,255),c(l+.5|0,0,255),c(h+.5|0,0,255),t[3]]}([Mn(t,e,i,o[0],r,a[0]),Mn(t,e,i,o[1],r,a[1]),o[2]+Mn(t,e,i,0,r,l),Mn(t,e,i,n[3],r,s[3])])}class En{constructor(t){this.opacity_=t.opacity,this.rotateWithView_=t.rotateWithView,this.rotation_=t.rotation,this.scale_=t.scale,this.scaleArray_=z(t.scale),this.displacement_=t.displacement,this.declutterMode_=t.declutterMode}clone(){const t=this.getScale();return new En({opacity:this.getOpacity(),scale:Array.isArray(t)?t.slice():t,rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()})}getOpacity(){return this.opacity_}getRotateWithView(){return this.rotateWithView_}getRotation(){return this.rotation_}getScale(){return this.scale_}getScaleArray(){return this.scaleArray_}getDisplacement(){return this.displacement_}getDeclutterMode(){return this.declutterMode_}getAnchor(){return Qe()}getImage(t){return Qe()}getHitDetectionImage(){return Qe()}getPixelRatio(t){return 1}getImageState(){return Qe()}getImageSize(){return Qe()}getOrigin(){return Qe()}getSize(){return Qe()}setDisplacement(t){this.displacement_=t}setOpacity(t){this.opacity_=t}setRotateWithView(t){this.rotateWithView_=t}setRotation(t){this.rotation_=t}setScale(t){this.scale_=t,this.scaleArray_=z(t)}listenImageChange(t){Qe()}load(){Qe()}unlistenImageChange(t){Qe()}ready(){return Promise.resolve()}}class kn extends En{constructor(t){super({opacity:1,rotateWithView:void 0!==t.rotateWithView&&t.rotateWithView,rotation:void 0!==t.rotation?t.rotation:0,scale:void 0!==t.scale?t.scale:1,displacement:void 0!==t.displacement?t.displacement:[0,0],declutterMode:t.declutterMode}),this.hitDetectionCanvas_=null,this.fill_=void 0!==t.fill?t.fill:null,this.origin_=[0,0],this.points_=t.points,this.radius=t.radius,this.radius2_=t.radius2,this.angle_=void 0!==t.angle?t.angle:0,this.stroke_=void 0!==t.stroke?t.stroke:null,this.size_,this.renderOptions_,this.imageState_=this.fill_&&this.fill_.loading()?Yi:Bi,this.imageState_===Yi&&this.ready().then(()=>this.imageState_=Bi),this.render()}clone(){const t=this.getScale(),e=new kn({fill:this.getFill()?this.getFill().clone():void 0,points:this.getPoints(),radius:this.getRadius(),radius2:this.getRadius2(),angle:this.getAngle(),stroke:this.getStroke()?this.getStroke().clone():void 0,rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),scale:Array.isArray(t)?t.slice():t,displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()});return e.setOpacity(this.getOpacity()),e}getAnchor(){const t=this.size_,e=this.getDisplacement(),i=this.getScaleArray();return[t[0]/2-e[0]/i[0],t[1]/2+e[1]/i[1]]}getAngle(){return this.angle_}getFill(){return this.fill_}setFill(t){this.fill_=t,this.render()}getHitDetectionImage(){return this.hitDetectionCanvas_||(this.hitDetectionCanvas_=this.createHitDetectionCanvas_(this.renderOptions_)),this.hitDetectionCanvas_}getImage(t){const e=this.fill_?.getKey(),i=`${t},${this.angle_},${this.radius},${this.radius2_},${this.points_},${e}`+Object.values(this.renderOptions_).join(\",\");let n=qi.get(i,null)?.getImage(1);if(!n){const e=this.renderOptions_,r=Math.ceil(e.size*t),s=a(r,r);this.draw_(e,s,t),n=s.canvas;const o=new Ki(n,void 0,null,Bi,null);qi.set(i,null,o),createImageBitmap(n).then(t=>{o.setImage(t)})}return n}getPixelRatio(t){return t}getImageSize(){return this.size_}getImageState(){return this.imageState_}getOrigin(){return this.origin_}getPoints(){return this.points_}getRadius(){return this.radius}setRadius(t){this.radius!==t&&(this.radius=t,this.render())}getRadius2(){return this.radius2_}setRadius2(t){this.radius2_!==t&&(this.radius2_=t,this.render())}getSize(){return this.size_}getStroke(){return this.stroke_}setStroke(t){this.stroke_=t,this.render()}listenImageChange(t){}load(){}unlistenImageChange(t){}calculateLineJoinSize_(t,e,i){if(0===e||this.points_===1/0||\"bevel\"!==t&&\"miter\"!==t)return e;let n=this.radius,r=void 0===this.radius2_?n:this.radius2_;if(n<r){const t=n;n=r,r=t}const s=void 0===this.radius2_?this.points_:2*this.points_,o=2*Math.PI/s,a=r*Math.sin(o),l=n-Math.sqrt(r*r-a*a),h=Math.sqrt(a*a+l*l),c=h/a;if(\"miter\"===t&&c<=i)return c*e;const u=e/2/c,f=e/2*(l/h),d=Math.sqrt((n+u)*(n+u)+f*f)-n;if(void 0===this.radius2_||\"bevel\"===t)return 2*d;const g=n*Math.sin(o),p=r-Math.sqrt(n*n-g*g),_=Math.sqrt(g*g+p*p)/g;if(_<=i){const t=_*e/2-r-n;return 2*Math.max(d,t)}return 2*d}createRenderOptions(){let t,e=hi,i=ui,n=0,r=null,s=0,o=0;this.stroke_&&(t=Zi(this.stroke_.getColor()??fi),o=this.stroke_.getWidth()??1,r=this.stroke_.getLineDash(),s=this.stroke_.getLineDashOffset()??0,i=this.stroke_.getLineJoin()??ui,e=this.stroke_.getLineCap()??hi,n=this.stroke_.getMiterLimit()??10);const a=this.calculateLineJoinSize_(i,o,n),l=Math.max(this.radius,this.radius2_||0);return{strokeStyle:t,strokeWidth:o,size:Math.ceil(2*l+a),lineCap:e,lineDash:r,lineDashOffset:s,lineJoin:i,miterLimit:n}}render(){this.renderOptions_=this.createRenderOptions();const t=this.renderOptions_.size;this.hitDetectionCanvas_=null,this.size_=[t,t]}draw_(t,e,i){if(e.scale(i,i),e.translate(t.size/2,t.size/2),this.createPath_(e),this.fill_){let t=this.fill_.getColor();null===t&&(t=li),e.fillStyle=Zi(t),e.fill()}t.strokeStyle&&(e.strokeStyle=t.strokeStyle,e.lineWidth=t.strokeWidth,t.lineDash&&(e.setLineDash(t.lineDash),e.lineDashOffset=t.lineDashOffset),e.lineCap=t.lineCap,e.lineJoin=t.lineJoin,e.miterLimit=t.miterLimit,e.stroke())}createHitDetectionCanvas_(t){let e;if(this.fill_){let i=this.fill_.getColor(),n=0;\"string\"==typeof i&&(i=F(i)),null===i?n=1:Array.isArray(i)&&(n=4===i.length?i[3]:1),0===n&&(e=a(t.size,t.size),this.drawHitDetectionCanvas_(t,e))}return e?e.canvas:this.getImage(1)}createPath_(t){let e=this.points_;const i=this.radius;if(e===1/0)t.arc(0,0,i,0,2*Math.PI);else{const n=void 0===this.radius2_?i:this.radius2_;void 0!==this.radius2_&&(e*=2);const r=this.angle_-Math.PI/2,s=2*Math.PI/e;for(let o=0;o<e;o++){const e=r+o*s,a=o%2==0?i:n;t.lineTo(a*Math.cos(e),a*Math.sin(e))}t.closePath()}}drawHitDetectionCanvas_(t,e){e.translate(t.size/2,t.size/2),this.createPath_(e),e.fillStyle=li,e.fill(),t.strokeStyle&&(e.strokeStyle=t.strokeStyle,e.lineWidth=t.strokeWidth,t.lineDash&&(e.setLineDash(t.lineDash),e.lineDashOffset=t.lineDashOffset),e.lineJoin=t.lineJoin,e.miterLimit=t.miterLimit,e.stroke())}ready(){return this.fill_?this.fill_.ready():Promise.resolve()}}class An extends kn{constructor(t){super({points:1/0,fill:(t=t||{radius:5}).fill,radius:t.radius,stroke:t.stroke,scale:void 0!==t.scale?t.scale:1,rotation:void 0!==t.rotation?t.rotation:0,rotateWithView:void 0!==t.rotateWithView&&t.rotateWithView,displacement:void 0!==t.displacement?t.displacement:[0,0],declutterMode:t.declutterMode})}clone(){const t=this.getScale(),e=new An({fill:this.getFill()?this.getFill().clone():void 0,stroke:this.getStroke()?this.getStroke().clone():void 0,radius:this.getRadius(),scale:Array.isArray(t)?t.slice():t,rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()});return e.setOpacity(this.getOpacity()),e}}class Pn{constructor(t){t=t||{},this.patternImage_=null,this.color_=null,void 0!==t.color&&this.setColor(t.color)}clone(){const t=this.getColor();return new Pn({color:Array.isArray(t)?t.slice():t||void 0})}getColor(){return this.color_}setColor(t){if(null!==t&&\"object\"==typeof t&&\"src\"in t){const e=Hi(null,t.src,{crossOrigin:\"anonymous\"},void 0,t.offset?null:t.color?t.color:null,!(t.offset&&t.size));e.ready().then(()=>{this.patternImage_=null}),e.getImageState()===Xi&&e.load(),e.getImageState()===Yi&&(this.patternImage_=e)}this.color_=t}getKey(){const t=this.getColor();return t?t instanceof CanvasPattern||t instanceof CanvasGradient?ei(t):\"object\"==typeof t&&\"src\"in t?t.src+\":\"+t.offset:F(t).toString():\"\"}loading(){return!!this.patternImage_}ready(){return this.patternImage_?this.patternImage_.ready():Promise.resolve()}}function On(t,e,i,n){return void 0!==i&&void 0!==n?[i/t,n/e]:void 0!==i?i/t:void 0!==n?n/e:1}class Rn extends En{constructor(t){const e=void 0!==(t=t||{}).opacity?t.opacity:1,i=void 0!==t.rotation?t.rotation:0,n=void 0!==t.scale?t.scale:1,r=void 0!==t.rotateWithView&&t.rotateWithView;super({opacity:e,rotation:i,scale:n,displacement:void 0!==t.displacement?t.displacement:[0,0],rotateWithView:r,declutterMode:t.declutterMode}),this.anchor_=void 0!==t.anchor?t.anchor:[.5,.5],this.normalizedAnchor_=null,this.anchorOrigin_=void 0!==t.anchorOrigin?t.anchorOrigin:\"top-left\",this.anchorXUnits_=void 0!==t.anchorXUnits?t.anchorXUnits:\"fraction\",this.anchorYUnits_=void 0!==t.anchorYUnits?t.anchorYUnits:\"fraction\",this.crossOrigin_=void 0!==t.crossOrigin?t.crossOrigin:null,this.referrerPolicy_=t.referrerPolicy;const s=void 0!==t.img?t.img:null;let o,a=t.src;if(Le(!(void 0!==a&&s),\"`image` and `src` cannot be provided at the same time\"),void 0!==a&&0!==a.length||!s||(a=s.src||ei(s)),Le(void 0!==a&&a.length>0,\"A defined and non-empty `src` or `image` must be provided\"),Le(!((void 0!==t.width||void 0!==t.height)&&void 0!==t.scale),\"`width` or `height` cannot be provided together with `scale`\"),void 0!==t.src?o=Xi:void 0!==s&&(o=\"complete\"in s?s.complete?s.src?Bi:Xi:Yi:Bi),this.color_=void 0!==t.color?F(t.color):null,this.iconImage_=Hi(s,a,{crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_},o,this.color_),this.offset_=void 0!==t.offset?t.offset:[0,0],this.offsetOrigin_=void 0!==t.offsetOrigin?t.offsetOrigin:\"top-left\",this.origin_=null,this.size_=void 0!==t.size?t.size:null,this.initialOptions_,void 0!==t.width||void 0!==t.height){let e,i;if(t.size)[e,i]=t.size;else{const n=this.getImage(1);if(n.width&&n.height)e=n.width,i=n.height;else if(n instanceof HTMLImageElement){this.initialOptions_=t;const e=()=>{if(this.unlistenImageChange(e),!this.initialOptions_)return;const i=this.iconImage_.getSize();this.setScale(On(i[0],i[1],t.width,t.height))};return void this.listenImageChange(e)}}void 0!==e&&this.setScale(On(e,i,t.width,t.height))}}clone(){let t,e,i;return this.initialOptions_?(e=this.initialOptions_.width,i=this.initialOptions_.height):(t=this.getScale(),t=Array.isArray(t)?t.slice():t),new Rn({anchor:this.anchor_.slice(),anchorOrigin:this.anchorOrigin_,anchorXUnits:this.anchorXUnits_,anchorYUnits:this.anchorYUnits_,color:this.color_&&this.color_.slice?this.color_.slice():this.color_||void 0,crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_,offset:this.offset_.slice(),offsetOrigin:this.offsetOrigin_,opacity:this.getOpacity(),rotateWithView:this.getRotateWithView(),rotation:this.getRotation(),scale:t,width:e,height:i,size:null!==this.size_?this.size_.slice():void 0,src:this.getSrc(),displacement:this.getDisplacement().slice(),declutterMode:this.getDeclutterMode()})}getAnchor(){let t=this.normalizedAnchor_;if(!t){t=this.anchor_;const e=this.getSize();if(\"fraction\"==this.anchorXUnits_||\"fraction\"==this.anchorYUnits_){if(!e)return null;t=this.anchor_.slice(),\"fraction\"==this.anchorXUnits_&&(t[0]*=e[0]),\"fraction\"==this.anchorYUnits_&&(t[1]*=e[1])}if(\"top-left\"!=this.anchorOrigin_){if(!e)return null;t===this.anchor_&&(t=this.anchor_.slice()),\"top-right\"!=this.anchorOrigin_&&\"bottom-right\"!=this.anchorOrigin_||(t[0]=-t[0]+e[0]),\"bottom-left\"!=this.anchorOrigin_&&\"bottom-right\"!=this.anchorOrigin_||(t[1]=-t[1]+e[1])}this.normalizedAnchor_=t}const e=this.getDisplacement(),i=this.getScaleArray();return[t[0]-e[0]/i[0],t[1]+e[1]/i[1]]}setAnchor(t){this.anchor_=t,this.normalizedAnchor_=null}getColor(){return this.color_}setColor(t){const e=t?F(t):null;if(this.color_===e||this.color_&&e&&this.color_.length===e.length&&this.color_.every((t,i)=>t===e[i]))return;this.color_=e;const i=this.getSrc(),n=void 0!==i?null:this.getHitDetectionImage(),r=void 0!==i?Xi:this.iconImage_.getImageState();this.iconImage_=Hi(n,i,{crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_},r,this.color_)}getImage(t){return this.iconImage_.getImage(t)}getPixelRatio(t){return this.iconImage_.getPixelRatio(t)}getImageSize(){return this.iconImage_.getSize()}getImageState(){return this.iconImage_.getImageState()}getHitDetectionImage(){return this.iconImage_.getHitDetectionImage()}getOrigin(){if(this.origin_)return this.origin_;let t=this.offset_;if(\"top-left\"!=this.offsetOrigin_){const e=this.getSize(),i=this.iconImage_.getSize();if(!e||!i)return null;t=t.slice(),\"top-right\"!=this.offsetOrigin_&&\"bottom-right\"!=this.offsetOrigin_||(t[0]=i[0]-e[0]-t[0]),\"bottom-left\"!=this.offsetOrigin_&&\"bottom-right\"!=this.offsetOrigin_||(t[1]=i[1]-e[1]-t[1])}return this.origin_=t,this.origin_}getSrc(){return this.iconImage_.getSrc()}setSrc(t){this.iconImage_=Hi(null,t,{crossOrigin:this.crossOrigin_,referrerPolicy:this.referrerPolicy_},Xi,this.color_)}getSize(){return this.size_?this.size_:this.iconImage_.getSize()}getWidth(){const t=this.getScaleArray();return this.size_?this.size_[0]*t[0]:this.iconImage_.getImageState()==Bi?this.iconImage_.getSize()[0]*t[0]:void 0}getHeight(){const t=this.getScaleArray();return this.size_?this.size_[1]*t[1]:this.iconImage_.getImageState()==Bi?this.iconImage_.getSize()[1]*t[1]:void 0}setScale(t){delete this.initialOptions_,super.setScale(t)}listenImageChange(t){this.iconImage_.addEventListener(je,t)}load(){this.iconImage_.load()}unlistenImageChange(t){this.iconImage_.removeEventListener(je,t)}ready(){return this.iconImage_.ready()}}class Ln{constructor(t){t=t||{},this.color_=void 0!==t.color?t.color:null,this.lineCap_=t.lineCap,this.lineDash_=void 0!==t.lineDash?t.lineDash:null,this.lineDashOffset_=t.lineDashOffset,this.lineJoin_=t.lineJoin,this.miterLimit_=t.miterLimit,this.offset_=t.offset,this.width_=t.width}clone(){const t=this.getColor();return new Ln({color:Array.isArray(t)?t.slice():t||void 0,lineCap:this.getLineCap(),lineDash:this.getLineDash()?this.getLineDash().slice():void 0,lineDashOffset:this.getLineDashOffset(),lineJoin:this.getLineJoin(),miterLimit:this.getMiterLimit(),offset:this.getOffset(),width:this.getWidth()})}getColor(){return this.color_}getLineCap(){return this.lineCap_}getLineDash(){return this.lineDash_}getLineDashOffset(){return this.lineDashOffset_}getLineJoin(){return this.lineJoin_}getMiterLimit(){return this.miterLimit_}getOffset(){return this.offset_}getWidth(){return this.width_}setColor(t){this.color_=t}setLineCap(t){this.lineCap_=t}setLineDash(t){this.lineDash_=t}setLineDashOffset(t){this.lineDashOffset_=t}setLineJoin(t){this.lineJoin_=t}setMiterLimit(t){this.miterLimit_=t}setOffset(t){this.offset_=t}setWidth(t){this.width_=t}}class Dn{constructor(t){t=t||{},this.geometry_=null,this.geometryFunction_=Fn,void 0!==t.geometry&&this.setGeometry(t.geometry),this.fill_=void 0!==t.fill?t.fill:null,this.image_=void 0!==t.image?t.image:null,this.renderer_=void 0!==t.renderer?t.renderer:null,this.hitDetectionRenderer_=void 0!==t.hitDetectionRenderer?t.hitDetectionRenderer:null,this.stroke_=void 0!==t.stroke?t.stroke:null,this.text_=void 0!==t.text?t.text:null,this.zIndex_=t.zIndex}clone(){let t=this.getGeometry();return t&&\"object\"==typeof t&&(t=t.clone()),new Dn({geometry:t??void 0,fill:this.getFill()?this.getFill().clone():void 0,image:this.getImage()?this.getImage().clone():void 0,renderer:this.getRenderer()??void 0,stroke:this.getStroke()?this.getStroke().clone():void 0,text:this.getText()?this.getText().clone():void 0,zIndex:this.getZIndex()})}getRenderer(){return this.renderer_}setRenderer(t){this.renderer_=t}setHitDetectionRenderer(t){this.hitDetectionRenderer_=t}getHitDetectionRenderer(){return this.hitDetectionRenderer_}getGeometry(){return this.geometry_}getGeometryFunction(){return this.geometryFunction_}getFill(){return this.fill_}setFill(t){this.fill_=t}getImage(){return this.image_}setImage(t){this.image_=t}getStroke(){return this.stroke_}setStroke(t){this.stroke_=t}getText(){return this.text_}setText(t){this.text_=t}getZIndex(){return this.zIndex_}setGeometry(t){\"function\"==typeof t?this.geometryFunction_=t:\"string\"==typeof t?this.geometryFunction_=function(e){return e.get(t)}:t?void 0!==t&&(this.geometryFunction_=function(){return t}):this.geometryFunction_=Fn,this.geometry_=t}setZIndex(t){this.zIndex_=t}}function Fn(t){return t.getGeometry()}class Tn{constructor(t){t=t||{},this.font_=t.font,this.rotation_=t.rotation,this.rotateWithView_=t.rotateWithView,this.keepUpright_=t.keepUpright,this.scale_=t.scale,this.scaleArray_=z(void 0!==t.scale?t.scale:1),this.text_=t.text,this.textAlign_=t.textAlign,this.justify_=t.justify,this.repeat_=t.repeat,this.textBaseline_=t.textBaseline,this.fill_=void 0!==t.fill?t.fill:new Pn({color:\"#333\"}),this.maxAngle_=void 0!==t.maxAngle?t.maxAngle:Math.PI/4,this.placement_=void 0!==t.placement?t.placement:\"point\",this.overflow_=!!t.overflow,this.stroke_=void 0!==t.stroke?t.stroke:null,this.offsetX_=void 0!==t.offsetX?t.offsetX:0,this.offsetY_=void 0!==t.offsetY?t.offsetY:0,this.backgroundFill_=t.backgroundFill?t.backgroundFill:null,this.backgroundStroke_=t.backgroundStroke?t.backgroundStroke:null,this.padding_=void 0===t.padding?null:t.padding,this.declutterMode_=t.declutterMode}clone(){const t=this.getScale();return new Tn({font:this.getFont(),placement:this.getPlacement(),repeat:this.getRepeat(),maxAngle:this.getMaxAngle(),overflow:this.getOverflow(),rotation:this.getRotation(),rotateWithView:this.getRotateWithView(),keepUpright:this.getKeepUpright(),scale:Array.isArray(t)?t.slice():t,text:this.getText(),textAlign:this.getTextAlign(),justify:this.getJustify(),textBaseline:this.getTextBaseline(),fill:this.getFill()instanceof Pn?this.getFill().clone():this.getFill(),stroke:this.getStroke()?this.getStroke().clone():void 0,offsetX:this.getOffsetX(),offsetY:this.getOffsetY(),backgroundFill:this.getBackgroundFill()?this.getBackgroundFill().clone():void 0,backgroundStroke:this.getBackgroundStroke()?this.getBackgroundStroke().clone():void 0,padding:this.getPadding()||void 0,declutterMode:this.getDeclutterMode()})}getOverflow(){return this.overflow_}getFont(){return this.font_}getMaxAngle(){return this.maxAngle_}getPlacement(){return this.placement_}getRepeat(){return this.repeat_}getOffsetX(){return this.offsetX_}getOffsetY(){return this.offsetY_}getFill(){return this.fill_}getRotateWithView(){return this.rotateWithView_}getKeepUpright(){return this.keepUpright_}getRotation(){return this.rotation_}getScale(){return this.scale_}getScaleArray(){return this.scaleArray_}getStroke(){return this.stroke_}getText(){return this.text_}getTextAlign(){return this.textAlign_}getJustify(){return this.justify_}getTextBaseline(){return this.textBaseline_}getBackgroundFill(){return this.backgroundFill_}getBackgroundStroke(){return this.backgroundStroke_}getPadding(){return this.padding_}getDeclutterMode(){return this.declutterMode_}setOverflow(t){this.overflow_=t}setFont(t){this.font_=t}setMaxAngle(t){this.maxAngle_=t}setOffsetX(t){this.offsetX_=t}setOffsetY(t){this.offsetY_=t}setPlacement(t){this.placement_=t}setRepeat(t){this.repeat_=t}setRotateWithView(t){this.rotateWithView_=t}setKeepUpright(t){this.keepUpright_=t}setFill(t){this.fill_=t}setRotation(t){this.rotation_=t}setScale(t){this.scale_=t,this.scaleArray_=z(void 0!==t?t:1)}setStroke(t){this.stroke_=t}setText(t){this.text_=t}setTextAlign(t){this.textAlign_=t}setJustify(t){this.justify_=t}setTextBaseline(t){this.textBaseline_=t}setBackgroundFill(t){this.backgroundFill_=t}setBackgroundStroke(t){this.backgroundStroke_=t}setPadding(t){this.padding_=t}}function zn(t){return!0}function $n(t,e){const i=function(t,e){const i=t.length,n=new Array(i);for(let r=0;r<i;++r){const i=t[r],s=\"filter\"in i?Cn(i.filter,W,e):zn;let o;if(Array.isArray(i.style)){const t=i.style.length;o=new Array(t);for(let n=0;n<t;++n)o[n]=Gn(i.style[n],e)}else o=[Gn(i.style,e)];n[r]={filter:s,styles:o}}return function(e){const r=[];let s=!1;for(let o=0;o<i;++o){if((0,n[o].filter)(e)&&(!t[o].else||!s)){s=!0;for(const t of n[o].styles){const i=t(e);i&&r.push(i)}}}return r}}(t,e=e??tt()),n={variables:{},properties:{},resolution:NaN,featureId:null,geometryType:\"\"};return function(t,r){if(n.properties=t.getPropertiesInternal(),n.resolution=r,e.featureId){const e=t.getId();n.featureId=void 0!==e?e:null}return e.geometryType&&(n.geometryType=se(t.getGeometry())),i(n)}}function Wn(t,e){e=e??tt();const i=t.length,n=new Array(i);for(let r=0;r<i;++r)n[r]=Gn(t[r],e);const r={variables:{},properties:{},resolution:NaN,featureId:null,geometryType:\"\"},s=new Array(i);return function(t,o){if(r.properties=t.getPropertiesInternal(),r.resolution=o,e.featureId){const e=t.getId();r.featureId=void 0!==e?e:null}e.geometryType&&(r.geometryType=se(t.getGeometry()));let a=0;for(let t=0;t<i;++t){const e=n[t](r);e&&(s[a]=e,a+=1)}return s.length=a,s}}function Gn(t,e){const i=Nn(t,\"\",e),n=Xn(t,\"\",e),r=function(t,e){const i=\"text-\",n=Un(t,i+\"value\",e);if(!n)return null;const r=Nn(t,i,e),s=Nn(t,i+\"background-\",e),o=Xn(t,i,e),a=Xn(t,i+\"background-\",e),l=Un(t,i+\"font\",e),h=Bn(t,i+\"max-angle\",e),c=Bn(t,i+\"offset-x\",e),u=Bn(t,i+\"offset-y\",e),f=jn(t,i+\"overflow\",e),d=Un(t,i+\"placement\",e),g=Bn(t,i+\"repeat\",e),p=Hn(t,i+\"scale\",e),_=jn(t,i+\"rotate-with-view\",e),m=Bn(t,i+\"rotation\",e),y=Un(t,i+\"align\",e),w=Un(t,i+\"justify\",e),x=Un(t,i+\"baseline\",e),v=jn(t,i+\"keep-upright\",e),S=qn(t,i+\"padding\",e),C=er(t,i+\"declutter-mode\"),b=new Tn({declutterMode:C});return function(t){if(b.setText(n(t)),r&&b.setFill(r(t)),s&&b.setBackgroundFill(s(t)),o&&b.setStroke(o(t)),a&&b.setBackgroundStroke(a(t)),l&&b.setFont(l(t)),h&&b.setMaxAngle(h(t)),c&&b.setOffsetX(c(t)),u&&b.setOffsetY(u(t)),f&&b.setOverflow(f(t)),d){const e=d(t);if(\"point\"!==e&&\"line\"!==e)throw new Error(\"Expected point or line for text-placement\");b.setPlacement(e)}if(g&&b.setRepeat(g(t)),p&&b.setScale(p(t)),_&&b.setRotateWithView(_(t)),m&&b.setRotation(m(t)),y){const e=y(t);if(\"left\"!==e&&\"center\"!==e&&\"right\"!==e&&\"end\"!==e&&\"start\"!==e)throw new Error(\"Expected left, right, center, start, or end for text-align\");b.setTextAlign(e)}if(w){const e=w(t);if(\"left\"!==e&&\"right\"!==e&&\"center\"!==e)throw new Error(\"Expected left, right, or center for text-justify\");b.setJustify(e)}if(x){const e=x(t);if(\"bottom\"!==e&&\"top\"!==e&&\"middle\"!==e&&\"alphabetic\"!==e&&\"hanging\"!==e)throw new Error(\"Expected bottom, top, middle, alphabetic, or hanging for text-baseline\");b.setTextBaseline(e)}return S&&b.setPadding(S(t)),v&&b.setKeepUpright(v(t)),b}}(t,e),s=function(t,e){if(\"icon-src\"in t)return function(t,e){const i=\"icon-\",n=i+\"src\",r=nr(t[n],n),s=Jn(t,i+\"anchor\",e),o=Hn(t,i+\"scale\",e),a=Bn(t,i+\"opacity\",e),l=Jn(t,i+\"displacement\",e),h=Bn(t,i+\"rotation\",e),c=jn(t,i+\"rotate-with-view\",e),u=Qn(t,i+\"anchor-origin\"),f=tr(t,i+\"anchor-x-units\"),d=tr(t,i+\"anchor-y-units\"),g=Yn(t,i+\"color\");let p,_=null;if(void 0!==g){Array.isArray(g)&&g.length>0&&\"string\"==typeof g[0]?_=Vn(t,i+\"color\",e):p=sr(g,i+\"color\")}const m=function(t,e){const i=t[e];if(void 0===i)return;if(\"string\"!=typeof i)throw new Error(`Expected a string for ${e}`);return i}(t,i+\"cross-origin\"),y=function(t,e){const i=t[e];if(void 0===i)return;return ir(i,e)}(t,i+\"offset\"),w=Qn(t,i+\"offset-origin\"),x=Zn(t,i+\"width\"),v=Zn(t,i+\"height\"),S=function(t,e){const i=t[e];if(void 0===i)return;if(\"number\"==typeof i)return z(i);if(!Array.isArray(i))throw new Error(`Expected a number or size array for ${e}`);if(2!==i.length||\"number\"!=typeof i[0]||\"number\"!=typeof i[1])throw new Error(`Expected a number or size array for ${e}`);return i}(t,i+\"size\"),C=er(t,i+\"declutter-mode\"),b={src:r,anchorOrigin:u,anchorXUnits:f,anchorYUnits:d,crossOrigin:m,offset:y,offsetOrigin:w,height:v,width:x,size:S,declutterMode:C};let M=null;return function(t){if(M)_&&M.setColor(_(t));else{const e=_?_(t):p;M=new Rn(void 0!==e?Object.assign({},b,{color:e}):Object.assign({},b))}return a&&M.setOpacity(a(t)),l&&M.setDisplacement(l(t)),h&&M.setRotation(h(t)),c&&M.setRotateWithView(c(t)),o&&M.setScale(o(t)),s&&M.setAnchor(s(t)),M}}(t,e);if(\"shape-points\"in t)return function(t,e){const i=\"shape-\",n=i+\"points\",r=i+\"radius\",s=rr(t[n],n);if(!(r in t))throw new Error(`Expected a number for ${r}`);const o=Bn(t,r,e),a=\"number\"==typeof t[r]?t[r]:5,l=i+\"radius2\",h=Bn(t,l,e),c=\"number\"==typeof t[l]?t[l]:void 0,u=Nn(t,i,e),f=Xn(t,i,e),d=Hn(t,i+\"scale\",e),g=Jn(t,i+\"displacement\",e),p=Bn(t,i+\"rotation\",e),_=jn(t,i+\"rotate-with-view\",e),m=Zn(t,i+\"angle\"),y=er(t,i+\"declutter-mode\"),w=new kn({points:s,radius:a,radius2:c,angle:m,declutterMode:y});return function(t){return o&&w.setRadius(o(t)),h&&w.setRadius2(h(t)),u&&w.setFill(u(t)),f&&w.setStroke(f(t)),g&&w.setDisplacement(g(t)),p&&w.setRotation(p(t)),_&&w.setRotateWithView(_(t)),d&&w.setScale(d(t)),w}}(t,e);if(\"circle-radius\"in t)return function(t,e){const i=\"circle-\",n=Nn(t,i,e),r=Xn(t,i,e),s=Bn(t,i+\"radius\",e),o=Hn(t,i+\"scale\",e),a=Jn(t,i+\"displacement\",e),l=Bn(t,i+\"rotation\",e),h=jn(t,i+\"rotate-with-view\",e),c=er(t,i+\"declutter-mode\"),u=new An({radius:5,declutterMode:c});return function(t){return s&&u.setRadius(s(t)),n&&u.setFill(n(t)),r&&u.setStroke(r(t)),a&&u.setDisplacement(a(t)),l&&u.setRotation(l(t)),h&&u.setRotateWithView(h(t)),o&&u.setScale(o(t)),u}}(t,e);return null}(t,e),o=Bn(t,\"z-index\",e);if(!(i||n||r||s||Xe(t)))throw new Error(\"No fill, stroke, point, or text symbolizer properties in style: \"+JSON.stringify(t));const a=new Dn;return function(t){let e=!0;if(i){const n=i(t);n&&(e=!1),a.setFill(n)}if(n){const i=n(t);i&&(e=!1),a.setStroke(i)}if(r){const i=r(t);i&&(e=!1),a.setText(i)}if(s){const i=s(t);i&&(e=!1),a.setImage(i)}return o&&a.setZIndex(o(t)),e?null:a}}function Nn(t,e,i){let n;if(e+\"fill-pattern-src\"in t)n=function(t,e,i){const n=Un(t,e+\"pattern-src\",i),r=Kn(t,e+\"pattern-offset\",i),s=Kn(t,e+\"pattern-size\",i),o=Vn(t,e+\"color\",i);return function(t){return{src:n(t),offset:r&&r(t),size:s&&s(t),color:o&&o(t)}}}(t,e+\"fill-\",i);else{if(\"none\"===t[e+\"fill-color\"])return t=>null;n=Vn(t,e+\"fill-color\",i)}if(!n)return null;const r=new Pn;return function(t){const e=n(t);return e===m?null:(r.setColor(e),r)}}function Xn(t,e,i){const n=Bn(t,e+\"stroke-width\",i),r=Vn(t,e+\"stroke-color\",i);if(!n&&!r)return null;const s=Un(t,e+\"stroke-line-cap\",i),o=Un(t,e+\"stroke-line-join\",i),a=qn(t,e+\"stroke-line-dash\",i),l=Bn(t,e+\"stroke-line-dash-offset\",i),h=Bn(t,e+\"stroke-miter-limit\",i),c=Bn(t,e+\"stroke-offset\",i),u=new Ln;return function(t){if(r){const e=r(t);if(e===m)return null;u.setColor(e)}if(n&&u.setWidth(n(t)),s){const e=s(t);if(\"butt\"!==e&&\"round\"!==e&&\"square\"!==e)throw new Error(\"Expected butt, round, or square line cap\");u.setLineCap(e)}if(o){const e=o(t);if(\"bevel\"!==e&&\"round\"!==e&&\"miter\"!==e)throw new Error(\"Expected bevel, round, or miter line join\");u.setLineJoin(e)}return a&&u.setLineDash(a(t)),l&&u.setLineDashOffset(l(t)),h&&u.setMiterLimit(h(t)),c&&u.setOffset(c(t)),u}}function Yn(t,e){if(!(e in t))return;const i=t[e];return void 0===i?void 0:i}function Bn(t,e,i){const n=Yn(t,e);if(void 0===n)return;const r=Cn(n,G,i);return function(t){return rr(r(t),e)}}function Un(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,N,i);return function(t){return nr(r(t),e)}}function jn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,W,i);return function(t){const i=r(t);if(\"boolean\"!=typeof i)throw new Error(`Expected a boolean for ${e}`);return i}}function Vn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,X,i);return function(t){return sr(r(t),e)}}function qn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;if(Array.isArray(n)&&(0===n.length||\"string\"!=typeof n[0])){const t=n.map((t,n)=>{if(\"number\"==typeof t)return()=>t;const r=Cn(t,G,i);return function(t){return rr(r(t),`${e}[${n}]`)}});return function(e){const i=new Array(t.length);for(let n=0;n<t.length;++n)i[n]=t[n](e);return i}}const r=Cn(n,Y,i);return function(t){return ir(r(t),e)}}function Jn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,Y,i);return function(t){const i=ir(r(t),e);if(2!==i.length)throw new Error(`Expected two numbers for ${e}`);return i}}function Kn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,Y,i);return function(t){return or(r(t),e)}}function Hn(t,e,i){const n=Yn(t,e);if(void 0===n)return null;const r=Cn(n,Y|G,i);return function(t){return function(t,e){if(\"number\"==typeof t)return t;return or(t,e)}(r(t),e)}}function Zn(t,e){const i=t[e];if(void 0!==i){if(\"number\"!=typeof i)throw new Error(`Expected a number for ${e}`);return i}}function Qn(t,e){const i=t[e];if(void 0!==i){if(\"bottom-left\"!==i&&\"bottom-right\"!==i&&\"top-left\"!==i&&\"top-right\"!==i)throw new Error(`Expected bottom-left, bottom-right, top-left, or top-right for ${e}`);return i}}function tr(t,e){const i=t[e];if(void 0!==i){if(\"pixels\"!==i&&\"fraction\"!==i)throw new Error(`Expected pixels or fraction for ${e}`);return i}}function er(t,e){const i=t[e];if(void 0!==i){if(\"string\"!=typeof i)throw new Error(`Expected a string for ${e}`);if(\"declutter\"!==i&&\"obstacle\"!==i&&\"none\"!==i)throw new Error(`Expected declutter, obstacle, or none for ${e}`);return i}}function ir(t,e){if(!Array.isArray(t))throw new Error(`Expected an array for ${e}`);const i=t.length;for(let n=0;n<i;++n)if(\"number\"!=typeof t[n])throw new Error(`Expected an array of numbers for ${e}`);return t}function nr(t,e){if(\"string\"!=typeof t)throw new Error(`Expected a string for ${e}`);return t}function rr(t,e){if(\"number\"!=typeof t)throw new Error(`Expected a number for ${e}`);return t}function sr(t,e){if(\"string\"==typeof t)return t;const i=ir(t,e),n=i.length;if(n<3||n>4)throw new Error(`Expected a color with 3 or 4 values for ${e}`);return i}function or(t,e){const i=ir(t,e);if(2!==i.length)throw new Error(`Expected an array of two numbers for ${e}`);return i}const ar=\"BUILD_INSTRUCTIONS\",lr=\"DISPOSE_INSTRUCTIONS\",hr=\"RENDER\",cr={radians:6370997/(2*Math.PI),degrees:2*Math.PI*6370997/360,ft:.3048,m:1,\"us-ft\":1200/3937};class ur{constructor(t){this.code_=t.code,this.units_=t.units,this.extent_=void 0!==t.extent?t.extent:null,this.worldExtent_=void 0!==t.worldExtent?t.worldExtent:null,this.axisOrientation_=void 0!==t.axisOrientation?t.axisOrientation:\"enu\",this.global_=void 0!==t.global&&t.global,this.canWrapX_=!(!this.global_||!this.extent_),this.getPointResolutionFunc_=t.getPointResolution,this.defaultTileGrid_=null,this.metersPerUnit_=t.metersPerUnit}canWrapX(){return this.canWrapX_}getCode(){return this.code_}getExtent(){return this.extent_}getUnits(){return this.units_}getMetersPerUnit(){return this.metersPerUnit_||cr[this.units_]}getWorldExtent(){return this.worldExtent_}getAxisOrientation(){return this.axisOrientation_}isGlobal(){return this.global_}setGlobal(t){this.global_=t,this.canWrapX_=!(!t||!this.extent_)}getDefaultTileGrid(){return this.defaultTileGrid_}setDefaultTileGrid(t){this.defaultTileGrid_=t}setExtent(t){this.extent_=t,this.canWrapX_=!(!this.global_||!t)}setWorldExtent(t){this.worldExtent_=t}setGetPointResolution(t){this.getPointResolutionFunc_=t}getPointResolutionFunc(){return this.getPointResolutionFunc_}}const fr=6378137,dr=Math.PI*fr,gr=[-dr,-dr,dr,dr],pr=[-180,-85,180,85],_r=fr*Math.log(Math.tan(Math.PI/2));class mr extends ur{constructor(t){super({code:t,units:\"m\",extent:gr,global:!0,worldExtent:pr,getPointResolution:function(t,e){return t/Math.cosh(e[1]/fr)}})}}const yr=[new mr(\"EPSG:3857\"),new mr(\"EPSG:102100\"),new mr(\"EPSG:102113\"),new mr(\"EPSG:900913\"),new mr(\"http://www.opengis.net/def/crs/EPSG/0/3857\"),new mr(\"http://www.opengis.net/gml/srs/epsg.xml#3857\")];function wr(t,e,i,n){const r=t.length;i=i>1?i:2,n=n??i,void 0===e&&(e=i>2?t.slice():new Array(r));for(let i=0;i<r;i+=n){e[i]=dr*t[i]/180;let n=fr*Math.log(Math.tan(Math.PI*(+t[i+1]+90)/360));n>_r?n=_r:n<-_r&&(n=-_r),e[i+1]=n}return e}function xr(t,e,i,n){const r=t.length;i=i>1?i:2,n=n??i,void 0===e&&(e=i>2?t.slice():new Array(r));for(let i=0;i<r;i+=n)e[i]=180*t[i]/dr,e[i+1]=360*Math.atan(Math.exp(t[i+1]/fr))/Math.PI-90;return e}const vr=[-180,-90,180,90],Sr=6378137*Math.PI/180;class Cr extends ur{constructor(t,e){super({code:t,units:\"degrees\",extent:vr,axisOrientation:e,global:!0,metersPerUnit:Sr,worldExtent:vr})}}const br=[new Cr(\"CRS:84\"),new Cr(\"EPSG:4326\",\"neu\"),new Cr(\"urn:ogc:def:crs:OGC:1.3:CRS84\"),new Cr(\"urn:ogc:def:crs:OGC:2:84\"),new Cr(\"http://www.opengis.net/def/crs/OGC/1.3/CRS84\"),new Cr(\"http://www.opengis.net/gml/srs/epsg.xml#4326\",\"neu\"),new Cr(\"http://www.opengis.net/def/crs/EPSG/0/4326\",\"neu\")];let Mr={};let Ir={};function Er(t,e,i){const n=t.getCode(),r=e.getCode();n in Ir||(Ir[n]={}),Ir[n][r]=i}function kr(t,e){return t in Ir&&e in Ir[t]?Ir[t][e]:null}const Ar=.9996,Pr=.00669438,Or=Pr*Pr,Rr=Or*Pr,Lr=Pr/(1-Pr),Dr=Math.sqrt(1-Pr),Fr=(1-Dr)/(1+Dr),Tr=Fr*Fr,zr=Tr*Fr,$r=zr*Fr,Wr=$r*Fr,Gr=.9983242984503243,Nr=15*Or/256+45*Rr/1024,Xr=35*Rr/3072,Yr=1.5*Fr-27/32*zr+269/512*Wr,Br=21/16*Tr-55/32*$r,Ur=151/96*zr-417/128*Wr,jr=1097/512*$r,Vr=6378137;function qr(t,e,i){const n=t-5e5,r=(i.north?e:e-1e7)/Ar/(Vr*Gr),s=r+Yr*Math.sin(2*r)+Br*Math.sin(4*r)+Ur*Math.sin(6*r)+jr*Math.sin(8*r),o=Math.sin(s),a=o*o,l=Math.cos(s),h=o/l,c=h*h,u=c*c,f=1-Pr*a,p=Math.sqrt(1-Pr*a),m=Lr*l**2,y=m*m,w=n/(Vr/p*Ar),x=w*w,v=x*w,S=v*w,C=S*w,b=s-h/((1-Pr)/f)*(x/2-S/24*(5+3*c+10*m-4*y-9*Lr))+C*w/720*(61+90*c+298*m+45*u-252*Lr-3*y);let M=(w-v/6*(1+2*c+m)+C/120*(5-2*m+28*c-3*y+8*Lr+24*u))/l;return M=_(M+g(Kr(i.number)),-Math.PI,Math.PI),[d(M),d(b)]}function Jr(t,e,i){t=_(t,-180,180),e<-80?e=-80:e>84&&(e=84);const n=g(e),r=Math.sin(n),s=Math.cos(n),o=r/s,a=o*o,l=a*a,h=g(t),c=g(Kr(i.number)),u=Vr/Math.sqrt(1-Pr*r**2),f=Lr*s**2,d=s*_(h-c,-Math.PI,Math.PI),p=d*d,m=p*d,y=m*d,w=y*d,x=w*d,v=Vr*(Gr*n-.002514607064228144*Math.sin(2*n)+Nr*Math.sin(4*n)-Xr*Math.sin(6*n)),S=Ar*u*(d+m/6*(1-a+f)+w/120*(5-18*a+l+72*f-58*Lr))+5e5;let C=Ar*(v+u*o*(p/2+y/24*(5-a+9*f+4*f**2)+x/720*(61-58*a+l+600*f-330*Lr)));return i.north||(C+=1e7),[S,C]}function Kr(t){return 6*(t-1)-180+3}const Hr=[/^EPSG:(\\d+)$/,/^urn:ogc:def:crs:EPSG::(\\d+)$/,/^http:\\/\\/www\\.opengis\\.net\\/def\\/crs\\/EPSG\\/0\\/(\\d+)$/];function Zr(t){let e=0;for(const i of Hr){const n=t.match(i);if(n){e=parseInt(n[1]);break}}if(!e)return null;let i=0,n=!1;return e>32700&&e<32761?i=e-32700:e>32600&&e<32661&&(n=!0,i=e-32600),i?{number:i,north:n}:null}function Qr(t,e){return function(i,n,r,s){const o=i.length;r=r>1?r:2,s=s??r,n||(n=r>2?i.slice():new Array(o));for(let r=0;r<o;r+=s){const s=i[r],o=i[r+1],a=t(s,o,e);n[r]=a[0],n[r+1]=a[1]}return n}}const ts=[function(t){const e=Zr(t.getCode());return e?{forward:Qr(Jr,e),inverse:Qr(qr,e)}:null}],es=[function(t){return Zr(t)?new ur({code:t,units:\"m\"}):null}];function is(t,e){if(void 0!==e)for(let i=0,n=t.length;i<n;++i)e[i]=t[i];else e=t.slice();return e}function ns(t){!function(t,e){Mr[t]=e}(t.getCode(),t),Er(t,t,is)}function rs(t){if(\"string\"!=typeof t)return t;const e=Mr[i=t]||Mr[i.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\\w+)$/,\"EPSG:$3\")]||null;var i;if(e)return e;for(const e of es){const i=e(t);if(i)return i}return null}function ss(t){!function(t){t.forEach(ns)}(t),t.forEach(function(e){t.forEach(function(t){e!==t&&Er(e,t,is)})})}function os(t,e){return function(i,n,r,s){return n=t(i,n,r,s),e(n,n,r,s)}}function as(t,e){return function(t,e){const i=t.getCode(),n=e.getCode();let r=kr(i,n);if(r)return r;let s=null,o=null;for(const i of ts)s||(s=i(t)),o||(o=i(e));if(!s&&!o)return null;const a=\"EPSG:4326\";if(o)if(s)r=os(s.inverse,o.forward);else{const t=kr(i,a);t&&(r=os(t,o.forward))}else{const t=kr(a,n);t&&(r=os(s.inverse,t))}return r&&(ns(t),ns(e),Er(t,e,r)),r}(rs(t),rs(e))}var ls,hs,cs;ss(yr),ss(br),ls=yr,hs=wr,cs=xr,br.forEach(function(t){ls.forEach(function(e){Er(t,e,hs),Er(e,t,cs)})});const us=Fe(),fs=[NaN,NaN];class ds extends ni{constructor(){super(),this.extent_=[1/0,1/0,-1/0,-1/0],this.extentRevision_=-1,this.simplifiedGeometryMaxMinSquaredTolerance=0,this.simplifiedGeometryRevision=0,this.simplifyTransformedInternal=Je((t,e,i)=>{if(!i)return this.getSimplifiedGeometry(e);const n=this.clone();return n.applyTransform(i),n.getSimplifiedGeometry(e)})}simplifyTransformed(t,e){return this.simplifyTransformedInternal(this.getRevision(),t,e)}clone(){return Qe()}closestPointXY(t,e,i,n){return Qe()}containsXY(t,e){return 0===this.closestPointXY(t,e,fs,Number.MIN_VALUE)}getClosestPoint(t,e){return e=e||[NaN,NaN],this.closestPointXY(t[0],t[1],e,1/0),e}intersectsCoordinate(t){return this.containsXY(t[0],t[1])}computeExtent(t){return Qe()}getExtent(t){if(this.extentRevision_!=this.getRevision()){const t=this.computeExtent(this.extent_);(isNaN(t[0])||isNaN(t[1]))&&me(t),this.extentRevision_=this.getRevision()}return function(t,e){return e?(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e):t}(this.extent_,t)}rotate(t,e){Qe()}scale(t,e,i){Qe()}simplify(t){return this.getSimplifiedGeometry(t*t)}getSimplifiedGeometry(t){return Qe()}getType(){return Qe()}applyTransform(t){Qe()}intersectsExtent(t){return Qe()}translate(t,e){Qe()}transform(t,e){const i=rs(t),n=\"tile-pixels\"==i.getUnits()?function(t,n,r){const s=i.getExtent(),o=i.getWorldExtent(),a=Ce(o)/Ce(s);$e(us,o[0],o[3],a,-a,0,0,0);const l=Ae(t,0,t.length,r,us,n),h=as(i,e);return h?h(l,l,r):l}:as(i,e);return this.applyTransform(n),this}}class gs extends ds{constructor(){super(),this.layout=\"XY\",this.stride=2,this.flatCoordinates}computeExtent(t){return we(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t)}getCoordinates(){return Qe()}getFirstCoordinate(){return this.flatCoordinates.slice(0,this.stride)}getFlatCoordinates(){return this.flatCoordinates}getLastCoordinate(){return this.flatCoordinates.slice(this.flatCoordinates.length-this.stride)}getLayout(){return this.layout}getSimplifiedGeometry(t){if(this.simplifiedGeometryRevision!==this.getRevision()&&(this.simplifiedGeometryMaxMinSquaredTolerance=0,this.simplifiedGeometryRevision=this.getRevision()),t<0||0!==this.simplifiedGeometryMaxMinSquaredTolerance&&t<=this.simplifiedGeometryMaxMinSquaredTolerance)return this;const e=this.getSimplifiedGeometryInternal(t);return e.getFlatCoordinates().length<this.flatCoordinates.length?e:(this.simplifiedGeometryMaxMinSquaredTolerance=t,this)}getSimplifiedGeometryInternal(t){return this}getStride(){return this.stride}setFlatCoordinates(t,e){this.stride=ps(t),this.layout=t,this.flatCoordinates=e}setCoordinates(t,e){Qe()}setLayout(t,e,i){let n;if(t)n=ps(t);else{for(let t=0;t<i;++t){if(0===e.length)return this.layout=\"XY\",void(this.stride=2);e=e[0]}n=e.length,t=function(t){let e;2==t?e=\"XY\":3==t?e=\"XYZ\":4==t&&(e=\"XYZM\");return e}(n)}this.layout=t,this.stride=n}applyTransform(t){this.flatCoordinates&&(t(this.flatCoordinates,this.flatCoordinates,this.layout.startsWith(\"XYZ\")?3:2,this.stride),this.changed())}rotate(t,e){const i=this.getFlatCoordinates();if(i){const n=this.getStride();Pe(i,0,i.length,n,t,e,i),this.changed()}}scale(t,e,i){void 0===e&&(e=t),i||(i=Se(this.getExtent()));const n=this.getFlatCoordinates();if(n){const r=this.getStride();!function(t,e,i,n,r,s,o,a){a=a||[];const l=o[0],h=o[1];let c=0;for(let o=e;o<i;o+=n){const e=t[o]-l,i=t[o+1]-h;a[c++]=l+r*e,a[c++]=h+s*i;for(let e=o+2;e<o+n;++e)a[c++]=t[e]}a&&a.length!=c&&(a.length=c)}(n,0,n.length,r,t,e,i,n),this.changed()}}translate(t,e){const i=this.getFlatCoordinates();if(i){const n=this.getStride();!function(t,e,i,n,r,s,o){o=o||[];let a=0;for(let l=e;l<i;l+=n){o[a++]=t[l]+r,o[a++]=t[l+1]+s;for(let e=l+2;e<l+n;++e)o[a++]=t[e]}o&&o.length!=a&&(o.length=a)}(i,0,i.length,n,t,e,i),this.changed()}}}function ps(t){let e;return\"XY\"==t?e=2:\"XYZ\"==t||\"XYM\"==t?e=3:\"XYZM\"==t&&(e=4),e}function _s(t,e,i,n,r,s,o){const a=t[e],l=t[e+1],h=t[i]-a,c=t[i+1]-l;let u;if(0===h&&0===c)u=e;else{const f=((r-a)*h+(s-l)*c)/(h*h+c*c);if(f>1)u=i;else{if(f>0){for(let r=0;r<n;++r)o[r]=p(t[e+r],t[i+r],f);return void(o.length=n)}u=e}}for(let e=0;e<n;++e)o[e]=t[u+e];o.length=n}function ms(t,e,i,n,r){let s=t[e],o=t[e+1];for(e+=n;e<i;e+=n){const i=t[e],n=t[e+1],a=f(s,o,i,n);a>r&&(r=a),s=i,o=n}return r}function ys(t,e,i,n,r,s,o,a,l,h,c){if(e==i)return h;let u,d;if(0===r){if(d=f(o,a,t[e],t[e+1]),d<h){for(u=0;u<n;++u)l[u]=t[e+u];return l.length=n,d}return h}c=c||[NaN,NaN];let g=e+n;for(;g<i;)if(_s(t,g-n,g,n,o,a,c),d=f(o,a,c[0],c[1]),d<h){for(h=d,u=0;u<n;++u)l[u]=c[u];l.length=n,g+=n}else g+=n*Math.max((Math.sqrt(d)-Math.sqrt(h))/r|0,1);if(s&&(_s(t,i-n,e,n,o,a,c),d=f(o,a,c[0],c[1]),d<h)){for(h=d,u=0;u<n;++u)l[u]=c[u];l.length=n}return h}function ws(t,e,i,n){for(let r=0,s=i.length;r<s;++r){const s=i[r];for(let i=0;i<n;++i)t[e++]=s[i]}return e}function xs(e,i,n,r,s,o,a){let l,h;const c=(n-i)/r;if(1===c)l=i;else if(2===c)l=i,h=s;else if(0!==c){let o=e[i],a=e[i+1],c=0;const u=[0];for(let t=i+r;t<n;t+=r){const i=e[t],n=e[t+1];c+=Math.sqrt((i-o)*(i-o)+(n-a)*(n-a)),u.push(c),o=i,a=n}const f=s*c,d=function(e,i,n){let r,s;n=n||t;let o=0,a=e.length,l=!1;for(;o<a;)r=o+(a-o>>1),s=+n(e[r],i),s<0?o=r+1:(a=r,l=!s);return l?o:~o}(u,f);d<0?(h=(f-u[-d-2])/(u[-d-1]-u[-d-2]),l=i+(-d-2)*r):l=i+d*r}a=a>1?a:2,o=o||new Array(a);for(let t=0;t<a;++t)o[t]=void 0===l?NaN:void 0===h?e[l+t]:p(e[l+t],e[l+r+t],h);return o}function vs(t,e,i,n,r){const s=function(t,e){let i;return i=e(function(t){return[t[0],t[1]]}(t)),i||(i=e(function(t){return[t[2],t[1]]}(t)),i||(i=e(function(t){return[t[2],t[3]]}(t)),i||(i=e(function(t){return[t[0],t[3]]}(t)),i||!1)))}(r,function(r){return!Ss(t,e,i,n,r[0],r[1])});return!s}function Ss(t,e,i,n,r,s){let o=0,a=t[i-n],l=t[i-n+1];for(;e<i;e+=n){const i=t[e],n=t[e+1];l<=s?n>s&&(i-a)*(s-l)-(r-a)*(n-l)>0&&o++:n<=s&&(i-a)*(s-l)-(r-a)*(n-l)<0&&o--,a=i,l=n}return 0!==o}function Cs(t,e,i,n,r,s){if(0===i.length)return!1;if(!Ss(t,e,i[0],n,r,s))return!1;for(let e=1,o=i.length;e<o;++e)if(Ss(t,i[e-1],i[e],n,r,s))return!1;return!0}function bs(t,e,i,n,r){let s;for(e+=n;e<i;e+=n)if(s=r(t.slice(e-n,e),t.slice(e,e+n)),s)return s;return!1}function Ms(t,e,i,n,r,s){return s=s??xe([1/0,1/0,-1/0,-1/0],t,e,i,n),!!be(r,s)&&(s[0]>=r[0]&&s[2]<=r[2]||s[1]>=r[1]&&s[3]<=r[3]||bs(t,e,i,n,function(t,e){return function(t,e,i){let n=!1;const r=pe(t,e),s=pe(t,i);if(r===ae||s===ae)n=!0;else{const o=t[0],a=t[1],l=t[2],h=t[3],c=e[0],u=e[1],f=i[0],d=i[1],g=(d-u)/(f-c);let p,_;s&le&&!(r&le)&&(p=f-(d-h)/g,n=p>=o&&p<=l),n||!(s&he)||r&he||(_=d-(f-l)*g,n=_>=a&&_<=h),n||!(s&ce)||r&ce||(p=f-(d-a)/g,n=p>=o&&p<=l),n||!(s&ue)||r&ue||(_=d-(f-o)*g,n=_>=a&&_<=h)}return n}(r,t,e)}))}function Is(t,e,i,n,r){if(!function(t,e,i,n,r){return!!(Ms(t,e,i,n,r)||Ss(t,e,i,n,r[0],r[1])||Ss(t,e,i,n,r[0],r[3])||Ss(t,e,i,n,r[2],r[1])||Ss(t,e,i,n,r[2],r[3]))}(t,e,i[0],n,r))return!1;if(1===i.length)return!0;for(let e=1,s=i.length;e<s;++e)if(vs(t,i[e-1],i[e],n,r)&&!Ms(t,i[e-1],i[e],n,r))return!1;return!0}function Es(t,e,i,n,r,s,o){const a=(i-e)/n;if(a<3){for(;e<i;e+=n)s[o++]=t[e],s[o++]=t[e+1];return o}const l=new Array(a);l[0]=1,l[a-1]=1;const h=[e,i-n];let c=0;for(;h.length>0;){const i=h.pop(),s=h.pop();let o=0;const a=t[s],f=t[s+1],d=t[i],g=t[i+1];for(let e=s+n;e<i;e+=n){const i=u(t[e],t[e+1],a,f,d,g);i>o&&(c=e,o=i)}o>r&&(l[(c-e)/n]=1,s+n<c&&h.push(s,c),c+n<i&&h.push(c,i))}for(let i=0;i<a;++i)l[i]&&(s[o++]=t[e+i*n],s[o++]=t[e+i*n+1]);return o}function ks(t,e){return e*Math.round(t/e)}function As(t,e,i,n,r,s,o){if(e==i)return o;let a,l,h=ks(t[e],r),c=ks(t[e+1],r);e+=n,s[o++]=h,s[o++]=c;do{if(a=ks(t[e],r),l=ks(t[e+1],r),(e+=n)==i)return s[o++]=a,s[o++]=l,o}while(a==h&&l==c);for(;e<i;){const i=ks(t[e],r),u=ks(t[e+1],r);if(e+=n,i==a&&u==l)continue;const f=a-h,d=l-c,g=i-h,p=u-c;f*p==d*g&&(f<0&&g<f||f==g||f>0&&g>f)&&(d<0&&p<d||d==p||d>0&&p>d)?(a=i,l=u):(s[o++]=a,s[o++]=l,h=a,c=l,a=i,l=u)}return s[o++]=a,s[o++]=l,o}function Ps(t,e,i,n,r,s,o,a){for(let l=0,h=i.length;l<h;++l){const h=i[l];o=As(t,e,h,n,r,s,o),a.push(o),e=h}return o}class Os extends gs{constructor(t,e){super(),this.flatMidpoint_=null,this.flatMidpointRevision_=-1,this.maxDelta_=-1,this.maxDeltaRevision_=-1,void 0===e||Array.isArray(t[0])?this.setCoordinates(t,e):this.setFlatCoordinates(e,t)}appendCoordinate(t){i(this.flatCoordinates,t),this.changed()}clone(){const t=new Os(this.flatCoordinates.slice(),this.layout);return t.applyProperties(this),t}closestPointXY(t,e,i,n){return n<fe(this.getExtent(),t,e)?n:(this.maxDeltaRevision_!=this.getRevision()&&(this.maxDelta_=Math.sqrt(ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,0)),this.maxDeltaRevision_=this.getRevision()),ys(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,this.maxDelta_,!1,t,e,i,n))}forEachSegment(t){return bs(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t)}getCoordinateAtM(t,e){return\"XYM\"!=this.layout&&\"XYZM\"!=this.layout?null:(e=void 0!==e&&e,function(t,e,i,n,r,s){if(i==e)return null;let o;if(r<t[e+n-1])return s?(o=t.slice(e,e+n),o[n-1]=r,o):null;if(t[i-1]<r)return s?(o=t.slice(i-n,i),o[n-1]=r,o):null;if(r==t[e+n-1])return t.slice(e,e+n);let a=e/n,l=i/n;for(;a<l;){const e=a+l>>1;r<t[(e+1)*n-1]?l=e:a=e+1}const h=t[a*n-1];if(r==h)return t.slice((a-1)*n,(a-1)*n+n);const c=(r-h)/(t[(a+1)*n-1]-h);o=[];for(let e=0;e<n-1;++e)o.push(p(t[(a-1)*n+e],t[a*n+e],c));return o.push(r),o}(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e))}getCoordinates(){return sn(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getCoordinateAt(t,e){return xs(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e,this.stride)}getLength(){return Me(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getFlatMidpoint(){return this.flatMidpointRevision_!=this.getRevision()&&(this.flatMidpoint_=this.getCoordinateAt(.5,this.flatMidpoint_??void 0),this.flatMidpointRevision_=this.getRevision()),this.flatMidpoint_}getSimplifiedGeometryInternal(t){const e=[];return e.length=Es(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e,0),new Os(e,\"XY\")}getType(){return\"LineString\"}intersectsExtent(t){return Ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,this.getExtent())}setCoordinates(t,e){this.setLayout(e,t,1),this.flatCoordinates||(this.flatCoordinates=[]),this.flatCoordinates.length=ws(this.flatCoordinates,0,t,this.stride),this.changed()}}class Rs extends gs{constructor(t,e){super(),this.setCoordinates(t,e)}clone(){const t=new Rs(this.flatCoordinates.slice(),this.layout);return t.applyProperties(this),t}closestPointXY(t,e,i,n){const r=this.flatCoordinates,s=f(t,e,r[0],r[1]);if(s<n){const t=this.stride;for(let e=0;e<t;++e)i[e]=r[e];return i.length=t,s}return n}getCoordinates(){return this.flatCoordinates.slice()}computeExtent(t){return ye(this.flatCoordinates,t)}getType(){return\"Point\"}intersectsExtent(t){return ge(t,this.flatCoordinates[0],this.flatCoordinates[1])}setCoordinates(t,e){this.setLayout(e,t,0),this.flatCoordinates||(this.flatCoordinates=[]),this.flatCoordinates.length=function(t,e,i){for(let n=0,r=i.length;n<r;++n)t[e++]=i[n];return e}(this.flatCoordinates,0,t,this.stride),this.changed()}}function Ls(t,e,i,n){let r=0;const s=t[i-n],o=t[i-n+1];let a=0,l=0;for(;e<i;e+=n){const i=t[e]-s,n=t[e+1]-o;r+=l*i-a*n,a=i,l=n}return r/2}class Ds extends gs{constructor(t,e){super(),this.maxDelta_=-1,this.maxDeltaRevision_=-1,void 0===e||Array.isArray(t[0])?this.setCoordinates(t,e):this.setFlatCoordinates(e,t)}clone(){return new Ds(this.flatCoordinates.slice(),this.layout)}closestPointXY(t,e,i,n){return n<fe(this.getExtent(),t,e)?n:(this.maxDeltaRevision_!=this.getRevision()&&(this.maxDelta_=Math.sqrt(ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,0)),this.maxDeltaRevision_=this.getRevision()),ys(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,this.maxDelta_,!0,t,e,i,n))}getArea(){return Ls(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getCoordinates(){return sn(this.flatCoordinates,0,this.flatCoordinates.length,this.stride)}getSimplifiedGeometryInternal(t){const e=[];return e.length=Es(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t,e,0),new Ds(e,\"XY\")}getType(){return\"LinearRing\"}intersectsExtent(t){return Ms(this.flatCoordinates,0,this.flatCoordinates.length,this.stride,t)}setCoordinates(t,e){this.setLayout(e,t,1),this.flatCoordinates||(this.flatCoordinates=[]),this.flatCoordinates.length=ws(this.flatCoordinates,0,t,this.stride),this.changed()}}function Fs(e,i,n,r,s,o,a){let l,h,c,u,f,d,g;const p=s[o+1],_=[];for(let t=0,s=n.length;t<s;++t){const s=n[t];for(u=e[s-r],d=e[s-r+1],l=i;l<s;l+=r)f=e[l],g=e[l+1],(p<=d&&g<=p||d<=p&&p<=g)&&(c=(p-d)/(g-d)*(f-u)+u,_.push(c)),u=f,d=g}let m=NaN,y=-1/0;for(_.sort(t),u=_[0],l=1,h=_.length;l<h;++l){f=_[l];const t=Math.abs(f-u);t>y&&(c=(u+f)/2,Cs(e,i,n,r,c,p)&&(m=c,y=t)),u=f}return isNaN(m)&&(m=s[o]),a?(a.push(m,p,y),a):[m,p,y]}function Ts(t,e,i,n){for(;e<i-n;){for(let r=0;r<n;++r){const s=t[e+r];t[e+r]=t[i-n+r],t[i-n+r]=s}e+=n,i-=n}}function zs(t,e,i,n){let r=0,s=t[i-n],o=t[i-n+1];for(;e<i;e+=n){const i=t[e],n=t[e+1];r+=(i-s)*(n+o),s=i,o=n}return 0===r?void 0:r>0}function $s(t,e,i,n,r){r=void 0!==r&&r;for(let s=0,o=i.length;s<o;++s){const o=i[s],a=zs(t,e,o,n);(0===s?r&&a||!r&&!a:r&&!a||!r&&a)&&Ts(t,e,o,n),e=o}return e}class Ws extends gs{constructor(t,e,i){super(),this.ends_=[],this.flatInteriorPointRevision_=-1,this.flatInteriorPoint_=null,this.maxDelta_=-1,this.maxDeltaRevision_=-1,this.orientedRevision_=-1,this.orientedFlatCoordinates_=null,void 0!==e&&i?(this.setFlatCoordinates(e,t),this.ends_=i):this.setCoordinates(t,e)}appendLinearRing(t){this.flatCoordinates?i(this.flatCoordinates,t.getFlatCoordinates()):this.flatCoordinates=t.getFlatCoordinates().slice(),this.ends_.push(this.flatCoordinates.length),this.changed()}clone(){const t=new Ws(this.flatCoordinates.slice(),this.layout,this.ends_.slice());return t.applyProperties(this),t}closestPointXY(t,e,i,n){return n<fe(this.getExtent(),t,e)?n:(this.maxDeltaRevision_!=this.getRevision()&&(this.maxDelta_=Math.sqrt(function(t,e,i,n,r){for(let s=0,o=i.length;s<o;++s){const o=i[s];r=ms(t,e,o,n,r),e=o}return r}(this.flatCoordinates,0,this.ends_,this.stride,0)),this.maxDeltaRevision_=this.getRevision()),function(t,e,i,n,r,s,o,a,l,h,c){c=c||[NaN,NaN];for(let u=0,f=i.length;u<f;++u){const f=i[u];h=ys(t,e,f,n,r,s,o,a,l,h,c),e=f}return h}(this.flatCoordinates,0,this.ends_,this.stride,this.maxDelta_,!0,t,e,i,n))}containsXY(t,e){return Cs(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride,t,e)}getArea(){return function(t,e,i,n){let r=0;for(let s=0,o=i.length;s<o;++s){const o=i[s];r+=Ls(t,e,o,n),e=o}return r}(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride)}getCoordinates(t){let e;return void 0!==t?(e=this.getOrientedFlatCoordinates().slice(),$s(e,0,this.ends_,this.stride,t)):e=this.flatCoordinates,on(e,0,this.ends_,this.stride)}getEnds(){return this.ends_}getFlatInteriorPoint(){if(this.flatInteriorPointRevision_!=this.getRevision()){const t=Se(this.getExtent());this.flatInteriorPoint_=Fs(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride,t,0),this.flatInteriorPointRevision_=this.getRevision()}return this.flatInteriorPoint_}getInteriorPoint(){return new Rs(this.getFlatInteriorPoint(),\"XYM\")}getLinearRingCount(){return this.ends_.length}getLinearRing(t){return t<0||this.ends_.length<=t?null:new Ds(this.flatCoordinates.slice(0===t?0:this.ends_[t-1],this.ends_[t]),this.layout)}getLinearRings(){const t=this.layout,e=this.flatCoordinates,i=this.ends_,n=[];let r=0;for(let s=0,o=i.length;s<o;++s){const o=i[s],a=new Ds(e.slice(r,o),t);n.push(a),r=o}return n}getOrientedFlatCoordinates(){if(this.orientedRevision_!=this.getRevision()){const t=this.flatCoordinates;!function(t,e,i,n,r){r=void 0!==r&&r;for(let s=0,o=i.length;s<o;++s){const o=i[s],a=zs(t,e,o,n);if(0===s){if(r&&a||!r&&!a)return!1}else if(r&&!a||!r&&a)return!1;e=o}return!0}(t,0,this.ends_,this.stride)?(this.orientedFlatCoordinates_=t.slice(),this.orientedFlatCoordinates_.length=$s(this.orientedFlatCoordinates_,0,this.ends_,this.stride)):this.orientedFlatCoordinates_=t,this.orientedRevision_=this.getRevision()}return this.orientedFlatCoordinates_}getSimplifiedGeometryInternal(t){const e=[],i=[];return e.length=Ps(this.flatCoordinates,0,this.ends_,this.stride,Math.sqrt(t),e,0,i),new Ws(e,\"XY\",i)}getType(){return\"Polygon\"}intersectsExtent(t){return Is(this.getOrientedFlatCoordinates(),0,this.ends_,this.stride,t)}setCoordinates(t,e){this.setLayout(e,t,2),this.flatCoordinates||(this.flatCoordinates=[]);const i=function(t,e,i,n,r){r=r||[];let s=0;for(let o=0,a=i.length;o<a;++o){const a=ws(t,e,i[o],n);r[s++]=a,e=a}return r.length=s,r}(this.flatCoordinates,0,t,this.stride,this.ends_);this.flatCoordinates.length=0===i.length?0:i[i.length-1],this.changed()}}const Gs=Fe();class Ns{constructor(t,e,i,n,r,s){this.styleFunction,this.extent_,this.id_=s,this.type_=t,this.flatCoordinates_=e,this.flatInteriorPoints_=null,this.flatMidpoints_=null,this.ends_=i||null,this.properties_=r,this.squaredTolerance_,this.stride_=n,this.simplifiedGeometry_}get(t){return this.properties_[t]}getExtent(){return this.extent_||(this.extent_=\"Point\"===this.type_?ye(this.flatCoordinates_):we(this.flatCoordinates_,0,this.flatCoordinates_.length,this.stride_)),this.extent_}getFlatInteriorPoint(){if(!this.flatInteriorPoints_){const t=Se(this.getExtent());this.flatInteriorPoints_=Fs(this.flatCoordinates_,0,this.ends_,this.stride_,t,0)}return this.flatInteriorPoints_}getFlatInteriorPoints(){if(!this.flatInteriorPoints_){const t=function(t,e){const i=[];let n,r=0,s=0;for(let o=0,a=e.length;o<a;++o){const a=e[o],l=zs(t,r,a,2);if(void 0===n&&(n=l),l===n)i.push(e.slice(s,o+1));else{if(0===i.length)continue;i[i.length-1].push(e[s])}s=o+1,r=a}return i}(this.flatCoordinates_,this.ends_),e=function(t,e,i,n){const r=[];let s=[1/0,1/0,-1/0,-1/0];for(let o=0,a=i.length;o<a;++o){const a=i[o];s=we(t,e,a[0],n),r.push((s[0]+s[2])/2,(s[1]+s[3])/2),e=a[a.length-1]}return r}(this.flatCoordinates_,0,t,this.stride_);this.flatInteriorPoints_=function(t,e,i,n,r){let s=[];for(let o=0,a=i.length;o<a;++o){const a=i[o];s=Fs(t,e,a,n,r,2*o,s),e=a[a.length-1]}return s}(this.flatCoordinates_,0,t,this.stride_,e)}return this.flatInteriorPoints_}getFlatMidpoint(){return this.flatMidpoints_||(this.flatMidpoints_=xs(this.flatCoordinates_,0,this.flatCoordinates_.length,this.stride_,.5)),this.flatMidpoints_}getFlatMidpoints(){if(!this.flatMidpoints_){this.flatMidpoints_=[];const t=this.flatCoordinates_;let e=0;const n=this.ends_;for(let r=0,s=n.length;r<s;++r){const s=n[r],o=xs(t,e,s,this.stride_,.5);i(this.flatMidpoints_,o),e=s}}return this.flatMidpoints_}getId(){return this.id_}getOrientedFlatCoordinates(){return this.flatCoordinates_}getGeometry(){return this}getSimplifiedGeometry(t){return this}simplifyTransformed(t,e){return this}getProperties(){return this.properties_}getPropertiesInternal(){return this.properties_}getStride(){return this.stride_}getStyleFunction(){return this.styleFunction}getType(){return this.type_}transform(t){const e=(t=rs(t)).getExtent(),i=t.getWorldExtent();if(e&&i){const t=Ce(i)/Ce(e);$e(Gs,i[0],i[3],t,-t,0,0,0),Ae(this.flatCoordinates_,0,this.flatCoordinates_.length,this.stride_,Gs,this.flatCoordinates_)}}applyTransform(t){t(this.flatCoordinates_,this.flatCoordinates_,this.stride_)}clone(){return new Ns(this.type_,this.flatCoordinates_.slice(),this.ends_?.slice(),this.stride_,Object.assign({},this.properties_),this.id_)}getEnds(){return this.ends_}enableSimplifyTransformed(){return this.simplifyTransformed=Je((t,e)=>{if(t===this.squaredTolerance_)return this.simplifiedGeometry_;this.simplifiedGeometry_=this.clone(),e&&this.simplifiedGeometry_.applyTransform(e);const i=this.simplifiedGeometry_.getFlatCoordinates();let n;switch(this.type_){case\"LineString\":i.length=Es(i,0,this.simplifiedGeometry_.flatCoordinates_.length,this.simplifiedGeometry_.stride_,t,i,0),n=[i.length];break;case\"MultiLineString\":n=[],i.length=function(t,e,i,n,r,s,o,a){for(let l=0,h=i.length;l<h;++l){const h=i[l];o=Es(t,e,h,n,r,s,o),a.push(o),e=h}return o}(i,0,this.simplifiedGeometry_.ends_,this.simplifiedGeometry_.stride_,t,i,0,n);break;case\"Polygon\":n=[],i.length=Ps(i,0,this.simplifiedGeometry_.ends_,this.simplifiedGeometry_.stride_,Math.sqrt(t),i,0,n)}return n&&(this.simplifiedGeometry_=new Ns(this.type_,i,n,this.stride_,this.properties_,this.id_)),this.squaredTolerance_=t,this.simplifiedGeometry_}),this}}function Xs(t){return Object.keys(t).reduce((e,i)=>e+(t[i].size||1),0)}Ns.prototype.getFlatCoordinates=Ns.prototype.getOrientedFlatCoordinates;const Ys={},Bs=new Ns(\"Point\",[0,0],[],2,Ys,\"dummy\"),Us=new TextDecoder;function js(t,e,i,n,r,s){const o=`prop_${t}`,a=i.findIndex(t=>t===o),l=i.slice(0,a).reduce((t,e)=>t+n[e].size,0),h=n[o].size;if(e===N){const t=r[l+1],e=r[l+2],i=s.slice(t,t+e);return Us.decode(i)}if(e===X){const t=(c=Array.from(r.slice(l,l+2)),[Math.min(Math.floor(c[0]/256)/255,1),Math.min(c[0]%256/255,1),Math.min(Math.floor(c[1]/256)/255,1),Math.min(c[1]%256/255,1)]);return t[0]*=255,t[1]*=255,t[2]*=255,t}var c;return h>1?Array.from(r.slice(l,l+h)):r[l]}function Vs(t,e,i,n,r){let s=t(i,1);if(s){s=Array.isArray(s)?s:[s];for(let t=0,o=s.length;t<o;t++){const o=s[t].getText();if(!o)continue;const a=o.getPlacement(),l=\"LineString\"===n.getType();\"line\"===a&&!l||\"line\"!==a&&l||(e.setTextStyle(o,r),e.drawText(n,i))}}}const qs=self;let Js=0;const Ks=new OffscreenCanvas(1,1),Hs=Ks.getContext(\"2d\"),Zs=new Map,Qs=Fe();function to(t,e,i,n,r,s){const o=n/2,a=r/2,l=1/e,h=-l,c=-t[0]+s,u=-t[1];return $e(Qs,o,a,l,h,-i,c,u)}qs.onmessage=t=>{const e=t.data;switch(e.type){case hr:{const t=(i=e.frameState,{...i,viewState:{...i.viewState,projection:rs(i.viewState.projection)}}),n=t.viewState,r=e.batchesToRender;Js&&cancelAnimationFrame(Js),Js=requestAnimationFrame(()=>{Js=0,t.size[0]!==Ks.width||t.size[1]!==Ks.height?(Ks.width=t.size[0],Ks.height=t.size[1]):Hs.clearRect(0,0,Ks.width,Ks.height);for(const i of r.values()){if(!Zs.has(i)){const t={type:hr,imageData:null,frameState:e.frameState,id:e.id};return void qs.postMessage(t)}const r=Zs.get(i);if(!r)continue;const s=to(n.center,n.resolution,n.rotation,Ks.width,Ks.height,0);Te(s,r.inverseTransform),r.executor.execute(Hs,t.size,s,t.viewState.rotation,!1)}const i=Ks.transferToImageBitmap(),s={type:hr,imageData:i,frameState:e.frameState,id:e.id};qs.postMessage(s,[i])});break}case ar:{const{polygonRenderInstructions:t,lineStringRenderInstructions:i,pointRenderInstructions:n,style:r,customAttributesSizes:s,renderInstructionsTransform:o,id:a,resolution:l}=e,h=1,c=l*o[0],u=Date.now().toString(),f=new Uint8Array(e.labelsArray),d=new un(1,[-1/0,-1/0,1/0,1/0],c,h),g=Object.keys(s).reduce((t,e)=>({...t,[e]:{size:s[e]}}),{}),p=tt();!function(t){function e(t){for(const e in t)e.startsWith(\"text-\")||\"z-index\"===e||delete t[e]}if(Array.isArray(t)){for(let i=0,n=t.length;i<n;i++){const n=t[i];if(\"style\"in n&&Array.isArray(n.style))for(let t=0,i=n.style.length;t<i;t++)e(n.style[t]);else e(\"style\"in n?n.style:n)}return t}e(t)}(r);const _=function(t,e){if(e=e??tt(),!Array.isArray(t))return Wn([t],e);const i=t.length;if(\"style\"in t[0]){const n=new Array(i);for(let e=0;e<i;++e){const i=t[e];if(!(\"style\"in i))throw new Error(\"Expected a list of rules with a style property\");n[e]=i}return $n(n,e)}return Wn(t,e)}(r,p);!function(t,e,i,n,r,s){const o=Object.keys(n),a=Xs(n),l={};let h=0;for(;h<t.length;){const c=new Float32Array(t.buffer,h*Float32Array.BYTES_PER_ELEMENT,a);h+=a;const u=t[h++];let f=0;const d=new Array(u);for(let e=0;e<u;e++)f+=t[h++],d[e]=2*f;const g=h+2*f,p=Array.from(new Float32Array(t.buffer,h*Float32Array.BYTES_PER_ELEMENT,2*f)),_=new Ws(p,\"XY\",d),m=Array.from(i.entries());for(let t=0;t<m.length;t++){const[i,r]=m[t];Ys[i]=js(i,r,o,n,c,e)}Vs(s,r,Bs,_,l),h=g}}(new Float32Array(t),f,p.properties,g,d,_),function(t,e,i,n,r,s){const o=Object.keys(n),a=Xs(n),l={};let h,c=0;for(;c<t.length;){const u=new Float32Array(t.buffer,c*Float32Array.BYTES_PER_ELEMENT,a);c+=a,h=t[c++];const f=Array.from(new Float32Array(t.buffer,c*Float32Array.BYTES_PER_ELEMENT,3*h)),d=new Os(f,\"XYM\"),g=Array.from(i.entries());for(let t=0;t<g.length;t++){const[i,r]=g[t];Ys[i]=js(i,r,o,n,u,e)}Vs(s,r,Bs,d,l),c+=3*h}}(new Float32Array(i),f,p.properties,g,d,_),function(t,e,i,n,r,s){const o=Object.keys(n),a=Xs(n),l={};let h=0;for(;h<t.length;){const c=[t.at(h),t.at(h+1)];h+=2;const u=new Float32Array(t.buffer,h*Float32Array.BYTES_PER_ELEMENT,a),f=new Rs(c,\"XY\"),d=Array.from(i.entries());for(let t=0;t<d.length;t++){const[i,r]=d[t];Ys[i]=js(i,r,o,n,u,e)}Vs(s,r,Bs,f,l),h+=a}}(new Float32Array(n),f,p.properties,g,d,_);const m=d.finish();if(0===m.instructions.length)Zs.set(u,null);else{const t=We(o),e=new Sn(c,h,!1,m);Zs.set(u,{inverseTransform:t,executor:e})}const y={type:ar,instructionsSetKey:u,id:a};qs.postMessage(y);break}case lr:{const{instructionsSetKey:t}=e;Zs.has(t)&&Zs.delete(t);break}}var i};";
	return new Worker(typeof Blob > "u" ? "data:application/javascript;base64," + Buffer.from(e, "binary").toString("base64") : URL.createObjectURL(new Blob([e], { type: "application/javascript" })));
}
//#endregion
//#region node_modules/ol/worker/webgl.js
function Bm() {
	let e = "const t=new Set;let e=!1;function n(e,n,i=2){const f=n&&n.length,u=f?n[0]*i:e.length;t.size&&t.clear();let s=r(e,0,u,i,!0);const v=[];if(!s||s.next===s.prev)return v;let b=0,A=0,m=0;if(f&&(s=function(e,n,x,i){const f=[];for(let o=0,x=n.length;o<x;o++){const u=r(e,n[o]*i,o<x-1?n[o+1]*i:e.length,i,!1);u===u.next&&t.add(u),f.push(B(u))}f.sort(l),function(t,e){const n=Math.ceil((t+2*e)/y)+e+2;h.length<4*n&&(h=new Float64Array(4*n));p=0}(e.length/i,n.length),M(x,x),c=!0;for(let t=0;t<f.length;t++)x=a(f[t],x);return c=!1,o(x)}(e,n,s,i)),e.length>80*i){b=e[0],A=e[1];let t=b,n=A;for(let r=i;r<u;r+=i){const o=e[r],x=e[r+1];o<b&&(b=o),x<A&&(A=x),o>t&&(t=o),x>n&&(n=x)}m=Math.max(t-b,n-A),m=0!==m?32767/m:0}return x(s,v,b,A,m),v}function r(t,e,n,r,o){let x=null;if(o===function(t,e,n,r){let o=0;for(let x=e,i=n-r;x<n;x+=r)o+=(t[i]-t[x])*(t[x+1]+t[i+1]),i=x;return o}(t,e,n,r)>0)for(let o=e;o<n;o+=r)x=G(o/r|0,t[o],t[o+1],x);else for(let o=n-r;o>=e;o-=r)x=G(o/r|0,t[o],t[o+1],x);return x&&N(x,x.next)&&(k(x),x=x.next),x}function o(n,r=n){const o=r===n;let x,i=n;do{x=!1,i===i.next||0!==t.size&&t.has(i)||!N(i,i.next)&&0!==S(i.prev,i,i.next)?(o||i!==r)&&(i=i.next,x=!o):((o||i===r)&&(r=i.prev),e=!0,k(i),i=i.prev,x=!0)}while(x||i!==r);return r}function x(t,n,r,x,c){c&&function(t,e,n,r){let o=t,x=0;do{o.z=I(o.x,o.y,e,n,r),w[x++]=o,o=o.next}while(o!==t);!function(t){if(t<=32){for(let e=1;e<t;e++){const t=w[e],n=t.z;let r=e-1;for(;r>=0&&w[r].z>n;)w[r+1]=w[r],r--;w[r+1]=t}return}F.length<t&&(F=new Uint32Array(t),Z=new Uint32Array(t),d=new Array(t));for(let e=0;e<t;e++)F[e]=w[e].z;E(t,w,F,d,Z,0),E(t,d,Z,w,F,8),E(t,w,F,d,Z,16),E(t,d,Z,w,F,24)}(x);let i=null;for(let t=0;t<x;t++){const e=w[t];e.prevZ=i,i&&(i.nextZ=e),i=e}i.nextZ=null}(t,r,x,c);let l=t,a=!1;for(;t.prev!==t.next;){const y=t.prev,h=t.next;if(S(y,t,h)<0&&(c?f(t,r,x,c):i(t)))n.push(y.i,t.i,h.i),k(t),t=h,l=h;else if((t=h)===l){if(e=!1,t=o(t),e){l=t;continue}if(!a){l=t=u(t,n),a=!0;continue}s(t,n,r,x,c);break}}}function i(t){const e=t.prev,n=t,r=t.next,o=e.x,x=n.x,i=r.x,f=e.y,u=n.y,s=r.y,c=Math.min(o,x,i),l=Math.min(f,u,s),a=Math.max(o,x,i),y=Math.max(f,u,s);let h=r.next;for(;h!==e;){if(h.x>=c&&h.x<=a&&h.y>=l&&h.y<=y&&(o!==h.x||f!==h.y)&&U(o,f,x,u,i,s,h.x,h.y)&&S(h.prev,h,h.next)>=0)return!1;h=h.next}return!0}function f(t,e,n,r){const o=t.prev,x=t,i=t.next,f=o.x,u=x.x,s=i.x,c=o.y,l=x.y,a=i.y,y=Math.min(f,u,s),h=Math.min(c,l,a),p=Math.max(f,u,s),v=Math.max(c,l,a),b=I(y,h,e,n,r),M=I(p,v,e,n,r);let A=t.prevZ;for(;A&&A.z>=b;){if(A.x>=y&&A.x<=p&&A.y>=h&&A.y<=v&&A!==i&&(f!==A.x||c!==A.y)&&U(f,c,u,l,s,a,A.x,A.y)&&S(A.prev,A,A.next)>=0)return!1;A=A.prevZ}let m=t.nextZ;for(;m&&m.z<=M;){if(m.x>=y&&m.x<=p&&m.y>=h&&m.y<=v&&m!==i&&(f!==m.x||c!==m.y)&&U(f,c,u,l,s,a,m.x,m.y)&&S(m.prev,m,m.next)>=0)return!1;m=m.nextZ}return!0}function u(t,e){let n=t,r=!1;do{const o=n.prev,x=n.next.next;R(o,n,n.next,x,!1)&&_(o,x)&&_(x,o)&&(e.push(o.i,n.i,x.i),k(n),k(n.next),n=t=x,r=!0),n=n.next}while(n!==t);return r?o(n):n}function s(t,e,n,r,i){let f=t;do{let t=f.next.next;for(;t!==f.prev;){if(f.i!==t.i&&P(f,t)){let u=O(f,t);return f=o(f,f.next),u=o(u,u.next),x(f,e,n,r,i),void x(u,e,n,r,i)}t=t.next}f=f.next}while(f!==t)}let c=!1;function l(t,e){return t.x-e.x||t.y-e.y||(t.next.y-t.y)/(t.next.x-t.x)-(e.next.y-e.y)/(e.next.x-e.x)}function a(t,e){const n=function(t,e){let n=e;const r=t.x,o=t.y;let x,i=-1/0;if(N(t,n))return n;for(let e=0,f=0;e<p;e++,f+=4){if(o<h[f+1]||o>h[f+3]||h[f]>r||h[f+2]<=i)continue;const u=A(e);n=m(e);do{if(n.prev.next===n){if(N(t,n.next))return n.next;if(o<=n.y&&o>=n.next.y&&n.next.y!==n.y){const t=n.x+(o-n.y)*(n.next.x-n.x)/(n.next.y-n.y);if(t<=r&&t>i&&(i=t,x=n.x<n.next.x?n:n.next,t===r))return x}}n=n.next}while(n!==u)}if(!x)return null;const f=x.x,u=x.y,s=Math.min(o,u),c=Math.max(o,u);let l=1/0;for(let e=0,a=0;e<p;e++,a+=4){if(h[a+2]<f||h[a]>r||h[a+3]<s||h[a+1]>c)continue;const y=A(e);n=m(e);do{if(n.prev.next===n&&r>=n.x&&n.x>=f&&r!==n.x&&U(o<u?r:i,o,f,u,o<u?i:r,o,n.x,n.y)){const e=Math.abs(o-n.y)/(r-n.x);(_(n,t)||n.y===o&&n.next.y===o&&n.next.x>r)&&(e<l||e===l&&(n.x>x.x||n.x===x.x&&g(x,n)))&&(x=n,l=e)}n=n.next}while(n!==y)}return x}(t,e);if(!n)return e;const r=O(n,t);return M(n,r.next.next),o(r,r.next),o(n,n.next)}const y=16;let h=new Float64Array(0),p=0;const v=[],b=[];function M(t,e){let n=t;do{const t=p++;v[t]=n;let r=1/0,o=1/0,x=-1/0,i=-1/0,f=0;do{const e=n.next;n.z=t,n.x<r&&(r=n.x),n.x>x&&(x=n.x),n.y<o&&(o=n.y),n.y>i&&(i=n.y),e.x<r&&(r=e.x),e.x>x&&(x=e.x),e.y<o&&(o=e.y),e.y>i&&(i=e.y),n=e}while(++f<y&&n!==e);b[t]=n;const u=4*t;h[u]=r,h[u+1]=o,h[u+2]=x,h[u+3]=i}while(n!==e)}function A(t){let e=b[t];for(;e.prev.next!==e;)e=e.next;return b[t]=e,e}function m(t){let e=v[t];for(;e.prev.next!==e;)e=e.next;return v[t]=e,e}function g(t,e){return S(t.prev,t,e.prev)<0&&S(e.next,t,t.next)<0}const w=[];let d=[],F=new Uint32Array(0),Z=new Uint32Array(0);const z=new Uint32Array(256);function E(t,e,n,r,o,x){z.fill(0);for(let e=0;e<t;e++)z[n[e]>>>x&255]++;let i=0;for(let t=0;t<256;t++){const e=z[t];z[t]=i,i+=e}for(let i=0;i<t;i++){const t=n[i],f=z[t>>>x&255]++;r[f]=e[i],o[f]=t}}function I(t,e,n,r,o){return(t=1431655765&((t=858993459&((t=252645135&((t=16711935&((t=(t-n)*o|0)|t<<8))|t<<4))|t<<2))|t<<1))|(e=1431655765&((e=858993459&((e=252645135&((e=16711935&((e=(e-r)*o|0)|e<<8))|e<<4))|e<<2))|e<<1))<<1}function B(t){let e=t,n=t;do{(e.x<n.x||e.x===n.x&&e.y<n.y)&&(n=e),e=e.next}while(e!==t);return n}function U(t,e,n,r,o,x,i,f){return(o-i)*(e-f)>=(t-i)*(x-f)&&(t-i)*(r-f)>=(n-i)*(e-f)&&(n-i)*(x-f)>=(o-i)*(r-f)}function P(t,e){const n=N(t,e)&&S(t.prev,t,t.next)>0&&S(e.prev,e,e.next)>0;return t.next.i!==e.i&&(n||_(t,e)&&_(e,t)&&(0!==S(t.prev,t,e.prev)||0!==S(t,e.prev,e)))&&!function(t,e){const n=Math.min(t.x,e.x),r=Math.max(t.x,e.x),o=Math.min(t.y,e.y),x=Math.max(t.y,e.y);let i=t;do{const f=i.next;if(i.x>r&&f.x>r||i.x<n&&f.x<n||i.y>x&&f.y>x||i.y<o&&f.y<o)i=f;else{if(i.i!==t.i&&f.i!==t.i&&i.i!==e.i&&f.i!==e.i&&R(i,f,t,e))return!0;i=f}}while(i!==t);return!1}(t,e)&&(n||function(t,e){let n=t,r=!1;const o=(t.x+e.x)/2,x=(t.y+e.y)/2;do{const t=n.next;n.y>x!=t.y>x&&o<(t.x-n.x)*(x-n.y)/(t.y-n.y)+n.x&&(r=!r),n=t}while(n!==t);return r}(t,e))}function S(t,e,n){return(e.y-t.y)*(n.x-e.x)-(e.x-t.x)*(n.y-e.y)}function N(t,e){return t.x===e.x&&t.y===e.y}function R(t,e,n,r,o=!0){const x=S(t,e,n),i=S(t,e,r),f=S(n,r,t),u=S(n,r,e);return(x>0&&i<0||x<0&&i>0)&&(f>0&&u<0||f<0&&u>0)||!!o&&(!(0!==x||!T(t,n,e))||(!(0!==i||!T(t,r,e))||(!(0!==f||!T(n,t,r))||!(0!==u||!T(n,e,r)))))}function T(t,e,n){return e.x<=Math.max(t.x,n.x)&&e.x>=Math.min(t.x,n.x)&&e.y<=Math.max(t.y,n.y)&&e.y>=Math.min(t.y,n.y)}function _(t,e){return S(t.prev,t,t.next)<0?S(t,e,t.next)>=0&&S(t,t.prev,e)>=0:S(t,e,t.prev)<0||S(t,t.next,e)<0}function O(t,e){const n=j(t.i,t.x,t.y),r=j(e.i,e.x,e.y),o=t.next,x=e.prev;return t.next=e,e.prev=t,n.next=o,o.prev=n,r.next=n,n.prev=r,x.next=r,r.prev=x,r}function G(t,e,n,r){const o=j(t,e,n);return r?(o.next=r.next,o.prev=r,r.next.prev=o,r.next=o):(o.prev=o,o.next=o),o}function k(t){t.next.prev=t.prev,t.prev.next=t.next,t.prevZ&&(t.prevZ.nextZ=t.nextZ),t.nextZ&&(t.nextZ.prevZ=t.prevZ),c&&function(t,e){const n=4*t.z;e.x<h[n]&&(h[n]=e.x),e.y<h[n+1]&&(h[n+1]=e.y),e.x>h[n+2]&&(h[n+2]=e.x),e.y>h[n+3]&&(h[n+3]=e.y)}(t.prev,t.next)}function j(t,e,n){return{i:t,x:e,y:n,prev:null,next:null,z:0,prevZ:null,nextZ:null}}function q(t,e,n){const r=Math.sqrt((e[0]-t[0])*(e[0]-t[0])+(e[1]-t[1])*(e[1]-t[1])),o=[(e[0]-t[0])/r,(e[1]-t[1])/r],x=[-o[1],o[0]],i=Math.sqrt((n[0]-t[0])*(n[0]-t[0])+(n[1]-t[1])*(n[1]-t[1])),f=[(n[0]-t[0])/i,(n[1]-t[1])/i];let u=0===r||0===i?0:Math.acos((s=f[0]*o[0]+f[1]*o[1],c=-1,l=1,Math.min(Math.max(s,c),l)));var s,c,l;u=Math.max(u,1e-5);return f[0]*x[0]+f[1]*x[1]>0?u:2*Math.PI-u}const L=[1,0,0,1,0,0];function Y(t,e){const n=e[0],r=e[1];return e[0]=t[0]*n+t[2]*r+t[4],e[1]=t[1]*n+t[3]*r+t[5],e}function C(t,e){const n=(r=e)[0]*r[3]-r[1]*r[2];var r;!function(t,e){if(!t)throw new Error(e)}(0!==n,\"Transformation matrix cannot be inverted\");const o=e[0],x=e[1],i=e[2],f=e[3],u=e[4],s=e[5];return t[0]=f/n,t[1]=-x/n,t[2]=-i/n,t[3]=o/n,t[4]=(i*s-f*u)/n,t[5]=-(o*s-x*u)/n,t}new Array(6);const D=[],H={vertexAttributesPosition:0,instanceAttributesPosition:0,indicesPosition:0};function J(t,e,n,r,o){const x=t[e++],i=t[e++],f=D;f.length=r;for(let n=0;n<f.length;n++)f[n]=t[e+n];let u=o?o.instanceAttributesPosition:0;return n[u++]=x,n[u++]=i,f.length&&(n.set(f,u),u+=f.length),H.instanceAttributesPosition=u,H}function K(t,e,n,r,o,x,i,f,u,s){const c=[t[e],t[e+1]],l=[t[n],t[n+1]],a=t[e+2],y=t[n+2],h=Y(f,[...c]),p=Y(f,[...l]);let v=-1,b=-1,M=s;const A=null!==o;if(null!==r){v=q(h,p,Y(f,[...[t[r],t[r+1]]])),Math.cos(v)<=.985&&(M+=Math.tan((v-Math.PI)/2))}if(A){b=q(p,h,Y(f,[...[t[o],t[o+1]]])),Math.cos(b)<=.985&&(M+=Math.tan((Math.PI-b)/2))}const m=Math.pow(2,24),g=u%m,w=Math.floor(u/m)*m;return x.push(c[0],c[1],a,l[0],l[1],y,v,b,g,w,s),x.push(...i),{length:u+Math.sqrt((p[0]-h[0])*(p[0]-h[0])+(p[1]-h[1])*(p[1]-h[1])),angle:M}}function Q(t,e,r,o,x){const i=2+x;let f=e;const u=t.slice(f,f+x);f+=x;const s=t[f++];let c=0;const l=new Array(s-1);for(let e=0;e<s;e++)c+=t[f++],e<s-1&&(l[e]=c);const a=t.slice(f,f+2*c),y=n(a,l,2);for(let t=0;t<y.length;t++)o.push(y[t]+r.length/i);for(let t=0;t<a.length;t+=2)r.push(a[t],a[t+1],...u);return f+2*c}const V=\"GENERATE_POLYGON_BUFFERS\",W=\"GENERATE_POINT_BUFFERS\",X=\"GENERATE_LINE_STRING_BUFFERS\",$=self;$.onmessage=t=>{const e=t.data;switch(e.type){case W:{const t=2,n=2,r=e.customAttributesSize,o=n+r,x=new Float32Array(e.renderInstructions),i=x.length/o*(t+r),f=Uint32Array.from([0,1,3,1,2,3]),u=Float32Array.from([-1,-1,1,-1,1,1,-1,1]),s=new Float32Array(i);let c;for(let t=0;t<x.length;t+=o)c=J(x,t,s,r,c);const l=Object.assign({indicesBuffer:f.buffer,vertexAttributesBuffer:u.buffer,instanceAttributesBuffer:s.buffer,renderInstructions:x.buffer},e);$.postMessage(l,[u.buffer,s.buffer,f.buffer,x.buffer]);break}case X:{const t=[],n=e.customAttributesSize,r=3,o=new Float32Array(e.renderInstructions);let x=0;const i=e.renderInstructionsTransform,f=L.slice(0);let u,s;for(C(f,i);x<o.length;){s=Array.from(o.slice(x,x+n)),x+=n,u=o[x++];const e=x,i=x+(u-1)*r,c=o[e]===o[i]&&o[e+1]===o[i+1];let l=0,a=0;for(let n=0;n<u-1;n++){let y=null;n>0?y=x+(n-1)*r:c&&(y=i-r);let h=null;n<u-2?h=x+(n+2)*r:c&&(h=e+r);const p=K(o,x+n*r,x+(n+1)*r,y,h,t,s,f,l,a);l=p.length,a=p.angle}x+=u*r}const c=Uint32Array.from([0,1,3,1,2,3]),l=Float32Array.from([-1,-1,1,-1,1,1,-1,1]),a=Float32Array.from(t),y=Object.assign({indicesBuffer:c.buffer,vertexAttributesBuffer:l.buffer,instanceAttributesBuffer:a.buffer,renderInstructions:o.buffer},e);$.postMessage(y,[l.buffer,a.buffer,c.buffer,o.buffer]);break}case V:{const t=[],n=[],r=e.customAttributesSize,o=new Float32Array(e.renderInstructions);let x=0;for(;x<o.length;)x=Q(o,x,t,n,r);const i=Uint32Array.from(n),f=Float32Array.from(t),u=Float32Array.from([]),s=Object.assign({indicesBuffer:i.buffer,vertexAttributesBuffer:f.buffer,instanceAttributesBuffer:u.buffer,renderInstructions:o.buffer},e);$.postMessage(s,[f.buffer,u.buffer,i.buffer,o.buffer]);break}}};";
	return new Worker(typeof Blob > "u" ? "data:application/javascript;base64," + Buffer.from(e, "binary").toString("base64") : URL.createObjectURL(new Blob([e], { type: "application/javascript" })));
}
//#endregion
//#region node_modules/ol/render/webgl/constants.js
var Vm = {
	GENERATE_POLYGON_BUFFERS: "GENERATE_POLYGON_BUFFERS",
	GENERATE_POINT_BUFFERS: "GENERATE_POINT_BUFFERS",
	GENERATE_LINE_STRING_BUFFERS: "GENERATE_LINE_STRING_BUFFERS"
}, Hm = {
	BUILD_INSTRUCTIONS: "BUILD_INSTRUCTIONS",
	DISPOSE_INSTRUCTIONS: "DISPOSE_INSTRUCTIONS",
	RENDER: "RENDER"
};
//#endregion
//#region node_modules/ol/render/webgl/encodeUtil.js
function Um(e, t) {
	t ||= [];
	let n = Math.floor(e / 256 / 256 / 256) / 255, r = Math.floor(e / 256 / 256) % 256 / 255, i = Math.floor(e / 256) % 256 / 255, a = e % 256 / 255;
	return t[0] = n * 256 * 255 + r * 255, t[1] = i * 256 * 255 + a * 255, t;
}
function Wm(e) {
	let t = 0;
	return t += Math.round(e[0] * 256 * 256 * 256 * 255), t += Math.round(e[1] * 256 * 256 * 255), t += Math.round(e[2] * 256 * 255), t += Math.round(e[3] * 255), t;
}
//#endregion
//#region node_modules/ol/render/webgl/renderinstructions.js
function Gm(e, t, n, r, i) {
	let a = 0;
	for (let o in n) {
		let s = n[o], c = s.callback.call(r, r.feature);
		if (typeof c == "string") {
			let [n, r] = t.push(c);
			e[i + a++] = sm(c), e[i + a++] = n, e[i + a++] = r;
			continue;
		}
		let l = c?.[0] ?? c;
		l === -9999999 && console.warn("The \"has\" operator might return false positives."), l === void 0 ? l = hm : l === null && (l = 0), e[i + a++] = l, s.size && s.size !== 1 && (e[i + a++] = c?.[1] ?? -9999999, !(s.size < 3) && (e[i + a++] = c?.[2] ?? -9999999, !(s.size < 4) && (e[i + a++] = c?.[3] ?? -9999999)));
	}
	return a;
}
function Km(e) {
	return Object.keys(e).reduce((t, n) => t + (e[n].size || 1), 0);
}
function qm(e, t, n, r, i) {
	let a = (2 + Km(r)) * e.geometriesCount;
	(!t || t.length !== a) && (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) o[0] = c.flatCoordss[e][0], o[1] = c.flatCoordss[e][1], z(i, o), t[s++] = o[0], t[s++] = o[1], s += Gm(t, n, r, c, s);
	}
	return t;
}
function Jm(e, t, n, r, i) {
	let a = 3 * e.verticesCount + (1 + Km(r)) * e.geometriesCount;
	(!t || t.length !== a) && (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) {
			o.length = c.flatCoordss[e].length, zr(c.flatCoordss[e], 0, o.length, 3, i, o, 3), s += Gm(t, n, r, c, s), t[s++] = o.length / 3;
			for (let e = 0, n = o.length; e < n; e += 3) t[s++] = o[e], t[s++] = o[e + 1], t[s++] = o[e + 2];
		}
	}
	return t;
}
function Ym(e, t, n, r, i) {
	let a = 2 * e.verticesCount + (1 + Km(r)) * e.geometriesCount + e.ringsCount;
	(!t || t.length !== a) && (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) {
			o.length = c.flatCoordss[e].length, zr(c.flatCoordss[e], 0, o.length, 2, i, o), s += Gm(t, n, r, c, s), t[s++] = c.ringsVerticesCounts[e].length;
			for (let n = 0, r = c.ringsVerticesCounts[e].length; n < r; n++) t[s++] = c.ringsVerticesCounts[e][n];
			for (let e = 0, n = o.length; e < n; e += 2) t[s++] = o[e], t[s++] = o[e + 1];
		}
	}
	return t;
}
//#endregion
//#region node_modules/ol/render/webgl/serialize.js
function Xm(e) {
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
function Zm(e) {
	return (JSON.stringify(e).split("").reduce((e, t) => (e << 5) - e + t.charCodeAt(0), 0) >>> 0).toString();
}
function Qm(e, t, n, r) {
	if (`${r}radius` in e && r !== "icon-") {
		let i = Q(n, e[`${r}radius`], W);
		if (`${r}radius2` in e) {
			let t = Q(n, e[`${r}radius2`], W);
			i = `max(${i}, ${t})`;
		}
		`${r}stroke-width` in e && (i = `(${i} + ${Q(n, e[`${r}stroke-width`], W)} * 0.5)`), t.setSymbolSizeExpression(`vec2(${i} * 2. + 0.5)`);
	}
	if (`${r}scale` in e) {
		let i = Q(n, e[`${r}scale`], Nl);
		t.setSymbolSizeExpression(`${t.getSymbolSizeExpression()} * ${i}`);
	}
	`${r}displacement` in e && t.setSymbolOffsetExpression(Q(n, e[`${r}displacement`], Ml)), `${r}rotation` in e && t.setSymbolRotationExpression(Q(n, e[`${r}rotation`], W)), `${r}rotate-with-view` in e && t.setSymbolRotateWithView(!!e[`${r}rotate-with-view`]);
}
function $m(e, t, n, r, i) {
	let a = "vec4(0.)";
	if (t !== null && (a = t), n !== null && r !== null) {
		let t = `smoothstep(-${r} + 0.63, -${r} - 0.58, ${e})`;
		a = `mix(${n}, ${a}, ${t})`;
	}
	let o = `(1.0 - smoothstep(-0.63, 0.58, ${e}))`, s = `${a} * vec4(1.0, 1.0, 1.0, ${o})`;
	return i !== null && (s = `${s} * vec4(1.0, 1.0, 1.0, ${i})`), s;
}
function eh(e, t, n, r, i) {
	let a = new Image();
	a.crossOrigin = e[`${r}cross-origin`] === void 0 ? "anonymous" : e[`${r}cross-origin`], R(typeof e[`${r}src`] == "string", `WebGL layers do not support expressions for the ${r}src style property`), a.src = e[`${r}src`], n[`u_texture${i}_size`] = () => a.complete ? [a.width, a.height] : [0, 0], t.addUniform(`u_texture${i}_size`, "vec2");
	let o = `u_texture${i}_size`;
	return n[`u_texture${i}`] = a, t.addUniform(`u_texture${i}`, "sampler2D"), o;
}
function th(e, t, n, r, i) {
	let a = Q(n, e[`${t}offset`], Nl);
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
function nh(e, t, n, r) {
	r.functions.circleDistanceField = "float circleDistanceField(vec2 point, float radius) {\n  return length(point) - radius;\n}", Qm(e, t, r, "circle-");
	let i = null;
	"circle-opacity" in e && (i = Q(r, e["circle-opacity"], W));
	let a = "coordsPx";
	"circle-scale" in e && (a = `coordsPx / ${Q(r, e["circle-scale"], Nl)}`);
	let o = null;
	"circle-fill-color" in e && (o = Q(r, e["circle-fill-color"], K));
	let s = null;
	"circle-stroke-color" in e && (s = Q(r, e["circle-stroke-color"], K));
	let c = Q(r, e["circle-radius"], W), l = null;
	"circle-stroke-width" in e && (l = Q(r, e["circle-stroke-width"], W), c = `(${c} + ${l} * 0.5)`);
	let u = $m(`circleDistanceField(${a}, ${c})`, o, s, l, i);
	t.setSymbolColorExpression(u);
}
function rh(e, t, n, r) {
	r.functions.round = "float round(float v) {\n  return sign(v) * floor(abs(v) + 0.5);\n}", r.functions.starDistanceField = "float starDistanceField(vec2 point, float numPoints, float radius, float radius2, float angle) {\n  float startAngle = -PI * 0.5 + angle; // tip starts upwards and rotates clockwise with angle\n  float c = cos(startAngle);\n  float s = sin(startAngle);\n  vec2 pointRotated = vec2(c * point.x - s * point.y, s * point.x + c * point.y);\n  float alpha = TWO_PI / numPoints; // the angle of one sector\n  float beta = atan(pointRotated.y, pointRotated.x);\n  float gamma = round(beta / alpha) * alpha; // angle in sector\n  c = cos(-gamma);\n  s = sin(-gamma);\n  vec2 inSector = vec2(c * pointRotated.x - s * pointRotated.y, abs(s * pointRotated.x + c * pointRotated.y));\n  vec2 tipToPoint = inSector + vec2(-radius, 0.);\n  vec2 edgeNormal = vec2(radius2 * sin(alpha * 0.5), -radius2 * cos(alpha * 0.5) + radius);\n  return dot(normalize(edgeNormal), tipToPoint);\n}", r.functions.regularDistanceField = "float regularDistanceField(vec2 point, float numPoints, float radius, float angle) {\n  float startAngle = -PI * 0.5 + angle; // tip starts upwards and rotates clockwise with angle\n  float c = cos(startAngle);\n  float s = sin(startAngle);\n  vec2 pointRotated = vec2(c * point.x - s * point.y, s * point.x + c * point.y);\n  float alpha = TWO_PI / numPoints; // the angle of one sector\n  float radiusIn = radius * cos(PI / numPoints);\n  float beta = atan(pointRotated.y, pointRotated.x);\n  float gamma = round((beta - alpha * 0.5) / alpha) * alpha + alpha * 0.5; // angle in sector from mid\n  c = cos(-gamma);\n  s = sin(-gamma);\n  vec2 inSector = vec2(c * pointRotated.x - s * pointRotated.y, abs(s * pointRotated.x + c * pointRotated.y));\n  return inSector.x - radiusIn;\n}", Qm(e, t, r, "shape-");
	let i = null;
	"shape-opacity" in e && (i = Q(r, e["shape-opacity"], W));
	let a = "coordsPx";
	"shape-scale" in e && (a = `coordsPx / ${Q(r, e["shape-scale"], Nl)}`);
	let o = null;
	"shape-fill-color" in e && (o = Q(r, e["shape-fill-color"], K));
	let s = null;
	"shape-stroke-color" in e && (s = Q(r, e["shape-stroke-color"], K));
	let c = null;
	"shape-stroke-width" in e && (c = Q(r, e["shape-stroke-width"], W));
	let l = Q(r, e["shape-points"], W), u = "0.";
	"shape-angle" in e && (u = Q(r, e["shape-angle"], W));
	let d, f = Q(r, e["shape-radius"], W);
	if (c !== null && (f = `${f} + ${c} * 0.5`), "shape-radius2" in e) {
		let t = Q(r, e["shape-radius2"], W);
		c !== null && (t = `${t} + ${c} * 0.5`), d = `starDistanceField(${a}, ${l}, ${f}, ${t}, ${u})`;
	} else d = `regularDistanceField(${a}, ${l}, ${f}, ${u})`;
	let p = $m(d, o, s, c, i);
	t.setSymbolColorExpression(p);
}
function ih(e, t, n, r) {
	let i = "vec4(1.0)";
	"icon-color" in e && (i = Q(r, e["icon-color"], K)), "icon-opacity" in e && (i = `${i} * vec4(1.0, 1.0, 1.0, ${Q(r, e["icon-opacity"], W)})`);
	let a = Zm(e["icon-src"]), o = eh(e, t, n, "icon-", a);
	if (t.setSymbolColorExpression(`${i} * texture2D(u_texture${a}, v_texCoord)`).setSymbolSizeExpression(o), "icon-width" in e && "icon-height" in e && t.setSymbolSizeExpression(`vec2(${Q(r, e["icon-width"], W)}, ${Q(r, e["icon-height"], W)})`), "icon-offset" in e && "icon-size" in e) {
		let n = Q(r, e["icon-size"], Ml), i = t.getSymbolSizeExpression();
		t.setSymbolSizeExpression(n);
		let a = th(e, "icon-", r, "v_quadSizePx", n);
		t.setTextureCoordinateExpression(`(vec4((${a}).xyxy) + vec4(0., 0., ${n})) / (${i}).xyxy`);
	}
	if (Qm(e, t, r, "icon-"), "icon-anchor" in e) {
		let n = Q(r, e["icon-anchor"], Ml), i = "1.0";
		"icon-scale" in e && (i = Q(r, e["icon-scale"], Nl));
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
function ah(e, t, n, r) {
	if ("stroke-color" in e && t.setStrokeColorExpression(Q(r, e["stroke-color"], K)), "stroke-pattern-src" in e) {
		let i = Zm(e["stroke-pattern-src"]), a = eh(e, t, n, "stroke-pattern-", i), o = a, s = "vec2(0.)";
		"stroke-pattern-offset" in e && "stroke-pattern-size" in e && (o = Q(r, e["stroke-pattern-size"], Ml), s = th(e, "stroke-pattern-", r, a, o));
		let c = "0.";
		"stroke-pattern-spacing" in e && (c = Q(r, e["stroke-pattern-spacing"], W));
		let l = "0.";
		"stroke-pattern-start-offset" in e && (l = Q(r, e["stroke-pattern-start-offset"], W)), r.functions.sampleStrokePattern = "vec4 sampleStrokePattern(sampler2D texture, vec2 textureSize, vec2 textureOffset, vec2 sampleSize, float spacingPx, float startOffsetPx, float currentLengthPx, float currentRadiusRatio, float lineWidth) {\n  float currentLengthScaled = (currentLengthPx - startOffsetPx) * sampleSize.y / lineWidth;\n  float spacingScaled = spacingPx * sampleSize.y / lineWidth;\n  float uCoordPx = mod(currentLengthScaled, (sampleSize.x + spacingScaled));\n  float isInsideOfPattern = step(uCoordPx, sampleSize.x);\n  float vCoordPx = (-currentRadiusRatio * 0.5 + 0.5) * sampleSize.y;\n  // make sure that we're not sampling too close to the borders to avoid interpolation with outside pixels\n  uCoordPx = clamp(uCoordPx, 0.5, sampleSize.x - 0.5);\n  vCoordPx = clamp(vCoordPx, 0.5, sampleSize.y - 0.5);\n  vec2 texCoord = (vec2(uCoordPx, vCoordPx) + textureOffset) / textureSize;\n  return texture2D(texture, texCoord) * vec4(1.0, 1.0, 1.0, isInsideOfPattern);\n}";
		let u = `u_texture${i}`, d = "1.";
		"stroke-color" in e && (d = t.getStrokeColorExpression()), t.setStrokeColorExpression(`${d} * sampleStrokePattern(${u}, ${a}, ${s}, ${o}, ${c}, ${l}, currentLengthPx, currentRadiusRatio, v_width)`), r.functions.computeStrokePatternLength = "float computeStrokePatternLength(vec2 sampleSize, float spacingPx, float lineWidth) {\n  float patternLengthPx = sampleSize.x / sampleSize.y * lineWidth;\n  return patternLengthPx + spacingPx;\n}", t.setStrokePatternLengthExpression(`computeStrokePatternLength(${o}, ${c}, v_width)`);
	}
	if ("stroke-width" in e && t.setStrokeWidthExpression(Q(r, e["stroke-width"], W)), "stroke-offset" in e && t.setStrokeOffsetExpression(Q(r, e["stroke-offset"], W)), "stroke-line-cap" in e && t.setStrokeCapExpression(Q(r, e["stroke-line-cap"], G)), "stroke-line-join" in e && t.setStrokeJoinExpression(Q(r, e["stroke-line-join"], G)), "stroke-miter-limit" in e && t.setStrokeMiterLimitExpression(Q(r, e["stroke-miter-limit"], W)), "stroke-line-dash" in e) {
		r.functions.getSingleDashDistance = `float getSingleDashDistance(float distance, float radius, float dashOffset, float dashLength, float dashLengthTotal, float capType, float lineWidth) {
  float localDistance = mod(distance, dashLengthTotal);
  float distanceSegment = abs(localDistance - dashOffset - dashLength * 0.5) - dashLength * 0.5;
  distanceSegment = min(distanceSegment, dashLengthTotal - localDistance);
  if (capType == ${cm("square")}) {
    distanceSegment -= lineWidth * 0.5;
  } else if (capType == ${cm("round")}) {
    distanceSegment = min(distanceSegment, sqrt(distanceSegment * distanceSegment + radius * radius) - lineWidth * 0.5);
  }
  return distanceSegment;
}`;
		let n = e["stroke-line-dash"].map((e) => Q(r, e, W));
		n.length % 2 == 1 && (n = [...n, ...n]);
		let i = "0.";
		"stroke-line-dash-offset" in e && (i = Q(r, e["stroke-line-dash-offset"], W));
		let a = `dashDistanceField_${Zm(e["stroke-line-dash"])}`, o = n.map((e, t) => `float dashLength${t}`).join(", "), s = n.map((e, t) => `dashLength${t}`).join(" + "), c = "0.", l = `getSingleDashDistance(distance, radius, ${c}, dashLength0, totalDashLength, capType, lineWidth)`;
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
function oh(e, t, n, r) {
	if ("fill-color" in e && t.setFillColorExpression(Q(r, e["fill-color"], K)), "fill-pattern-src" in e) {
		let i = Zm(e["fill-pattern-src"]), a = eh(e, t, n, "fill-pattern-", i);
		t.setFillPatternSizeExpression(a);
		let o = "vec2(0.)";
		if ("fill-pattern-offset" in e && "fill-pattern-size" in e) {
			let n = Q(r, e["fill-pattern-size"], Ml);
			t.setFillPatternSizeExpression(n), o = th(e, "fill-pattern-", r, a, "v_patternSizePx");
		}
		r.functions.sampleFillPattern = "vec4 sampleFillPattern(sampler2D texture, vec2 textureSize, vec2 textureOffset, vec2 sampleSize, vec2 patternOriginPx, vec2 pxPosition, float sampleScaleRatio) {\n  vec2 pxRelativePos = pxPosition - patternOriginPx;\n\n  // rotate the relative position from origin by the current view rotation\n  pxRelativePos = vec2(pxRelativePos.x * cos(u_rotation) - pxRelativePos.y * sin(u_rotation), pxRelativePos.x * sin(u_rotation) + pxRelativePos.y * cos(u_rotation));\n  // sample position is computed according to the sample offset & size\n  vec2 samplePos = mod(pxRelativePos / sampleScaleRatio, sampleSize);\n  // also make sure that we're not sampling too close to the borders to avoid interpolation with outside pixels\n  samplePos = clamp(samplePos, vec2(0.5), sampleSize - vec2(0.5));\n  samplePos.y = sampleSize.y - samplePos.y; // invert y axis so that images appear upright\n  return texture2D(texture, (samplePos + textureOffset) / textureSize);\n}";
		let s = `u_texture${i}`, c = "1.";
		"fill-color" in e && (c = t.getFillColorExpression()), t.setFillColorExpression(`${c} * sampleFillPattern(${s}, ${a}, ${o}, v_patternSizePx, v_patternOriginPx, pxPos, df_float(u_df_patternScaleRatio))`);
	}
}
function sh(e, t, n, r) {
	function i(...e) {
		try {
			Q(...e);
		} catch {}
	}
	"text-value" in e && i(r, e["text-value"], G), "text-font" in e && i(r, e["text-font"], G), "text-max-angle" in e && i(r, e["text-max-angle"], W), "text-offset-x" in e && i(r, e["text-offset-x"], W), "text-offset-y" in e && i(r, e["text-offset-y"], W), "text-overflow" in e && i(r, e["text-overflow"], jl), "text-placement" in e && i(r, e["text-placement"], G), "text-repeat" in e && i(r, e["text-repeat"], W), "text-scale" in e && i(r, e["text-scale"], Nl), "text-rotate-with-view" in e && i(r, e["text-rotate-with-view"], jl), "text-rotation" in e && i(r, e["text-rotation"], W), "text-align" in e && i(r, e["text-align"], G), "text-justify" in e && i(r, e["text-justify"], G), "text-baseline" in e && i(r, e["text-baseline"], G), "text-padding" in e && i(r, e["text-padding"], Ml), "text-fill-color" in e && i(r, e["text-fill-color"], K), "text-stroke-color" in e && i(r, e["text-stroke-color"], K), "text-stroke-line-cap" in e && i(r, e["text-stroke-line-cap"], G), "text-stroke-line-join" in e && i(r, e["text-stroke-line-join"], G), "text-stroke-line-dash" in e && i(r, e["text-stroke-line-dash"], Ml), "text-stroke-line-dash-offset" in e && i(r, e["text-stroke-line-dash-offset"], W), "text-stroke-miter-limit" in e && i(r, e["text-stroke-miter-limit"], W), "text-stroke-width" in e && i(r, e["text-stroke-width"], W), "text-background-fill-color" in e && i(r, e["text-background-fill-color"], K), "text-background-stroke-color" in e && i(r, e["text-background-stroke-color"], K), "text-background-stroke-line-cap" in e && i(r, e["text-background-stroke-line-cap"], G), "text-background-stroke-line-join" in e && i(r, e["text-background-stroke-line-join"], G), "text-background-stroke-line-dash" in e && i(r, e["text-background-stroke-line-dash"], Ml), "text-background-stroke-line-dash-offset" in e && i(r, e["text-background-stroke-line-dash-offset"], W), "text-background-stroke-miter-limit" in e && i(r, e["text-background-stroke-miter-limit"], W), "text-background-stroke-width" in e && i(r, e["text-background-stroke-width"], W), "z-index" in e && i(r, e["z-index"], W);
}
function ch(e, t, n) {
	let r = um(t), i = new Mm(), a = {};
	if ("icon-src" in e ? ih(e, i, a, r) : "shape-points" in e ? rh(e, i, a, r) : "circle-radius" in e && nh(e, i, a, r), ah(e, i, a, r), oh(e, i, a, r), sh(e, i, a, r), n) {
		let e = Wl(t), a = Q(r, n, jl, e);
		e.mCoordinate ? i.setFragmentDiscardExpression(`!${a}`) : i.setShapeDiscardExpression(`!${a}`);
	}
	let o = {};
	function s(e, t, n, a) {
		if (!r[e]) return;
		let s = wm(n), c = Cm(n);
		i.addAttribute(`a_${t}`, s), o[t] = {
			size: c,
			callback: a
		};
	}
	return s("geometryType", mm, G, (e) => sm(uu(e.getGeometry()))), s("featureId", pm, G | W, (e) => {
		let t = e.getId() ?? null;
		return typeof t == "string" ? sm(t) : t;
	}), Tm(i, r), {
		builder: i,
		attributes: {
			...o,
			...Dm(r)
		},
		uniforms: {
			...a,
			...Em(r, t)
		}
	};
}
//#endregion
//#region node_modules/ol/render/webgl/textUtil.js
var lh = {
	TEXT_OVERLAY_TEXTURE: "u_textOverlay",
	TEXT_OVERLAY_MATRIX: "u_textOverlayMatrix"
};
function uh(e) {
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
function dh(e, t) {
	let n = gp();
	return {
		fragmentShader: `
      precision mediump float;

      uniform sampler2D u_image;
      uniform sampler2D ${lh.TEXT_OVERLAY_TEXTURE};
      uniform mat4 ${lh.TEXT_OVERLAY_MATRIX};

      varying vec2 v_texCoord;

      void main() {
        vec4 color = texture2D(u_image, v_texCoord);

        vec2 coords = v_texCoord * 2. - vec2(1.);
        coords = (${lh.TEXT_OVERLAY_MATRIX} * vec4(coords.xy, 0., 1.)).xy;
        coords = coords * 0.5 + vec2(0.5);
        float outOfBounds = clamp(step(1., coords.x) + step(1., coords.y) + step(0., -coords.x) + step(0., -coords.y), 0., 1.);

        vec4 textColor = texture2D(${lh.TEXT_OVERLAY_TEXTURE}, vec2(coords.x, 1. - coords.y));
        textColor.a *= 1. - outOfBounds; // if we're sampling out of the text overlay, make alpha 0 to avoid drawing anything

        gl_FragColor = textColor.a * textColor + (1. - textColor.a) * color;
      }`,
		uniforms: {
			[lh.TEXT_OVERLAY_TEXTURE]: e,
			[lh.TEXT_OVERLAY_MATRIX]: (r) => {
				let i = e(), a = t();
				if (!i || !a) return n;
				let o = a.viewState, s = r.viewState, c = s.center, l = s.resolution, u = s.rotation, d = r.size, f = o.center, p = o.resolution, m = o.rotation, h = i.width, g = i.height;
				return _p(n), yp(n, 1 / p / (h / 2), 1 / p / (g / 2), 1, n), xp(n, m, n), bp(n, c[0] - f[0], c[1] - f[1], 0, n), xp(n, -u, n), yp(n, l * d[0] / 2, l * d[1] / 2, 1, n), n;
			}
		}
	};
}
new ea("Point", [0, 0], [], 2, {}, "dummy"), new TextDecoder();
//#endregion
//#region node_modules/ol/render/webgl/VectorStyleRenderer.js
var fh = [], ph;
function mh() {
	return ph ||= Bm(), ph;
}
var hh = 0;
function gh(e, t, n) {
	let r = hh++;
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
var _h = {
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
}, vh = class extends c {
	constructor(e, t, n, r) {
		super(), this.helper_, this.hitDetectionEnabled_ = !!r, this.flatStyle = yh(e), this.styleShaders = bh(e, t), this.customAttributes_ = {}, this.uniforms_ = {}, this.hitDetectionEnabled_ && (this.customAttributes_.hitColor = {
			callback() {
				return Um(this.ref, fh);
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
				type: Vp.FLOAT
			}));
			return e.builder.getFillVertexShader() && (t.fillRenderPass = {
				vertexShader: e.builder.getFillVertexShader(),
				fragmentShader: e.builder.getFillFragmentShader(),
				attributesDesc: [{
					name: _h.POSITION,
					size: 2,
					type: Vp.FLOAT
				}, ...n],
				instancedAttributesDesc: [],
				instancePrimitiveVertexCount: 3
			}), e.builder.getStrokeVertexShader() && (t.strokeRenderPass = {
				vertexShader: e.builder.getStrokeVertexShader(),
				fragmentShader: e.builder.getStrokeFragmentShader(),
				attributesDesc: [{
					name: _h.LOCAL_POSITION,
					size: 2,
					type: Vp.FLOAT
				}],
				instancedAttributesDesc: [
					{
						name: _h.SEGMENT_START,
						size: 2,
						type: Vp.FLOAT
					},
					{
						name: _h.MEASURE_START,
						size: 1,
						type: Vp.FLOAT
					},
					{
						name: _h.SEGMENT_END,
						size: 2,
						type: Vp.FLOAT
					},
					{
						name: _h.MEASURE_END,
						size: 1,
						type: Vp.FLOAT
					},
					{
						name: _h.JOIN_ANGLES,
						size: 2,
						type: Vp.FLOAT
					},
					{
						name: _h.DISTANCE_LOW,
						size: 1,
						type: Vp.FLOAT
					},
					{
						name: _h.DISTANCE_HIGH,
						size: 1,
						type: Vp.FLOAT
					},
					{
						name: _h.ANGLE_TANGENT_SUM,
						size: 1,
						type: Vp.FLOAT
					},
					...n
				],
				instancePrimitiveVertexCount: 6
			}), e.builder.getSymbolVertexShader() && (t.symbolRenderPass = {
				vertexShader: e.builder.getSymbolVertexShader(),
				fragmentShader: e.builder.getSymbolFragmentShader(),
				attributesDesc: [{
					name: _h.LOCAL_POSITION,
					size: 2,
					type: Vp.FLOAT
				}],
				instancedAttributesDesc: [{
					name: _h.POSITION,
					size: 2,
					type: Vp.FLOAT
				}, ...n],
				instancePrimitiveVertexCount: 6
			}), t;
		}), this.hasFill_ = this.renderPasses_.some((e) => e.fillRenderPass), this.hasStroke_ = this.renderPasses_.some((e) => e.strokeRenderPass), this.hasSymbol_ = this.renderPasses_.some((e) => e.symbolRenderPass), this.hasText_ = this.flatStyle && uh(this.flatStyle), this.hasText_ && (this.textOverlayCanvas_ = I().canvas, this.textOverlayContext_ = this.textOverlayCanvas_.getContext("2d"), this.textOverlayRenderFrameState_ = null, this.textOverlayWorker_ = zm(), this.textOverlayRenderList_ = /* @__PURE__ */ new Set()), this.setHelper(n);
	}
	async generateBuffers(e, t, n) {
		let r = Nr(Tr(), t);
		if (e.isEmpty()) return {
			polygonBuffers: null,
			lineStringBuffers: null,
			pointBuffers: null,
			invertVerticesTransform: r,
			textInstructionsKey: null
		};
		let i = new Rm(), a = this.generateRenderInstructions_(e, i, t), [o, s, c, l] = await Promise.all([
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
			polygonInstructions: this.hasFill_ || this.hasText_ ? Ym(e.polygonBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null,
			lineStringInstructions: this.hasStroke_ || this.hasText_ ? Jm(e.lineStringBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null,
			pointInstructions: this.hasSymbol_ || this.hasText_ ? qm(e.pointBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null
		};
	}
	generateBuffersForType_(e, t, n) {
		if (e === null) return null;
		let r;
		switch (t) {
			case "Polygon":
				r = Vm.GENERATE_POLYGON_BUFFERS;
				break;
			case "LineString":
				r = Vm.GENERATE_LINE_STRING_BUFFERS;
				break;
			case "Point": r = Vm.GENERATE_POINT_BUFFERS;
		}
		let i = {
			type: r,
			renderInstructions: e.buffer,
			renderInstructionsTransform: n,
			customAttributesSize: Km(this.customAttributes_)
		};
		return gh(mh(), i, [e.buffer]).then((e) => {
			if (!this.helper_.getGL()) return;
			let t = e, n = new Pp(Cp, Ep).fromArrayBuffer(t.indicesBuffer), r = new Pp(Sp, Ep).fromArrayBuffer(t.vertexAttributesBuffer), i = new Pp(Sp, Ep).fromArrayBuffer(t.instanceAttributesBuffer);
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
			type: Hm.BUILD_INSTRUCTIONS,
			polygonRenderInstructions: a,
			lineStringRenderInstructions: o,
			pointRenderInstructions: s,
			labelsArray: t.getArray(),
			style: this.flatStyle,
			customAttributesSizes: c,
			renderInstructionsTransform: n,
			resolution: r
		};
		return gh(this.textOverlayWorker_, l, i).then((e) => e.instructionsSetKey);
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
			type: Hm.RENDER,
			frameState: Xm(e),
			batchesToRender: this.textOverlayRenderList_
		};
		return gh(this.textOverlayWorker_, t).then((e) => {
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
			type: Hm.DISPOSE_INSTRUCTIONS,
			instructionsSetKey: e
		});
	}
	disposeInternal() {
		this.textOverlayWorker_?.terminate(), super.disposeInternal();
	}
};
function yh(e) {
	return Array.isArray(e) ? e.some((e) => "builder" in e && !("sourceRule" in e)) ? null : e.some((e) => "builder" in e) ? e.map((e) => e.sourceRule) : e : "builder" in e ? "sourceRule" in e ? [e.sourceRule] : null : e;
}
function bh(e, t) {
	let n = Array.isArray(e) ? e : [e];
	if ("style" in n[0]) {
		let e = [], r = n, i = [];
		for (let n of r) {
			let r = Array.isArray(n.style) ? n.style : [n.style], a = n.filter;
			n.else && i.length && (a = ["all", ...i.map((e) => ["!", e])], n.filter && a.push(n.filter), a.length < 3 && (a = a[1])), n.filter && i.push(n.filter);
			let o = r.map((e) => ({
				...ch(e, t, a),
				sourceRule: n
			}));
			e.push(...o);
		}
		return e;
	}
	return "builder" in n[0] ? n : n.map((e) => ({
		...ch(e, t, null),
		sourceRule: { style: e }
	}));
}
//#endregion
//#region node_modules/ol/webgl/RenderTarget.js
var xh = /* @__PURE__ */ new Uint8Array(4), Sh = class {
	constructor(e, t) {
		this.helper_ = e;
		let n = e.getGL();
		this.texture_ = n.createTexture(), this.framebuffer_ = n.createFramebuffer(), this.depthbuffer_ = n.createRenderbuffer(), this.size_ = t || [1, 1], this.data_ = /* @__PURE__ */ new Uint8Array(), this.dataCacheDirty_ = !0, this.updateSize_();
	}
	setSize(e) {
		h(e, this.size_) || (this.size_[0] = e[0], this.size_[1] = e[1], this.updateSize_());
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
		if (e < 0 || t < 0 || e > this.size_[0] || t >= this.size_[1]) return xh[0] = 0, xh[1] = 0, xh[2] = 0, xh[3] = 0, xh;
		this.readAll();
		let n = Math.floor(e) + (this.size_[1] - Math.floor(t) - 1) * this.size_[0];
		return xh[0] = this.data_[n * 4], xh[1] = this.data_[n * 4 + 1], xh[2] = this.data_[n * 4 + 2], xh[3] = this.data_[n * 4 + 3], xh;
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
}, Ch = {
	PATTERN_ORIGIN_X_DOUBLE: "u_df_patternOriginX",
	PATTERN_ORIGIN_Y_DOUBLE: "u_df_patternOriginY",
	PATTERN_SCALE_RATIO_DOUBLE: "u_df_patternScaleRatio",
	ONE: "u_one"
}, wh = [0, 0], Th = [0, 0], Eh = Tr(), Dh = gp();
function Oh(e, t, n, r) {
	kr(Eh, t), Dr(Eh, n), e.setUniformMatrixValue(Bp.PROJECTION_MATRIX, vp(Dh, Eh)), Nr(Eh, Eh), e.setUniformMatrixValue(Bp.INVERT_PROJECTION_MATRIX, vp(Dh, Eh)), wh[0] = 0, wh[1] = 0;
	let i = r.size, a = r.viewState.resolution, o = r.viewState.center;
	Mr(Eh, i[0] / 2, i[1] / 2, 1 / a, 1 / a, 0, -o[0], -o[1]), z(Eh, wh), Th[0] = km(wh[0]), Th[1] = Om(wh[0]), e.setUniformFloatVec2(Ch.PATTERN_ORIGIN_X_DOUBLE, Th), Th[0] = km(wh[1]), Th[1] = Om(wh[1]), e.setUniformFloatVec2(Ch.PATTERN_ORIGIN_Y_DOUBLE, Th);
	let s = 2 ** ((r.viewState.zoom + .5) % 1 - .5);
	wh[0] = km(s), wh[1] = Om(s), e.setUniformFloatVec2(Ch.PATTERN_SCALE_RATIO_DOUBLE, wh);
}
//#endregion
//#region node_modules/ol/renderer/webgl/worldUtil.js
function kh(e, t) {
	let n = e.viewState.projection, r = t.getSource().getWrapX() && n.canWrapX(), i = n.getExtent(), a = e.extent, o = r ? L(i) : null, s = r ? Math.ceil((a[2] - i[2]) / o) + 1 : 1;
	return [
		r ? Math.floor((a[0] - i[0]) / o) : 0,
		s,
		o
	];
}
//#endregion
//#region node_modules/ol/renderer/webgl/VectorLayer.js
var Ah = {
	...Bp,
	...Ch,
	...lh,
	RENDER_EXTENT: "u_renderExtent",
	GLOBAL_ALPHA: "u_globalAlpha"
}, jh = class extends Zp {
	constructor(e, t) {
		let n = {
			[Ah.RENDER_EXTENT]: [
				0,
				0,
				0,
				0
			],
			[Ah.GLOBAL_ALPHA]: 1,
			[Ah.ONE]: 1
		};
		super(e, {
			uniforms: n,
			postProcesses: t.postProcesses ?? []
		}), this.hitDetectionEnabled_ = !t.disableHitDetection, this.hitRenderTarget_, this.sourceRevision_ = -1, this.layerRevision_ = -1, this.skipNextTextRender_ = !1, this.previousExtent_ = Ke(), this.currentTransform_ = Tr(), this.currentFrameStateTransform_ = Tr(), this.styleVariables_ = {}, this.style_ = [], this.hasText_ = !1, this.styleRenderer_ = null, this.buffers_ = null, this.batch_ = new Nm(), this.initialFeaturesAdded_ = !1, this.sourceListenKeys_ = null, this.applyOptions_(t);
	}
	addInitialFeatures_(e) {
		let t = this.getLayer().getSource(), n = gr(), r;
		n && (r = ur(n, e.viewState.projection)), this.batch_.addFeatures(t.getFeatures(), r), this.sourceListenKeys_ = [
			i(t, Pf.ADDFEATURE, this.handleSourceFeatureAdded_.bind(this, r)),
			i(t, Pf.CHANGEFEATURE, this.handleSourceFeatureChanged_.bind(this, r), this),
			i(t, Pf.REMOVEFEATURE, this.handleSourceFeatureDelete_, this),
			i(t, Pf.CLEAR, this.handleSourceFeatureClear_, this)
		];
	}
	applyOptions_(e) {
		this.styleVariables_ = e.variables, this.style_ = e.style;
		let t = yh(this.style_), n = !!t && uh(t);
		n && !this.hasText_ ? this.setPostProcesses([dh(() => this.styleRenderer_.getTextOverlayCanvas(), () => this.styleRenderer_.getTextOverlayFrameState()), ...this.getPostProcesses()]) : !n && this.hasText_ && this.setPostProcesses(this.getPostProcesses().slice(1)), this.hasText_ = n;
	}
	createRenderers_() {
		this.buffers_ = null, this.styleRenderer_ = new vh(this.style_, this.styleVariables_, this.helper, this.hitDetectionEnabled_);
	}
	reset(e) {
		this.applyOptions_(e), this.helper && this.createRenderers_(), super.reset(e);
	}
	afterHelperCreated() {
		this.styleRenderer_ ? this.styleRenderer_.setHelper(this.helper, this.buffers_) : this.createRenderers_(), this.hitDetectionEnabled_ && (this.hitRenderTarget_ = new Sh(this.helper));
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
		Oh(this.helper, this.currentFrameStateTransform_, e, t);
	}
	renderFrame(e) {
		let t = this.helper.getGL();
		this.preRender(t, e);
		let n = this.getLayer(), [r, i, a] = kh(e, n);
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
		let t = this.getLayer(), n = t.getSource(), r = e.viewState, i = !e.viewHints[Jo.ANIMATING] && !e.viewHints[Jo.INTERACTING], a = !Ze(this.previousExtent_, e.extent), o = this.sourceRevision_ < n.getRevision(), s = this.layerRevision_ < t.getRevision();
		if (this.sourceRevision_ = n.getRevision(), this.layerRevision_ = t.getRevision(), (s || a || o) && (this.skipNextTextRender_ = !1), i && (a || o)) {
			let i = r.projection, a = r.resolution, o = t instanceof yd ? t.getRenderBuffer() : 0, s = ze(e.extent, o * a), c = gr();
			c ? n.loadFeatures(yr(s, c), xr(a, i), c) : n.loadFeatures(s, a, i), this.ready = !1;
			let l = this.helper.makeProjectionTransform(e, Tr(), !0);
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
			this.helper.makeProjectionTransform(e, this.currentFrameStateTransform_), jr(this.currentFrameStateTransform_, a * i, 0), this.buffers_ && this.styleRenderer_.render(this.buffers_, e, () => {
				this.applyUniforms_(this.buffers_.invertVerticesTransform, e), this.helper.applyHitDetectionUniform(t);
			});
		while (++a < r);
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		if (R(this.hitDetectionEnabled_, "`forEachFeatureAtCoordinate` cannot be used on a WebGL layer if the hit detection logic has been disabled using the `disableHitDetection: true` option."), !this.styleRenderer_ || !this.hitDetectionEnabled_) return;
		let a = z(t.coordinateToPixelTransform, e.slice()), o = this.hitRenderTarget_.readPixel(a[0] / 2, a[1] / 2), s = Wm([
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
			o(e);
		}), null), this.styleRenderer_ && this.styleRenderer_.dispose(), super.disposeInternal();
	}
	renderDeclutter() {}
}, Mh = {
	BLUR: "blur",
	GRADIENT: "gradient",
	RADIUS: "radius"
}, Nh = [
	"#00f",
	"#0ff",
	"#0f0",
	"#ff0",
	"#f00"
], Ph = class extends yd {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.gradient, delete t.radius, delete t.blur, delete t.weight, super(t), this.on, this.once, this.un, this.filter_ = e.filter ?? !0, this.styleVariables_ = e.variables || {}, this.gradient_ = null, this.addChangeListener(Mh.GRADIENT, this.handleGradientChanged_), this.setGradient(e.gradient ? e.gradient : Nh), this.setBlur(e.blur === void 0 ? 15 : e.blur), this.setRadius(e.radius === void 0 ? 8 : e.radius);
		let n = e.weight ? e.weight : "weight";
		this.weight_ = n, this.setRenderOrder(null);
	}
	getBlur() {
		return this.get(Mh.BLUR);
	}
	getGradient() {
		return this.get(Mh.GRADIENT);
	}
	getRadius() {
		return this.get(Mh.RADIUS);
	}
	handleGradientChanged_() {
		this.gradient_ = Fh(this.getGradient());
	}
	setBlur(e) {
		let t = this.get(Mh.BLUR);
		if (this.set(Mh.BLUR, e), typeof e == "number" && typeof t == "number") {
			this.changed();
			return;
		}
		this.clearRenderer();
	}
	setGradient(e) {
		this.set(Mh.GRADIENT, e);
	}
	setRadius(e) {
		let t = this.get(Mh.RADIUS);
		if (this.set(Mh.RADIUS, e), typeof e == "number" && typeof t == "number") {
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
		let e = new Mm(), t = um(this.styleVariables_), n = Wl(this.styleVariables_), r = Q(t, this.filter_, jl, n), i = Q(t, this.getRadius(), W), a = Q(t, this.getBlur(), W), o = {};
		typeof this.getBlur() == "number" && (a = "a_blur", o.a_blur = () => this.getBlur(), e.addUniform("a_blur", "float")), typeof this.getRadius() == "number" && (i = "a_radius", o.a_radius = () => this.getRadius(), e.addUniform("a_radius", "float"));
		let s = {}, c = null;
		if (typeof this.weight_ == "string" || typeof this.weight_ == "function") {
			let t = typeof this.weight_ == "string" ? (e) => e.get(this.weight_) : this.weight_;
			s.prop_weight = {
				size: 1,
				callback: (e) => {
					let n = t(e);
					return n === void 0 ? 1 : Dt(n, 0, 1);
				}
			}, c = "a_prop_weight", e.addAttribute("a_prop_weight", "float");
		} else c = Q(t, [
			"clamp",
			this.weight_,
			0,
			1
		], W);
		let l = `(${i} / max(1., ${a}))`;
		e.setSymbolSizeExpression(`vec2(${i} + ${a}) * 2.`).setSymbolColorExpression(`vec4(smoothstep(0., 1., (1. - length(coordsPx * 2. / v_quadSizePx)) * ${l}) * ${c})`).setStrokeColorExpression(`vec4(smoothstep(0., 1., (1. - length(currentRadiusPx * 2. / v_width)) * ${l}) * ${c})`).setStrokeWidthExpression(`(${i} + ${a}) * 2.`).setFillColorExpression(`vec4(${c})`), n.mCoordinate ? e.setFragmentDiscardExpression(`!${r}`) : e.setShapeDiscardExpression(`!${r}`), Tm(e, t);
		let u = Dm(t), d = Em(t, this.styleVariables_);
		return new jh(this, {
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
function Fh(e) {
	let t = I(1, 256), n = t.createLinearGradient(0, 0, 1, 256), r = 1 / (e.length - 1);
	for (let t = 0, i = e.length; t < i; ++t) n.addColorStop(t * r, e[t]);
	return t.fillStyle = n, t.fillRect(0, 0, 1, 256), t.canvas;
}
//#endregion
//#region node_modules/ol-ext/util/input/Base.js
var Ih = class extends A {
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
}, Lh = class extends Ih {
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
}, Rh = class extends Lh {
	constructor(e) {
		e ||= {}, super(e), this.element.className = ("ol-ext-toggle-switch " + (e.className || "")).trim();
	}
}, zh = class extends Lh {
	constructor(e) {
		e ||= {}, super(e), this.element.className = ("ol-ext-check ol-ext-radio " + (e.className || "")).trim();
	}
}, $ = {};
$.create = function(e, t) {
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
			if (/select/i.test(e)) for (var i in t.options) $.create("OPTION", {
				html: i,
				value: t.options[i],
				parent: n
			});
			break;
		case "style":
			$.setStyle(n, t.style);
			break;
		case "change":
		case "click":
			$.addListener(n, r, t[r]);
			break;
		case "on":
			for (var a in t.on) $.addListener(n, a, t.on[a]);
			break;
		case "checked":
			n.checked = !!t.checked;
			break;
		default: n.setAttribute(r, t[r]);
	}
	return n;
}, $.createSwitch = function(e) {
	var t = $.create("INPUT", {
		type: "checkbox",
		on: e.on,
		click: e.click,
		change: e.change,
		parent: e.parent
	});
	return new Rh(Object.assign({ input: t }, e || {})), t;
}, $.createCheck = function(e) {
	var t = $.create("INPUT", {
		name: e.name,
		type: e.type === "radio" ? "radio" : "checkbox",
		on: e.on,
		parent: e.parent
	}), n = Object.assign({ input: t }, e || {});
	return e.type === "radio" ? new zh(n) : new Lh(n), t;
}, $.setHTML = function(e, t) {
	t instanceof Element ? e.appendChild(t) : t !== void 0 && (e.innerHTML = t);
}, $.appendText = function(e, t) {
	e.appendChild(document.createTextNode(t || ""));
}, $.addListener = function(e, t, n, r) {
	typeof t == "string" && (t = t.split(" ")), t.forEach(function(t) {
		e.addEventListener(t, n, r);
	});
}, $.removeListener = function(e, t, n) {
	typeof t == "string" && (t = t.split(" ")), t.forEach(function(t) {
		e.removeEventListener(t, n);
	});
}, $.show = function(e) {
	e.style.display = "";
}, $.hide = function(e) {
	e.style.display = "none";
}, $.hidden = function(e) {
	return $.getStyle(e, "display") === "none";
}, $.toggle = function(e) {
	e.style.display = e.style.display === "none" ? "" : "none";
}, $.setStyle = function(e, t) {
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
}, $.getStyle = function(e, t) {
	var n, r = (e.ownerDocument || document).defaultView;
	if (r && r.getComputedStyle) t = t.replace(/([A-Z])/g, "-$1").toLowerCase(), n = r.getComputedStyle(e, null).getPropertyValue(t);
	else if (e.currentStyle && (t = t.replace(/-(\w)/g, function(e, t) {
		return t.toUpperCase();
	}), n = e.currentStyle[t], /^\d+(em|pt|%|ex)?$/i.test(n))) return (function(t) {
		var n = e.style.left, r = e.runtimeStyle.left;
		return e.runtimeStyle.left = e.currentStyle.left, e.style.left = t || 0, t = e.style.pixelLeft + "px", e.style.left = n, e.runtimeStyle.left = r, t;
	})(n);
	return /px$/.test(n) ? parseInt(n) : n;
}, $.outerHeight = function(e) {
	return e.offsetHeight + $.getStyle(e, "marginBottom");
}, $.outerWidth = function(e) {
	return e.offsetWidth + $.getStyle(e, "marginLeft");
}, $.offsetRect = function(e) {
	var t = e.getBoundingClientRect();
	return {
		top: t.top + (window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0),
		left: t.left + (window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft || 0),
		height: t.height || t.bottom - t.top,
		width: t.width || t.right - t.left
	};
}, $.getFixedOffset = function(e) {
	var t = {
		left: 0,
		top: 0
	}, n = function(e) {
		if (!e) return t;
		if ($.getStyle(e, "position") === "absolute" && $.getStyle(e, "transform") !== "none") {
			var r = e.getBoundingClientRect();
			return t.left += r.left, t.top += r.top, t;
		}
		return n(e.offsetParent);
	};
	return n(e.offsetParent);
}, $.positionRect = function(e, t) {
	var n = 0, r = 0, i = function(a) {
		if (a) return n += a.offsetLeft, r += a.offsetTop, i(a.offsetParent);
		var o = {
			top: e.offsetTop + r,
			left: e.offsetLeft + n
		};
		return t && (o.top -= window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0, o.left -= window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft || 0), o.bottom = o.top + e.offsetHeight, o.right = o.top + e.offsetWidth, o;
	};
	return i(e.offsetParent);
}, $.scrollDiv = function(e, t) {
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
		t.target.classList.contains("ol-noscroll") || (l = !1, n = t[s], a = /* @__PURE__ */ new Date(), e.classList.add("ol-move"), t.preventDefault(), window.addEventListener("pointermove", g), $.addListener(window, ["pointerup", "pointercancel"], x));
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
			e.removeEventListener("pointermove", b), e.parentNode.classList.add("ol-miniscroll"), y = $.create("DIV"), v = $.create("DIV", {
				className: "ol-scroll",
				html: y
			}), e.parentNode.insertBefore(v, e), y.addEventListener("pointerdown", function(e) {
				d = !0, h(e);
			}), t.mousewheel && ($.addListener(v, [
				"mousewheel",
				"DOMMouseScroll",
				"onmousewheel"
			], function(e) {
				S(e);
			}), $.addListener(y, [
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
	e.style["touch-action"] = "none", e.style.overflow = "hidden", e.classList.add("ol-scrolldiv"), $.addListener(e, ["pointerdown"], function(e) {
		d = !1, h(e);
	}), e.addEventListener("click", function(t) {
		e.classList.contains("ol-move") && (t.preventDefault(), t.stopPropagation());
	}, !0);
	var x = function(i) {
		a = /* @__PURE__ */ new Date() - a, a > 100 || d ? r = 0 : a > 0 && (r = ((r || 0) + (n - i[s]) / a) / 2), _(t.animate === !1 ? 0 : r * 200), n = !1, r = 0, a = 0, e.classList.contains("ol-move") ? e.classList.remove("ol-hasClick") : (e.classList.add("ol-hasClick"), setTimeout(function() {
			e.classList.remove("ol-hasClick");
		}, 500)), d = !1, window.removeEventListener("pointermove", g), $.removeListener(window, ["pointerup", "pointercancel"], x);
	}, S = function(t) {
		var n = Math.max(-1, Math.min(1, t.wheelDelta || -t.detail));
		return e.classList.add("ol-move"), e[c] -= n * 30, e.classList.remove("ol-move"), !1;
	};
	return t.mousewheel && $.addListener(e, [
		"mousewheel",
		"DOMMouseScroll",
		"onmousewheel"
	], S), { refresh: p };
}, $.dispatchEvent = function(e, t) {
	var n;
	try {
		n = new CustomEvent(e);
	} catch {
		n = document.createEvent("CustomEvent"), n.initCustomEvent(e, !0, !0, {});
	}
	t.dispatchEvent(n);
}, $.setCursor = function(e, t) {
	e instanceof Tf && (e = e.getTargetElement()), !("ontouchstart" in window) && e instanceof Element && (e.style.cursor = t);
};
//#endregion
//#region node_modules/ol-ext/control/LayerSwitcher.js
var Bh = class extends De {
	constructor(e) {
		e ||= {};
		var t = $.create("DIV", { className: e.switcherClass || "ol-layerswitcher" });
		super({
			element: t,
			target: e.target
		});
		var n = this;
		this.dcount = 0, this.show_progress = e.show_progress, this.oninfo = typeof e.oninfo == "function" ? e.oninfo : null, this.onextent = typeof e.onextent == "function" ? e.onextent : null, this.hasextent = e.extent || e.onextent, this.hastrash = e.trash, this.reordering = e.reordering !== !1, this._layers = [], this._layerGroup = e.layerGroup && e.layerGroup.getLayers ? e.layerGroup : null, this.onchangeCheck = typeof e.onchangeCheck == "function" ? e.onchangeCheck : null, typeof e.displayInLayerSwitcher == "function" && (this.displayInLayerSwitcher = e.displayInLayerSwitcher), e.target || (t.classList.add("ol-unselectable"), t.classList.add("ol-control"), t.classList.add(e.collapsed === !1 ? "ol-forceopen" : "ol-collapsed"), e.counter && t.classList.add("ol-counter"), this.counter = $.create("SPAN", {
			class: "ol-counter",
			text: 0,
			parent: t
		}), this.button = $.create("BUTTON", {
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
		})), e.minibar && (e.noScroll = !0), e.noScroll || (this.topv = $.create("DIV", {
			className: "ol-switchertopdiv",
			parent: t,
			click: function() {
				n.overflow("+50%");
			}
		}), this.botv = $.create("DIV", {
			className: "ol-switcherbottomdiv",
			parent: t,
			click: function() {
				n.overflow("-50%");
			}
		})), this._noScroll = e.noScroll), this.panel_ = $.create("UL", { className: "panel" }), this.panelContainer_ = $.create("DIV", {
			className: "panel-container",
			html: this.panel_,
			parent: t
		}), !e.target && !e.noScroll && $.addListener(this.panel_, "mousewheel DOMMouseScroll onmousewheel", function(e) {
			n.overflow(Math.max(-1, Math.min(1, e.wheelDelta || -e.detail))) && (e.stopPropagation(), e.preventDefault());
		}), this.header_ = $.create("LI", {
			className: "ol-header",
			parent: this.panel_
		}), this.set("drawDelay", e.drawDelay || 0), this.set("selection", e.selection), e.minibar && setTimeout(function() {
			var e = $.scrollDiv(this.panelContainer_, {
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
		if (super.setMap(e), this.drawPanel(), this._listener) for (var t in this._listener) T(this._listener[t]);
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
		$.setHTML(this.header_, e);
	}
	overflow(e) {
		if (this.button && !this._noScroll) {
			if ($.hidden(this.panel_)) {
				$.setStyle(this.element, { height: "auto" });
				return;
			}
			var t = $.outerHeight(this.element), n = $.outerHeight(this.panel_), r = this.button.offsetTop + $.outerHeight(this.button), i = this.panel_.offsetTop - r;
			if (n > t - r) {
				$.setStyle(this.element, { height: "100%" });
				var a = this.panel_.querySelectorAll("li.ol-visible .li-content")[0], o = a ? 2 * $.getStyle(a, "height") : 0;
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
				return i + n <= t - 3 * r / 2 ? (i = t - 3 * r / 2 - n, $.hide(this.botv)) : $.show(this.botv), i >= 0 ? (i = 0, $.hide(this.topv)) : $.show(this.topv), $.setStyle(this.panel_, { top: i + "px" }), !0;
			}
			return $.setStyle(this.element, { height: "auto" }), $.setStyle(this.panel_, { top: 0 }), $.hide(this.botv), $.hide(this.topv), !1;
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
				T(e);
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
		return !r || ht(this.getMap().getView().calculateExtent(this.getMap().getSize()), r);
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
			}), n.classList.remove("drag"), n.parentNode.classList.remove("drag"), t.element.classList.remove("drag"), c && c.remove(), $.removeListener(document, "mousemove touchmove", f), $.removeListener(document, "mouseup touchend touchcancel", d);
		}
		function f(e) {
			if (a = e.pageY || e.touches && e.touches.length && e.touches[0].pageY || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageY, r && Math.abs(o - a) > 2 && (r = !1, n.classList.add("drag"), l = t._getLayerForLI(n), s = !1, u = t._getLayerForLI(n.parentNode.parentNode), c = $.create("LI", {
				className: "ol-dragover",
				html: n.innerHTML,
				style: {
					position: "absolute",
					"z-index": 1e4,
					left: n.offsetLeft,
					opacity: .5,
					width: $.outerWidth(n),
					height: $.getStyle(n, "height")
				},
				parent: i
			}), t.element.classList.add("drag"), t.dispatchEvent({
				type: "reorder-start",
				layer: l,
				group: u
			})), !r) {
				e.preventDefault(), e.stopPropagation(), $.setStyle(c, { top: a - $.offsetRect(i).top + i.scrollTop + 5 });
				var d;
				if (!e.touches) d = e.target, e.target.shadowRoot && (d = e.composedPath()[0]);
				else for (d = document.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY); d.shadowRoot;) d = d.shadowRoot.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY);
				for (d.classList.contains("ol-switcherbottomdiv") ? t.overflow(-1) : d.classList.contains("ol-switchertopdiv") && t.overflow(1); d && d.tagName !== "LI";) d = d.parentNode;
				(!d || !d.classList.contains("dropover")) && n.parentNode.querySelectorAll("li").forEach(function(e) {
					e.classList.remove("dropover"), e.classList.remove("dropover-after"), e.classList.remove("dropover-before");
				}), d && d.parentNode.classList.contains("drag") && d !== n ? (s = t._getLayerForLI(d), s && !s.get("allwaysOnTop") == !l.get("allwaysOnTop") ? (d.classList.add("dropover"), d.classList.add(n.offsetTop < d.offsetTop ? "dropover-after" : "dropover-before")) : s = !1, $.show(c)) : (s = !1, d === n ? $.hide(c) : $.show(c)), s ? c.classList.remove("forbidden") : c.classList.add("forbidden");
			}
		}
		$.addListener(document, "mousemove touchmove", f), $.addListener(document, "mouseup touchend touchcancel", d);
	}
	dragOpacity_(e) {
		e.stopPropagation(), e.preventDefault();
		var t = this, n = e.target, r = this._getLayerForLI(n.parentNode.parentNode.parentNode);
		if (!r) return;
		var i = e.pageX || e.touches && e.touches.length && e.touches[0].pageX || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageX, a = $.getStyle(n, "left") - i;
		t.dragging_ = !0;
		function o() {
			$.removeListener(document, "mouseup touchend touchcancel", o), $.removeListener(document, "mousemove touchmove", s), t.dragging_ = !1;
		}
		function s(e) {
			var t = (a + (e.pageX || e.touches && e.touches.length && e.touches[0].pageX || e.changedTouches && e.changedTouches.length && e.changedTouches[0].pageX)) / $.getStyle(n.parentNode, "width"), i = Math.max(0, Math.min(1, t));
			$.setStyle(n, { left: i * 100 + "%" }), n.parentNode.nextElementSibling.innerHTML = Math.round(i * 100), r.setOpacity(i);
		}
		$.addListener(document, "mouseup touchend touchcancel", o), $.addListener(document, "mousemove touchmove", s);
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
			var u = $.create("LI", {
				className: (c.getVisible() ? "ol-visible " : " ") + (c.get("baseLayer") ? "baselayer" : ""),
				parent: e
			});
			this._setLayerForLI(u, c), this._selectedLayer === c && u.classList.add("ol-layer-select");
			var d = $.create("DIV", {
				className: "ol-layerswitcher-buttons",
				parent: u
			}), f = $.create("DIV", {
				className: "li-content",
				parent: u
			}), p = $.create("INPUT", {
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
			var m = $.create("LABEL", {
				title: c.get("title") || c.get("name"),
				click: i,
				style: { userSelect: "none" },
				parent: f
			});
			if (m.addEventListener("selectstart", function() {
				return !1;
			}), $.create("SPAN", {
				html: c.get("title") || c.get("name"),
				click: function(e) {
					this.get("selection") && (e.stopPropagation(), this.selectLayer(c));
				}.bind(this),
				parent: m
			}), this.reordering && (l < r.length - 1 && (c.get("allwaysOnTop") || !r[l + 1].get("allwaysOnTop")) || l > 0 && (!c.get("allwaysOnTop") || r[l - 1].get("allwaysOnTop"))) && $.create("DIV", {
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
				}), h && $.create("DIV", {
					className: c.get("openInLayerSwitcher") ? "collapse-layers" : "expend-layers",
					title: this.tip.plus,
					click: function() {
						var e = n._getLayerForLI(this.parentNode.parentNode);
						e.set("openInLayerSwitcher", !e.get("openInLayerSwitcher"));
					},
					parent: d
				});
			}
			if (this.oninfo && $.create("DIV", {
				className: "layerInfo",
				title: this.tip.info,
				click: a,
				parent: d
			}), this.hastrash && !c.get("noSwitcherDelete") && $.create("DIV", {
				className: "layerTrash",
				title: this.tip.trash,
				click: s,
				parent: d
			}), this.hasextent && r[l].getExtent()) {
				var g = r[l].getExtent();
				g.length == 4 && g[0] < g[2] && g[1] < g[3] && $.create("DIV", {
					className: "layerExtent",
					title: this.tip.extent,
					click: o,
					parent: d
				});
			}
			if (this.show_progress && c instanceof bs) {
				var _ = $.create("DIV", {
					className: "layerswitcher-progress",
					parent: f
				});
				this.setprogress_(c), c.layerswitcher_progress = $.create("DIV", { parent: _ });
			}
			var v = $.create("DIV", {
				className: "layerswitcher-opacity",
				click: function(e) {
					if (e.target === this) {
						e.stopPropagation(), e.preventDefault();
						var t = Math.max(0, Math.min(1, e.offsetX / $.getStyle(this, "width")));
						n._getLayerForLI(this.parentNode.parentNode).setOpacity(t), this.parentNode.querySelectorAll(".layerswitcher-opacity-label")[0].innerHTML = Math.round(t * 100);
					}
				},
				parent: f
			});
			if ($.create("DIV", {
				className: "layerswitcher-opacity-cursor ol-noscroll",
				style: { left: c.getOpacity() * 100 + "%" },
				on: { "mousedown touchstart": function(e) {
					n.dragOpacity_(e);
				} },
				parent: v
			}), $.create("DIV", {
				className: "layerswitcher-opacity-label",
				html: Math.round(c.getOpacity() * 100),
				parent: f
			}), c.getLayers && (u.classList.add("ol-layer-group"), c.get("openInLayerSwitcher") === !0)) {
				var y = $.create("UL", { parent: u });
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
		return e ? e.getLayers ? "ol-layer-group" : e instanceof xd ? "ol-layer-vector" : e instanceof fp ? "ol-layer-vectortile" : e instanceof bs ? "ol-layer-tile" : e instanceof hp ? "ol-layer-image" : e instanceof Ph ? "ol-layer-heatmap" : e.getFeatures ? "ol-layer-vectorimage" : "unknown" : "none";
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
				n === t ? (n = t = 0, $.setStyle(e.layerswitcher_progress, { width: 0 })) : $.setStyle(e.layerswitcher_progress, { width: (t / n * 100).toFixed(1) + "%" });
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
Bh.prototype.tip = {
	up: "up/down",
	down: "down",
	info: "informations...",
	extent: "zoom to extent",
	trash: "remove layer",
	plus: "expand/shrink"
};
//#endregion
//#region src/api/tileserver.js
async function Vh() {
	let e = await fetch("/tileserver/session_id");
	if (!e.ok) throw Error("Failed to create TileServer session.");
	return (await e.json()).session_id;
}
async function Hh(e) {
	let t = new FormData();
	if (t.append("slide_path", e), !(await fetch("/tileserver/slide", {
		method: "PUT",
		body: t
	})).ok) throw Error(`Failed to load slide: ${e}`);
	let n = await fetch("/tileserver/slide");
	if (!n.ok) throw Error("Failed to retrieve slide metadata.");
	return n.json();
}
async function Uh(e) {
	let t = await fetch(`/tileserver/files/${e}`);
	if (!t.ok) throw Error(`Failed to get configured ${e} files.`);
	return t.json();
}
async function Wh() {
	if (!(await fetch("/tileserver/clear_overlays", { method: "PUT" })).ok) throw Error("Failed to clear overlays.");
}
async function Gh() {
	if (!(await fetch("/tileserver/slide", { method: "DELETE" })).ok) throw Error("Failed to remove the current slide.");
}
async function Kh(e, t) {
	let n = new FormData();
	n.append("overlay_path", e), n.append("layer_name", t);
	let r = await fetch("/tileserver/overlay", {
		method: "PUT",
		body: n
	});
	if (!r.ok) throw Error(`Failed to load overlay: ${e}`);
	return r.json();
}
async function qh(e) {
	if (!(await fetch(`/tileserver/overlay/${encodeURIComponent(e)}`, { method: "DELETE" })).ok) throw Error(`Failed to remove overlay: ${e}`);
}
function Jh(e, t) {
	return t === null ? e : `${e}?${new URLSearchParams({ layer: t })}`;
}
async function Yh(e, t = null) {
	let n = new FormData();
	if (n.append("val", JSON.stringify(e)), !(await fetch(Jh("/tileserver/renderer/where", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation visibility.");
}
async function Xh(e, t = null) {
	let n = e instanceof Map ? [...e.entries()] : Object.entries(e), r = new FormData();
	if (r.append("cmap", JSON.stringify({
		keys: n.map(([e]) => e),
		values: n.map(([, e]) => e)
	})), !(await fetch(Jh("/tileserver/cmap", t), {
		method: "PUT",
		body: r
	})).ok) throw Error("Failed to update annotation colours.");
}
async function Zh(e, t = null) {
	let n = e instanceof Map ? [...e.entries()] : Object.entries(e), r = new FormData();
	if (r.append("opacities", JSON.stringify({
		keys: n.map(([e]) => e),
		values: n.map(([, e]) => e)
	})), !(await fetch(Jh("/tileserver/annotation_opacities", t), {
		method: "PUT",
		body: r
	})).ok) throw Error("Failed to update annotation opacity.");
}
async function Qh(e, t = null) {
	let n = new FormData();
	n.append("types", JSON.stringify(e)), t !== null && n.append("palette", t);
	let r = await fetch("/tileserver/annotation_colours", {
		method: "PUT",
		body: n
	});
	if (!r.ok) throw Error("Failed to generate annotation colours.");
	let i = await r.json();
	return new Map(i.keys.map((e, t) => [e, i.values[t]]));
}
async function $h(e) {
	let t = new URLSearchParams({ layer: e }), n = await fetch(`/tileserver/prop_names/all?${t}`);
	if (!n.ok) throw Error("Failed to get annotation properties.");
	return n.json();
}
async function eg(e, t) {
	let n = new URLSearchParams({ layer: e }), r = await fetch(`/tileserver/prop_values/${encodeURIComponent(t)}/all?${n}`);
	if (!r.ok) throw Error("Failed to get annotation property values.");
	return r.json();
}
async function tg(e, t, n, { details: r = !1 } = {}) {
	let i = new URLSearchParams({ layer: e });
	r && i.set("details", "1");
	let a = await fetch(`/tileserver/tap_query/${t}/${n}?${i}`);
	if (!a.ok) throw Error("Failed to inspect annotation.");
	return a.json();
}
async function ng(e, t = null) {
	let n = new FormData();
	if (n.append("val", JSON.stringify(e)), !(await fetch(Jh("/tileserver/renderer/score_prop", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation property.");
}
async function rg(e, t = null) {
	let n = new FormData();
	if (n.append("cmap", JSON.stringify(e)), !(await fetch(Jh("/tileserver/cmap", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation colour map.");
}
async function ig(e, t = null) {
	let n = new FormData();
	if (n.append("range", JSON.stringify(e)), !(await fetch(Jh("/tileserver/prop_range", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation property range.");
}
async function ag(e, t, n, r, i = null) {
	let a = new FormData();
	if (a.append("type_id", JSON.stringify(e)), a.append("prop", t), a.append("cmap", JSON.stringify(n)), a.append("range", JSON.stringify(r)), !(await fetch(Jh("/tileserver/secondary_cmap", i), {
		method: "PUT",
		body: a
	})).ok) throw Error("Failed to update secondary annotation colour map.");
}
async function og(e = null) {
	let t = new FormData();
	if (t.append("val", JSON.stringify(null)), !(await fetch(Jh("/tileserver/renderer/secondary_cmap", e), {
		method: "PUT",
		body: t
	})).ok) throw Error("Failed to clear secondary annotation colour map.");
}
//#endregion
//#region node_modules/ol/control/FullScreen.js
var sg = ["fullscreenchange", "webkitfullscreenchange"], cg = {
	ENTERFULLSCREEN: "enterfullscreen",
	LEAVEFULLSCREEN: "leavefullscreen"
}, lg = class extends De {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			target: e.target
		}), this.on, this.once, this.un, this.keys_ = e.keys !== void 0 && e.keys, this.source_ = e.source, this.isInFullscreen_ = !1, this.boundHandleMapTargetChange_ = this.handleMapTargetChange_.bind(this), this.cssClassName_ = e.className === void 0 ? "ol-full-screen" : e.className, this.documentListeners_ = [], this.activeClassName_ = e.activeClassName === void 0 ? [this.cssClassName_ + "-true"] : e.activeClassName.split(" "), this.inactiveClassName_ = e.inactiveClassName === void 0 ? [this.cssClassName_ + "-false"] : e.inactiveClassName.split(" ");
		let t = e.label === void 0 ? "⤢" : e.label;
		this.labelNode_ = typeof t == "string" ? document.createTextNode(t) : t;
		let n = e.labelActive === void 0 ? "×" : e.labelActive;
		this.labelActiveNode_ = typeof n == "string" ? document.createTextNode(n) : n;
		let r = e.tipLabel ? e.tipLabel : "Toggle full-screen";
		this.button_ = document.createElement("button"), this.button_.title = r, this.button_.setAttribute("type", "button"), this.button_.appendChild(this.labelNode_), this.button_.addEventListener(s.CLICK, this.handleClick_.bind(this), !1), this.setClassName_(this.button_, this.isInFullscreen_), this.element.className = `${this.cssClassName_} ${F} ${ne}`, this.element.appendChild(this.button_);
	}
	handleClick_(e) {
		e.preventDefault(), this.handleFullScreen_();
	}
	handleFullScreen_() {
		let e = this.getMap();
		if (!e) return;
		let t = e.getOwnerDocument();
		if (ug(t)) {
			if (dg(t)) mg(t);
			else {
				let n;
				n = this.source_ ? typeof this.source_ == "string" ? t.getElementById(this.source_) : this.source_ : e.getTargetElement(), this.keys_ ? pg(n) : fg(n);
			}
		}
	}
	handleFullScreenChange_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.isInFullscreen_;
		this.isInFullscreen_ = dg(e.getOwnerDocument()), t !== this.isInFullscreen_ && (this.setClassName_(this.button_, this.isInFullscreen_), this.isInFullscreen_ ? (xe(this.labelActiveNode_, this.labelNode_), this.dispatchEvent(cg.ENTERFULLSCREEN)) : (xe(this.labelNode_, this.labelActiveNode_), this.dispatchEvent(cg.LEAVEFULLSCREEN)), e.updateSize());
	}
	setClassName_(e, t) {
		t ? (e.classList.remove(...this.inactiveClassName_), e.classList.add(...this.activeClassName_)) : (e.classList.remove(...this.activeClassName_), e.classList.add(...this.inactiveClassName_));
	}
	setMap(e) {
		let t = this.getMap();
		t && t.removeChangeListener(Dd.TARGET, this.boundHandleMapTargetChange_), super.setMap(e), this.handleMapTargetChange_(), e && e.addChangeListener(Dd.TARGET, this.boundHandleMapTargetChange_);
	}
	handleMapTargetChange_() {
		let e = this.documentListeners_;
		for (let t = 0, n = e.length; t < n; ++t) o(e[t]);
		e.length = 0;
		let t = this.getMap();
		if (t) {
			let n = t.getOwnerDocument();
			ug(n) ? this.element.classList.remove(te) : this.element.classList.add(te);
			for (let t = 0, r = sg.length; t < r; ++t) e.push(i(n, sg[t], this.handleFullScreenChange_, this));
			this.handleFullScreenChange_();
		}
	}
};
function ug(e) {
	let t = e.body;
	return !!(t.webkitRequestFullscreen || t.requestFullscreen && e.fullscreenEnabled);
}
function dg(e) {
	return !!(e.webkitIsFullScreen || e.fullscreenElement);
}
function fg(e) {
	e.requestFullscreen ? e.requestFullscreen() : e.webkitRequestFullscreen && e.webkitRequestFullscreen();
}
function pg(e) {
	e.webkitRequestFullscreen ? e.webkitRequestFullscreen() : fg(e);
}
function mg(e) {
	e.exitFullscreen ? e.exitFullscreen() : e.webkitExitFullscreen && e.webkitExitFullscreen();
}
//#endregion
//#region node_modules/ol/control/MousePosition.js
var hg = "projection", gg = "coordinateFormat", _g = class extends De {
	constructor(e) {
		e ||= {};
		let t = document.createElement("div");
		t.className = e.className === void 0 ? "ol-mouse-position" : e.className, super({
			element: t,
			render: e.render,
			target: e.target
		}), this.on, this.once, this.un, this.addChangeListener(hg, this.handleProjectionChanged_), e.coordinateFormat && this.setCoordinateFormat(e.coordinateFormat), e.projection && this.setProjection(e.projection), this.renderOnMouseOut_ = e.placeholder !== void 0, this.placeholder_ = this.renderOnMouseOut_ ? e.placeholder : "&#160;", this.renderedHTML_ = t.innerHTML, this.mapProjection_ = null, this.transform_ = null, this.wrapX_ = e.wrapX !== !1;
	}
	handleProjectionChanged_() {
		this.transform_ = null;
	}
	getCoordinateFormat() {
		return this.get(gg);
	}
	getProjection() {
		return this.get(hg);
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
			this.listenerKeys.push(i(t, Td.POINTERMOVE, this.handleMouseMove, this)), this.renderOnMouseOut_ && this.listenerKeys.push(i(t, Td.POINTEROUT, this.handleMouseOut, this)), this.updateHTML_(null);
		}
	}
	setCoordinateFormat(e) {
		this.set(gg, e);
	}
	setProjection(e) {
		this.set(hg, rr(e));
	}
	updateHTML_(e) {
		let t = this.placeholder_;
		if (e && this.mapProjection_) {
			if (!this.transform_) {
				let e = this.getProjection();
				this.transform_ = e ? ur(this.mapProjection_, e) : er;
			}
			let n = this.getMap().getCoordinateFromPixelInternal(e);
			if (n) {
				let e = gr();
				e && (this.transform_ = ur(this.mapProjection_, e)), this.transform_(n, n), this.wrapX_ && Wt(n, e || this.getProjection() || this.mapProjection_);
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
};
//#endregion
//#region src/controls/map-controls.js
function vg({ map: e, viewerApp: t, getSlideSource: n, zoomVisibleInput: r, zoomLevelVisibleInput: i, rotationVisibleInput: a, resetViewButton: o, resetViewControl: s, resetViewVisibleInput: c, fullscreenVisibleInput: l, mousePositionVisibleInput: u, mouseWheelZoomSensitivitySelect: d, zoomButtonStepSelect: f }) {
	let p = {
		low: {
			deltaPerZoom: 600,
			maxDelta: 1
		},
		default: {
			deltaPerZoom: 300,
			maxDelta: 1
		},
		high: {
			deltaPerZoom: 150,
			maxDelta: 2
		}
	};
	function m() {
		let e = p[d.value] ?? p.default, t = new ff({ maxDelta: e.maxDelta });
		return t.deltaPerZoom_ = e.deltaPerZoom, t.setActive(n() !== null), t;
	}
	let h = m();
	e.addInteraction(h);
	function g() {
		return new Pe({ delta: Number(f.value) });
	}
	let _ = g();
	e.addControl(_);
	let v = _.element, y = v.querySelector(".ol-zoom-out");
	/* v8 ignore if */
	if (y === null) throw Error("The OpenLayers zoom control could not be found.");
	let b = document.createElement("input");
	b.type = "number", b.className = "ol-zoom-level", b.step = "1", b.setAttribute("aria-label", "Zoom level"), b.title = "Zoom level", v.insertBefore(b, y);
	function x() {
		let t = e.getView().getZoom();
		if (t === void 0) {
			b.value = "";
			return;
		}
		b.value = Number.isInteger(t) ? t.toString() : t.toFixed(1);
	}
	function S() {
		let t = Number.parseFloat(b.value);
		if (!Number.isFinite(t)) {
			x();
			return;
		}
		let n = e.getView(), r = Math.min(Math.max(t, n.getMinZoom()), n.getMaxZoom());
		n.setZoom(r), x();
	}
	b.addEventListener("focus", () => {
		b.select();
	}), b.addEventListener("blur", () => {
		S();
	}), b.addEventListener("keydown", (e) => {
		if (e.key === "Enter") {
			e.preventDefault(), b.blur();
			return;
		}
		e.key === "Escape" && (e.preventDefault(), x(), b.blur());
	}), x();
	function C() {
		let t = n();
		if (t === null) return;
		let r = t.getTileGrid().getExtent(), i = e.getView();
		i.setRotation(0), i.fit(r, { size: e.getSize() });
	}
	o.addEventListener("click", () => {
		C();
	});
	let w = new _g({
		coordinateFormat: (e) => Bt([e[0], -e[1]], "{x}, {y}", 0),
		className: "ol-mouse-position",
		placeholder: "\xA0"
	});
	e.addControl(w);
	let T = new Ne({
		autoHide: !1,
		className: "ol-rotate"
	});
	e.addControl(T);
	let E = new lg({ source: t });
	e.addControl(E);
	let D = document.createElement("div");
	D.className = "bottom-controls-group ol-unselectable", t.append(D), D.append(s, v, E.element);
	let O = n() !== null;
	function k(e) {
		O = e;
		let t = v.querySelector(".ol-zoom-in"), n = v.querySelector(".ol-zoom-out"), r = T.element.querySelector("button");
		for (let i of [
			t,
			n,
			r,
			o
		]) i !== null && (i.disabled = !e);
		b.disabled = !e, v.classList.toggle("viewer-control-disabled", !e), h.setActive(e), w.element.classList.toggle("viewer-control-hidden", !e || !u.checked);
	}
	function A() {
		let e = n() !== null;
		v.classList.toggle("viewer-control-hidden", !r.checked), b.classList.toggle("viewer-control-hidden", !i.checked), T.element.classList.toggle("viewer-control-hidden", !a.checked), s.classList.toggle("viewer-control-hidden", !c.checked), E.element.classList.toggle("viewer-control-hidden", !l.checked), w.element.classList.toggle("viewer-control-hidden", !e || !u.checked);
	}
	function j() {
		e.removeInteraction(h), h = m(), h.setActive(O), e.addInteraction(h);
	}
	function M() {
		/* v8 ignore if */
		if (e.removeControl(_), _ = g(), e.addControl(_), v = _.element, y = v.querySelector(".ol-zoom-out"), y === null) throw Error("The OpenLayers zoom control could not be found.");
		v.insertBefore(b, y), D.insertBefore(v, E.element), k(O), A();
	}
	return d.addEventListener("change", () => {
		j();
	}), f.addEventListener("change", () => {
		M();
	}), {
		fullscreen: E,
		mousePositionControl: w,
		rotate: T,
		setViewerEnabled: k,
		updateMouseWheelZoomSensitivity: j,
		updateVisibility: A,
		updateZoomButtonStep: M,
		updateZoomLevel: x
	};
}
//#endregion
//#region node_modules/ol-ext/util/getMapCanvas.js
var yg = function(e) {
	if (!e) return null;
	var t = e.getViewport().getElementsByClassName("ol-fixedoverlay")[0];
	return t || (e.getViewport().querySelector(".ol-layers") ? (t = document.createElement("canvas"), t.className = "ol-fixedoverlay", e.getViewport().querySelector(".ol-layers").after(t), e.on("precompose", function(n) {
		t.width = e.getSize()[0] * n.frameState.pixelRatio, t.height = e.getSize()[1] * n.frameState.pixelRatio;
	})) : t = e.getViewport().querySelector("canvas")), t;
}, bg = class extends De {
	constructor(e) {
		e ||= {}, super(e), this.setStyle(e.style);
	}
	setMap(e) {
		this.getCanvas(e);
		var t = this.getMap();
		if (this._listener &&= (T(this._listener), null), super.setMap(e), t) try {
			t.renderSync();
		} catch {}
		e && (this._listener = e.on("postcompose", this._draw.bind(this)));
	}
	getCanvas(e) {
		return yg(e);
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
		this._style = e || new ku({});
	}
	getStyle() {
		return this._style;
	}
	getStroke() {
		return this._style.getStroke() || this._style.setStroke(new Ou({
			color: "#000",
			width: 1.25
		})), this._style.getStroke();
	}
	getFill() {
		return this._style.getFill() || this._style.setFill(new Du({ color: "#fff" })), this._style.getFill();
	}
	getTextStroke() {
		var e = this._style.getText();
		return e ||= new Fu({}), e.getStroke() || e.setStroke(new Ou({
			color: "#fff",
			width: 3
		})), e.getStroke();
	}
	getTextFill() {
		var e = this._style.getText();
		return e ||= new Fu({}), e.getFill() || e.setFill(new Du({ color: "#fff" })), e.getFill();
	}
	getTextFont() {
		var e = this._style.getText();
		return e ||= new Fu({}), e.getFont() || e.setFont("12px sans-serif"), e.getFont();
	}
	_draw() {
		console.warn("[CanvasBase] draw function not implemented.");
	}
}, xg = class extends bg {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = "ol-graticule ol-unselectable ol-hidden", super({ element: t }), this.set("projection", e.projection || "EPSG:4326");
		var n = new Jt({ code: this.get("projection") }).getMetersPerUnit();
		for (this.fac = 1; n / this.fac > 10;) this.fac *= 10;
		this.fac = 1e4 / this.fac, this.set("maxResolution", e.maxResolution || Infinity), this.set("step", e.step || .1), this.set("stepCoord", e.stepCoord || 1), this.set("spacing", e.spacing || 40), this.set("intervals", e.intervals), this.set("precision", e.precision), this.set("margin", e.margin || 0), this.set("borderWidth", e.borderWidth || 5), this.set("stroke", e.stroke !== !1), this.formatCoord = e.formatCoord || function(e) {
			return e;
		}, e.style instanceof ku ? this.setStyle(e.style) : this.setStyle(new ku({
			stroke: new Ou({
				color: "#000",
				width: 1
			}),
			fill: new Du({ color: "#fff" }),
			text: new Fu({
				stroke: new Ou({
					color: "#fff",
					width: 2
				}),
				fill: new Du({ color: "#000" })
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
			], l = -Infinity, u = Infinity, d = -Infinity, f = Infinity, p = 0, m; m = c[p]; p++) c[p] = pr(m, s.getView().getProjection(), o), l = Math.max(l, c[p][0]), u = Math.min(u, c[p][0]), d = Math.max(d, c[p][1]), f = Math.min(f, c[p][1]);
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
			var T = rr(o).getExtent();
			T && (u < T[0] && (u = T[0]), f < T[1] && (f = T[1]), l > T[2] && (l = T[2] + g), d > T[3] && (d = T[3] + g));
			var E = this.getStyle().getStroke() && this.get("stroke"), D = this.getStyle().getText(), O = this.getStyle().getFill();
			t.save(), t.scale(r, r), t.beginPath(), t.rect(y, y, i - 2 * y, a - 2 * y), t.clip(), t.beginPath();
			for (var k = {
				top: [],
				left: [],
				bottom: [],
				right: []
			}, A = u, j, M, N, ee; A < l; A += g) for (N = pr([A, f], o, s.getView().getProjection()), N = s.getPixelFromCoordinate(N), E && t.moveTo(N[0], N[1]), M = N, j = f + w; j <= d; j += w) ee = pr([A, j], o, s.getView().getProjection()), ee = s.getPixelFromCoordinate(ee), E && t.lineTo(ee[0], ee[1]), M[1] > 0 && ee[1] < 0 && k.top.push([A, M]), M[1] > a && ee[1] < a && k.bottom.push([A, M]), M = ee;
			for (j = f; j < d; j += g) for (N = pr([u, j], o, s.getView().getProjection()), N = s.getPixelFromCoordinate(N), E && t.moveTo(N[0], N[1]), M = N, A = u + w; A <= l; A += w) ee = pr([A, j], o, s.getView().getProjection()), ee = s.getPixelFromCoordinate(ee), E && t.lineTo(ee[0], ee[1]), M[0] < 0 && ee[0] > 0 && k.left.push([j, M]), M[0] < i && ee[0] > i && k.right.push([j, M]), M = ee;
			if (E && (t.strokeStyle = this.getStyle().getStroke().getColor(), t.lineWidth = this.getStyle().getStroke().getWidth(), t.stroke()), D) {
				t.fillStyle = this.getStyle().getText().getFill().getColor(), t.strokeStyle = this.getStyle().getText().getStroke().getColor(), t.lineWidth = this.getStyle().getText().getStroke().getWidth(), t.font = this.getStyle().getText().getFont(), t.textAlign = "center", t.textBaseline = "hanging";
				var P, F, te = (O ? v : 0) + y + 2;
				for (p = 0; P = k.top[p]; p++) Math.round(P[0] / this.get("step")) % _ || (F = this.formatCoord(P[0], "top"), t.strokeText(F, P[1][0], te), t.fillText(F, P[1][0], te));
				for (t.textBaseline = "alphabetic", p = 0; P = k.bottom[p]; p++) Math.round(P[0] / this.get("step")) % _ || (F = this.formatCoord(P[0], "bottom"), t.strokeText(F, P[1][0], a - te), t.fillText(F, P[1][0], a - te));
				for (t.textBaseline = "middle", t.textAlign = "left", p = 0; P = k.left[p]; p++) Math.round(P[0] / this.get("step")) % _ || (F = this.formatCoord(P[0], "left"), t.strokeText(F, te, P[1][1]), t.fillText(F, te, P[1][1]));
				for (t.textAlign = "right", p = 0; P = k.right[p]; p++) Math.round(P[0] / this.get("step")) % _ || (F = this.formatCoord(P[0], "right"), t.strokeText(F, i - te, P[1][1]), t.fillText(F, i - te, P[1][1]));
			}
			if (O) {
				var ne = this.getStyle().getFill().getColor(), re, ie;
				for ((ie = this.getStyle().getStroke()) ? re = this.getStyle().getStroke().getColor() : (re = ne, ne = "#fff"), t.strokeStyle = re, t.lineWidth = ie ? ie.getWidth() : 1, p = 1; p < k.top.length; p++) t.beginPath(), t.rect(k.top[p - 1][1][0], y, k.top[p][1][0] - k.top[p - 1][1][0], v), t.fillStyle = Math.round(k.top[p][0] / g) % 2 ? re : ne, t.fill(), t.stroke();
				for (p = 1; p < k.bottom.length; p++) t.beginPath(), t.rect(k.bottom[p - 1][1][0], a - v - y, k.bottom[p][1][0] - k.bottom[p - 1][1][0], v), t.fillStyle = Math.round(k.bottom[p][0] / g) % 2 ? re : ne, t.fill(), t.stroke();
				for (p = 1; p < k.left.length; p++) t.beginPath(), t.rect(y, k.left[p - 1][1][1], v, k.left[p][1][1] - k.left[p - 1][1][1]), t.fillStyle = Math.round(k.left[p][0] / g) % 2 ? re : ne, t.fill(), t.stroke();
				for (p = 1; p < k.right.length; p++) t.beginPath(), t.rect(i - v - y, k.right[p - 1][1][1], v, k.right[p][1][1] - k.right[p - 1][1][1]), t.fillStyle = Math.round(k.right[p][0] / g) % 2 ? re : ne, t.fill(), t.stroke();
				t.beginPath(), t.fillStyle = re, t.rect(y, y, v, v), t.rect(y, a - v - y, v, v), t.rect(i - v - y, y, v, v), t.rect(i - v - y, a - v - y, v, v), t.fill();
			}
			t.restore();
		}
	}
}, Sg = class extends De {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = (e.className || "") + " ol-button ol-unselectable ol-control", super({
			element: t,
			target: e.target
		});
		var n = this, r = this.button_ = document.createElement(/ol-text-button/.test(e.className) ? "div" : "button");
		for (var i in e.id ? r.setAttribute("id", e.id) : r.setAttribute("id", "ol-button-" + O(this)), this.button_.className = e.classButton || "", r.type = "button", e.title && (r.title = e.title), e.name && (r.name = e.name), e.html instanceof Element ? r.appendChild(e.html) : r.innerHTML = e.html || "", r.addEventListener("click", function(t) {
			t && t.preventDefault && (t.preventDefault(), t.stopPropagation()), e.handleClick && e.handleClick.call(n, t);
		}), t.appendChild(r), !e.title && r.firstElementChild && (r.title = r.firstElementChild.title), e.title && this.set("title", e.title), e.title && this.set("title", e.title), e.name && this.set("name", e.name), e.attributes || {}) this.button_.setAttribute(i, e.attributes[i]);
	}
	setVisible(e) {
		e ? $.show(this.element) : $.hide(this.element);
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
		$.setHTML(this.button_, e);
	}
	getButtonElement() {
		return this.button_;
	}
}, Cg = class extends Sg {
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
//#region src/utils/colours.js
function wg(e) {
	let t = e.replace("#", "");
	return /^[0-9a-fA-F]{6}$/.test(t) ? {
		r: Number.parseInt(t.slice(0, 2), 16),
		g: Number.parseInt(t.slice(2, 4), 16),
		b: Number.parseInt(t.slice(4, 6), 16)
	} : null;
}
function Tg({ r: e, g: t, b: n }) {
	let r = [
		e,
		t,
		n
	].map((e) => {
		let t = e / 255;
		return t <= .04045 ? t / 12.92 : ((t + .055) / 1.055) ** 2.4;
	});
	return .2126 * r[0] + .7152 * r[1] + .0722 * r[2];
}
function Eg(e) {
	let t = Tg(e);
	return 1.05 / (t + .05) >= (t + .05) / .05 ? "#ffffff" : "#000000";
}
function Dg(e, t, n) {
	return {
		r: Math.round(e.r + (t - e.r) * n),
		g: Math.round(e.g + (t - e.g) * n),
		b: Math.round(e.b + (t - e.b) * n)
	};
}
function Og(e, t) {
	return `rgba(${e.r}, ${e.g}, ${e.b}, ${t})`;
}
//#endregion
//#region src/controls/grid.js
var kg = {
	fine: 32,
	default: 64,
	coarse: 128
}, Ag = 64, jg = {
	light: {
		line: {
			r: 255,
			g: 255,
			b: 255
		},
		label: "rgba(255, 255, 255, 1)",
		outline: "rgba(20, 20, 20, 1)"
	},
	dark: {
		line: {
			r: 20,
			g: 20,
			b: 20
		},
		label: "rgba(20, 20, 20, 1)",
		outline: "rgba(255, 255, 255, 1)"
	},
	"light-contrast": {
		line: {
			r: 0,
			g: 170,
			b: 200
		},
		label: "rgba(0, 170, 200, 1)",
		outline: "rgba(20, 20, 20, 1)"
	},
	"dark-contrast": {
		line: {
			r: 145,
			g: 55,
			b: 0
		},
		label: "rgba(145, 55, 0, 1)",
		outline: "rgba(255, 255, 255, 1)"
	}
};
function Mg({ map: e, projection: t, themeSelect: n, gridThemeSelect: r, gridOpacityInput: i, gridOpacityValue: a, gridSpacingSelect: o, gridLabelsVisibleInput: s, graticuleVisibleInput: c, screenSpaceGraticuleVisibleInput: l, onGraticulesChange: u }) {
	function d() {
		return kg[o.value] ?? kg.default;
	}
	function f() {
		return r.value === "default" ? n.value === "light" ? "light" : n.value === "high-contrast" ? "dark-contrast" : "dark" : r.value;
	}
	let p = new Fu({
		font: "12px Calibri,sans-serif",
		fill: new Du({ color: "rgba(0, 0, 0, 1)" }),
		stroke: new Ou({
			color: "rgba(255, 255, 255, 1)",
			width: 3
		})
	}), m = new ku({
		stroke: new Ou({
			color: "rgba(0, 0, 0, 0.5)",
			width: 1
		}),
		text: p
	});
	function h() {
		let t = jg[f()] ?? jg.dark, n = Number(i.value) / 100, r = m.getStroke(), o = p;
		r.setColor(Og(t.line, n)), o.getFill().setColor(t.label), o.getStroke().setColor(t.outline), a.textContent = `${i.value}%`, e.renderSync();
	}
	function g() {
		m.setText(s.checked ? p : null), e.renderSync();
	}
	function _(e) {
		return new xg({
			projection: e,
			margin: Ag,
			style: m,
			spacing: d(),
			/* v8 ignore start */
			formatCoord: (e, t) => (e = t === "left" || t === "right" ? -Math.floor(e) : Math.floor(e), e >= 1e6 && (e = e.toExponential(3), e = e.replace("+", "")), e)
		});
	}
	let v = Ag;
	function y(t) {
		let n = d();
		return new xg({
			projection: t.getCode(),
			spacing: n,
			margin: v,
			style: m,
			/* v8 ignore start */
			formatCoord(t, r) {
				let i = e.getView().calculateExtent(e.getSize()), a = e.getView().getResolution(), o = i[0] + a * v, s = i[3] - a * v, c;
				if (c = r === "left" || r === "right" ? -(t - s) : t - o, c = Math.floor(c / a / n), r === "left" || r === "right") {
					let e = "";
					do
						e += String.fromCharCode(65 + c % 26), c = Math.floor(c / 26);
					while (c > 0);
					return e.split("").reverse().join("");
				}
				return c;
			}
		});
	}
	let b = _(t), x = y(t);
	function S() {
		u(b, x);
	}
	let C = new Cg({
		html: "<i class=\"fas fa-ruler-combined\"></i>",
		className: "ol-graticule",
		title: "Toggle Graticule",
		onToggle(t) {
			C.element.classList.toggle("active", t), t ? (w.setActive(!1), w.element.classList.remove("active"), x.setMap(null), b.setMap(e)) : b.setMap(null);
		}
	});
	e.addControl(C);
	let w = new Cg({
		html: "<i class=\"fas fa-border-all\"></i>",
		className: "ol-screen-space-graticule",
		title: "Toggle Screen Space Graticule",
		onToggle(t) {
			w.element.classList.toggle("active", t), t ? (C.setActive(!1), C.element.classList.remove("active"), b.setMap(null), x.setMap(e)) : x.setMap(null);
		}
	});
	e.addControl(w);
	function T(t, n) {
		let r = !1, i = !1;
		n ? (r = C.getActive(), i = w.getActive()) : (C.setActive(!1), w.setActive(!1), C.element.classList.remove("active"), w.element.classList.remove("active")), b.setMap(null), x.setMap(null), b = _(t), x = y(t), r && b.setMap(e), i && x.setMap(e), S(), e.renderSync();
	}
	function E() {
		T(e.getView().getProjection(), !0);
	}
	function D(e, { preserveActive: t = !0 } = {}) {
		T(e, t);
	}
	function O() {
		C.element.classList.toggle("viewer-control-hidden", !c.checked), w.element.classList.toggle("viewer-control-hidden", !l.checked), c.checked || (C.setActive(!1), C.element.classList.remove("active"), b.setMap(null)), l.checked || (w.setActive(!1), w.element.classList.remove("active"), x.setMap(null));
	}
	function k(e) {
		let t = C.element.querySelector("button"), n = w.element.querySelector("button");
		for (let r of [t, n]) r !== null && (r.disabled = !e);
		e || (C.setActive(!1), w.setActive(!1), C.element.classList.remove("active"), w.element.classList.remove("active"), b.setMap(null), x.setMap(null));
	}
	return r.addEventListener("change", () => {
		h();
	}), i.addEventListener("input", () => {
		h();
	}), s.addEventListener("change", () => {
		g();
	}), o.addEventListener("change", () => {
		E();
	}), h(), g(), S(), {
		graticuleToggle: C,
		screenSpaceGraticuleToggle: w,
		setProjection: D,
		setViewerEnabled: k,
		updateAppearance: h,
		updateLabels: g,
		updateSpacing: E,
		updateVisibility: O
	};
}
//#endregion
//#region node_modules/ol/control/ScaleLine.js
var Ng = "units", Pg = [
	1,
	2,
	5
], Fg = 25.4 / .28, Ig = class extends De {
	constructor(e) {
		e ||= {};
		let t = document.createElement("div");
		t.style.pointerEvents = "none", super({
			element: t,
			render: e.render,
			target: e.target
		}), this.on, this.once, this.un;
		let n = e.className === void 0 ? e.bar ? "ol-scale-bar" : "ol-scale-line" : e.className;
		this.innerElement_ = document.createElement("div"), this.innerElement_.className = n + "-inner", this.element.className = n + " " + F, this.element.appendChild(this.innerElement_), this.viewState_ = null, this.minWidth_ = e.minWidth === void 0 ? 64 : e.minWidth, this.maxWidth_ = e.maxWidth, this.renderedVisible_ = !1, this.renderedWidth_ = void 0, this.renderedHTML_ = "", this.addChangeListener(Ng, this.handleUnitsChanged_), this.setUnits(e.units || "metric"), this.scaleBar_ = e.bar || !1, this.scaleBarSteps_ = e.steps || 4, this.scaleBarText_ = e.text || !1, this.dpi_ = e.dpi || void 0;
	}
	getUnits() {
		return this.get(Ng);
	}
	handleUnitsChanged_() {
		this.updateElement_();
	}
	setUnits(e) {
		this.set(Ng, e);
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
		let t = e.center, n = e.projection, r = this.getUnits(), i = r == "degrees" ? "degrees" : "m", a = ir(n, e.resolution, t, i), o = this.minWidth_ * (this.dpi_ || Fg) / Fg, s = this.maxWidth_ === void 0 ? void 0 : this.maxWidth_ * (this.dpi_ || Fg) / Fg, c = o * a, l = "";
		if (r == "degrees") {
			let e = qt.degrees;
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
			if (d = Pg[(u % 3 + 3) % 3] * e, f = Math.round(d / a), isNaN(f)) {
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
		let e = ir(this.viewState_.projection, this.viewState_.resolution, this.viewState_.center, "m"), t = this.dpi_ || Fg;
		return 1e3 / 25.4 * e * t;
	}
	render(e) {
		let t = e.frameState;
		this.viewState_ = t ? t.viewState : null, this.updateElement_();
	}
}, Lg = {
	small: 70,
	default: 100,
	large: 140
};
function Rg({ map: e, hasSlide: t, enabledInput: n, colourInput: r, opacityInput: i, opacityValue: a, sizeSelect: o, unitsSelect: s, onControlChange: c }) {
	function l() {
		let e = Lg[o.value] ?? Lg.default;
		return new Ig({
			units: s.value,
			minWidth: e
		});
	}
	let u = l();
	e.addControl(u), c(u);
	function d() {
		u.element.classList.toggle("viewer-control-hidden", !t() || !n.checked);
	}
	function f() {
		let e = r.value, t = u.element.querySelector(".ol-scale-line-inner");
		t !== null && (t.style.color = e, t.style.borderColor = e);
	}
	function p() {
		let e = Number(i.value) / 100, t = wg(r.value);
		if (t === null) return;
		let n = Eg(t) === "#ffffff" ? {
			r: 255,
			g: 255,
			b: 255
		} : {
			r: 17,
			g: 17,
			b: 17
		};
		u.element.style.backgroundColor = Og(n, e), a.textContent = `${i.value}%`;
	}
	function m() {
		e.removeControl(u), u = l(), e.addControl(u), c(u), d(), f(), p();
	}
	function h() {
		u.setUnits(s.value);
	}
	function g(e) {
		u.element.classList.toggle("viewer-control-hidden", !e || !n.checked);
	}
	return n.addEventListener("change", () => {
		d();
	}), r.addEventListener("input", () => {
		f(), p();
	}), i.addEventListener("input", () => {
		p();
	}), o.addEventListener("change", () => {
		m();
	}), s.addEventListener("change", () => {
		h();
	}), f(), p(), {
		setViewerEnabled: g,
		updateColour: f,
		updateOpacity: p,
		updateSize: m,
		updateUnits: h,
		updateVisibility: d
	};
}
//#endregion
//#region node_modules/ol/Overlay.js
var zg = {
	ELEMENT: "element",
	MAP: "map",
	OFFSET: "offset",
	POSITION: "position",
	POSITIONING: "positioning"
}, Bg = class extends A {
	constructor(e) {
		super(), this.on, this.once, this.un, this.options = e, this.id = e.id, this.insertFirst = e.insertFirst === void 0 || e.insertFirst, this.stopEvent = e.stopEvent === void 0 || e.stopEvent, this.element = document.createElement("div"), this.element.className = e.className === void 0 ? "ol-overlay-container " + P : e.className, this.element.style.position = "absolute", this.element.style.pointerEvents = "auto", this.autoPan = e.autoPan === !0 ? {} : e.autoPan || void 0, this.rendered = {
			transform_: "",
			visible: !0
		}, this.mapPostrenderListenerKey = null, this.addChangeListener(zg.ELEMENT, this.handleElementChanged), this.addChangeListener(zg.MAP, this.handleMapChanged), this.addChangeListener(zg.OFFSET, this.handleOffsetChanged), this.addChangeListener(zg.POSITION, this.handlePositionChanged), this.addChangeListener(zg.POSITIONING, this.handlePositioningChanged), e.element !== void 0 && this.setElement(e.element), this.setOffset(e.offset === void 0 ? [0, 0] : e.offset), this.setPositioning(e.positioning || "top-left"), e.position !== void 0 && this.setPosition(e.position);
	}
	getElement() {
		return this.get(zg.ELEMENT);
	}
	getId() {
		return this.id;
	}
	getMap() {
		return this.get(zg.MAP) || null;
	}
	getOffset() {
		return this.get(zg.OFFSET);
	}
	getPosition() {
		return this.get(zg.POSITION);
	}
	getPositioning() {
		return this.get(zg.POSITIONING);
	}
	handleElementChanged() {
		Se(this.element);
		let e = this.getElement();
		e && this.element.appendChild(e);
	}
	handleMapChanged() {
		this.mapPostrenderListenerKey &&= (this.element?.remove(), o(this.mapPostrenderListenerKey), null);
		let e = this.getMap();
		if (e) {
			this.mapPostrenderListenerKey = i(e, Ee.POSTRENDER, this.render, this), this.updatePixelPosition();
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
		this.set(zg.ELEMENT, e);
	}
	setMap(e) {
		this.set(zg.MAP, e);
	}
	setOffset(e) {
		this.set(zg.OFFSET, e);
	}
	setPosition(e) {
		this.set(zg.POSITION, e);
	}
	performAutoPan() {
		this.autoPan && this.panIntoView(this.autoPan);
	}
	panIntoView(e) {
		let t = this.getMap();
		if (!t || !t.getTargetElement() || !this.get(zg.POSITION)) return;
		let n = this.getRect(t.getTargetElement(), t.getSize()), r = this.getElement(), i = this.getRect(r, [ye(r), be(r)]);
		e ||= {};
		let a = e.margin === void 0 ? 20 : e.margin;
		if (!Ue(n, i)) {
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
		this.set(zg.POSITIONING, e);
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
}, Vg = .75, Hg = .1, Ug = class extends De {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		}), this.boundHandleRotationChanged_ = this.handleRotationChanged_.bind(this), this.collapsed_ = e.collapsed === void 0 || e.collapsed, this.collapsible_ = e.collapsible === void 0 || e.collapsible, this.collapsible_ || (this.collapsed_ = !1), this.rotateWithView_ = e.rotateWithView !== void 0 && e.rotateWithView, this.viewExtent_ = void 0;
		let t = e.className === void 0 ? "ol-overviewmap" : e.className, n = e.tipLabel === void 0 ? "Overview map" : e.tipLabel, r = e.collapseLabel === void 0 ? "‹" : e.collapseLabel;
		typeof r == "string" ? (this.collapseLabel_ = document.createElement("span"), this.collapseLabel_.textContent = r) : this.collapseLabel_ = r;
		let i = e.label === void 0 ? "›" : e.label;
		typeof i == "string" ? (this.label_ = document.createElement("span"), this.label_.textContent = i) : this.label_ = i;
		let a = this.collapsible_ && !this.collapsed_ ? this.collapseLabel_ : this.label_, o = document.createElement("button");
		o.setAttribute("type", "button"), o.title = n, o.appendChild(a), o.addEventListener(s.CLICK, this.handleClick_.bind(this), !1), this.ovmapDiv_ = document.createElement("div"), this.ovmapDiv_.className = "ol-overviewmap-map", this.view_ = e.view;
		let c = new Tf({
			view: e.view,
			controls: new N(),
			interactions: new N()
		});
		this.ovmap_ = c, e.layers && e.layers.forEach(function(e) {
			c.addLayer(e);
		});
		let l = document.createElement("div");
		l.className = "ol-overviewmap-box", l.style.boxSizing = "border-box", this.boxOverlay_ = new Bg({
			position: [0, 0],
			positioning: "center-center",
			element: l
		}), this.ovmap_.addOverlay(this.boxOverlay_);
		let u = t + " " + F + " " + ne + (this.collapsed_ && this.collapsible_ ? " " + re : "") + (this.collapsible_ ? "" : " ol-uncollapsible"), d = this.element;
		d.className = u, d.appendChild(this.ovmapDiv_), d.appendChild(o);
		let f = this.boxOverlay_, p = this.boxOverlay_.getElement(), m = (e) => ({
			clientX: e.clientX,
			clientY: e.clientY
		}), h = function(e) {
			let t = m(e), n = c.getEventCoordinate(t);
			f.setPosition(n);
		}, g = (e) => {
			let t = c.getEventCoordinateInternal(e), n = this.getMap();
			n.getView().setCenterInternal(t);
			let r = n.getOwnerDocument();
			r.removeEventListener("pointermove", h), r.removeEventListener("pointerup", g);
		};
		this.ovmapDiv_.addEventListener("pointerdown", (e) => {
			let t = this.getMap().getOwnerDocument();
			e.target === p && t.addEventListener("pointermove", h), t.addEventListener("pointerup", g);
		});
	}
	setMap(e) {
		let n = this.getMap();
		if (e !== n) {
			if (n) {
				let e = n.getView();
				e && this.unbindView_(e), this.ovmap_.setTarget(null);
			}
			if (super.setMap(e), e) {
				this.ovmap_.setTarget(this.ovmapDiv_), this.listenerKeys.push(i(e, t.PROPERTYCHANGE, this.handleMapPropertyChange_, this));
				let n = e.getView();
				n && this.bindView_(n), this.ovmap_.isRendered() || this.updateBoxAfterOvmapIsRendered_();
			}
		}
	}
	handleMapPropertyChange_(e) {
		if (e.key === Dd.VIEW) {
			let t = e.oldValue;
			t && this.unbindView_(t);
			let n = this.getMap().getView();
			this.bindView_(n);
		} else !this.ovmap_.isRendered() && (e.key === Dd.TARGET || e.key === Dd.SIZE) && this.ovmap_.updateSize();
	}
	bindView_(e) {
		if (!this.view_) {
			let t = new cs({ projection: e.getProjection() });
			this.ovmap_.setView(t);
		}
		e.addChangeListener(Yo.ROTATION, this.boundHandleRotationChanged_), this.handleRotationChanged_(), e.isDef() && (this.ovmap_.updateSize(), this.resetExtent_());
	}
	unbindView_(e) {
		e.removeChangeListener(Yo.ROTATION, this.boundHandleRotationChanged_);
	}
	handleRotationChanged_() {
		this.rotateWithView_ && this.ovmap_.getView().setRotation(this.getMap().getView().getRotation());
	}
	validateExtent_() {
		let e = this.getMap(), t = this.ovmap_;
		if (!e.isRendered() || !t.isRendered()) return;
		let n = e.getSize(), r = e.getView().calculateExtentInternal(n);
		if (this.viewExtent_ && Ze(r, this.viewExtent_)) return;
		this.viewExtent_ = r;
		let i = t.getSize(), a = t.getView().calculateExtentInternal(i), o = t.getPixelFromCoordinateInternal(pt(r)), s = t.getPixelFromCoordinateInternal(at(r)), c = Math.abs(o[0] - s[0]), l = Math.abs(o[1] - s[1]), u = i[0], d = i[1];
		c < u * Hg || l < d * Hg || c > u * Vg || l > d * Vg ? this.resetExtent_() : Ue(a, r) || this.recenter_();
	}
	resetExtent_() {
		let e = this.getMap(), t = this.ovmap_, n = e.getSize(), r = e.getView().calculateExtentInternal(n), i = t.getView();
		vt(r, 1 / (2 ** (Math.log(Vg / Hg) / Math.LN2 / 2) * Hg)), i.fitInternal(Xi(r));
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
		this.ovmapPostrenderKey_ ||= a(this.ovmap_, Ee.POSTRENDER, (e) => {
			delete this.ovmapPostrenderKey_, this.updateBox_();
		});
	}
	handleClick_(e) {
		e.preventDefault(), this.handleToggle_();
	}
	handleToggle_() {
		this.element.classList.toggle(re), this.collapsed_ ? xe(this.collapseLabel_, this.label_) : xe(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_;
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
		this.collapsible_ && this.collapsed_ !== e && this.handleToggle_();
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
}, Wg = {
	small: {
		width: 220,
		height: 180
	},
	default: {
		width: 300,
		height: 250
	},
	large: {
		width: 380,
		height: 320
	}
};
function Gg({ map: e, source: t, projection: n, extent: r, sizeSelect: i, visibleInput: a, hasSlide: o }) {
	let s = new bs();
	t !== null && s.setSource(t);
	function c() {
		return Wg[i.value] ?? Wg.default;
	}
	function l(e, t) {
		let n = c(), r = [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2], i = t[2] - t[0], a = t[3] - t[1], o = Math.max(i / n.width, a / n.height), s = new cs({
			projection: e,
			center: r,
			resolution: o,
			resolutions: [o],
			constrainOnlyCenter: !0
		});
		return s.on("change:center", () => {
			let e = s.getCenter();
			e !== void 0 && (e[0] !== r[0] || e[1] !== r[1]) && s.setCenter(r);
		}), s;
	}
	let u = document.createElement("span");
	u.className = "overview-toggle-icon", u.innerHTML = "<i class=\"fas fa-chevron-up\"></i>";
	let d = document.createElement("span");
	d.className = "overview-toggle-icon", d.innerHTML = "<i class=\"fas fa-chevron-down\"></i>";
	let f = new Ug({
		className: "ol-overviewmap ol-custom-overviewmap",
		layers: [s],
		collapsed: !1,
		collapsible: !0,
		collapseLabel: u,
		label: d,
		rotateWithView: !1,
		tipLabel: "Toggle overview map",
		view: l(n, r)
	});
	e.addControl(f);
	let p = f.getOverviewMap();
	function m() {
		requestAnimationFrame(() => {
			p.updateSize(), p.renderSync();
		});
	}
	function h() {
		let t = c();
		f.element.style.setProperty("--overview-map-width", `${t.width}px`), f.element.style.setProperty("--overview-map-height", `${t.height}px`), p.updateSize();
		let n = s.getSource();
		if (n !== null) {
			let t = e.getView().getProjection(), r = n.getTileGrid().getExtent();
			p.setView(l(t, r));
		}
		p.renderSync();
	}
	function g() {
		let e = o() && a.checked;
		f.element.classList.toggle("viewer-control-hidden", !e), e && m();
	}
	function _(e) {
		f.element.classList.toggle("viewer-control-hidden", !e || !a.checked), e && m();
	}
	function v(e) {
		s.setSource(e);
	}
	function y(e, t) {
		p.setView(l(e, t));
	}
	return i.addEventListener("change", () => {
		h();
	}), h(), {
		control: f,
		refresh: m,
		setSource: v,
		setView: y,
		setViewerEnabled: _,
		updateSize: h,
		updateVisibility: g
	};
}
//#endregion
//#region src/components/file-select.js
var Kg = 0;
function qg(e) {
	let t = document.createElement("div");
	t.className = "viewer-file-select";
	let n = document.createElement("button");
	n.type = "button", n.className = "viewer-file-select-button", n.setAttribute("aria-haspopup", "listbox"), n.setAttribute("aria-expanded", "false");
	let r = document.createElement("span");
	r.className = "viewer-file-select-label", r.textContent = e, n.append(r);
	let i = document.createElement("div");
	i.className = "viewer-file-select-menu", i.hidden = !0;
	let a = document.createElement("input");
	a.type = "text", a.className = "viewer-file-select-search", a.placeholder = "Search", a.autocomplete = "off", a.spellcheck = !1, a.setAttribute("aria-label", `Search ${e.toLowerCase()}`);
	let o = document.createElement("div");
	o.className = "viewer-file-select-options", o.id = `viewer-file-select-${Kg}`, o.setAttribute("role", "listbox"), Kg += 1, n.setAttribute("aria-controls", o.id), a.setAttribute("aria-controls", o.id), i.append(a, o), t.append(n, i);
	let s = [], c = "", l = e, u = [], d = -1;
	function f(e) {
		t.classList.toggle("open", e), i.hidden = !e, n.setAttribute("aria-expanded", e.toString());
	}
	function p() {
		if (c === "") {
			r.textContent = l, r.title = "";
			return;
		}
		let e = s.find((e) => e.path === c)?.name ?? c.split(/[\\/]/).pop() ?? c;
		r.textContent = e, r.title = c;
	}
	function m(e) {
		c = e.path, p(), _(), t.dispatchEvent(new CustomEvent("change", { detail: e.path }));
	}
	function h() {
		let e = a.value.trim().toLocaleLowerCase();
		if (u = s.filter((t) => t.name.toLocaleLowerCase().includes(e)), o.replaceChildren(), u.length === 0) {
			let e = document.createElement("div");
			e.className = "viewer-file-select-empty", e.textContent = "No matches", o.append(e);
			return;
		}
		u.forEach((e, t) => {
			let n = document.createElement("button");
			n.type = "button", n.className = "viewer-file-select-option", n.textContent = e.name, n.title = e.path, n.setAttribute("role", "option"), n.setAttribute("aria-selected", (e.path === c).toString()), e.path === c && n.classList.add("selected"), t === d && n.classList.add("active"), n.addEventListener("mousedown", (e) => {
				e.preventDefault();
			}), n.addEventListener("click", (t) => {
				t.stopPropagation(), m(e);
			}), o.append(n);
		}), o.querySelector(".viewer-file-select-option.active")?.scrollIntoView({ block: "nearest" });
	}
	function g() {
		if (!(n.disabled || s.length === 0)) {
			for (let e of document.querySelectorAll(".viewer-file-select.open")) e !== t && e.close?.();
			a.value = "", d = -1, h(), f(!0), requestAnimationFrame(() => {
				a.focus();
			});
		}
	}
	function _() {
		a.value = "", d = -1, f(!1);
	}
	return t.close = _, t.setFiles = (e, n) => {
		s = e, c = "", l = n, p(), _(), t.disabled = s.length === 0;
	}, Object.defineProperty(t, "value", {
		get() {
			return c;
		},
		set(e) {
			c = e, p(), _();
		}
	}), Object.defineProperty(t, "disabled", {
		get() {
			return n.disabled;
		},
		set(e) {
			n.disabled = e, t.classList.toggle("disabled", e), e && _();
		}
	}), n.addEventListener("click", () => {
		if (t.classList.contains("open")) {
			_();
			return;
		}
		g();
	}), n.addEventListener("keydown", (e) => {
		e.key === "ArrowDown" && (e.preventDefault(), g());
	}), a.addEventListener("input", () => {
		d = -1, h();
	}), a.addEventListener("keydown", (e) => {
		if (e.key === "Escape") {
			e.preventDefault(), _(), n.focus();
			return;
		}
		if (u.length !== 0) {
			if (e.key === "ArrowDown") {
				e.preventDefault(), d = Math.min(d + 1, u.length - 1), h();
				return;
			}
			if (e.key === "ArrowUp") {
				e.preventDefault(), d = d <= 0 ? u.length - 1 : d - 1, h();
				return;
			}
			if (e.key === "Enter") {
				e.preventDefault();
				let t = u[d >= 0 ? d : 0];
				t !== void 0 && m(t);
			}
		}
	}), document.addEventListener("click", (e) => {
		t.contains(e.target) || _();
	}), t.disabled = !0, t;
}
//#endregion
//#region src/utils/paths.js
function Jg(e) {
	let t = e.split(/[\\/]/).pop() ?? e, n = t.lastIndexOf(".");
	return n <= 0 ? t : t.slice(0, n);
}
//#endregion
//#region src/panels/files.js
function Yg({ panel: e, toggle: t, container: n, configuredSlides: r, configuredOverlays: i, getCurrentSlidePath: a, hasSlide: o, hasOverlays: s, onSlideSelected: c, onOverlaySelected: l, onClearSlide: u, onClearOverlays: d, onOpen: f }) {
	let p = document.createElement("div");
	p.className = "viewer-file-selectors";
	let m = qg("Select slide"), h = qg("Load overlay");
	p.append(m, h);
	let g = document.createElement("div");
	g.className = "viewer-file-actions";
	function _(e) {
		let t = document.createElement("button");
		return t.type = "button", t.textContent = e, t;
	}
	let v = _("Clear Slide"), y = _("Clear Overlays");
	v.disabled = !0, y.disabled = !0, g.append(v, y), n.append(p, g);
	function b(n) {
		if (n && f(), e.classList.toggle("hidden", !n), t.classList.toggle("active", n), t.innerHTML = n ? "<i class=\"fas fa-folder-open\"></i>" : "<i class=\"fas fa-folder\"></i>", !n) for (let t of e.querySelectorAll(".viewer-file-select.open")) t.close?.();
	}
	function x(e, t, n) {
		e.setFiles(t, n);
	}
	function S(e) {
		let t = Jg(e);
		return i.files.filter((e) => (e.name.split(/[\\/]/).pop() ?? e.name).includes(t));
	}
	function C() {
		if (i.directory === null) {
			x(h, [], "No overlay directory configured");
			return;
		}
		let e = a();
		if (e === null) {
			x(h, [], "Select slide first");
			return;
		}
		let t = S(e);
		x(h, t, t.length === 0 ? "No matching overlays" : "Load overlay");
	}
	function w(e) {
		m.value = e;
	}
	function T() {
		let e = o(), t = s(), n = a(), i = n !== null && S(n).length > 0;
		v.disabled = !e, y.disabled = !e || !t, m.disabled = r.files.length === 0, h.disabled = !e || !i;
	}
	function E(e) {
		if (!e) {
			T();
			return;
		}
		m.disabled = !0, h.disabled = !0, v.disabled = !0, y.disabled = !0;
	}
	return m.addEventListener("change", async (e) => {
		let t = e.detail ?? m.value;
		if (t !== "") {
			E(!0);
			try {
				await c(t), h.value = "";
			} catch (e) {
				console.error(e);
			} finally {
				E(!1);
			}
		}
	}), h.addEventListener("change", async (e) => {
		let t = e.detail ?? h.value;
		if (t !== "") {
			E(!0);
			try {
				await l(t), h.value = "";
			} catch (e) {
				console.error(e);
			} finally {
				E(!1);
			}
		}
	}), v.addEventListener("click", async () => {
		E(!0);
		try {
			await u(), m.value = "", h.value = "", C();
		} catch (e) {
			console.error(e);
		} finally {
			E(!1);
		}
	}), y.addEventListener("click", async () => {
		E(!0);
		try {
			await d(), h.value = "";
		} catch (e) {
			console.error(e);
		} finally {
			E(!1);
		}
	}), t.addEventListener("click", () => {
		b(e.classList.contains("hidden"));
	}), x(m, r.files, r.directory === null ? "No slide directory configured" : "Select slide"), C(), T(), {
		setOpen: b,
		setSlide: w,
		updateActionState: T,
		updateOverlaySelect: C
	};
}
//#endregion
//#region src/panels/layers.js
function Xg({ panel: e, toggle: t, list: n, getSlideLayer: r, getCurrentSlidePath: i, getOverlayLayers: a, onRemoveLayer: o, onOpen: s }) {
	function c(n) {
		n && s(), e.classList.toggle("hidden", !n), t.classList.toggle("active", n);
	}
	function l() {
		let e = [], t = r();
		t.getSource() !== null && e.push({
			id: "slide",
			name: Jg(i() ?? "slide"),
			layer: t
		});
		let n = Object.entries(a()).map(([e, t]) => ({
			id: e,
			name: e,
			layer: t
		})).sort((e, t) => (e.layer.getZIndex() ?? 0) - (t.layer.getZIndex() ?? 0));
		return e.push(...n), e;
	}
	function u(e, t) {
		/* v8 ignore if */
		if (e === "slide") return;
		let n = l().filter((e) => e.id !== "slide"), r = n.findIndex((t) => t.id === e);
		/* v8 ignore if */
		if (r === -1) return;
		let i = t === "up" ? r - 1 : r + 1;
		/* v8 ignore if */
		if (i < 0 || i >= n.length) return;
		let a = n[r].layer, o = n[i].layer, s = a.getZIndex() ?? 0, c = o.getZIndex() ?? 0;
		a.setZIndex(c), o.setZIndex(s), d();
	}
	function d() {
		n.replaceChildren();
		let e = l(), t = e.filter((e) => e.id !== "slide");
		if (e.length === 0) {
			let e = document.createElement("div");
			e.className = "layer-editor-empty", e.textContent = "No layers loaded", n.appendChild(e);
			return;
		}
		e.forEach(({ id: e, name: r, layer: i }) => {
			let a = document.createElement("div");
			a.className = "layer-editor-item";
			let s = document.createElement("div");
			s.className = "layer-editor-item-header";
			let c = document.createElement("input");
			c.className = "layer-editor-visibility", c.type = "checkbox", c.checked = i.getVisible(), c.title = `Toggle ${r}`, c.addEventListener("change", () => {
				i.setVisible(c.checked);
			});
			let l = document.createElement("span");
			if (l.className = "layer-editor-name", l.textContent = r, l.title = r, s.append(c, l), e !== "slide") {
				let n = t.findIndex((t) => t.id === e), i = document.createElement("div");
				i.className = "layer-editor-order";
				let a = document.createElement("button");
				a.type = "button", a.title = "Move layer up", a.innerHTML = "<i class=\"fas fa-chevron-up\"></i>", a.disabled = n === 0, a.addEventListener("click", () => {
					u(e, "up");
				});
				let c = document.createElement("button");
				c.type = "button", c.title = "Move layer down", c.innerHTML = "<i class=\"fas fa-chevron-down\"></i>", c.disabled = n === t.length - 1, c.addEventListener("click", () => {
					u(e, "down");
				});
				let l = document.createElement("button");
				l.type = "button", l.title = `Remove ${r}`, l.innerHTML = "<i class=\"fas fa-times\"></i>", l.addEventListener("click", () => {
					o(e).catch((e) => {
						console.error(e);
					});
				}), i.append(a, c, l), s.appendChild(i);
			}
			let d = document.createElement("div");
			d.className = "layer-editor-opacity";
			let f = document.createElement("input");
			f.className = "layer-editor-slider", f.type = "range", f.min = "0", f.max = "1", f.step = "0.05", f.value = i.getOpacity().toString();
			let p = document.createElement("span");
			p.className = "layer-editor-value", p.textContent = `${Math.round(i.getOpacity() * 100)}%`, f.addEventListener("input", () => {
				let e = Number(f.value);
				i.setOpacity(e), p.textContent = `${Math.round(e * 100)}%`;
			}), d.append(f, p), a.append(s, d), n.appendChild(a);
		});
	}
	return t.addEventListener("click", () => {
		c(e.classList.contains("hidden"));
	}), {
		render: d,
		setOpen: c
	};
}
//#endregion
//#region src/panels/annotations.js
function Zg(e) {
	return typeof e == "string" ? e : `#${e.slice(0, 3).map((e) => Math.round(e * 255).toString(16).padStart(2, "0")).join("")}`;
}
function Qg({ panel: e, toggle: t, list: n, selectAllButton: r, deselectAllButton: i, importButton: a, importInput: o, exportButton: s, colourBySelect: c, paletteField: l, paletteSelect: u, colourMapField: d, colourMapSelect: f, secondaryTypeField: p, secondaryTypeSelect: m, propertyField: h, propertySelect: g, propertyLegend: _, propertyLegendCaption: v, propertyMin: y, propertyMax: b, linkOpacityInput: x, getAnnotationGroups: S, getAnnotationTypes: C, getAnnotationColour: w, getDisplayMode: T, getPalette: E, getColourMap: D, getAnnotationProperties: O, getAnnotationProperty: k, getSecondaryType: A, getPropertyRange: j, isAnnotationTypeVisible: M, getAnnotationOpacity: N, getOpacityLinked: ee, onColourChange: P, onVisibilityChange: F, onOpacityChange: te, onOpacityLinkChange: ne, onPaletteChange: re, onColourMapChange: ie, onDisplayModeChange: ae, onPropertyChange: oe, onSecondaryTypeChange: se, onSetAllVisibility: ce, onImport: le, onExport: ue, onOpen: de }) {
	function fe(n) {
		n && (de(), I()), e.classList.toggle("hidden", !n), t.classList.toggle("active", n);
	}
	function pe(e) {
		e().then(() => {
			I();
		}).catch((e) => {
			console.error(e), I();
		});
	}
	function me(e) {
		let t = [...g.options].map((e) => e.value);
		if (t.length === e.length && t.every((t, n) => t === e[n])) return;
		let n = e.map((e) => {
			let t = document.createElement("option");
			return t.value = e, t.textContent = e, t;
		});
		g.replaceChildren(...n);
	}
	function he(e) {
		let t = e.flatMap(({ layerName: e, annotationTypes: t }) => t.map((t) => ({
			layerName: e,
			annotationType: t
		}))), n = t.map(({ layerName: e, annotationType: t }) => JSON.stringify([e, t])), r = [...m.options].map((e) => e.value);
		if (r.length === n.length && r.every((e, t) => e === n[t])) return;
		let i = t.map(({ layerName: e, annotationType: t }) => {
			let n = document.createElement("option");
			return n.value = JSON.stringify([e, t]), n.textContent = `${e} · ${String(t)}`, n;
		});
		m.replaceChildren(...i);
	}
	function I() {
		n.replaceChildren();
		let e = S(), t = T(), o = E(), ne = D(), re = O(), ie = C(), ae = A(), oe = e.length > 0;
		x.checked = ee(), x.disabled = !oe, x.closest(".annotations-panel-link-opacity")?.classList.toggle("disabled", !oe), l.hidden = t !== "type", d.hidden = t === "type", u.value = o, f.value = ne, u.disabled = !oe, f.disabled = !oe || re.length === 0, l.classList.toggle("disabled", !oe), d.classList.toggle("disabled", !oe || re.length === 0), c.value = t, c.disabled = !oe, c.closest(".annotations-panel-display-field")?.classList.toggle("disabled", !oe);
		let se = c.querySelector("option[value=\"property\"]"), ce = c.querySelector("option[value=\"secondary\"]");
		se.disabled = !oe || re.length === 0, ce.disabled = !oe || ie.length === 0 || re.length === 0, p.hidden = t !== "secondary", h.hidden = t === "type", me(re), he(e);
		let le = k();
		if (le !== null && re.includes(le) && (g.value = le), ae !== null) {
			let e = JSON.stringify([ae.layerName, ae.annotationType]);
			[...m.options].some((t) => t.value === e) && (m.value = e);
		}
		m.disabled = ie.length === 0;
		let ue = j(), de = (t === "property" || t === "secondary") && ue !== null;
		if (_.hidden = !de, _.dataset.colourMap = ne, de) {
			let [e, n] = ue;
			v.textContent = t === "secondary" && ae !== null ? `${ae.layerName} · ${String(ae.annotationType)} · ${le} values · low → high` : `${le} values · low → high`, y.textContent = Number(e.toPrecision(4)).toString(), b.textContent = Number(n.toPrecision(4)).toString();
		}
		if (g.disabled = re.length === 0, r.disabled = !oe || t === "secondary", i.disabled = !oe || t === "secondary", a.disabled = !oe || t !== "type", s.disabled = !oe || t !== "type", e.length === 0) {
			let e = document.createElement("div");
			e.className = "annotations-panel-empty", e.textContent = "No annotations loaded", n.appendChild(e);
			return;
		}
		for (let { layerName: r, annotationTypes: i } of e) {
			let e = document.createElement("section");
			e.className = "annotations-panel-group";
			let a = document.createElement("div");
			a.className = "annotations-panel-group-title", a.textContent = r, a.title = r, e.appendChild(a);
			for (let a of i) {
				let i = String(a), o = t === "secondary" && ae !== null && r === ae.layerName && Object.is(a, ae.annotationType), s = document.createElement("div");
				s.className = "annotations-panel-item";
				let c = document.createElement("div");
				c.className = "annotations-panel-item-header";
				let l = document.createElement("input");
				l.type = "checkbox", l.className = "annotations-panel-visibility", l.checked = t === "secondary" ? o : M(r, a), l.disabled = t === "secondary", l.title = `Toggle ${i}`, l.addEventListener("change", () => {
					pe(() => F(r, a, l.checked));
				});
				let u = document.createElement("input");
				u.type = "color", u.className = "annotations-panel-colour", u.value = Zg(w(r, a)), u.title = `Change ${i} colour`, u.disabled = t === "property" || t === "secondary", u.addEventListener("change", () => {
					pe(() => P(r, a, u.value));
				});
				let d = document.createElement("span");
				d.className = "annotations-panel-name", d.textContent = i, d.title = i, c.append(l, u, d);
				let f = document.createElement("div");
				f.className = "annotations-panel-opacity";
				let p = document.createElement("span");
				p.className = "annotations-panel-opacity-label", p.textContent = "Fill opacity";
				let m = document.createElement("input");
				m.type = "range", m.className = "annotations-panel-slider", m.min = "0", m.max = "1", m.step = "0.05", m.value = N(r, a).toString(), m.disabled = t === "secondary" && !o;
				let h = document.createElement("span");
				h.className = "annotations-panel-value", h.textContent = `${Math.round(Number(m.value) * 100)}%`, m.addEventListener("input", () => {
					let e = `${Math.round(Number(m.value) * 100)}%`;
					if (h.textContent = e, x.checked) for (let t of n.querySelectorAll(".annotations-panel-slider")) t.value = m.value, t.closest(".annotations-panel-opacity")?.querySelector(".annotations-panel-value")?.replaceChildren(e);
				}), m.addEventListener("change", () => {
					pe(() => te(r, a, Number(m.value)));
				}), f.append(p, m, h), s.append(c, f), e.appendChild(s);
			}
			n.appendChild(e);
		}
	}
	return x.addEventListener("change", () => {
		pe(() => ne(x.checked));
	}), a.addEventListener("click", () => {
		o.click();
	}), o.addEventListener("change", () => {
		let e = o.files?.[0];
		o.value = "", e !== void 0 && pe(() => le(e));
	}), u.addEventListener("change", () => {
		pe(() => re(u.value));
	}), f.addEventListener("change", () => {
		pe(() => ie(f.value));
	}), c.addEventListener("change", () => {
		pe(() => ae(c.value));
	}), g.addEventListener("change", () => {
		pe(() => oe(g.value));
	}), m.addEventListener("change", () => {
		let [e, t] = JSON.parse(m.value);
		pe(() => se(e, t));
	}), r.addEventListener("click", () => {
		pe(() => ce(!0));
	}), i.addEventListener("click", () => {
		pe(() => ce(!1));
	}), s.addEventListener("click", () => {
		ue();
	}), t.addEventListener("click", () => {
		fe(e.classList.contains("hidden"));
	}), {
		render: I,
		setOpen: fe
	};
}
//#endregion
//#region src/panels/settings.js
var $g = {
	dark: "#111111",
	light: "#f2f2f2",
	"high-contrast": "#000000"
}, e_ = "tiatoolbox-openlayers-settings";
function t_({ viewerApp: e, panel: t, toggle: n, closeButton: r, tabs: i, tabPanels: a, resetDefaultsButton: o, themeSelect: s, controlOpacityInput: c, controlOpacityValue: l, annotationInspectionEnabledInput: u, multipleAnnotationSelectionInput: d, zoomVisibleInput: f, zoomLevelVisibleInput: p, rotationVisibleInput: m, graticuleVisibleInput: h, screenSpaceGraticuleVisibleInput: g, resetViewVisibleInput: _, fullscreenVisibleInput: v, mousePositionVisibleInput: y, overviewMapVisibleInput: b, overviewMapSizeSelect: x, mouseWheelZoomSensitivitySelect: S, zoomButtonStepSelect: C, gridThemeSelect: w, gridOpacityInput: T, gridSpacingSelect: E, gridLabelsVisibleInput: D, scaleBarEnabledInput: O, scaleBarColourInput: k, scaleBarOpacityInput: A, scaleBarSizeSelect: j, scaleBarUnitsSelect: M, onAnnotationInspectionChange: N, onMultipleAnnotationSelectionChange: ee, onThemeChange: P, onControlVisibilityChange: F, onReset: te }) {
	let ne = !1;
	function re(e) {
		t.classList.toggle("hidden", !e), n.classList.toggle("active", e);
	}
	function ie() {
		let t = wg($g[s.value] ?? $g.dark);
		/* v8 ignore if */
		if (t === null) return;
		let n = Number(c.value) / 100, r = Eg(t), i = r === "#ffffff" ? {
			r: 255,
			g: 255,
			b: 255
		} : {
			r: 0,
			g: 0,
			b: 0
		}, a = r === "#ffffff" ? {
			r: 0,
			g: 0,
			b: 0
		} : {
			r: 255,
			g: 255,
			b: 255
		}, o = r === "#ffffff" ? 255 : 0, u = Dg(t, o, .08), d = Dg(t, o, .16), f = Dg(t, o, .28), p = Dg(t, o, .4), m = Dg(t, o, .58);
		e.style.setProperty("--viewer-control-background", Og(t, n)), e.style.setProperty("--viewer-control-surface-background", Og(u, n)), e.style.setProperty("--viewer-control-hover-background", Og(d, n)), e.style.setProperty("--viewer-control-pressed-background", Og(f, n)), e.style.setProperty("--viewer-control-foreground", r), e.style.setProperty("--viewer-control-hover-foreground", Eg(d)), e.style.setProperty("--viewer-control-pressed-foreground", Eg(f)), e.style.setProperty("--viewer-control-muted-foreground", Og(i, .7)), e.style.setProperty("--viewer-control-subtle-foreground", Og(i, .55)), e.style.setProperty("--viewer-control-border", Og(p, Math.max(n, .7))), e.style.setProperty("--viewer-control-focus-border", Og(m, Math.max(n, .9))), e.style.setProperty("--viewer-control-foreground-shadow", Og(a, .75)), l.textContent = `${c.value}%`;
	}
	function ae() {
		let e = {
			theme: s.value,
			interfaceOpacity: c.value,
			controls: {
				annotationInspection: u.checked,
				multipleAnnotationSelection: d.checked,
				zoom: f.checked,
				zoomLevel: p.checked,
				rotation: m.checked,
				graticule: h.checked,
				screenSpaceGraticule: g.checked,
				resetView: _.checked,
				fullscreen: v.checked,
				mousePosition: y.checked,
				overviewMap: b.checked
			},
			navigation: {
				mouseWheelZoomSensitivity: S.value,
				zoomButtonStep: C.value
			},
			overviewMap: { size: x.value },
			grid: {
				theme: w.value,
				opacity: T.value,
				spacing: E.value,
				labels: D.checked
			},
			scaleBar: {
				enabled: O.checked,
				colour: k.value,
				opacity: A.value,
				size: j.value,
				units: M.value
			}
		};
		try {
			window.localStorage.setItem(e_, JSON.stringify(e));
		} catch {}
	}
	function oe() {
		let e;
		try {
			let t = window.localStorage.getItem(e_);
			if (t === null) return;
			e = JSON.parse(t);
		} catch {
			return;
		}
		if (typeof e != "object" || !e) return;
		[
			"dark",
			"light",
			"high-contrast"
		].includes(e.theme) && (s.value = e.theme), [
			"40",
			"45",
			"50",
			"55",
			"60",
			"65",
			"70",
			"75",
			"80",
			"85",
			"90",
			"95",
			"100"
		].includes(e.interfaceOpacity) && (c.value = e.interfaceOpacity);
		let t = e.controls;
		typeof t == "object" && t && (typeof t.annotationInspection == "boolean" && (u.checked = t.annotationInspection), typeof t.multipleAnnotationSelection == "boolean" && (d.checked = t.multipleAnnotationSelection), typeof t.zoom == "boolean" && (f.checked = t.zoom), typeof t.zoomLevel == "boolean" && (p.checked = t.zoomLevel), typeof t.rotation == "boolean" && (m.checked = t.rotation), typeof t.graticule == "boolean" && (h.checked = t.graticule), typeof t.screenSpaceGraticule == "boolean" && (g.checked = t.screenSpaceGraticule), typeof t.resetView == "boolean" && (_.checked = t.resetView), typeof t.fullscreen == "boolean" && (v.checked = t.fullscreen), typeof t.mousePosition == "boolean" && (y.checked = t.mousePosition), typeof t.overviewMap == "boolean" && (b.checked = t.overviewMap));
		let n = e.navigation;
		typeof n == "object" && n && ([
			"low",
			"default",
			"high"
		].includes(n.mouseWheelZoomSensitivity) && (S.value = n.mouseWheelZoomSensitivity), [
			"0.1",
			"0.5",
			"1",
			"2"
		].includes(n.zoomButtonStep) && (C.value = n.zoomButtonStep));
		let r = e.overviewMap;
		typeof r == "object" && r && [
			"small",
			"default",
			"large"
		].includes(r.size) && (x.value = r.size);
		let i = e.grid;
		if (typeof i == "object" && i) {
			[
				"default",
				"light",
				"dark",
				"light-contrast",
				"dark-contrast"
			].includes(i.theme) && (w.value = i.theme);
			let e = Number(i.opacity);
			Number.isFinite(e) && e >= 0 && e <= 100 && (T.value = e.toString()), [
				"fine",
				"default",
				"coarse"
			].includes(i.spacing) && (E.value = i.spacing), typeof i.labels == "boolean" && (D.checked = i.labels);
		}
		let a = e.scaleBar;
		if (typeof a == "object" && a) {
			typeof a.enabled == "boolean" && (O.checked = a.enabled), typeof a.colour == "string" && /^#[0-9a-fA-F]{6}$/.test(a.colour) && (k.value = a.colour);
			let e = Number(a.opacity);
			Number.isFinite(e) && e >= 0 && e <= 100 && (A.value = e.toString()), [
				"small",
				"default",
				"large"
			].includes(a.size) && (j.value = a.size), ["metric", "imperial"].includes(a.units) && (M.value = a.units);
		}
	}
	function se() {
		s.value = "dark", x.value = "default", S.value = "default", C.value = "1", w.value = "default", T.value = "50", E.value = "default", D.checked = !0, c.value = "100", d.checked = !1;
		for (let e of [
			u,
			f,
			p,
			m,
			h,
			g,
			_,
			v,
			y,
			b
		]) e.checked = !0;
		O.checked = !0, k.value = "#ffffff", A.value = "100", j.value = "default", M.value = "metric";
	}
	function ce() {
		try {
			window.localStorage.removeItem(e_);
		} catch {}
	}
	function le() {
		if (!ne) {
			ne = !0, n.addEventListener("click", () => {
				re(t.classList.contains("hidden"));
			}), r.addEventListener("click", () => {
				re(!1);
			});
			for (let e of i) e.addEventListener("click", () => {
				let t = e.dataset.settingsTab;
				for (let t of i) t.classList.toggle("active", t === e);
				for (let e of a) e.classList.toggle("hidden", e.dataset.settingsPanel !== t);
			});
			s.addEventListener("change", () => {
				ie(), P(), ae();
			}), c.addEventListener("input", () => {
				ie(), ae();
			}), u.addEventListener("change", () => {
				N(), ae();
			}), d.addEventListener("change", () => {
				ee(), ae();
			});
			for (let e of [
				f,
				p,
				m,
				h,
				g,
				_,
				v,
				y,
				b
			]) e.addEventListener("change", () => {
				F(), ae();
			});
			for (let e of [
				w,
				E,
				D,
				x,
				S,
				C,
				O,
				j,
				M
			]) e.addEventListener("change", () => {
				ae();
			});
			for (let e of [
				T,
				k,
				A
			]) e.addEventListener("input", () => {
				ae();
			});
			o.addEventListener("click", () => {
				te();
			});
		}
	}
	return {
		bindEvents: le,
		clearSavedSettings: ce,
		load: oe,
		resetValues: se,
		setOpen: re,
		updateAppearance: ie
	};
}
//#endregion
//#region src/utils/annotation-colours.js
async function n_(e, t, n, r = {}) {
	let i = [...new Set(t)].filter((t) => !e.has(t));
	if (i.length === 0) return e;
	let a = i.filter((e) => !Object.hasOwn(r, String(e))), o = a.length === 0 ? /* @__PURE__ */ new Map() : await n(a);
	for (let t of i) {
		let n = r[String(t)];
		e.set(t, n === void 0 ? o.get(t) : [
			n[0] / 255,
			n[1] / 255,
			n[2] / 255,
			1
		]);
	}
	return e;
}
function r_(e) {
	return typeof e == "object" && !!e && !Array.isArray(e);
}
function i_(e, t) {
	if (!r_(e)) throw Error(`${t} must be an object.`);
	for (let [n, r] of Object.entries(e)) if (!Array.isArray(r) || r.length !== 3 && r.length !== 4 || !r.every((e) => typeof e == "number" && Number.isFinite(e) && e >= 0 && e <= 255)) throw Error(`Invalid annotation colour for ${t}.${n}.`);
}
function a_(e) {
	if (!r_(e)) throw Error("Annotation colour config must be an object.");
	let t = Object.hasOwn(e, "color_dict"), n = Object.hasOwn(e, "layer_color_dicts");
	if (!t && !n) throw Error("Annotation colour config must contain color_dict or layer_color_dicts.");
	let r = t ? e.color_dict : {};
	i_(r, "color_dict");
	let i = {};
	if (n) {
		if (!r_(e.layer_color_dicts)) throw Error("layer_color_dicts must be an object.");
		for (let [t, n] of Object.entries(e.layer_color_dicts)) i_(n, `layer_color_dicts.${t}`), i[t] = n;
	}
	return {
		colorDict: r,
		layerColorDicts: i
	};
}
function o_(e, t, n, r = {}) {
	let i = new Map(e);
	for (let e of new Set(t)) {
		let t = String(e), a = Object.hasOwn(r, t) ? r[t] : n[t];
		if (a === void 0) continue;
		let o = i.get(e)?.[3] ?? 1;
		i.set(e, [
			a[0] / 255,
			a[1] / 255,
			a[2] / 255,
			o
		]);
	}
	return i;
}
function s_(e, t) {
	let n = {};
	for (let r of new Set(t)) {
		let t = e.get(r);
		t !== void 0 && (n[String(r)] = [
			Math.round(t[0] * 255),
			Math.round(t[1] * 255),
			Math.round(t[2] * 255),
			255
		]);
	}
	return { color_dict: n };
}
//#endregion
//#region src/utils/annotation-filters.js
function c_(e, t) {
	let n = e.filter((e) => t.get(e) ?? !0);
	if (n.length === e.length) return null;
	if (n.length === 0) {
		let t = JSON.stringify(e[0]);
		return `(props["type"]==${t}) & (props["type"]!=${t})`;
	}
	return n.map((e) => `(props["type"]==${JSON.stringify(e)})`).join(" | ");
}
//#endregion
//#region src/utils/numbers.js
function l_(e) {
	if (e.length === 0) return null;
	let t = Infinity, n = -Infinity;
	for (let r of e) {
		if (typeof r != "number" || !Number.isFinite(r)) return null;
		t = Math.min(t, r), n = Math.max(n, r);
	}
	return [t, n];
}
//#endregion
//#region src/main.js
function u_(e, t, n) {
	return new cp({
		url: `/tileserver/layer/slide/${e}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${n}`,
		size: t.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
}
var d_ = document.getElementById("map"), f_ = document.querySelector(".viewer-app"), p_ = document.getElementById("viewer-panel"), m_ = document.getElementById("viewer-panel-toggle"), h_ = document.getElementById("viewer-files"), g_ = document.getElementById("layer-editor"), __ = document.getElementById("layer-editor-toggle"), v_ = document.getElementById("layer-editor-list"), y_ = document.getElementById("annotations-panel"), b_ = document.getElementById("annotations-toggle"), x_ = document.getElementById("annotations-panel-list"), S_ = document.getElementById("annotations-colour-by"), C_ = document.getElementById("annotations-palette-field"), w_ = document.getElementById("annotations-palette"), T_ = document.getElementById("annotations-colour-map-field"), E_ = document.getElementById("annotations-colour-map"), D_ = document.getElementById("annotations-secondary-type-field"), O_ = document.getElementById("annotations-secondary-type"), k_ = document.getElementById("annotations-property-field"), A_ = document.getElementById("annotations-property"), j_ = document.getElementById("annotations-property-legend"), M_ = document.getElementById("annotations-property-legend-caption"), N_ = document.getElementById("annotations-property-min"), P_ = document.getElementById("annotations-property-max"), F_ = document.getElementById("annotations-link-opacity"), I_ = document.getElementById("annotation-inspectors"), L_ = document.getElementById("annotations-select-all"), R_ = document.getElementById("annotations-deselect-all"), z_ = document.getElementById("annotations-import-colours"), B_ = document.getElementById("annotations-import-colours-file"), V_ = document.getElementById("annotations-export-colours"), H_ = document.getElementById("settings-panel"), U_ = document.getElementById("settings-toggle"), W_ = document.getElementById("settings-close"), G_ = document.querySelectorAll(".settings-tab"), K_ = document.querySelectorAll(".settings-tab-panel"), q_ = document.getElementById("settings-annotation-inspection"), J_ = document.getElementById("settings-multiple-annotation-selection"), Y_ = document.getElementById("settings-zoom-visible"), X_ = document.getElementById("settings-zoom-level-visible"), Z_ = document.getElementById("settings-rotation-visible"), Q_ = document.getElementById("settings-graticule-visible"), $_ = document.getElementById("settings-screen-space-graticule-visible"), ev = document.getElementById("reset-view-button"), tv = document.querySelector(".reset-view-control"), nv = document.getElementById("settings-reset-view-visible"), rv = document.getElementById("settings-fullscreen-visible"), iv = document.getElementById("settings-mouse-position-visible"), av = document.getElementById("settings-overview-map-visible"), ov = document.getElementById("settings-overview-map-size"), sv = document.getElementById("settings-mouse-wheel-zoom-sensitivity"), cv = document.getElementById("settings-zoom-button-step"), lv = document.getElementById("settings-scale-bar-enabled"), uv = document.getElementById("settings-theme"), dv = document.getElementById("settings-grid-theme"), fv = document.getElementById("settings-grid-opacity"), pv = document.getElementById("settings-grid-opacity-value"), mv = document.getElementById("settings-grid-spacing"), hv = document.getElementById("settings-grid-labels-visible"), gv = document.getElementById("settings-control-opacity"), _v = document.getElementById("settings-control-opacity-value"), vv = document.getElementById("settings-reset-defaults"), yv = document.getElementById("settings-scale-bar-colour"), bv = document.getElementById("settings-scale-bar-opacity"), xv = document.getElementById("settings-scale-bar-opacity-value"), Sv = document.getElementById("settings-scale-bar-size"), Cv = document.getElementById("settings-scale-bar-units");
if (d_ === null || f_ === null) throw Error("The OpenLayers viewer could not be found.");
if (p_ === null || m_ === null || g_ === null || __ === null || v_ === null || y_ === null || b_ === null || x_ === null || L_ === null || R_ === null || z_ === null || B_ === null || V_ === null || S_ === null || C_ === null || w_ === null || T_ === null || E_ === null || k_ === null || A_ === null || j_ === null || M_ === null || N_ === null || P_ === null || F_ === null || I_ === null || D_ === null || O_ === null || H_ === null || U_ === null || W_ === null || q_ === null || J_ === null || Y_ === null || X_ === null || Z_ === null || Q_ === null || $_ === null || ev === null || tv === null || nv === null || rv === null || iv === null || av === null || ov === null || sv === null || cv === null || uv === null || dv === null || fv === null || pv === null || mv === null || hv === null || gv === null || _v === null || vv === null || lv === null || yv === null || bv === null || xv === null || Sv === null || Cv === null || h_ === null) throw Error("The OpenLayers viewer controls could not be found.");
var wv = 0, Tv = 0, Ev = /* @__PURE__ */ new Map(), Dv = new If();
function Ov() {
	wv += 1, Tv += 1;
}
function kv(e) {
	let t = Ev.get(e);
	t !== void 0 && (Dv.removeFeature(t.feature), t.card.remove(), Ev.delete(e));
}
function Av() {
	Dv.clear(), I_.replaceChildren(), Ev.clear();
}
function jv(e) {
	for (let [t, n] of Ev) n.layerName === e && kv(t);
}
function Mv() {
	if (Ev.size <= 1) return;
	let e = null, t = -1;
	for (let [n, r] of Ev) r.order > t && (e = n, t = r.order);
	for (let t of [...Ev.keys()]) t !== e && kv(t);
}
function Nv() {
	Ov(), Av();
}
var Pv = {
	dark: "#ffffff",
	light: "#000000",
	"high-contrast": "#ffffff"
}, Fv = t_({
	viewerApp: f_,
	panel: H_,
	toggle: U_,
	closeButton: W_,
	tabs: G_,
	tabPanels: K_,
	resetDefaultsButton: vv,
	themeSelect: uv,
	controlOpacityInput: gv,
	controlOpacityValue: _v,
	annotationInspectionEnabledInput: q_,
	multipleAnnotationSelectionInput: J_,
	zoomVisibleInput: Y_,
	zoomLevelVisibleInput: X_,
	rotationVisibleInput: Z_,
	graticuleVisibleInput: Q_,
	screenSpaceGraticuleVisibleInput: $_,
	resetViewVisibleInput: nv,
	fullscreenVisibleInput: rv,
	mousePositionVisibleInput: iv,
	overviewMapVisibleInput: av,
	overviewMapSizeSelect: ov,
	mouseWheelZoomSensitivitySelect: sv,
	zoomButtonStepSelect: cv,
	gridThemeSelect: dv,
	gridOpacityInput: fv,
	gridSpacingSelect: mv,
	gridLabelsVisibleInput: hv,
	scaleBarEnabledInput: lv,
	scaleBarColourInput: yv,
	scaleBarOpacityInput: bv,
	scaleBarSizeSelect: Sv,
	scaleBarUnitsSelect: Cv,
	onAnnotationInspectionChange() {
		q_.checked || Nv();
	},
	onMultipleAnnotationSelectionChange() {
		Ov(), J_.checked || Mv();
	},
	onThemeChange() {
		yv.value = Pv[uv.value] ?? Pv.dark, nb.updateColour(), nb.updateOpacity(), cb.updateAppearance();
	},
	onControlVisibilityChange() {
		fb();
	},
	onReset() {
		pb();
	}
});
Fv.load(), Fv.updateAppearance();
var Iv = Xg({
	panel: g_,
	toggle: __,
	list: v_,
	getSlideLayer: () => Vy,
	getCurrentSlidePath: () => Hv,
	getOverlayLayers: () => Uv,
	onRemoveLayer: (e) => Sb(e),
	onOpen() {
		Ly.setOpen(!1), Ry.setOpen(!1);
	}
}), Lv = JSON.parse(d_.dataset.layers ?? "[]"), Rv = null, zv = Date.now(), Bv = Date.now(), Vv = null, Hv = null, Uv = {}, Wv = /* @__PURE__ */ new Set(), Gv = /* @__PURE__ */ new Map(), Kv = /* @__PURE__ */ new Map(), qv = /* @__PURE__ */ new Map(), Jv = /* @__PURE__ */ new Map(), Yv = /* @__PURE__ */ new Map(), Xv = "type", Zv = null, Qv = null, $v = "automatic", ey = "viridis", ty = [], ny = /* @__PURE__ */ new Map(), ry = !1;
function iy() {
	let e = /* @__PURE__ */ new Set();
	for (let t of qv.values()) for (let n of t) e.add(n);
	return [...e];
}
function ay(e = $v) {
	return e === "automatic" || e === "custom" ? null : e;
}
function oy() {
	return [...qv.entries()].map(([e, t]) => ({
		layerName: e,
		annotationTypes: t
	}));
}
function sy() {
	for (let [e, t] of qv) if (t.length > 0) return {
		layerName: e,
		annotationType: t[0]
	};
	return null;
}
async function cy() {
	let e = [...Wv];
	if (e.length === 0) return [];
	let t = await Promise.all(e.map((e) => $h(e))), n = t[0].filter((e) => !["type", "class"].includes(e.toLowerCase()) && t.slice(1).every((t) => t.includes(e))), r = [];
	for (let t of n) {
		let n = l_((await Promise.all(e.map((e) => eg(e, t)))).flat());
		n !== null && (ny.set(t, n), r.push(t));
	}
	return r.sort();
}
function ly(e) {
	let t = ny.get(e);
	if (t === void 0) throw Error(`Annotation property range is not available: ${e}`);
	let [n, r] = t;
	return n === r ? [n, n + 1] : t;
}
async function uy(e) {
	let t = Kv.get(e);
	if (t === void 0) throw Error(`Annotation colours are not available for layer: ${e}`);
	await og(e), await ng("type", e), await ig(null, e), await Xh(t, e);
}
async function dy({ refresh: e = !0 } = {}) {
	await Promise.all([...Wv].map((e) => uy(e))), Xv = "type", Zv = null, Qv = null, await wy({ refresh: e });
}
async function fy() {
	await og(), await ng("type"), await ig(null);
}
async function py(e, { refresh: t = !0 } = {}) {
	let n = ly(e);
	await Promise.all([...Wv].map(async (t) => {
		await og(t), await ng(e, t), await rg(ey, t), await ig(n, t);
	})), Xv = "property", Zv = e, Qv = null, await wy({ refresh: t });
}
async function my(e, t, n, { refresh: r = !0 } = {}) {
	let i = ly(n);
	await Promise.all([...Wv].map((e) => uy(e))), await ag(t, n, ey, i, e), Xv = "secondary", Zv = n, Qv = {
		layerName: e,
		annotationType: t
	}, await wy({ refresh: r });
}
async function hy({ refresh: e = !0 } = {}) {
	if (Wv.size === 0) {
		await fy(), ty = [], ny.clear(), Xv = "type", Zv = null, Qv = null, Ry.render();
		return;
	}
	if (ty = [], ny.clear(), Ry.render(), ty = await cy(), Xv === "property" && (Zv === null || !ty.includes(Zv) ? await dy({ refresh: e }) : await py(Zv, { refresh: e })), Xv === "secondary") {
		let t = Qv, n = t === null ? void 0 : qv.get(t.layerName);
		t === null || n === void 0 || !n.some((e) => Object.is(e, t.annotationType)) || Zv === null || !ty.includes(Zv) ? await dy({ refresh: e }) : await my(t.layerName, t.annotationType, Zv, { refresh: e });
	}
	Ry.render();
}
async function gy(e) {
	let t = await Qh(iy(), ay(e)), n = [];
	for (let [e, r] of qv) {
		let i = Kv.get(e);
		if (i === void 0) continue;
		let a = Yv.get(e), o = new Map(i);
		for (let e of r) {
			let n = t.get(e);
			if (n === void 0) continue;
			let r = i.get(e), s = a?.get(e) ?? r?.[3] ?? 1;
			o.set(e, [
				n[0],
				n[1],
				n[2],
				s
			]);
		}
		n.push({
			layerName: e,
			previousColours: new Map(i),
			updatedColours: o
		});
	}
	try {
		await Promise.all(n.map(({ layerName: e, updatedColours: t }) => Xh(t, e)));
	} catch (e) {
		throw await Promise.allSettled(n.map(({ layerName: e, previousColours: t }) => Xh(t, e))), e;
	}
	for (let { layerName: e, updatedColours: t } of n) Kv.set(e, t);
	for (let [e, n] of t) {
		let t = Gv.get(e);
		Gv.set(e, [
			n[0],
			n[1],
			n[2],
			t?.[3] ?? n[3] ?? 1
		]);
	}
	$v = e, xy();
}
async function _y(e) {
	let t;
	try {
		t = JSON.parse(await e.text());
	} catch {
		throw Error("Annotation colour file is not valid JSON.");
	}
	let { colorDict: n, layerColorDicts: r } = a_(t), i = [];
	for (let [e, t] of qv) {
		let a = Kv.get(e);
		if (a === void 0) continue;
		let o = o_(a, t, n, r[e] ?? {});
		i.push({
			layerName: e,
			previousColours: new Map(a),
			updatedColours: o
		});
	}
	try {
		await Promise.all(i.map(({ layerName: e, updatedColours: t }) => Xh(t, e)));
	} catch (e) {
		throw await Promise.allSettled(i.map(({ layerName: e, previousColours: t }) => Xh(t, e))), e;
	}
	for (let { layerName: e, updatedColours: t } of i) Kv.set(e, t);
	$v = "custom", xy();
}
function vy() {
	let e = /* @__PURE__ */ new Map(), t = {};
	for (let [n, r] of qv) {
		let i = Kv.get(n);
		if (i !== void 0) {
			t[n] = s_(i, r).color_dict;
			for (let t of r) {
				if (e.has(t)) continue;
				let n = i.get(t);
				n !== void 0 && e.set(t, n);
			}
		}
	}
	let n = s_(e, iy());
	n.layer_color_dicts = t;
	let r = `${JSON.stringify(n, null, 4)}\n`, i = new Blob([r], { type: "application/json" }), a = URL.createObjectURL(i), o = document.createElement("a");
	o.href = a, o.download = "annotation_config.json", document.body.appendChild(o), o.click(), o.remove(), URL.revokeObjectURL(a);
}
function yy(e, t, n) {
	let r = new Map(t.map((e) => [e, !0])), i = new Map(t.map((e) => [e, n.get(e)?.[3] ?? 1]));
	Kv.set(e, n), Jv.set(e, r), Yv.set(e, i);
}
function by(e) {
	Kv.delete(e), Jv.delete(e), Yv.delete(e);
}
function xy(e = null) {
	Bv += 1;
	for (let t of Wv) {
		if (t === e) continue;
		let n = Uv[t];
		if (n === void 0) continue;
		let r = new cp({
			url: `/tileserver/layer/${encodeURIComponent(t)}/${Rv}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${Bv}`,
			size: Vv.slide_dimensions,
			crossOrigin: "anonymous",
			zDirection: -1
		});
		n.setSource(r);
	}
	Yy.render();
}
function Sy(e, t) {
	return Xv === "secondary" && Qv !== null ? e === Qv.layerName && Object.is(t, Qv.annotationType) : Jv.get(e)?.get(t) ?? !0;
}
async function Cy(e, { refresh: t = !0 } = {}) {
	if (!Wv.has(e)) return;
	let n = qv.get(e) ?? [], r = Jv.get(e);
	if (r === void 0) throw Error(`Annotation visibility is not available for layer: ${e}`);
	await Yh(c_(n, Xv === "secondary" ? new Map(n.map((t) => [t, Sy(e, t)])) : r), e), t && xy();
}
async function wy({ refresh: e = !0 } = {}) {
	await Promise.all([...Wv].map((e) => Cy(e, { refresh: !1 }))), e && xy();
}
async function Ty(e) {
	let [t, n] = e, r = [...Wv].filter((e) => Uv[e]?.getVisible() !== !1).sort((e, t) => (Uv[t]?.getZIndex() ?? 0) - (Uv[e]?.getZIndex() ?? 0));
	for (let e of r) {
		let r = await tg(e, t, -n, { details: !0 });
		if (Object.keys(r).length !== 0) {
			if (r.id === void 0 || r.geometry === void 0 || r.properties === void 0) throw Error("Annotation inspection response is incomplete.");
			if (r.properties.type === void 0 || Sy(e, r.properties.type)) return {
				layerName: e,
				annotationId: r.id,
				geometry: r.geometry,
				properties: r.properties
			};
		}
	}
	return null;
}
function Ey(e) {
	return {
		box: "Bounding box (px)",
		centroid: "Centroid (px)",
		prob: "Probability",
		type: "Type"
	}[e] ?? e;
}
function Dy(e, t) {
	if (e === "prob" && typeof t == "number") return `${t.toFixed(4)} (${(t * 100).toFixed(2)}%)`;
	if (e === "centroid" && Array.isArray(t) && t.length >= 2) {
		let [e, n] = t;
		return `x: ${typeof e == "number" ? e.toFixed(2) : e}, y: ${typeof n == "number" ? n.toFixed(2) : n}`;
	}
	return e === "box" && Array.isArray(t) && t.length === 4 ? `x: ${t[0]}–${t[2]}, y: ${t[1]}–${t[3]}` : Array.isArray(t) ? t.join(", ") : typeof t == "number" && !Number.isInteger(t) ? t.toFixed(4) : typeof t == "object" && t ? JSON.stringify(t) : String(t);
}
function Oy(e, t, n) {
	let r = Math.max(8, f_.clientWidth - e.offsetWidth - 8), i = Math.max(8, f_.clientHeight - e.offsetHeight - 8);
	e.style.left = `${Math.min(Math.max(t, 8), r)}px`, e.style.top = `${Math.min(Math.max(n, 8), i)}px`;
}
function ky(e, t) {
	let n = !1, r = 0, i = 0;
	t.addEventListener("pointerdown", (a) => {
		if (a.target.closest("button") !== null) return;
		a.preventDefault();
		let o = e.getBoundingClientRect();
		n = !0, r = a.clientX - o.left, i = a.clientY - o.top, t.setPointerCapture(a.pointerId), e.classList.add("dragging");
	}), t.addEventListener("pointermove", (t) => {
		if (!n) return;
		let a = f_.getBoundingClientRect();
		Oy(e, t.clientX - a.left - r, t.clientY - a.top - i);
	});
	let a = () => {
		n = !1, e.classList.remove("dragging");
	};
	t.addEventListener("pointerup", a), t.addEventListener("pointercancel", a);
}
function Ay(e) {
	let t = new da().readFeature({
		type: "Feature",
		geometry: e.geometry,
		properties: {}
	}), n = t.getGeometry();
	if (n === void 0) throw Error("Annotation geometry is not available.");
	return n.scale(1, -1, [0, 0]), t;
}
function jy(e, t) {
	let n = document.createElement("aside");
	n.className = "annotation-inspector";
	let r = document.createElement("div");
	r.className = "annotation-inspector-header";
	let i = document.createElement("span");
	i.textContent = t.layerName;
	let a = document.createElement("button");
	a.className = "annotation-inspector-close", a.type = "button", a.title = "Close annotation information", a.setAttribute("aria-label", "Close annotation information");
	let o = document.createElement("i");
	o.className = "fas fa-times", a.appendChild(o), r.append(i, a);
	let s = document.createElement("div");
	s.className = "annotation-inspector-properties";
	for (let [e, n] of Object.entries(t.properties)) {
		let t = document.createElement("div");
		t.className = "annotation-inspector-property";
		let r = document.createElement("div");
		r.className = "annotation-inspector-property-name", r.textContent = Ey(e);
		let i = document.createElement("div");
		i.className = "annotation-inspector-property-value", i.textContent = Dy(e, n), t.append(r, i), s.appendChild(t);
	}
	return n.append(r, s), a.addEventListener("click", () => {
		kv(e);
	}), ky(n, r), n;
}
function My(e, t, n) {
	let r = JSON.stringify([e.layerName, String(e.annotationId)]), i = J_.checked;
	i || Av();
	let a = Ev.get(r), o = d_.getBoundingClientRect(), s = f_.getBoundingClientRect();
	if (a !== void 0) {
		a.order = n, Ev.delete(r), Ev.set(r, a), I_.appendChild(a.card), Oy(a.card, o.left - s.left + t[0] + 12, o.top - s.top + t[1] + 12);
		return;
	}
	let c = Ay(e), l = jy(r, e), u = i ? Ev.size * 18 : 0;
	Ev.set(r, {
		layerName: e.layerName,
		feature: c,
		card: l,
		order: n
	}), Dv.addFeature(c), I_.appendChild(l), Oy(l, o.left - s.left + t[0] + 12 + u, o.top - s.top + t[1] + 12 + u);
}
var Ny = await Uh("slide"), Py = await Uh("overlay"), Fy = Py.config?.color_dict ?? {}, Iy = Py.config?.layer_color_dicts ?? {}, Ly = Yg({
	panel: p_,
	toggle: m_,
	container: h_,
	configuredSlides: Ny,
	configuredOverlays: Py,
	getCurrentSlidePath: () => Hv,
	hasSlide: () => Vv !== null,
	hasOverlays: () => Object.keys(Uv).length > 0,
	onSlideSelected: (e) => bb(e),
	onOverlaySelected: (e) => xb(e),
	onClearSlide: () => yb(),
	onClearOverlays: () => gb(),
	onOpen() {
		Iv.setOpen(!1), Ry.setOpen(!1);
	}
}), Ry = Qg({
	panel: y_,
	toggle: b_,
	list: x_,
	colourBySelect: S_,
	paletteField: C_,
	paletteSelect: w_,
	colourMapField: T_,
	colourMapSelect: E_,
	secondaryTypeField: D_,
	secondaryTypeSelect: O_,
	propertyField: k_,
	propertySelect: A_,
	propertyLegend: j_,
	propertyLegendCaption: M_,
	propertyMin: N_,
	propertyMax: P_,
	linkOpacityInput: F_,
	selectAllButton: L_,
	deselectAllButton: R_,
	importButton: z_,
	importInput: B_,
	exportButton: V_,
	getAnnotationGroups: oy,
	getAnnotationTypes: iy,
	getDisplayMode: () => Xv,
	getPalette: () => $v,
	getColourMap: () => ey,
	getAnnotationProperties: () => ty,
	getAnnotationProperty: () => Zv,
	getSecondaryType: () => Qv,
	getPropertyRange: () => Zv === null ? null : ny.get(Zv) ?? null,
	getAnnotationColour: (e, t) => Kv.get(e)?.get(t) ?? [
		0,
		0,
		0,
		1
	],
	isAnnotationTypeVisible: (e, t) => Jv.get(e)?.get(t) ?? !0,
	getAnnotationOpacity: (e, t) => Yv.get(e)?.get(t) ?? 1,
	getOpacityLinked: () => ry,
	async onPaletteChange(e) {
		await gy(e);
	},
	async onColourMapChange(e) {
		let t = ey;
		ey = e;
		try {
			if (Xv === "property") {
				if (Zv === null) throw Error("No annotation property is selected.");
				await py(Zv);
				return;
			}
			if (Xv === "secondary") {
				if (Zv === null || Qv === null) throw Error("Class + Property selection is not available.");
				await my(Qv.layerName, Qv.annotationType, Zv);
			}
		} catch (e) {
			throw ey = t, e;
		}
	},
	async onDisplayModeChange(e) {
		if (e === "type") {
			await dy();
			return;
		}
		let t = Zv ?? ty[0];
		if (t === void 0) throw Error("No annotation properties are available.");
		if (e === "property") {
			await py(t);
			return;
		}
		if (e === "secondary") {
			let e = Qv ?? sy();
			if (e === null) throw Error("No annotation classes are available.");
			await my(e.layerName, e.annotationType, t);
			return;
		}
		throw Error(`Unknown annotation display mode: ${e}`);
	},
	async onPropertyChange(e) {
		if (Xv === "secondary") {
			if (Qv === null) throw Error("No secondary annotation class is selected.");
			await my(Qv.layerName, Qv.annotationType, e);
			return;
		}
		await py(e);
	},
	async onSecondaryTypeChange(e, t) {
		let n = Zv ?? ty[0];
		if (n === void 0) throw Error("No annotation properties are available.");
		await my(e, t, n);
	},
	async onColourChange(e, t, n) {
		let r = wg(n), i = Kv.get(e);
		if (i === void 0) throw Error(`Annotation colours are not available for layer: ${e}`);
		let a = i.get(t)?.[3] ?? 1, o = new Map(i);
		o.set(t, [
			r.r / 255,
			r.g / 255,
			r.b / 255,
			a
		]), await Xh(o, e), Kv.set(e, o), $v = "custom", xy();
	},
	async onVisibilityChange(e, t, n) {
		let r = Jv.get(e);
		if (r === void 0) throw Error(`Annotation visibility is not available for layer: ${e}`);
		let i = r.get(t) ?? !0;
		r.set(t, n);
		try {
			await Cy(e);
		} catch (e) {
			throw r.set(t, i), e;
		}
	},
	async onOpacityChange(e, t, n) {
		let r = [], i = (e, t) => {
			let i = Yv.get(e);
			if (i === void 0) throw Error(`Annotation opacity is not available for layer: ${e}`);
			let a = Kv.get(e);
			if (a === void 0) throw Error(`Annotation colours are not available for layer: ${e}`);
			let o = new Map(i), s = new Map(i), c = new Map(a);
			for (let e of t) {
				s.set(e, n);
				let t = c.get(e);
				t !== void 0 && c.set(e, [
					t[0],
					t[1],
					t[2],
					n
				]);
			}
			r.push({
				layerName: e,
				previousOpacities: o,
				updatedOpacities: s,
				updatedColours: c
			});
		};
		if (ry) for (let e of Wv) i(e, qv.get(e) ?? []);
		else i(e, [t]);
		try {
			await Promise.all(r.map(({ layerName: e, updatedOpacities: t }) => Zh(t, e)));
		} catch (e) {
			throw await Promise.allSettled(r.map(({ layerName: e, previousOpacities: t }) => Zh(t, e))), e;
		}
		for (let { layerName: e, updatedOpacities: t, updatedColours: n } of r) Yv.set(e, t), Kv.set(e, n);
		xy();
	},
	async onOpacityLinkChange(e) {
		ry = e;
	},
	async onSetAllVisibility(e) {
		let t = /* @__PURE__ */ new Map();
		for (let n of Wv) {
			let r = Jv.get(n);
			if (r === void 0) throw Error(`Annotation visibility is not available for layer: ${n}`);
			t.set(n, new Map(r));
			let i = qv.get(n) ?? [];
			for (let t of i) r.set(t, e);
		}
		try {
			await Promise.all([...Wv].map((e) => Cy(e, { refresh: !1 })));
		} catch (e) {
			for (let [e, n] of t) Jv.set(e, n);
			throw await Promise.allSettled([...Wv].map((e) => Cy(e, { refresh: !1 }))), xy(), e;
		}
		xy();
	},
	async onImport(e) {
		await _y(e);
	},
	onExport() {
		vy();
	},
	onOpen() {
		Ly.setOpen(!1), Iv.setOpen(!1);
	}
});
Ly.setOpen(!0);
var zy = new URLSearchParams(window.location.search).get("slide") ?? (Lv.length === 0 ? Ny.files[0]?.path ?? null : null);
if (zy !== null) {
	Hv = zy, Rv = await Vh();
	let e = await Hh(zy);
	Vv = e, Ly.setSlide(zy), Ly.updateOverlaySelect(), Lv = [{
		name: "slide",
		url: `/tileserver/layer/slide/${Rv}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${zv}`,
		size: e.slide_dimensions,
		mpp: e.mpp[0]
	}];
} else Lv.length === 0 && (Rv = await Vh());
Ly.updateActionState();
var By = Lv.map((e) => {
	let t = new cp({
		url: e.url,
		size: e.size,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	return new bs({
		title: e.name,
		source: t
	});
}), Vy = By[0];
Vy === void 0 && (Vy = new bs({ title: "slide" }), By.push(Vy)), Vy.setZIndex(0);
var Hy = Vy.getSource(), Uy, Wy, Gy;
if (Hy !== null) {
	let e = Hy.getTileGrid();
	Uy = e.getResolutions(), Wy = e.getExtent(), Gy = new Jt({
		code: "ZoomifyProjection",
		units: "pixels",
		extent: Wy,
		metersPerUnit: Lv[0].mpp * 1e-6,
		getPointResolution(e) {
			return e;
		}
	});
} else Uy = [1], Wy = [
	0,
	-1,
	1,
	0
], Gy = new Jt({
	code: "ZoomifyProjectionEmpty",
	units: "pixels",
	extent: Wy,
	metersPerUnit: 1,
	getPointResolution(e) {
		return e;
	}
});
var Ky = .1;
function qy(e) {
	let t = e[2] - e[0], n = e[3] - e[1], r = t * Ky, i = n * Ky;
	return [
		e[0] - r,
		e[1] - i,
		e[2] + r,
		e[3] + i
	];
}
tr(Gy);
var Jy = new cs({
	projection: Gy,
	resolutions: Uy,
	extent: qy(Wy),
	constrainOnlyCenter: !0,
	smoothExtentConstraint: !0,
	smoothResolutionConstraint: !1,
	center: [.5, -.5],
	resolution: Uy[0]
}), Yy = new Tf({
	target: d_,
	layers: By,
	view: Jy,
	controls: Fe({
		zoom: !1,
		rotate: !1
	}),
	interactions: hf({ mouseWheelZoom: !1 })
}), Xy = new xd({
	source: Dv,
	style: new ku({
		fill: new Du({ color: "rgba(255, 215, 0, 0.22)" }),
		stroke: new Ou({
			color: "#ffd700",
			width: 3
		}),
		image: new Eu({
			radius: 7,
			fill: new Du({ color: "rgba(255, 215, 0, 0.45)" }),
			stroke: new Ou({
				color: "#ffd700",
				width: 3
			})
		})
	})
});
Xy.setZIndex(1e4), Yy.addLayer(Xy);
var Zy = vg({
	map: Yy,
	viewerApp: f_,
	getSlideSource: () => Vy.getSource(),
	zoomVisibleInput: Y_,
	zoomLevelVisibleInput: X_,
	rotationVisibleInput: Z_,
	resetViewButton: ev,
	resetViewControl: tv,
	resetViewVisibleInput: nv,
	fullscreenVisibleInput: rv,
	mousePositionVisibleInput: iv,
	mouseWheelZoomSensitivitySelect: sv,
	zoomButtonStepSelect: cv
}), { fullscreen: Qy, mousePositionControl: $y, rotate: eb } = Zy, tb = null, nb = Rg({
	map: Yy,
	hasSlide: () => Vy.getSource() !== null,
	enabledInput: lv,
	colourInput: yv,
	opacityInput: bv,
	opacityValue: xv,
	sizeSelect: Sv,
	unitsSelect: Cv,
	onControlChange(e) {
		tb = e, window.scaleLineControl = e;
	}
}), rb = Gg({
	map: Yy,
	source: Hy,
	projection: Gy,
	extent: Wy,
	sizeSelect: ov,
	visibleInput: av,
	hasSlide: () => Vy.getSource() !== null
}), ib = rb.control, ab = new Bh();
Yy.addControl(ab);
var ob = null, sb = null, cb = Mg({
	map: Yy,
	projection: Gy,
	themeSelect: uv,
	gridThemeSelect: dv,
	gridOpacityInput: fv,
	gridOpacityValue: pv,
	gridSpacingSelect: mv,
	gridLabelsVisibleInput: hv,
	graticuleVisibleInput: Q_,
	screenSpaceGraticuleVisibleInput: $_,
	onGraticulesChange(e, t) {
		ob = e, sb = t, window.graticule = ob, window.screenSpaceGraticule = sb;
	}
}), { graticuleToggle: lb, screenSpaceGraticuleToggle: ub } = cb, db = document.createElement("div");
db.className = "viewer-tools-group ol-unselectable", Yy.getOverlayContainerStopEvent().append(db), db.append(eb.element, lb.element, ub.element);
function fb() {
	Zy.updateVisibility(), cb.updateVisibility(), rb.updateVisibility();
}
function pb() {
	Fv.resetValues(), Ov(), Mv(), Fv.updateAppearance(), cb.updateAppearance(), cb.updateSpacing(), cb.updateLabels(), fb(), rb.updateSize(), Zy.updateMouseWheelZoomSensitivity(), Zy.updateZoomButtonStep(), nb.updateSize(), nb.updateUnits(), nb.updateVisibility(), nb.updateColour(), nb.updateOpacity(), Fv.clearSavedSettings();
}
Fv.bindEvents();
function mb(e) {
	Zy.setViewerEnabled(e), cb.setViewerEnabled(e), nb.setViewerEnabled(e), rb.setViewerEnabled(e);
}
if (mb(Hy !== null), fb(), Hy !== null) {
	Yy.getView().fit(Wy);
	let e = _b();
	e !== null && (Yy.getView().setCenter(e.center), Yy.getView().setZoom(e.zoom));
}
Yy.on("moveend", () => {
	vb(), Zy.updateZoomLevel();
}), Yy.on("singleclick", async (e) => {
	if (!q_.checked || Wv.size === 0) return;
	Tv += 1;
	let t = Tv, n = wv, r = J_.checked;
	try {
		let i = await Ty(e.coordinate);
		if (n !== wv || !q_.checked || Wv.size === 0 || !r && t !== Tv || i === null) return;
		My(i, e.pixel, t);
	} catch (e) {
		if (n !== wv) return;
		console.error("Failed to inspect annotation.", e);
	}
});
function hb() {
	for (let e of Object.values(Uv)) {
		e.setSource(null), Yy.removeLayer(e);
		let t = By.indexOf(e);
		t !== -1 && By.splice(t, 1);
	}
	for (let e of Object.keys(Uv)) delete Uv[e];
	Wv.clear(), qv.clear(), Kv.clear(), Jv.clear(), Yv.clear(), Nv(), Xv = "type", Zv = null, Qv = null, ty = [], ny.clear(), Ry.render(), Iv.render(), Ly.updateActionState();
}
async function gb() {
	await Wh(), hb(), await fy();
}
function _b() {
	let e = new URLSearchParams(window.location.search), t = Number(e.get("x")), n = Number(e.get("y")), r = Number(e.get("zoom"));
	return e.get("x") === null || e.get("y") === null || e.get("zoom") === null || !Number.isFinite(t) || !Number.isFinite(n) || !Number.isFinite(r) ? null : {
		center: [t, n],
		zoom: r
	};
}
function vb() {
	if (Hv === null) return;
	let e = Yy.getView(), t = e.getCenter(), n = e.getZoom();
	if (t === void 0 || n === void 0) return;
	let r = new URL(window.location.href);
	r.searchParams.set("slide", Hv), r.searchParams.set("x", t[0].toFixed(2)), r.searchParams.set("y", t[1].toFixed(2)), r.searchParams.set("zoom", n.toString());
	let i = r.searchParams.toString().replace(/%2F/gi, "/");
	window.history.replaceState({}, "", `${r.pathname}?${i}${r.hash}`);
}
async function yb() {
	if (Rv === null) throw Error("No TileServer session is available.");
	await Gh(), hb(), await fy(), Hv = null, Vv = null, Lv.length = 0, zv += 1, Bv += 1, Vy.setSource(null), rb.setSource(null), Iv.render();
	let e = [
		0,
		-1,
		1,
		0
	], t = [1], n = new Jt({
		code: "ZoomifyProjectionEmpty",
		units: "pixels",
		extent: e,
		metersPerUnit: 1,
		getPointResolution(e) {
			return e;
		}
	});
	tr(n);
	let r = new cs({
		projection: n,
		resolutions: t,
		constrainOnlyCenter: !0,
		center: [.5, -.5],
		resolution: t[0]
	});
	Yy.setView(r), rb.setView(n, e), cb.setProjection(n, { preserveActive: !1 });
	let i = new URL(window.location.href);
	i.search = "", i.hash = "", window.history.replaceState({}, "", i), mb(!1), Zy.updateZoomLevel(), Ly.updateActionState();
}
async function bb(e) {
	if (Rv === null) throw Error("Dynamic slide switching requires a TileServer session.");
	hb(), await fy();
	let t = await Hh(e);
	Vv = t, Hv = e, Ly.setSlide(e), Ly.updateOverlaySelect(), Ly.updateActionState(), zv += 1;
	let n = u_(Rv, t, zv), r = n.getTileGrid(), i = r.getExtent(), a = r.getResolutions(), o = new Jt({
		code: "ZoomifyProjection",
		units: "pixels",
		extent: i,
		metersPerUnit: t.mpp[0] * 1e-6
	});
	tr(o);
	let s = [(i[0] + i[2]) / 2, (i[1] + i[3]) / 2], c = new cs({
		projection: o,
		resolutions: a,
		extent: qy(i),
		constrainOnlyCenter: !0,
		smoothExtentConstraint: !0,
		smoothResolutionConstraint: !1,
		center: s,
		resolution: a[0]
	});
	c.fit(i, { size: Yy.getSize() }), Yy.setView(c), rb.setView(o, i), cb.setProjection(o), Vy.setSource(n), rb.setSource(n), Iv.render(), mb(!0), vb(), Zy.updateZoomLevel();
}
Iv.render();
async function xb(e) {
	if (Rv === null || Vv === null) throw Error("Dynamic overlay loading requires a loaded slide.");
	let t = e.split(".").pop().toLowerCase();
	if (t === "npy" || t === "mha") throw Error("Registration overlays are not supported yet.");
	let n = [
		"db",
		"dat",
		"geojson"
	].includes(t), r = Jg(e), i = Wv.has(r);
	if (r === "slide") throw Error("The overlay name \"slide\" is reserved.");
	let a = await Kh(e, r);
	if (i && (Ov(), jv(r)), n) {
		Wv.add(r);
		let e = [...new Set(a)];
		qv.set(r, e);
		let t = /* @__PURE__ */ new Map(), n = {
			...Fy,
			...Iy[r] ?? {}
		}, i = e.some((e) => Object.hasOwn(n, String(e)));
		await n_(t, e, (e) => Qh(e, ay()), n), i && ($v = "custom");
		for (let [e, n] of t) Gv.has(e) || Gv.set(e, n);
		yy(r, e, t), await Zh(Yv.get(r), r), Xv === "type" && await Xh(Kv.get(r), r);
	} else i && (Wv.delete(r), qv.delete(r), by(r));
	n || i ? (await hy({ refresh: !1 }), n && await Cy(r, { refresh: !1 }), xy(r)) : Bv += 1;
	let o = new cp({
		url: `/tileserver/layer/${encodeURIComponent(r)}/${Rv}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${Bv}`,
		size: Vv.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	if (Uv[r] !== void 0) Uv[r].setSource(o), Uv[r].setVisible(!0);
	else {
		let e = [Vy, ...Object.values(Uv)], t = Math.max(...e.map((e) => e.getZIndex() ?? 0)), n = new bs({
			title: r,
			source: o,
			opacity: .75
		});
		n.setZIndex(t + 1), Uv[r] = n, Yy.addLayer(n), By.push(n);
	}
	return Iv.render(), Ly.updateActionState(), a;
}
async function Sb(e) {
	let t = Uv[e], n = Wv.has(e);
	if (t === void 0) throw Error(`Overlay is not loaded: ${e}`);
	let r = t.getSource();
	t.setVisible(!1), t.setSource(null), Yy.removeLayer(t);
	try {
		await qh(e);
	} catch (e) {
		throw t.setSource(r), t.setVisible(!0), Yy.addLayer(t), e;
	}
	let i = By.indexOf(t);
	i !== -1 && By.splice(i, 1), Wv.delete(e), qv.delete(e), by(e), delete Uv[e], n && (Ov(), jv(e), await hy({ refresh: !1 }), xy()), Iv.render(), Ly.updateActionState();
}
async function Cb(e) {
	if (Wv.size === 0) throw Error("No annotation overlay is loaded.");
	let t = e instanceof Map ? [...e.entries()] : Object.entries(e), n = new Map(t);
	await Xh(n), Gv.clear();
	for (let [e, t] of n) Gv.set(e, t);
	xy();
}
Object.assign(window, {
	clearOverlays: gb,
	extent: Wy,
	fullscreen: Qy,
	graticule: ob,
	graticuleToggle: lb,
	layerSwitcher: ab,
	layers: By,
	layersData: Lv,
	loadOverlay: xb,
	map: Yy,
	mousePositionControl: $y,
	overlayLayers: Uv,
	overviewMapControl: ib,
	projection: Gy,
	removeOverlay: Sb,
	removeSlide: yb,
	resolutions: Uy,
	rotate: eb,
	scaleLineControl: tb,
	screenSpaceGraticule: sb,
	screenSpaceGraticuleToggle: ub,
	setAnnotationColors: Cb,
	switchSlide: bb,
	view: Jy
});
//#endregion
