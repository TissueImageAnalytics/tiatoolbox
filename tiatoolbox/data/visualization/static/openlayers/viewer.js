// node_modules/ol/CollectionEventType.js
var e = {
	ADD: "add",
	REMOVE: "remove"
}, t = { PROPERTYCHANGE: "propertychange" };
// node_modules/ol/obj.js
function n(e) {
	for (let t in e) delete e[t];
}
function r(e) {
	let t;
	for (t in e) return !1;
	return !t;
}
// node_modules/ol/events.js
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
// node_modules/ol/events/EventType.js
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
	KEYUP: "keyup",
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
// node_modules/ol/array.js
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
// node_modules/ol/util.js
function _() {
	throw Error("Unimplemented abstract method.");
}
var v = 0;
function y(e) {
	let t = e;
	return t.ol_uid ||= String(++v);
}
function b(e) {
	return !!e && typeof e.then == "function" && typeof e.catch == "function";
}
// node_modules/ol/functions.js
function x() {
	return !0;
}
function S() {
	return !1;
}
function C() {}
function w(e) {
	let t, n, r;
	return function() {
		let i = Array.prototype.slice.call(arguments);
		return n && this === r && h(i, n) || (r = this, n = i, t = e.apply(this, i)), t;
	};
}
function T(e) {
	function t() {
		let t;
		try {
			t = e();
		} catch (e) {
			return Promise.reject(e);
		}
		return b(t) ? t : Promise.resolve(t);
	}
	return t();
}
// node_modules/ol/events/Event.js
var E = class {
	constructor(e) {
		this.propagationStopped, this.defaultPrevented, this.type = e, this.target = null;
	}
	preventDefault() {
		this.defaultPrevented = !0;
	}
	stopPropagation() {
		this.propagationStopped = !0;
	}
}, D = class extends c {
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
		let i = t ? new E(e) : e;
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
			for (delete o[n]; e--;) this.removeEventListener(n, C);
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
		r !== -1 && (this.pendingRemovals_ && e in this.pendingRemovals_ ? (n[r] = C, ++this.pendingRemovals_[e]) : (n.splice(r, 1), n.length === 0 && delete this.listeners_[e]));
	}
}, O = class extends D {
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
		if (n) k(n);
		else if (Array.isArray(e)) for (let n = 0, r = e.length; n < r; ++n) this.removeEventListener(e[n], t);
		else this.removeEventListener(e, t);
	}
};
O.prototype.on, O.prototype.once, O.prototype.un;
function k(e) {
	if (Array.isArray(e)) for (let t = 0, n = e.length; t < n; ++t) o(e[t]);
	else o(e);
}
// node_modules/ol/Object.js
var A = class extends E {
	constructor(e, t, n) {
		super(e), this.key = t, this.oldValue = n;
	}
}, j = class extends O {
	constructor(e) {
		super(), this.on, this.once, this.un, y(this), this.values_ = null, e !== void 0 && this.setProperties(e);
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
		r = `change:${e}`, this.hasListener(r) && this.dispatchEvent(new A(r, e, n)), r = t.PROPERTYCHANGE, this.hasListener(r) && this.dispatchEvent(new A(r, e, n));
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
}, M = { LENGTH: "length" }, N = class extends E {
	constructor(e, t, n) {
		super(e), this.element = t, this.index = n;
	}
}, P = class extends j {
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
		return this.get(M.LENGTH);
	}
	insertAt(t, n) {
		if (t < 0 || t > this.getLength()) throw Error("Index out of bounds: " + t);
		this.unique_ && this.assertUnique_(n), this.array_.splice(t, 0, n), this.updateLength_(), this.dispatchEvent(new N(e.ADD, n, t));
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
		return this.array_.splice(t, 1), this.updateLength_(), this.dispatchEvent(new N(e.REMOVE, n, t)), n;
	}
	setAt(t, n) {
		if (t >= this.getLength()) {
			this.insertAt(t, n);
			return;
		}
		if (t < 0) throw Error("Index out of bounds: " + t);
		this.unique_ && this.assertUnique_(n, t);
		let r = this.array_[t];
		this.array_[t] = n, this.dispatchEvent(new N(e.REMOVE, r, t)), this.dispatchEvent(new N(e.ADD, n, t));
	}
	updateLength_() {
		this.set(M.LENGTH, this.array_.length);
	}
	assertUnique_(e, t) {
		let n = this.array_;
		for (let r = 0, i = n.length; r < i; ++r) if (n[r] === e && r !== t) throw Error("Duplicate item added to a unique collection");
	}
}, F = "ol-hidden", ee = "ol-selectable", te = "ol-unselectable", ne = "ol-unsupported", re = "ol-control", ie = "ol-collapsed", ae = /* @__PURE__ */ RegExp("^\\s*(?=(?:(?:[-a-z]+\\s*){0,2}(italic|oblique))?)(?=(?:(?:[-a-z]+\\s*){0,2}(small-caps))?)(?=(?:(?:[-a-z]+\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)(?:(?:normal|\\1|\\2|\\3)\\s*){0,3}((?:xx?-)?(?:small|large)|medium|smaller|larger|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx]))(?:\\s*\\/\\s*(normal|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx])?))?\\s*([-,\\\"\\'\\sa-z0-9]+?)\\s*$", "i"), oe = [
	"style",
	"variant",
	"weight",
	"size",
	"lineHeight",
	"family"
], se = {
	normal: 400,
	bold: 700
}, ce = function(e) {
	let t = e.match(ae);
	if (!t) return null;
	let n = {
		lineHeight: "normal",
		size: "1.2em",
		style: "normal",
		weight: "400",
		variant: "normal"
	};
	for (let e = 0, r = oe.length; e < r; ++e) {
		let r = t[e + 1];
		r !== void 0 && (n[oe[e]] = typeof r == "string" ? r.trim() : r);
	}
	return isNaN(Number(n.weight)) && n.weight in se && (n.weight = se[n.weight]), n.families = n.family.split(/,\s?/).map((e) => e.trim().replace(/^['"]|['"]$/g, "")), n;
}, le = typeof navigator < "u" && navigator.userAgent !== void 0 ? navigator.userAgent.toLowerCase() : "", ue = le.includes("safari") && !le.includes("chrom") && (le.includes("version/15.4") || /cpu (os|iphone os) 15_4 like mac os x/.test(le)), de = le.includes("webkit") && !le.includes("edge"), fe = le.includes("macintosh"), pe = typeof devicePixelRatio < "u" ? devicePixelRatio : 1, me = typeof WorkerGlobalScope < "u" && typeof OffscreenCanvas < "u" && self instanceof WorkerGlobalScope, he = typeof Image < "u" && Image.prototype.decode, ge = (function() {
	let e = !1;
	try {
		let t = Object.defineProperty({}, "passive", { get: function() {
			e = !0;
		} });
		window.addEventListener("_", null, t), window.removeEventListener("_", null, t);
	} catch {}
	return e;
})();
// node_modules/ol/dom.js
function _e(e, t, n, r) {
	let i;
	return i = n && n.length ? n.shift() : me ? new class extends OffscreenCanvas {
		style = {};
	}(e ?? 300, t ?? 150) : document.createElement("canvas"), e && (i.width = e), t && (i.height = t), i.getContext("2d", r);
}
var ve;
function ye() {
	return ve ||= _e(1, 1), ve;
}
function be(e) {
	let t = e.canvas;
	t.width = 1, t.height = 1, e.clearRect(0, 0, 1, 1);
}
function xe(e) {
	let t = e.offsetWidth, n = getComputedStyle(e);
	return t += parseInt(n.marginLeft, 10) + parseInt(n.marginRight, 10), t;
}
function Se(e) {
	let t = e.offsetHeight, n = getComputedStyle(e);
	return t += parseInt(n.marginTop, 10) + parseInt(n.marginBottom, 10), t;
}
function Ce(e, t) {
	let n = t.parentNode;
	n && n.replaceChild(e, t);
}
function we(e) {
	for (; e.lastChild;) e.lastChild.remove();
}
function Te(e, t) {
	let n = e.childNodes;
	for (let r = 0;; ++r) {
		let i = n[r], a = t[r];
		if (!i && !a) break;
		i !== a && (i ? a ? e.insertBefore(a, i) : (e.removeChild(i), --r) : e.appendChild(a));
	}
}
function Ee() {
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
function De(e) {
	return typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof OffscreenCanvas < "u" && e instanceof OffscreenCanvas;
}
// node_modules/ol/MapEventType.js
var Oe = {
	POSTRENDER: "postrender",
	MOVESTART: "movestart",
	MOVEEND: "moveend",
	LOADSTART: "loadstart",
	LOADEND: "loadend"
}, ke = class extends j {
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
			this.element && t && t.appendChild(this.element), this.render !== C && this.listenerKeys.push(i(e, Oe.POSTRENDER, this.render, this)), e.render();
		}
	}
	render(e) {}
	setTarget(e) {
		this.target_ = typeof e == "string" ? document.getElementById(e) : e;
	}
}, Ae = class extends ke {
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
		let l = t + " " + te + " " + re + (this.collapsed_ && this.collapsible_ ? " " + ie : "") + (this.collapsible_ ? "" : " ol-uncollapsible"), u = this.element;
		u.className = l, u.appendChild(this.toggleButton_), u.appendChild(this.ulElement_), this.renderedAttributions_ = [], this.renderedVisible_ = !0;
	}
	collectSourceAttributions_(e) {
		let t = this.getMap();
		if (!t) return [];
		let n = t.getAllLayers(), r = new Set(n.flatMap((t) => t.getAttributions(e)));
		if (this.attributions_ !== void 0 && (Array.isArray(this.attributions_) ? this.attributions_.forEach((e) => r.add(e)) : r.add(this.attributions_)), !this.overrideCollapsible_) {
			let e = !n.some((e) => e.getSource()?.getAttributionsCollapsible() === !1);
			this.setCollapsible(e);
		}
		return Array.from(r);
	}
	async updateElement_(e) {
		let t = this.element;
		if (!t) return;
		if (!e) {
			this.renderedVisible_ &&= (t.style.display = "none", !1);
			return;
		}
		let n = await Promise.all(this.collectSourceAttributions_(e).map((e) => T(() => e))), r = n.length > 0;
		if (this.renderedVisible_ != r && (t.style.display = r ? "" : "none", this.renderedVisible_ = r), !h(n, this.renderedAttributions_)) {
			we(this.ulElement_);
			for (let e = 0, t = n.length; e < t; ++e) {
				let t = document.createElement("li");
				t.innerHTML = n[e], this.ulElement_.appendChild(t);
			}
			this.renderedAttributions_ = n;
		}
	}
	handleClick_(e) {
		e.preventDefault(), this.handleToggle_(), this.userCollapsed_ = this.collapsed_;
	}
	handleToggle_() {
		let e = this.element;
		e && (e.classList.toggle(ie), this.collapsed_ ? Ce(this.collapseLabel_, this.label_) : Ce(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_, this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_)));
	}
	getCollapsible() {
		return this.collapsible_;
	}
	setCollapsible(e) {
		if (this.collapsible_ === e) return;
		this.collapsible_ = e;
		let t = this.element;
		t && (t.classList.toggle("ol-uncollapsible"), this.userCollapsed_ && this.handleToggle_());
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
// node_modules/ol/easing.js
function je(e) {
	return e ** 3;
}
function Me(e) {
	return 1 - je(1 - e);
}
function Ne(e) {
	return 3 * e * e - 2 * e * e * e;
}
function Pe(e) {
	return e;
}
// node_modules/ol/control/Rotate.js
var Fe = class extends ke {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			render: e.render,
			target: e.target
		});
		let t = e.className === void 0 ? "ol-rotate" : e.className, n = e.label === void 0 ? "⇧" : e.label, r = e.compassClassName === void 0 ? "ol-compass" : e.compassClassName;
		this.label_, typeof n == "string" ? (this.label_ = document.createElement("span"), this.label_.className = r, this.label_.textContent = n) : (this.label_ = n, this.label_.classList.add(r));
		let i = e.tipLabel ? e.tipLabel : "Reset rotation", a = document.createElement("button");
		a.className = t + "-reset", a.setAttribute("type", "button"), a.title = i, a.appendChild(this.label_), a.addEventListener(s.CLICK, this.handleClick_.bind(this), !1);
		let o = t + " " + te + " " + re, c = this.element;
		c.className = o, c.appendChild(a), this.callResetNorth_ = e.resetNorth ? e.resetNorth : void 0, this.duration_ = e.duration === void 0 ? 250 : e.duration, this.autoHide_ = e.autoHide === void 0 || e.autoHide, this.rotation_ = void 0, this.autoHide_ && this.element.classList.add(F);
	}
	handleClick_(e) {
		e.preventDefault(), this.callResetNorth_ === void 0 ? this.resetNorth_() : this.callResetNorth_();
	}
	resetNorth_() {
		let e = this.getMap();
		if (!e) return;
		let t = e.getView();
		if (!t) return;
		let n = t.getRotation();
		n !== void 0 && (this.duration_ > 0 && n % (2 * Math.PI) != 0 ? t.animate({
			rotation: 0,
			duration: this.duration_,
			easing: Me
		}) : t.setRotation(0));
	}
	render(e) {
		let t = e.frameState;
		if (!t) return;
		let n = t.viewState.rotation;
		if (n != this.rotation_) {
			let e = "rotate(" + n + "rad)", t = this.element;
			if (this.autoHide_ && t) {
				let e = t.classList.contains(F);
				!e && n === 0 ? t.classList.add(F) : e && n !== 0 && t.classList.remove(F);
			}
			this.label_.style.transform = e;
		}
		this.rotation_ = n;
	}
}, Ie = class extends ke {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			target: e.target
		});
		let t = e.className === void 0 ? "ol-zoom" : e.className, n = e.delta === void 0 ? 1 : e.delta, r = e.zoomInClassName === void 0 ? t + "-in" : e.zoomInClassName, i = e.zoomOutClassName === void 0 ? t + "-out" : e.zoomOutClassName, a = e.zoomInLabel === void 0 ? "+" : e.zoomInLabel, o = e.zoomOutLabel === void 0 ? "–" : e.zoomOutLabel, c = e.zoomInTipLabel === void 0 ? "Zoom in" : e.zoomInTipLabel, l = e.zoomOutTipLabel === void 0 ? "Zoom out" : e.zoomOutTipLabel, u = document.createElement("button");
		u.className = r, u.setAttribute("type", "button"), u.title = c, u.appendChild(typeof a == "string" ? document.createTextNode(a) : a), u.addEventListener(s.CLICK, this.handleClick_.bind(this, n), !1);
		let d = document.createElement("button");
		d.className = i, d.setAttribute("type", "button"), d.title = l, d.appendChild(typeof o == "string" ? document.createTextNode(o) : o), d.addEventListener(s.CLICK, this.handleClick_.bind(this, -n), !1);
		let f = t + " " + te + " " + re, p = this.element;
		p.className = f, p.appendChild(u), p.appendChild(d), this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleClick_(e, t) {
		t.preventDefault(), this.zoomByDelta_(e);
	}
	zoomByDelta_(e) {
		let t = this.getMap();
		if (!t) return;
		let n = t.getView();
		if (!n) return;
		let r = n.getZoom();
		if (r !== void 0) {
			let t = n.getConstrainedZoom(r + e);
			if (t === void 0) return;
			this.duration_ > 0 ? (n.getAnimating() && n.cancelAnimations(), n.animate({
				zoom: t,
				duration: this.duration_,
				easing: Me
			})) : n.setZoom(t);
		}
	}
};
// node_modules/ol/control/defaults.js
function Le(e) {
	e ||= {};
	let t = new P();
	return (e.zoom === void 0 || e.zoom) && t.push(new Ie(e.zoomOptions)), (e.rotate === void 0 || e.rotate) && t.push(new Fe(e.rotateOptions)), (e.attribution === void 0 || e.attribution) && t.push(new Ae(e.attributionOptions)), t;
}
// node_modules/ol/extent/Relationship.js
var Re = {
	UNKNOWN: 0,
	INTERSECTING: 1,
	ABOVE: 2,
	RIGHT: 4,
	BELOW: 8,
	LEFT: 16
};
// node_modules/ol/extent.js
function ze(e) {
	let t = Je();
	for (let n = 0, r = e.length; n < r; ++n) tt(t, e[n]);
	return t;
}
function Be(e, t, n) {
	return Ye(Math.min.apply(null, e), Math.min.apply(null, t), Math.max.apply(null, e), Math.max.apply(null, t), n);
}
function Ve(e, t, n) {
	return n ? (n[0] = e[0] - t, n[1] = e[1] - t, n[2] = e[2] + t, n[3] = e[3] + t, n) : [
		e[0] - t,
		e[1] - t,
		e[2] + t,
		e[3] + t
	];
}
function He(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e.slice();
}
function Ue(e, t, n) {
	let r, i;
	return r = t < e[0] ? e[0] - t : e[2] < t ? t - e[2] : 0, i = n < e[1] ? e[1] - n : e[3] < n ? n - e[3] : 0, r * r + i * i;
}
function We(e, t) {
	return Ke(e, t[0], t[1]);
}
function Ge(e, t) {
	return e[0] <= t[0] && t[2] <= e[2] && e[1] <= t[1] && t[3] <= e[3];
}
function Ke(e, t, n) {
	return e[0] <= t && t <= e[2] && e[1] <= n && n <= e[3];
}
function qe(e, t) {
	let n = e[0], r = e[1], i = e[2], a = e[3], o = t[0], s = t[1], c = Re.UNKNOWN;
	return o < n ? c |= Re.LEFT : o > i && (c |= Re.RIGHT), s < r ? c |= Re.BELOW : s > a && (c |= Re.ABOVE), c === Re.UNKNOWN && (c = Re.INTERSECTING), c;
}
function Je() {
	return [
		Infinity,
		Infinity,
		-Infinity,
		-Infinity
	];
}
function Ye(e, t, n, r, i) {
	return i ? (i[0] = e, i[1] = t, i[2] = n, i[3] = r, i) : [
		e,
		t,
		n,
		r
	];
}
function Xe(e) {
	return Ye(Infinity, Infinity, -Infinity, -Infinity, e);
}
function Ze(e, t) {
	let n = e[0], r = e[1];
	return Ye(n, r, n, r, t);
}
function Qe(e, t, n, r, i) {
	return nt(Xe(i), e, t, n, r);
}
function $e(e, t) {
	return e[0] == t[0] && e[2] == t[2] && e[1] == t[1] && e[3] == t[3];
}
function et(e, t) {
	return t[0] < e[0] && (e[0] = t[0]), t[2] > e[2] && (e[2] = t[2]), t[1] < e[1] && (e[1] = t[1]), t[3] > e[3] && (e[3] = t[3]), e;
}
function tt(e, t) {
	t[0] < e[0] && (e[0] = t[0]), t[0] > e[2] && (e[2] = t[0]), t[1] < e[1] && (e[1] = t[1]), t[1] > e[3] && (e[3] = t[1]);
}
function nt(e, t, n, r, i) {
	for (; n < r; n += i) rt(e, t[n], t[n + 1]);
	return e;
}
function rt(e, t, n) {
	e[0] = Math.min(e[0], t), e[1] = Math.min(e[1], n), e[2] = Math.max(e[2], t), e[3] = Math.max(e[3], n);
}
function it(e, t) {
	let n;
	return n = t(ot(e)), n || (n = t(st(e)), n) || (n = t(gt(e)), n) || (n = t(ht(e)), n) ? n : !1;
}
function at(e) {
	let t = 0;
	return vt(e) || (t = I(e) * ft(e)), t;
}
function ot(e) {
	return [e[0], e[1]];
}
function st(e) {
	return [e[2], e[1]];
}
function ct(e) {
	return [(e[0] + e[2]) / 2, (e[1] + e[3]) / 2];
}
function lt(e, t) {
	let n;
	if (t === "bottom-left") n = ot(e);
	else if (t === "bottom-right") n = st(e);
	else if (t === "top-left") n = ht(e);
	else if (t === "top-right") n = gt(e);
	else throw Error("Invalid corner");
	return n;
}
function ut(e, t, n, r, i) {
	let [a, o, s, c, l, u, d, f] = dt(e, t, n, r);
	return Ye(Math.min(a, s, l, d), Math.min(o, c, u, f), Math.max(a, s, l, d), Math.max(o, c, u, f), i);
}
function dt(e, t, n, r) {
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
function ft(e) {
	return e[3] - e[1];
}
function pt(e, t, n) {
	let r = n || Je();
	return _t(e, t) ? (r[0] = e[0] > t[0] ? e[0] : t[0], r[1] = e[1] > t[1] ? e[1] : t[1], r[2] = e[2] < t[2] ? e[2] : t[2], r[3] = e[3] < t[3] ? e[3] : t[3]) : Xe(r), r;
}
function mt(e, t) {
	if (!_t(e, t)) return [e.slice()];
	if (Ge(t, e)) return [];
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
function ht(e) {
	return [e[0], e[3]];
}
function gt(e) {
	return [e[2], e[3]];
}
function I(e) {
	return e[2] - e[0];
}
function _t(e, t) {
	return e[0] <= t[2] && e[2] >= t[0] && e[1] <= t[3] && e[3] >= t[1];
}
function vt(e) {
	return e[2] < e[0] || e[3] < e[1];
}
function yt(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e;
}
function bt(e, t) {
	let n = (e[2] - e[0]) / 2 * (t - 1), r = (e[3] - e[1]) / 2 * (t - 1);
	e[0] -= n, e[2] += n, e[1] -= r, e[3] += r;
}
function xt(e, t, n) {
	let r = !1, i = qe(e, t), a = qe(e, n);
	if (i === Re.INTERSECTING || a === Re.INTERSECTING) r = !0;
	else {
		let o = e[0], s = e[1], c = e[2], l = e[3], u = t[0], d = t[1], f = n[0], p = n[1], m = (p - d) / (f - u), h, g;
		a & Re.ABOVE && !(i & Re.ABOVE) && (h = f - (p - l) / m, r = h >= o && h <= c), !r && a & Re.RIGHT && !(i & Re.RIGHT) && (g = p - (f - c) * m, r = g >= s && g <= l), !r && a & Re.BELOW && !(i & Re.BELOW) && (h = f - (p - s) / m, r = h >= o && h <= c), !r && a & Re.LEFT && !(i & Re.LEFT) && (g = p - (f - o) * m, r = g >= s && g <= l);
	}
	return r;
}
function St(e, t, n, r) {
	if (vt(e)) return Xe(n);
	let i = [];
	if (r !== void 0 && r > 1) {
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
	return Be(a, o, n);
}
function Ct(e, t) {
	let n = t.getExtent(), r = ct(e);
	if (t.canWrapX() && (r[0] < n[0] || r[0] >= n[2])) {
		let t = I(n), i = Math.floor((r[0] - n[0]) / t) * t;
		e[0] -= i, e[2] -= i;
	}
	return e;
}
function wt(e, t, n) {
	if (t.canWrapX()) {
		let r = t.getExtent();
		if (!isFinite(e[0]) || !isFinite(e[2])) return [[
			r[0],
			e[1],
			r[2],
			e[3]
		]];
		Ct(e, t);
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
function Tt(e, t) {
	let n = [e];
	for (let e = 0, r = t.length; e < r && n.length > 0; ++e) {
		let r = [];
		for (let i = 0, a = n.length; i < a; ++i) r.push(...mt(n[i], t[e]));
		n = r;
	}
	return n;
}
// node_modules/ol/console.js
var Et = {
	info: 1,
	warn: 2,
	error: 3,
	none: 4
}, Dt = Et.info;
function Ot(...e) {
	Dt > Et.warn || console.warn(...e);
}
// node_modules/ol/math.js
function kt(e, t, n) {
	return Math.min(Math.max(e, t), n);
}
function At(e, t, n, r, i, a) {
	let o = i - n, s = a - r;
	if (o !== 0 || s !== 0) {
		let c = ((e - n) * o + (t - r) * s) / (o * o + s * s);
		c > 1 ? (n = i, r = a) : c > 0 && (n += o * c, r += s * c);
	}
	return jt(e, t, n, r);
}
function jt(e, t, n, r) {
	let i = n - e, a = r - t;
	return i * i + a * a;
}
function Mt(e) {
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
function Nt(e) {
	return e * 180 / Math.PI;
}
function Pt(e) {
	return e * Math.PI / 180;
}
function Ft(e, t) {
	let n = e % t;
	return n * t < 0 ? n + t : n;
}
function It(e, t, n) {
	return e + n * (t - e);
}
function Lt(e, t) {
	let n = 10 ** t;
	return Math.round(e * n) / n;
}
function Rt(e, t) {
	return Math.floor(Lt(e, t));
}
function zt(e, t) {
	return Math.ceil(Lt(e, t));
}
function Bt(e, t, n) {
	if (e >= t && e < n) return e;
	let r = n - t;
	return ((e - t) % r + r) % r + t;
}
// node_modules/ol/coordinate.js
function Vt(e, t) {
	return e[0] += +t[0], e[1] += +t[1], e;
}
function Ht(e, t, n) {
	return e ? t.replace("{x}", e[0].toFixed(n)).replace("{y}", e[1].toFixed(n)) : "";
}
function Ut(e, t) {
	let n = !0;
	for (let r = e.length - 1; r >= 0; --r) if (e[r] != t[r]) {
		n = !1;
		break;
	}
	return n;
}
function Wt(e, t) {
	let n = Math.cos(t), r = Math.sin(t), i = e[0] * n - e[1] * r, a = e[1] * n + e[0] * r;
	return e[0] = i, e[1] = a, e;
}
function Gt(e, t) {
	return e[0] *= t, e[1] *= t, e;
}
function Kt(e, t) {
	if (t.canWrapX()) {
		let n = I(t.getExtent()), r = Jt(e, t, n);
		r && (e[0] -= r * n);
	}
	return e;
}
function qt(e, t, n) {
	if (t.canWrapX()) {
		let r = I(t.getExtent()), i = Math.round((n - e[0]) / r);
		i !== 0 && (e[0] += i * r);
	}
	return e;
}
function Jt(e, t, n) {
	let r = t.getExtent(), i = 0;
	return t.canWrapX() && (e[0] < r[0] || e[0] > r[2]) && (n ||= I(r), i = Math.floor((e[0] - r[0]) / n)), i;
}
function Yt(e, t, n) {
	let r = Math.sqrt((t[0] - e[0]) * (t[0] - e[0]) + (t[1] - e[1]) * (t[1] - e[1])), i = [(t[0] - e[0]) / r, (t[1] - e[1]) / r], a = [-i[1], i[0]], o = Math.sqrt((n[0] - e[0]) * (n[0] - e[0]) + (n[1] - e[1]) * (n[1] - e[1])), s = [(n[0] - e[0]) / o, (n[1] - e[1]) / o], c = r === 0 || o === 0 ? 0 : Math.acos(kt(s[0] * i[0] + s[1] * i[1], -1, 1));
	return c = Math.max(c, 1e-5), s[0] * a[0] + s[1] * a[1] > 0 ? c : Math.PI * 2 - c;
}
// node_modules/ol/proj/Units.js
var Xt = {
	radians: 6370997 / (2 * Math.PI),
	degrees: 2 * Math.PI * 6370997 / 360,
	ft: .3048,
	m: 1,
	"us-ft": 1200 / 3937
}, Zt = class {
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
		return this.metersPerUnit_ || Xt[this.units_];
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
}, Qt = 6378137, $t = Math.PI * Qt, en = [
	-$t,
	-$t,
	$t,
	$t
], tn = [
	-180,
	-85,
	180,
	85
], nn = Qt * Math.log(Math.tan(Math.PI / 2)), rn = class extends Zt {
	constructor(e) {
		super({
			code: e,
			units: "m",
			extent: en,
			global: !0,
			worldExtent: tn,
			getPointResolution: function(e, t) {
				return e / Math.cosh(t[1] / Qt);
			}
		});
	}
}, an = [
	new rn("EPSG:3857"),
	new rn("EPSG:102100"),
	new rn("EPSG:102113"),
	new rn("EPSG:900913"),
	new rn("http://www.opengis.net/def/crs/EPSG/0/3857"),
	new rn("http://www.opengis.net/gml/srs/epsg.xml#3857")
];
function on(e, t, n, r) {
	let i = e.length;
	n = n !== void 0 && n > 1 ? n : 2, r ??= n, t === void 0 && (t = n > 2 ? e.slice() : Array(i));
	for (let n = 0; n < i; n += r) {
		t[n] = $t * e[n] / 180;
		let r = Qt * Math.log(Math.tan(Math.PI * (+e[n + 1] + 90) / 360));
		r > nn ? r = nn : r < -nn && (r = -nn), t[n + 1] = r;
	}
	return t;
}
function sn(e, t, n, r) {
	let i = e.length;
	n = n !== void 0 && n > 1 ? n : 2, r ??= n, t === void 0 && (t = n > 2 ? e.slice() : Array(i));
	for (let n = 0; n < i; n += r) t[n] = 180 * e[n] / $t, t[n + 1] = 360 * Math.atan(Math.exp(e[n + 1] / Qt)) / Math.PI - 90;
	return t;
}
// node_modules/ol/proj/epsg4326.js
var cn = 6378137, ln = [
	-180,
	-90,
	180,
	90
], un = Math.PI * cn / 180, dn = class extends Zt {
	constructor(e, t) {
		super({
			code: e,
			units: "degrees",
			extent: ln,
			axisOrientation: t,
			global: !0,
			metersPerUnit: un,
			worldExtent: ln
		});
	}
}, fn = [
	new dn("CRS:84"),
	new dn("EPSG:4326", "neu"),
	new dn("urn:ogc:def:crs:OGC:1.3:CRS84"),
	new dn("urn:ogc:def:crs:OGC:2:84"),
	new dn("http://www.opengis.net/def/crs/OGC/1.3/CRS84"),
	new dn("http://www.opengis.net/gml/srs/epsg.xml#4326", "neu"),
	new dn("http://www.opengis.net/def/crs/EPSG/0/4326", "neu")
], pn = {};
function mn(e) {
	return pn[e] || pn[e.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\w+)$/, "EPSG:$3")] || null;
}
function hn(e, t) {
	pn[e] = t;
}
// node_modules/ol/proj/transforms.js
var gn = {};
function _n(e, t, n) {
	let r = e.getCode(), i = t.getCode();
	r in gn || (gn[r] = {}), gn[r][i] = n;
}
function vn(e, t) {
	return e in gn && t in gn[e] ? gn[e][t] : null;
}
// node_modules/ol/proj/utm.js
var yn = .9996, bn = .00669438, xn = bn * bn, Sn = xn * bn, Cn = bn / .99330562, wn = Math.sqrt(.99330562), Tn = (1 - wn) / (1 + wn), En = Tn * Tn, Dn = En * Tn, On = Dn * Tn, kn = On * Tn, An = 1 - bn / 4 - 3 * xn / 64 - 5 * Sn / 256, jn = .002514607064228144, Mn = 26390466021299826e-22, Nn = 35 * Sn / 3072, Pn = 3 / 2 * Tn - 27 / 32 * Dn + 269 / 512 * kn, Fn = 21 / 16 * En - 55 / 32 * On, In = 151 / 96 * Dn - 417 / 128 * kn, Ln = 1097 / 512 * On, Rn = 6378137;
function zn(e, t, n) {
	let r = e - 5e5, i = (n.north ? t : t - 1e7) / yn / (Rn * An), a = i + Pn * Math.sin(2 * i) + Fn * Math.sin(4 * i) + In * Math.sin(6 * i) + Ln * Math.sin(8 * i), o = Math.sin(a), s = o * o, c = Math.cos(a), l = o / c, u = l * l, d = u * u, f = 1 - bn * s, p = Rn / Math.sqrt(1 - bn * s), m = .99330562 / f, h = Cn * c ** 2, g = h * h, _ = r / (p * yn), v = _ * _, y = v * _, b = y * _, x = b * _, S = x * _, C = a - l / m * (v / 2 - b / 24 * (5 + 3 * u + 10 * h - 4 * g - 9 * Cn)) + S / 720 * (61 + 90 * u + 298 * h + 45 * d - 252 * Cn - 3 * g), w = (_ - y / 6 * (1 + 2 * u + h) + x / 120 * (5 - 2 * h + 28 * u - 3 * g + 8 * Cn + 24 * d)) / c;
	return w = Bt(w + Pt(Gn(n.number)), -Math.PI, Math.PI), [Nt(w), Nt(C)];
}
var Bn = -80, Vn = 84, Hn = -180, Un = 180;
function Wn(e, t, n) {
	e = Bt(e, Hn, Un), t < Bn ? t = Bn : t > Vn && (t = Vn);
	let r = Pt(t), i = Math.sin(r), a = Math.cos(r), o = i / a, s = o * o, c = s * s, l = Pt(e), u = Pt(Gn(n.number)), d = Rn / Math.sqrt(1 - bn * i ** 2), f = Cn * a ** 2, p = a * Bt(l - u, -Math.PI, Math.PI), m = p * p, h = m * p, g = h * p, _ = g * p, v = _ * p, y = Rn * (An * r - jn * Math.sin(2 * r) + Mn * Math.sin(4 * r) - Nn * Math.sin(6 * r)), b = yn * d * (p + h / 6 * (1 - s + f) + _ / 120 * (5 - 18 * s + c + 72 * f - 58 * Cn)) + 5e5, x = yn * (y + d * o * (m / 2 + g / 24 * (5 - s + 9 * f + 4 * f ** 2) + v / 720 * (61 - 58 * s + c + 600 * f - 330 * Cn)));
	return n.north || (x += 1e7), [b, x];
}
function Gn(e) {
	return (e - 1) * 6 - 180 + 3;
}
var Kn = [
	/^EPSG:(\d+)$/,
	/^urn:ogc:def:crs:EPSG::(\d+)$/,
	/^http:\/\/www\.opengis\.net\/def\/crs\/EPSG\/0\/(\d+)$/
];
function qn(e) {
	let t = 0;
	for (let n of Kn) {
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
function Jn(e, t) {
	return function(n, r, i, a) {
		let o = n.length;
		i = i !== void 0 && i > 1 ? i : 2, a ??= i, r ||= i > 2 ? n.slice() : Array(o);
		for (let i = 0; i < o; i += a) {
			let a = n[i], o = n[i + 1], s = e(a, o, t);
			r[i] = s[0], r[i + 1] = s[1];
		}
		return r;
	};
}
function Yn(e) {
	return qn(e) ? new Zt({
		code: e,
		units: "m"
	}) : null;
}
function Xn(e) {
	let t = qn(e.getCode());
	return t ? {
		forward: Jn(Wn, t),
		inverse: Jn(zn, t)
	} : null;
}
function Zn(e, t, n) {
	n ||= 6371008.8;
	let r = Pt(e[1]), i = Pt(t[1]), a = (i - r) / 2, o = Pt(t[0] - e[0]) / 2, s = Math.sin(a) * Math.sin(a) + Math.sin(o) * Math.sin(o) * Math.cos(r) * Math.cos(i);
	return 2 * n * Math.atan2(Math.sqrt(s), Math.sqrt(1 - s));
}
// node_modules/ol/proj.js
var Qn = [Xn], $n = [Yn], er = !0;
function tr(e) {
	er = !(e === void 0 || e);
}
function nr(e, t) {
	if (t !== void 0) {
		for (let n = 0, r = e.length; n < r; ++n) t[n] = e[n];
		t = t;
	} else t = e.slice();
	return t;
}
function rr(e, t) {
	if (t !== void 0 && e !== t) {
		for (let n = 0, r = e.length; n < r; ++n) t[n] = e[n];
		e = t;
	}
	return e;
}
function ir(e) {
	hn(e.getCode(), e), _n(e, e, nr);
}
function ar(e) {
	e.forEach(ir);
}
function or(e) {
	if (typeof e != "string") return e === void 0 ? null : e;
	let t = mn(e);
	if (t) return t;
	for (let t of $n) {
		let n = t(e);
		if (n) return n;
	}
	return null;
}
function sr(e, t, n, r) {
	let i = or(e);
	if (!i) return t;
	let a, o = i.getPointResolutionFunc();
	if (o) {
		if (a = o(t, n), r && r !== i.getUnits()) {
			let e = i.getMetersPerUnit();
			e && (a = a * e / Xt[r]);
		}
	} else {
		let e = i.getUnits();
		if (e == "degrees" && !r || r == "degrees") a = t;
		else {
			let o = pr(i, or("EPSG:4326"));
			if (!o && e !== "degrees") a = t * (i.getMetersPerUnit() ?? 1);
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
				o && (e = o(e, e, 2)), a = (Zn(e.slice(0, 2), e.slice(2, 4)) + Zn(e.slice(4, 6), e.slice(6, 8))) / 2;
			}
			let s = r ? Xt[r] : i.getMetersPerUnit();
			s !== void 0 && (a /= s);
		}
	}
	return a;
}
function cr(e) {
	ar(e), e.forEach(function(t) {
		e.forEach(function(e) {
			t !== e && _n(t, e, nr);
		});
	});
}
function lr(e, t, n, r) {
	e.forEach(function(e) {
		t.forEach(function(t) {
			_n(e, t, n), _n(t, e, r);
		});
	});
}
function ur(e, t) {
	return e ? typeof e == "string" ? or(e) : e : or(t);
}
function dr(e) {
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
function fr(e, t) {
	if (e === t) return !0;
	let n = e.getUnits() === t.getUnits();
	return (e.getCode() === t.getCode() || pr(e, t) === nr) && n;
}
function pr(e, t) {
	if (!e || !t) return null;
	let n = e.getCode(), r = t.getCode(), i = vn(n, r);
	if (i) return i;
	let a = null, o = null;
	for (let n of Qn) a ||= n(e), o ||= n(t);
	if (!a && !o) return null;
	let s = "EPSG:4326";
	if (!o && a) {
		let e = vn(s, r);
		e && (i = mr(a.inverse, e));
	} else if (!a && o) {
		let e = vn(n, s);
		e && (i = mr(e, o.forward));
	} else a && o && (i = mr(a.inverse, o.forward));
	return i && (ir(e), ir(t), _n(e, t, i)), i;
}
function mr(e, t) {
	return function(n, r, i, a) {
		return r = e(n, r, i, a), t(r, r, i, a);
	};
}
function hr(e, t) {
	return pr(or(e), or(t));
}
function gr(e, t, n) {
	let r = hr(t, n);
	if (!r) {
		let e = or(t)?.getCode() ?? String(t), r = or(n)?.getCode() ?? String(n);
		throw Error(`No transform available between ${e} and ${r}`);
	}
	return r(e, void 0, e.length);
}
function _r(e, t, n, r) {
	let i = hr(t, n);
	return i ? St(e, i, void 0, r) : e.slice();
}
var vr = null;
function yr() {
	return vr;
}
function br(e, t) {
	return e;
}
function xr(e, t) {
	return er && !Ut(e, [0, 0]) && e[0] >= -180 && e[0] <= 180 && e[1] >= -90 && e[1] <= 90 && (er = !1, Ot("Call useGeographic() from ol/proj once to work with [longitude, latitude] coordinates.")), e;
}
function Sr(e, t) {
	return e;
}
function Cr(e, t) {
	return e;
}
function wr(e, t) {
	return e;
}
function Tr() {
	cr(an), cr(fn), lr(fn, an, on, sn);
}
Tr();
// node_modules/ol/asserts.js
function L(e, t) {
	if (!e) throw Error(t);
}
// node_modules/ol/transform.js
var Er = [
	1,
	0,
	0,
	1,
	0,
	0
], Dr = [
	,
	,
	,
	,
	,
	,
];
function Or() {
	return Er.slice(0);
}
function kr(e) {
	return jr(e, 1, 0, 0, 1, 0, 0);
}
function Ar(e, t) {
	let n = e[0], r = e[1], i = e[2], a = e[3], o = e[4], s = e[5], c = t[0], l = t[1], u = t[2], d = t[3], f = t[4], p = t[5];
	return e[0] = n * c + i * l, e[1] = r * c + a * l, e[2] = n * u + i * d, e[3] = r * u + a * d, e[4] = n * f + i * p + o, e[5] = r * f + a * p + s, e;
}
function jr(e, t, n, r, i, a, o) {
	return e[0] = t, e[1] = n, e[2] = r, e[3] = i, e[4] = a, e[5] = o, e;
}
function Mr(e, t) {
	return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e;
}
function R(e, t) {
	let n = t[0], r = t[1];
	return t[0] = e[0] * n + e[2] * r + e[4], t[1] = e[1] * n + e[3] * r + e[5], t;
}
function Nr(e, t, n) {
	return Ar(e, jr(Dr, t, 0, 0, n, 0, 0));
}
function Pr(e, t, n) {
	return Ar(e, jr(Dr, 1, 0, 0, 1, t, n));
}
function Fr(e, t, n, r, i, a, o, s) {
	let c = Math.sin(a), l = Math.cos(a);
	return e[0] = r * l, e[1] = i * c, e[2] = -r * c, e[3] = i * l, e[4] = o * r * l - s * r * c + t, e[5] = o * i * c + s * i * l + n, e;
}
function Ir(e, t) {
	let n = Lr(t);
	L(n !== 0, "Transformation matrix cannot be inverted");
	let r = t[0], i = t[1], a = t[2], o = t[3], s = t[4], c = t[5];
	return e[0] = o / n, e[1] = -i / n, e[2] = -a / n, e[3] = r / n, e[4] = (a * c - o * s) / n, e[5] = -(r * c - i * s) / n, e;
}
function Lr(e) {
	return e[0] * e[3] - e[1] * e[2];
}
var Rr = [
	1e5,
	1e5,
	1e5,
	1e5,
	2,
	2
];
function zr(e) {
	return "matrix(" + e.join(", ") + ")";
}
function Br(e) {
	return e.substring(7, e.length - 1).split(",").map(parseFloat);
}
function Vr(e, t) {
	let n = Br(e), r = Br(t);
	for (let e = 0; e < 6; ++e) if (Math.round((n[e] - r[e]) * Rr[e]) !== 0) return !1;
	return !0;
}
// node_modules/ol/geom/flat/transform.js
function Hr(e, t, n, r, i, a, o) {
	a ||= [], o ||= 2;
	let s = 0;
	for (let c = t; c < n; c += r) {
		let t = e[c], n = e[c + 1];
		a[s++] = i[0] * t + i[2] * n + i[4], a[s++] = i[1] * t + i[3] * n + i[5];
		for (let t = 2; t < o; t++) a[s++] = e[c + t];
	}
	return a && a.length != s && (a.length = s), a;
}
function Ur(e, t, n, r, i, a, o) {
	o ||= [];
	let s = Math.cos(i), c = Math.sin(i), l = a[0], u = a[1], d = 0;
	for (let i = t; i < n; i += r) {
		let t = e[i] - l, n = e[i + 1] - u;
		o[d++] = l + t * s - n * c, o[d++] = u + t * c + n * s;
		for (let t = i + 2; t < i + r; ++t) o[d++] = e[t];
	}
	return o && o.length != d && (o.length = d), o;
}
function Wr(e, t, n, r, i, a, o, s) {
	s ||= [];
	let c = o[0], l = o[1], u = 0;
	for (let o = t; o < n; o += r) {
		let t = e[o] - c, n = e[o + 1] - l;
		s[u++] = c + i * t, s[u++] = l + a * n;
		for (let t = o + 2; t < o + r; ++t) s[u++] = e[t];
	}
	return s && s.length != u && (s.length = u), s;
}
function Gr(e, t, n, r, i, a, o) {
	o ||= [];
	let s = 0;
	for (let c = t; c < n; c += r) {
		o[s++] = e[c] + i, o[s++] = e[c + 1] + a;
		for (let t = c + 2; t < c + r; ++t) o[s++] = e[t];
	}
	return o && o.length != s && (o.length = s), o;
}
// node_modules/ol/geom/Geometry.js
var Kr = Or(), qr = [NaN, NaN], Jr = class extends j {
	constructor() {
		super(), this.extent_ = Je(), this.extentRevision_ = -1, this.simplifiedGeometryMaxMinSquaredTolerance = 0, this.simplifiedGeometryRevision = 0, this.simplifyTransformedInternal = w((e, t, n) => {
			if (!n) return this.getSimplifiedGeometry(t);
			let r = this.clone();
			return r.applyTransform(n), r.getSimplifiedGeometry(t);
		});
	}
	simplifyTransformed(e, t) {
		return this.simplifyTransformedInternal(this.getRevision(), e, t);
	}
	clone() {
		return _();
	}
	closestPointXY(e, t, n, r) {
		return _();
	}
	containsXY(e, t) {
		return this.closestPointXY(e, t, qr, Number.MIN_VALUE) === 0;
	}
	getClosestPoint(e, t) {
		return t ||= [NaN, NaN], this.closestPointXY(e[0], e[1], t, Infinity), t;
	}
	intersectsCoordinate(e) {
		return this.containsXY(e[0], e[1]);
	}
	computeExtent(e) {
		return _();
	}
	getExtent(e) {
		if (this.extentRevision_ != this.getRevision()) {
			let e = this.computeExtent(this.extent_);
			(isNaN(e[0]) || isNaN(e[1])) && Xe(e), this.extentRevision_ = this.getRevision();
		}
		return yt(this.extent_, e);
	}
	rotate(e, t) {
		_();
	}
	scale(e, t, n) {
		_();
	}
	simplify(e) {
		return this.getSimplifiedGeometry(e * e);
	}
	getSimplifiedGeometry(e) {
		return _();
	}
	getType() {
		return _();
	}
	applyTransform(e) {
		_();
	}
	intersectsExtent(e) {
		return _();
	}
	translate(e, t) {
		_();
	}
	transform(e, t) {
		let n = or(e);
		if (!n) return this;
		let r = n.getUnits() == "tile-pixels" ? function(e, r, i) {
			let a = n.getExtent(), o = n.getWorldExtent(), s = ft(o) / ft(a);
			Fr(Kr, o[0], o[3], s, -s, 0, 0, 0);
			let c = Hr(e, 0, e.length, i, Kr, r), l = hr(n, t);
			return l ? l(c, c, i) : c;
		} : hr(n, t);
		return r && this.applyTransform(r), this;
	}
}, Yr = class extends Jr {
	constructor() {
		super(), this.layout = "XY", this.stride = 2, this.flatCoordinates;
	}
	computeExtent(e) {
		return Qe(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	getCoordinates() {
		return _();
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
		this.stride = Zr(e), this.layout = e, this.flatCoordinates = t;
	}
	setCoordinates(e, t) {
		_();
	}
	setLayout(e, t, n) {
		let r;
		if (e) r = Zr(e);
		else {
			for (let e = 0; e < n; ++e) {
				if (t.length === 0) {
					this.layout = "XY", this.stride = 2;
					return;
				}
				t = t[0];
			}
			r = t.length, e = Xr(r);
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
			Ur(n, 0, n.length, r, e, t, n), this.changed();
		}
	}
	scale(e, t, n) {
		t === void 0 && (t = e), n ||= ct(this.getExtent());
		let r = this.getFlatCoordinates();
		if (r) {
			let i = this.getStride();
			Wr(r, 0, r.length, i, e, t, n, r), this.changed();
		}
	}
	translate(e, t) {
		let n = this.getFlatCoordinates();
		if (n) {
			let r = this.getStride();
			Gr(n, 0, n.length, r, e, t, n), this.changed();
		}
	}
};
function Xr(e) {
	let t;
	return e == 2 ? t = "XY" : e == 3 ? t = "XYZ" : e == 4 && (t = "XYZM"), t;
}
function Zr(e) {
	let t;
	return e == "XY" ? t = 2 : e == "XYZ" || e == "XYM" ? t = 3 : e == "XYZM" && (t = 4), t;
}
function Qr(e, t, n) {
	let r = e.getFlatCoordinates();
	if (!r) return null;
	let i = e.getStride();
	return Hr(r, 0, r.length, i, t, n);
}
// node_modules/ol/geom/flat/deflate.js
function $r(e, t, n, r) {
	for (let i = 0; i < r; ++i) e[t++] = n[i];
	return t;
}
function ei(e, t, n, r) {
	for (let i = 0, a = n.length; i < a; ++i) {
		let a = n[i];
		for (let n = 0; n < r; ++n) e[t++] = a[n];
	}
	return t;
}
function ti(e, t, n, r, i) {
	i ||= [];
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = ei(e, t, n[o], r);
		i[a++] = s, t = s;
	}
	return i.length = a, i;
}
function ni(e, t, n, r, i) {
	i ||= [];
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = ti(e, t, n[o], r, i[a]);
		s.length === 0 && (s[0] = t), i[a++] = s, t = s[s.length - 1];
	}
	return i.length = a, i;
}
// node_modules/ol/Feature.js
var ri = class e extends j {
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
		this.style_ = e ?? null, this.styleFunction_ = e ? ii(e) : void 0, this.changed();
	}
	setId(e) {
		this.id_ = e, this.changed();
	}
	setGeometryName(e) {
		e !== this.geometryName_ && (this.removeChangeListener(this.geometryName_, this.handleGeometryChanged_), this.geometryName_ = e, this.addChangeListener(this.geometryName_, this.handleGeometryChanged_), this.handleGeometryChanged_());
	}
};
function ii(e) {
	if (typeof e == "function") return e;
	let t;
	return Array.isArray(e) ? t = e : (L(typeof e.getZIndex == "function", "Expected an `ol/style/Style` or an array of `ol/style/Style.js`"), t = [e]), function() {
		return t;
	};
}
// node_modules/ol/geom/flat/closest.js
function ai(e, t, n, r, i, a, o) {
	let s = e[t], c = e[t + 1], l = e[n] - s, u = e[n + 1] - c, d;
	if (l === 0 && u === 0) d = t;
	else {
		let f = ((i - s) * l + (a - c) * u) / (l * l + u * u);
		if (f > 1) d = n;
		else if (f > 0) {
			for (let i = 0; i < r; ++i) o[i] = It(e[t + i], e[n + i], f);
			o.length = r;
			return;
		} else d = t;
	}
	for (let t = 0; t < r; ++t) o[t] = e[d + t];
	o.length = r;
}
function oi(e, t, n, r, i) {
	let a = e[t], o = e[t + 1];
	for (t += r; t < n; t += r) {
		let n = e[t], r = e[t + 1], s = jt(a, o, n, r);
		s > i && (i = s), a = n, o = r;
	}
	return i;
}
function si(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i = oi(e, t, o, r, i), t = o;
	}
	return i;
}
function ci(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i = si(e, t, o, r, i), t = o[o.length - 1];
	}
	return i;
}
function li(e, t, n, r, i, a, o, s, c, l, u) {
	if (t == n) return l;
	let d, f;
	if (i === 0) {
		if (f = jt(o, s, e[t], e[t + 1]), f < l) {
			for (d = 0; d < r; ++d) c[d] = e[t + d];
			return c.length = r, f;
		}
		return l;
	}
	u ||= [NaN, NaN];
	let p = t + r;
	for (; p < n;) if (ai(e, p - r, p, r, o, s, u), f = jt(o, s, u[0], u[1]), f < l) {
		for (l = f, d = 0; d < r; ++d) c[d] = u[d];
		c.length = r, p += r;
	} else p += r * Math.max((Math.sqrt(f) - Math.sqrt(l)) / i | 0, 1);
	if (a && (ai(e, n - r, t, r, o, s, u), f = jt(o, s, u[0], u[1]), f < l)) {
		for (l = f, d = 0; d < r; ++d) c[d] = u[d];
		c.length = r;
	}
	return l;
}
function ui(e, t, n, r, i, a, o, s, c, l, u) {
	u ||= [NaN, NaN];
	for (let d = 0, f = n.length; d < f; ++d) {
		let f = n[d];
		l = li(e, t, f, r, i, a, o, s, c, l, u), t = f;
	}
	return l;
}
function di(e, t, n, r, i, a, o, s, c, l, u) {
	u ||= [NaN, NaN];
	for (let d = 0, f = n.length; d < f; ++d) {
		let f = n[d];
		l = ui(e, t, f, r, i, a, o, s, c, l, u), t = f[f.length - 1];
	}
	return l;
}
// node_modules/ol/geom/flat/inflate.js
function fi(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = t; o < n; o += r) i[a++] = e.slice(o, o + r);
	return i.length = a, i;
}
function pi(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		i[a++] = fi(e, t, s, r, i[a]), t = s;
	}
	return i.length = a, i;
}
function mi(e, t, n, r, i) {
	i = i === void 0 ? [] : i;
	let a = 0;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		i[a++] = s.length === 1 && s[0] === t ? [] : pi(e, t, s, r, i[a]), t = s[s.length - 1];
	}
	return i.length = a, i;
}
// node_modules/ol/geom/flat/interpolate.js
function hi(e, t, n, r, i, a, o) {
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
	o = o !== void 0 && o > 1 ? o : 2, a ||= Array(o);
	for (let t = 0; t < o; ++t) a[t] = s === void 0 ? NaN : c === void 0 ? e[s + t] : It(e[s + t], e[s + r + t], c);
	return a;
}
function gi(e, t, n, r, i, a) {
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
	for (let t = 0; t < r - 1; ++t) o.push(It(e[(s - 1) * r + t], e[s * r + t], d));
	return o.push(i), o;
}
function _i(e, t, n, r, i, a, o) {
	if (o) return gi(e, t, n[n.length - 1], r, i, a);
	let s;
	if (i < e[r - 1]) return a ? (s = e.slice(0, r), s[r - 1] = i, s) : null;
	if (e[e.length - 1] < i) return a ? (s = e.slice(e.length - r), s[r - 1] = i, s) : null;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		if (t != o) {
			if (i < e[t + r - 1]) return null;
			if (i <= e[o - 1]) return gi(e, t, o, r, i, !1);
			t = o;
		}
	}
	return null;
}
// node_modules/ol/geom/flat/contains.js
function vi(e, t, n, r, i) {
	return !it(i, function(i) {
		return !yi(e, t, n, r, i[0], i[1]);
	});
}
function yi(e, t, n, r, i, a) {
	let o = 0, s = e[n - r], c = e[n - r + 1];
	for (; t < n; t += r) {
		let n = e[t], r = e[t + 1];
		c <= a ? r > a && (n - s) * (a - c) - (i - s) * (r - c) > 0 && o++ : r <= a && (n - s) * (a - c) - (i - s) * (r - c) < 0 && o--, s = n, c = r;
	}
	return o !== 0;
}
function bi(e, t, n, r, i, a) {
	if (n.length === 0 || !yi(e, t, n[0], r, i, a)) return !1;
	for (let t = 1, o = n.length; t < o; ++t) if (yi(e, n[t - 1], n[t], r, i, a)) return !1;
	return !0;
}
function xi(e, t, n, r, i, a) {
	if (n.length === 0) return !1;
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		if (bi(e, t, s, r, i, a)) return !0;
		t = s[s.length - 1];
	}
	return !1;
}
// node_modules/ol/geom/flat/segments.js
function Si(e, t, n, r, i) {
	let a;
	for (t += r; t < n; t += r) if (a = i(e.slice(t - r, t), e.slice(t, t + r)), a) return a;
	return !1;
}
// node_modules/ol/geom/flat/intersectsextent.js
function Ci(e, t, n, r, i, a) {
	return a ??= nt(Je(), e, t, n, r), _t(i, a) ? a[0] >= i[0] && a[2] <= i[2] || a[1] >= i[1] && a[3] <= i[3] || Si(e, t, n, r, function(e, t) {
		return xt(i, e, t);
	}) : !1;
}
function wi(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		if (Ci(e, t, n[a], r, i)) return !0;
		t = n[a];
	}
	return !1;
}
function Ti(e, t, n, r, i) {
	return !!(Ci(e, t, n, r, i) || yi(e, t, n, r, i[0], i[1]) || yi(e, t, n, r, i[0], i[3]) || yi(e, t, n, r, i[2], i[1]) || yi(e, t, n, r, i[2], i[3]));
}
function Ei(e, t, n, r, i) {
	if (!Ti(e, t, n[0], r, i)) return !1;
	if (n.length === 1) return !0;
	for (let t = 1, a = n.length; t < a; ++t) if (vi(e, n[t - 1], n[t], r, i) && !Ci(e, n[t - 1], n[t], r, i)) return !1;
	return !0;
}
function Di(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		if (Ei(e, t, o, r, i)) return !0;
		t = o[o.length - 1];
	}
	return !1;
}
// node_modules/ol/geom/flat/length.js
function Oi(e, t, n, r) {
	let i = e[t], a = e[t + 1], o = 0;
	for (let s = t + r; s < n; s += r) {
		let t = e[s], n = e[s + 1];
		o += Math.sqrt((t - i) * (t - i) + (n - a) * (n - a)), i = t, a = n;
	}
	return o;
}
// node_modules/ol/geom/flat/simplify.js
function ki(e, t, n, r, i, a, o) {
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
			let n = e[t], r = e[t + 1], i = At(n, r, s, d, f, p);
			i > o && (u = t, o = i);
		}
		o > i && (c[(u - t) / r] = 1, a + r < u && l.push(a, u), u + r < n && l.push(u, n));
	}
	for (let n = 0; n < s; ++n) c[n] && (a[o++] = e[t + n * r], a[o++] = e[t + n * r + 1]);
	return o;
}
function Ai(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c];
		o = ki(e, t, l, r, i, a, o), s.push(o), t = l;
	}
	return o;
}
function ji(e, t) {
	return t * Math.round(e / t);
}
function Mi(e, t, n, r, i, a, o) {
	if (t == n) return o;
	let s = ji(e[t], i), c = ji(e[t + 1], i);
	t += r, a[o++] = s, a[o++] = c;
	let l, u;
	do
		if (l = ji(e[t], i), u = ji(e[t + 1], i), t += r, t == n) return a[o++] = l, a[o++] = u, o;
	while (l == s && u == c);
	for (; t < n;) {
		let n = ji(e[t], i), d = ji(e[t + 1], i);
		if (t += r, n == l && d == u) continue;
		let f = l - s, p = u - c, m = n - s, h = d - c;
		f * h == p * m && (f < 0 && m < f || f == m || f > 0 && m > f) && (p < 0 && h < p || p == h || p > 0 && h > p) ? (l = n, u = d) : (a[o++] = l, a[o++] = u, s = l, c = u, l = n, u = d);
	}
	return a[o++] = l, a[o++] = u, o;
}
function Ni(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c];
		o = Mi(e, t, l, r, i, a, o), s.push(o), t = l;
	}
	return o;
}
function Pi(e, t, n, r, i, a, o, s) {
	for (let c = 0, l = n.length; c < l; ++c) {
		let l = n[c], u = [];
		o = Ni(e, t, l, r, i, a, o, u), s.push(u), t = l[l.length - 1];
	}
	return o;
}
// node_modules/ol/geom/LineString.js
var Fi = class e extends Yr {
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
		return r < Ue(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(oi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), li(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !1, e, t, n, r));
	}
	forEachSegment(e) {
		return Si(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	getCoordinateAtM(e, t) {
		return this.layout != "XYM" && this.layout != "XYZM" ? null : (t = t !== void 0 && t, gi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, t));
	}
	getCoordinates() {
		return fi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getCoordinateAt(e, t) {
		return hi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, t, this.stride);
	}
	getLength() {
		return Oi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getFlatMidpoint() {
		return this.flatMidpointRevision_ != this.getRevision() && (this.flatMidpoint_ = this.getCoordinateAt(.5, this.flatMidpoint_ ?? void 0), this.flatMidpointRevision_ = this.getRevision()), this.flatMidpoint_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [];
		return n.length = ki(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, t, n, 0), new e(n, "XY");
	}
	getType() {
		return "LineString";
	}
	intersectsExtent(e) {
		return Ci(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, this.getExtent());
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = ei(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
}, Ii = class e extends Yr {
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
		return r < Ue(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(si(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), ui(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !1, e, t, n, r));
	}
	getCoordinateAtM(e, t, n) {
		return this.layout != "XYM" && this.layout != "XYZM" || this.flatCoordinates.length === 0 ? null : (t = t !== void 0 && t, n = n !== void 0 && n, _i(this.flatCoordinates, 0, this.ends_, this.stride, e, t, n));
	}
	getCoordinates() {
		return pi(this.flatCoordinates, 0, this.ends_, this.stride);
	}
	getEnds() {
		return this.ends_;
	}
	getLineString(e) {
		return e < 0 || this.ends_.length <= e ? null : new Fi(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}
	getLineStrings() {
		let e = this.flatCoordinates, t = this.ends_, n = this.layout, r = [], i = 0;
		for (let a = 0, o = t.length; a < o; ++a) {
			let o = t[a], s = new Fi(e.slice(i, o), n);
			r.push(s), i = o;
		}
		return r;
	}
	getLength() {
		let e = this.ends_, t = 0, n = 0;
		for (let r = 0, i = e.length; r < i; ++r) n += Oi(this.flatCoordinates, t, e[r], this.stride), t = e[r];
		return n;
	}
	getFlatMidpoints() {
		let e = [], t = this.flatCoordinates, n = 0, r = this.ends_, i = this.stride;
		for (let a = 0, o = r.length; a < o; ++a) {
			let o = r[a];
			m(e, hi(t, n, o, i, .5)), n = o;
		}
		return e;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = Ai(this.flatCoordinates, 0, this.ends_, this.stride, t, n, 0, r), new e(n, "XY", r);
	}
	getType() {
		return "MultiLineString";
	}
	intersectsExtent(e) {
		return wi(this.flatCoordinates, 0, this.ends_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		let n = ti(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}
}, Li = class e extends Yr {
	constructor(e, t) {
		super(), this.setCoordinates(e, t);
	}
	clone() {
		let t = new e(this.flatCoordinates.slice(), this.layout);
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		let i = this.flatCoordinates, a = jt(e, t, i[0], i[1]);
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
		return Ze(this.flatCoordinates, e);
	}
	getType() {
		return "Point";
	}
	intersectsExtent(e) {
		return Ke(e, this.flatCoordinates[0], this.flatCoordinates[1]);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 0), this.flatCoordinates ||= [], this.flatCoordinates.length = $r(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
}, Ri = class e extends Yr {
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
		if (r < Ue(this.getExtent(), e, t)) return r;
		let i = this.flatCoordinates, a = this.stride;
		for (let o = 0, s = i.length; o < s; o += a) {
			let s = jt(e, t, i[o], i[o + 1]);
			if (s < r) {
				r = s;
				for (let e = 0; e < a; ++e) n[e] = i[o + e];
				n.length = a;
			}
		}
		return r;
	}
	getCoordinates() {
		return fi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getPoint(e) {
		let t = this.flatCoordinates.length / this.stride;
		return e < 0 || t <= e ? null : new Li(this.flatCoordinates.slice(e * this.stride, (e + 1) * this.stride), this.layout);
	}
	getPoints() {
		let e = this.flatCoordinates, t = this.layout, n = this.stride, r = [];
		for (let i = 0, a = e.length; i < a; i += n) {
			let a = new Li(e.slice(i, i + n), t);
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
			if (Ke(e, n, i)) return !0;
		}
		return !1;
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = ei(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
};
// node_modules/ol/geom/flat/area.js
function zi(e, t, n, r) {
	let i = 0, a = e[n - r], o = e[n - r + 1], s = 0, c = 0;
	for (; t < n; t += r) {
		let n = e[t] - a, r = e[t + 1] - o;
		i += c * n - s * r, s = n, c = r;
	}
	return i / 2;
}
function Bi(e, t, n, r) {
	let i = 0;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i += zi(e, t, o, r), t = o;
	}
	return i;
}
function Vi(e, t, n, r) {
	let i = 0;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		i += Bi(e, t, o, r), t = o[o.length - 1];
	}
	return i;
}
// node_modules/ol/geom/LinearRing.js
var Hi = class e extends Yr {
	constructor(e, t) {
		super(), this.maxDelta_ = -1, this.maxDeltaRevision_ = -1, t !== void 0 && !Array.isArray(e[0]) ? this.setFlatCoordinates(t, e) : this.setCoordinates(e, t);
	}
	clone() {
		return new e(this.flatCoordinates.slice(), this.layout);
	}
	closestPointXY(e, t, n, r) {
		return r < Ue(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(oi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), li(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	getArea() {
		return zi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getCoordinates() {
		return fi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}
	getSimplifiedGeometryInternal(t) {
		let n = [];
		return n.length = ki(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, t, n, 0), new e(n, "XY");
	}
	getType() {
		return "LinearRing";
	}
	intersectsExtent(e) {
		return Ci(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = ei(this.flatCoordinates, 0, e, this.stride), this.changed();
	}
};
// node_modules/ol/geom/flat/interiorpoint.js
function Ui(e, t, n, r, i, a, o) {
	let s, c, l, d, f, p, m, h = i[a + 1], g = [];
	for (let i = 0, a = n.length; i < a; ++i) {
		let a = n[i];
		for (d = e[a - r], p = e[a - r + 1], s = t; s < a; s += r) f = e[s], m = e[s + 1], (h <= p && m <= h || p <= h && h <= m) && (l = (h - p) / (m - p) * (f - d) + d, g.push(l)), d = f, p = m;
	}
	let _ = NaN, v = -Infinity;
	for (g.sort(u), d = g[0], s = 1, c = g.length; s < c; ++s) {
		f = g[s];
		let i = Math.abs(f - d);
		i > v && (l = (d + f) / 2, bi(e, t, n, r, l, h) && (_ = l, v = i)), d = f;
	}
	return isNaN(_) && (_ = i[a]), o ? (o.push(_, h, v), o) : [
		_,
		h,
		v
	];
}
function Wi(e, t, n, r, i) {
	let a = [];
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		a = Ui(e, t, s, r, i, 2 * o, a), t = s[s.length - 1];
	}
	return a;
}
// node_modules/ol/geom/flat/reverse.js
function Gi(e, t, n, r) {
	for (; t < n - r;) {
		for (let i = 0; i < r; ++i) {
			let a = e[t + i];
			e[t + i] = e[n - r + i], e[n - r + i] = a;
		}
		t += r, n -= r;
	}
}
// node_modules/ol/geom/flat/orient.js
function Ki(e, t, n, r) {
	let i = 0, a = e[n - r], o = e[n - r + 1];
	for (; t < n; t += r) {
		let n = e[t], r = e[t + 1];
		i += (n - a) * (r + o), a = n, o = r;
	}
	return i === 0 ? void 0 : i > 0;
}
function qi(e, t, n, r, i) {
	i = i !== void 0 && i;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a], s = Ki(e, t, o, r);
		if (a === 0) {
			if (i && s || !i && !s) return !1;
		} else if (i && !s || !i && s) return !1;
		t = o;
	}
	return !0;
}
function Ji(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a];
		if (!qi(e, t, o, r, i)) return !1;
		o.length && (t = o[o.length - 1]);
	}
	return !0;
}
function Yi(e, t, n, r, i) {
	i = i !== void 0 && i;
	for (let a = 0, o = n.length; a < o; ++a) {
		let o = n[a], s = Ki(e, t, o, r);
		(a === 0 ? i && s || !i && !s : i && !s || !i && s) && Gi(e, t, o, r), t = o;
	}
	return t;
}
function Xi(e, t, n, r, i) {
	for (let a = 0, o = n.length; a < o; ++a) t = Yi(e, t, n[a], r, i);
	return t;
}
function Zi(e, t) {
	let n = [], r = 0, i = 0, a;
	for (let o = 0, s = t.length; o < s; ++o) {
		let s = t[o], c = Ki(e, r, s, 2);
		if (a === void 0 && (a = c), c === a) n.push(t.slice(i, o + 1));
		else {
			if (n.length === 0) continue;
			n[n.length - 1].push(t[i]);
		}
		i = o + 1, r = s;
	}
	return n;
}
// node_modules/ol/geom/Polygon.js
var Qi = class e extends Yr {
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
		return r < Ue(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(si(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), ui(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	containsXY(e, t) {
		return bi(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, t);
	}
	getArea() {
		return Bi(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride);
	}
	getCoordinates(e) {
		let t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), Yi(t, 0, this.ends_, this.stride, e)), pi(t, 0, this.ends_, this.stride);
	}
	getEnds() {
		return this.ends_;
	}
	getFlatInteriorPoint() {
		if (this.flatInteriorPointRevision_ != this.getRevision()) {
			let e = ct(this.getExtent());
			this.flatInteriorPoint_ = Ui(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, 0), this.flatInteriorPointRevision_ = this.getRevision();
		}
		return this.flatInteriorPoint_;
	}
	getInteriorPoint() {
		return new Li(this.getFlatInteriorPoint(), "XYM");
	}
	getLinearRingCount() {
		return this.ends_.length;
	}
	getLinearRing(e) {
		return e < 0 || this.ends_.length <= e ? null : new Hi(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}
	getLinearRings() {
		let e = this.layout, t = this.flatCoordinates, n = this.ends_, r = [], i = 0;
		for (let a = 0, o = n.length; a < o; ++a) {
			let o = n[a], s = new Hi(t.slice(i, o), e);
			r.push(s), i = o;
		}
		return r;
	}
	getOrientedFlatCoordinates() {
		if (this.orientedRevision_ != this.getRevision()) {
			let e = this.flatCoordinates;
			qi(e, 0, this.ends_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = Yi(this.orientedFlatCoordinates_, 0, this.ends_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = Ni(this.flatCoordinates, 0, this.ends_, this.stride, Math.sqrt(t), n, 0, r), new e(n, "XY", r);
	}
	getType() {
		return "Polygon";
	}
	intersectsExtent(e) {
		return Ei(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		let n = ti(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}
};
function $i(e) {
	if (vt(e)) throw Error("Cannot create polygon from empty extent");
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
// node_modules/ol/geom/flat/center.js
function ea(e, t, n, r) {
	let i = [], a = Je();
	for (let o = 0, s = n.length; o < s; ++o) {
		let s = n[o];
		a = Qe(e, t, s[0], r), i.push((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), t = s[s.length - 1];
	}
	return i;
}
// node_modules/ol/geom/MultiPolygon.js
var ta = class e extends Yr {
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
		return r < Ue(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(ci(this.flatCoordinates, 0, this.endss_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), di(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}
	containsXY(e, t) {
		return xi(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, e, t);
	}
	getArea() {
		return Vi(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride);
	}
	getCoordinates(e) {
		let t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), Xi(t, 0, this.endss_, this.stride, e)), mi(t, 0, this.endss_, this.stride);
	}
	getEndss() {
		return this.endss_;
	}
	getFlatInteriorPoints() {
		if (this.flatInteriorPointsRevision_ != this.getRevision()) {
			let e = ea(this.flatCoordinates, 0, this.endss_, this.stride);
			this.flatInteriorPoints_ = Wi(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, e), this.flatInteriorPointsRevision_ = this.getRevision();
		}
		return this.flatInteriorPoints_;
	}
	getInteriorPoints() {
		return new Ri(this.getFlatInteriorPoints().slice(), "XYM");
	}
	getOrientedFlatCoordinates() {
		if (this.orientedRevision_ != this.getRevision()) {
			let e = this.flatCoordinates;
			Ji(e, 0, this.endss_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = Xi(this.orientedFlatCoordinates_, 0, this.endss_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}
	getSimplifiedGeometryInternal(t) {
		let n = [], r = [];
		return n.length = Pi(this.flatCoordinates, 0, this.endss_, this.stride, Math.sqrt(t), n, 0, r), new e(n, "XY", r);
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
		return new Qi(this.flatCoordinates.slice(t, r), this.layout, n);
	}
	getPolygons() {
		let e = this.layout, t = this.flatCoordinates, n = this.endss_, r = [], i = 0;
		for (let a = 0, o = n.length; a < o; ++a) {
			let o = n[a].slice(), s = o[o.length - 1];
			if (i !== 0) for (let e = 0, t = o.length; e < t; ++e) o[e] -= i;
			let c = new Qi(t.slice(i, s), e, o);
			r.push(c), i = s;
		}
		return r;
	}
	getType() {
		return "MultiPolygon";
	}
	intersectsExtent(e) {
		return Di(this.getOrientedFlatCoordinates(), 0, this.endss_, this.stride, e);
	}
	setCoordinates(e, t) {
		this.setLayout(t, e, 3), this.flatCoordinates ||= [];
		let n = ni(this.flatCoordinates, 0, e, this.stride, this.endss_);
		if (n.length === 0) this.flatCoordinates.length = 0;
		else {
			let e = n[n.length - 1];
			this.flatCoordinates.length = e.length === 0 ? 0 : e[e.length - 1];
		}
		this.changed();
	}
}, na = Or(), ra = class e {
	constructor(e, t, n, r, i, a) {
		this.styleFunction, this.extent_, this.id_ = a, this.type_ = e, this.flatCoordinates_ = t, this.flatInteriorPoints_ = null, this.flatMidpoints_ = null, this.ends_ = n || null, this.properties_ = i, this.squaredTolerance_, this.stride_ = r, this.simplifiedGeometry_;
	}
	get(e) {
		return this.properties_[e];
	}
	getExtent() {
		return this.extent_ ||= this.type_ === "Point" ? Ze(this.flatCoordinates_) : Qe(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_), this.extent_;
	}
	getFlatInteriorPoint() {
		if (!this.flatInteriorPoints_) {
			let e = ct(this.getExtent());
			this.flatInteriorPoints_ = Ui(this.flatCoordinates_, 0, this.ends_, this.stride_, e, 0);
		}
		return this.flatInteriorPoints_;
	}
	getFlatInteriorPoints() {
		if (!this.flatInteriorPoints_) {
			let e = Zi(this.flatCoordinates_, this.ends_), t = ea(this.flatCoordinates_, 0, e, this.stride_);
			this.flatInteriorPoints_ = Wi(this.flatCoordinates_, 0, e, this.stride_, t);
		}
		return this.flatInteriorPoints_;
	}
	getFlatMidpoint() {
		return this.flatMidpoints_ ||= hi(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_, .5), this.flatMidpoints_;
	}
	getFlatMidpoints() {
		if (!this.flatMidpoints_) {
			this.flatMidpoints_ = [];
			let e = this.flatCoordinates_, t = 0, n = this.ends_;
			for (let r = 0, i = n.length; r < i; ++r) {
				let i = n[r], a = hi(e, t, i, this.stride_, .5);
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
		e = or(e);
		let t = e.getExtent(), n = e.getWorldExtent();
		if (t && n) {
			let e = ft(n) / ft(t);
			Fr(na, n[0], n[3], e, -e, 0, 0, 0), Hr(this.flatCoordinates_, 0, this.flatCoordinates_.length, this.stride_, na, this.flatCoordinates_);
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
		return this.simplifyTransformed = w((t, n) => {
			if (t === this.squaredTolerance_) return this.simplifiedGeometry_;
			this.simplifiedGeometry_ = this.clone(), n && this.simplifiedGeometry_.applyTransform(n);
			let r = this.simplifiedGeometry_.getFlatCoordinates(), i;
			switch (this.type_) {
				case "LineString":
					r.length = ki(r, 0, this.simplifiedGeometry_.flatCoordinates_.length, this.simplifiedGeometry_.stride_, t, r, 0), i = [r.length];
					break;
				case "MultiLineString":
					i = [], r.length = Ai(r, 0, this.simplifiedGeometry_.ends_, this.simplifiedGeometry_.stride_, t, r, 0, i);
					break;
				case "Polygon": i = [], r.length = Ni(r, 0, this.simplifiedGeometry_.ends_, this.simplifiedGeometry_.stride_, Math.sqrt(t), r, 0, i);
			}
			return i && (this.simplifiedGeometry_ = new e(this.type_, r, i, this.stride_, this.properties_, this.id_)), this.squaredTolerance_ = t, this.simplifiedGeometry_;
		}), this;
	}
};
ra.prototype.getFlatCoordinates = ra.prototype.getOrientedFlatCoordinates;
// node_modules/ol/geom/GeometryCollection.js
var ia = class e extends Jr {
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
		let t = new e(aa(this.geometries_));
		return t.applyProperties(this), t;
	}
	closestPointXY(e, t, n, r) {
		if (r < Ue(this.getExtent(), e, t)) return r;
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
		Xe(e);
		let t = this.geometries_;
		for (let n = 0, r = t.length; n < r; ++n) et(e, t[n].getExtent());
		return e;
	}
	getGeometries() {
		return aa(this.geometries_);
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
		n ||= ct(this.getExtent());
		let r = this.geometries_;
		for (let i = 0, a = r.length; i < a; ++i) r[i].scale(e, t, n);
		this.changed();
	}
	setGeometries(e) {
		this.setGeometriesArray(aa(e));
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
function aa(e) {
	return e.map((e) => e.clone());
}
// node_modules/ol/format/Feature.js
var oa = class {
	constructor() {
		this.dataProjection = void 0, this.defaultFeatureProjection = void 0, this.featureClass = ri, this.supportedMediaTypes = void 0;
	}
	getReadOptions(e, t) {
		if (t) {
			let n = t.dataProjection ? or(t.dataProjection) : this.readProjection(e);
			if (t.extent && n && n.getUnits() === "tile-pixels") {
				let e = or(n);
				e && (e.setWorldExtent(t.extent), n = e);
			}
			t = {
				dataProjection: n ?? void 0,
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
		return _();
	}
	readFeature(e, t) {
		return _();
	}
	readFeatures(e, t) {
		return _();
	}
	readGeometry(e, t) {
		return _();
	}
	readProjection(e) {
		return _();
	}
	writeFeature(e, t) {
		return _();
	}
	writeFeatures(e, t) {
		return _();
	}
	writeGeometry(e, t) {
		return _();
	}
};
function sa(e, t, n) {
	let r = n ? or(n.featureProjection) : null, i = n ? or(n.dataProjection) : null, a = e;
	if (r && i && !fr(r, i)) {
		t && (a = e.clone());
		let n = t ? r : i, o = t ? i : r;
		if (n.getUnits() === "tile-pixels") a.transform(n, o);
		else {
			let e = hr(n, o);
			e && a.applyTransform(e);
		}
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
function ca(e, t, n) {
	if (Array.isArray(t[0])) {
		let r = t;
		return Ji(e, 0, r, n) || (e = e.slice(), Xi(e, 0, r, n)), e;
	}
	let r = t;
	return qi(e, 0, r, n) || (e = e.slice(), Yi(e, 0, r, n)), e;
}
function la(e, t) {
	let n = e.geometry;
	if (!n) return [];
	if (Array.isArray(n)) return n.map((t) => la({
		...e,
		geometry: t
	})).flat();
	let r = n.type === "MultiPolygon" ? "Polygon" : n.type;
	if (r === "GeometryCollection" || r === "Circle") throw Error("Unsupported geometry type: " + r);
	let i = (n.layout ?? "XY").length;
	return sa(new ra(r, r === "Polygon" ? ca(n.flatCoordinates, n.ends, i) : n.flatCoordinates, n.ends?.flat() ?? [], i, e.properties || {}, e.id).enableSimplifyTransformed(), !1, t);
}
function ua(e, t) {
	if (!e) return null;
	if (Array.isArray(e)) return new ia(e.map((e) => ua(e, t)));
	let n = e.type, r = e.layout || "XY", i;
	switch (n) {
		case "Point":
			i = new Li(e.flatCoordinates, r);
			break;
		case "LineString":
			i = new Fi(e.flatCoordinates, r);
			break;
		case "Polygon":
			i = new Qi(e.flatCoordinates, r, e.ends);
			break;
		case "MultiPoint":
			i = new Ri(e.flatCoordinates, r);
			break;
		case "MultiLineString":
			i = new Ii(e.flatCoordinates, r, e.ends);
			break;
		case "MultiPolygon":
			i = new ta(e.flatCoordinates, r, e.ends);
			break;
		default: throw Error("Unsupported geometry type: " + n);
	}
	return sa(i, !1, t);
}
// node_modules/ol/format/JSONFeature.js
var da = class extends oa {
	constructor() {
		super();
	}
	getType() {
		return "json";
	}
	readFeature(e, t) {
		let n = fa(e);
		return n ? this.readFeatureFromObject(n, this.getReadOptions(e, t)) : [];
	}
	readFeatures(e, t) {
		let n = fa(e);
		return n ? this.readFeaturesFromObject(n, this.getReadOptions(e, t)) : [];
	}
	readFeatureFromObject(e, t) {
		return _();
	}
	readFeaturesFromObject(e, t) {
		return _();
	}
	readGeometry(e, t) {
		let n = fa(e);
		if (!n) throw Error("Expected a JSON object");
		return this.readGeometryFromObject(n, this.getReadOptions(e, t));
	}
	readGeometryFromObject(e, t) {
		return _();
	}
	readProjection(e) {
		let t = fa(e);
		if (!t) throw Error("Expected a JSON object");
		return this.readProjectionFromObject(t);
	}
	readProjectionFromObject(e) {
		return _();
	}
	writeFeature(e, t) {
		return JSON.stringify(this.writeFeatureObject(e, t));
	}
	writeFeatureObject(e, t) {
		return _();
	}
	writeFeatures(e, t) {
		return JSON.stringify(this.writeFeaturesObject(e, t));
	}
	writeFeaturesObject(e, t) {
		return _();
	}
	writeGeometry(e, t) {
		return JSON.stringify(this.writeGeometryObject(e, t));
	}
	writeGeometryObject(e, t) {
		return _();
	}
};
function fa(e) {
	return typeof e == "string" ? JSON.parse(e) || null : e === null ? null : e;
}
// node_modules/ol/format/GeoJSON.js
var pa = class extends da {
	constructor(e) {
		e ||= {}, super(), this.dataProjection = or(e.dataProjection ? e.dataProjection : "EPSG:4326") ?? void 0, e.featureProjection && (this.defaultFeatureProjection = or(e.featureProjection) ?? void 0), e.featureClass && (this.featureClass = e.featureClass), this.geometryName_ = e.geometryName, this.extractGeometryName_ = e.extractGeometryName, this.supportedMediaTypes = ["application/geo+json", "application/vnd.geo+json"];
	}
	readFeatureFromObject(e, t) {
		let n = e, r;
		r = n.type === "Feature" ? n : {
			type: "Feature",
			geometry: e,
			properties: null
		};
		let i = ma(r.geometry, t);
		if (this.featureClass === ra) return la({
			geometry: i ?? void 0,
			id: r.id,
			properties: r.properties ?? void 0
		}, t);
		let a = this.featureClass, o = new a();
		return this.geometryName_ ? o.setGeometryName(this.geometryName_) : this.extractGeometryName_ && "geometry_name" in r && r.geometry_name && o.setGeometryName(r.geometry_name), o.setGeometry(ua(i, t)), "id" in r && o.setId(r.id), r.properties && o.setProperties(r.properties, !0), o;
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
		return ha(e, t);
	}
	readProjectionFromObject(e) {
		let t = e.crs, n;
		if (t) {
			if (t.type == "name") n = or(t.properties.name);
			else if (t.type === "EPSG") n = or("EPSG:" + t.properties.code);
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
		return o && (n.geometry = Ca(o, t), delete a[e.getGeometryName()]), r(a) || (n.properties = a), n;
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
		return Ca(e, this.adaptOptions(t));
	}
};
function ma(e, t) {
	if (!e || e.type !== "GeometryCollection" && Array.isArray(e.coordinates) && !e.coordinates.length) return null;
	let n;
	switch (e.type) {
		case "Point":
			n = _a(e);
			break;
		case "LineString":
			n = va(e);
			break;
		case "Polygon":
			n = Sa(e);
			break;
		case "MultiPoint":
			n = ba(e);
			break;
		case "MultiLineString":
			n = ya(e);
			break;
		case "MultiPolygon":
			n = xa(e);
			break;
		case "GeometryCollection":
			n = ga(e);
			break;
		default: throw Error("Unsupported GeoJSON type: " + e.type);
	}
	return n;
}
function ha(e, t) {
	let n = ma(e, t);
	return n ? ua(n, t) : null;
}
function ga(e, t) {
	let n = e.geometries, r = [];
	for (let e of n) {
		let n = ma(e, t);
		n && r.push(n);
	}
	return r;
}
function _a(e) {
	let t = e.coordinates;
	return {
		type: "Point",
		flatCoordinates: t,
		layout: Xr(t.length)
	};
}
function va(e) {
	let t = e.coordinates, n = t.flat();
	return {
		type: "LineString",
		flatCoordinates: n,
		ends: [n.length],
		layout: Xr(t[0]?.length || 2)
	};
}
function ya(e) {
	let t = e.coordinates, n = t[0]?.[0]?.length || 2, r = [];
	return {
		type: "MultiLineString",
		flatCoordinates: r,
		ends: ti(r, 0, t, n),
		layout: Xr(n)
	};
}
function ba(e) {
	let t = e.coordinates;
	return {
		type: "MultiPoint",
		flatCoordinates: t.flat(),
		layout: Xr(t[0]?.length || 2)
	};
}
function xa(e) {
	let t = e.coordinates, n = [], r = t[0]?.[0]?.[0].length || 2;
	return {
		type: "MultiPolygon",
		flatCoordinates: n,
		ends: ni(n, 0, t, r),
		layout: Xr(r)
	};
}
function Sa(e) {
	let t = e.coordinates, n = [], r = t[0]?.[0]?.length;
	return {
		type: "Polygon",
		flatCoordinates: n,
		ends: ti(n, 0, t, r),
		layout: Xr(r)
	};
}
function Ca(e, t) {
	e = sa(e, !0, t);
	let n = e.getType(), r;
	switch (n) {
		case "Point":
			r = ka(e, t);
			break;
		case "LineString":
			r = Ta(e, t);
			break;
		case "Polygon":
			r = Aa(e, t);
			break;
		case "MultiPoint":
			r = Da(e, t);
			break;
		case "MultiLineString":
			r = Ea(e, t);
			break;
		case "MultiPolygon":
			r = Oa(e, t);
			break;
		case "GeometryCollection":
			r = wa(e, t);
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
function wa(e, t) {
	return t = Object.assign({}, t), delete t.featureProjection, {
		type: "GeometryCollection",
		geometries: e.getGeometriesArray().map(function(e) {
			return Ca(e, t);
		})
	};
}
function Ta(e, t) {
	return {
		type: "LineString",
		coordinates: e.getCoordinates()
	};
}
function Ea(e, t) {
	return {
		type: "MultiLineString",
		coordinates: e.getCoordinates()
	};
}
function Da(e, t) {
	return {
		type: "MultiPoint",
		coordinates: e.getCoordinates()
	};
}
function Oa(e, t) {
	let n;
	return t && (n = t.rightHanded), {
		type: "MultiPolygon",
		coordinates: e.getCoordinates(n)
	};
}
function ka(e, t) {
	return {
		type: "Point",
		coordinates: e.getCoordinates()
	};
}
function Aa(e, t) {
	let n;
	return t && (n = t.rightHanded), {
		type: "Polygon",
		coordinates: e.getCoordinates(n)
	};
}
// node_modules/ol/TileState.js
var z = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
}, ja = class extends D {
	constructor(e, t, n) {
		super(), n ||= {}, this.tileCoord = e, this.state = t, this.key = "", this.transition_ = n.transition === void 0 ? 250 : n.transition, this.transitionStarts_ = {}, this.interpolate = !!n.interpolate;
	}
	changed() {
		this.dispatchEvent(s.CHANGE);
	}
	release() {
		this.setState(z.EMPTY);
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
		if (this.state !== z.EMPTY) {
			if (this.state !== z.ERROR && this.state > e) throw Error("Tile load sequence violation");
			this.state = e, this.changed();
		}
	}
	load() {
		_();
	}
	getAlpha(e, t) {
		if (!this.transition_) return 1;
		let n = this.transitionStarts_[e];
		if (!n) n = t, this.transitionStarts_[e] = n;
		else if (n === -1) return 1;
		let r = t - n + 1e3 / 60;
		return r >= this.transition_ ? 1 : je(r / this.transition_);
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
// node_modules/ol/DataTile.js
function Ma(e) {
	return e instanceof Image || e instanceof HTMLCanvasElement || e instanceof HTMLVideoElement || e instanceof ImageBitmap ? e : null;
}
function Na(e) {
	return e instanceof Uint8Array || e instanceof Uint8ClampedArray || e instanceof Float32Array || e instanceof DataView ? e : null;
}
var Pa = /* @__PURE__ */ Error("disposed"), Fa = [256, 256], Ia = class extends ja {
	constructor(e) {
		let t = z.IDLE;
		super(e.tileCoord, t, {
			transition: e.transition,
			interpolate: e.interpolate
		}), this.loader_ = e.loader, this.data_ = null, this.error_ = null, this.size_ = e.size || null, this.controller_ = e.controller || null;
	}
	getSize() {
		if (this.size_) return this.size_;
		let e = Ma(this.data_);
		return e ? [e.width, e.height] : Fa;
	}
	getData() {
		return this.data_;
	}
	getError() {
		return this.error_;
	}
	load() {
		if (this.state !== z.IDLE && this.state !== z.ERROR) return;
		this.state = z.LOADING, this.changed();
		let e = this;
		this.loader_().then(function(t) {
			e.data_ = t, e.state = z.LOADED, e.changed();
		}).catch(function(t) {
			e.error_ = t, e.state = z.ERROR, e.changed();
		});
	}
	disposeInternal() {
		this.controller_ &&= (this.controller_.abort(Pa), null), super.disposeInternal();
	}
}, B = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
};
// node_modules/ol/Image.js
function La(e, t, n) {
	let r = e, i = !0, c = !1, l = !1, u = [a(r, s.LOAD, function() {
		l = !0, c || t();
	})];
	return r.src && he ? (c = !0, r.decode().then(function() {
		i && t();
	}).catch(function(e) {
		i && (l ? t() : n());
	})) : u.push(a(r, s.ERROR, n)), function() {
		i = !1, u.forEach(o);
	};
}
function Ra(e, t) {
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
function za(e, t) {
	return t && (e.src = t), e.src && he ? new Promise((t, n) => e.decode().then(() => t(e)).catch((r) => e.complete && e.width ? t(e) : n(r))) : Ra(e);
}
// node_modules/ol/ImageTile.js
var Ba = class extends ja {
	constructor(e, t, n, r, i, a) {
		super(e, t, a), this.crossOrigin_ = r?.crossOrigin, this.referrerPolicy_ = r?.referrerPolicy, this.src_ = n, this.key = n, this.image_, me ? this.image_ = new OffscreenCanvas(1, 1) : (this.image_ = new Image(), this.crossOrigin_ !== null && this.crossOrigin_ !== void 0 && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_)), this.unlisten_ = null, this.tileLoadFunction_ = i;
	}
	getImage() {
		return this.image_;
	}
	setImage(e) {
		this.image_ = e, this.state = z.LOADED, this.unlistenImage_(), this.changed();
	}
	getCrossOrigin() {
		return this.crossOrigin_ ?? null;
	}
	getReferrerPolicy() {
		return this.referrerPolicy_;
	}
	handleImageError_() {
		this.state = z.ERROR, this.unlistenImage_(), this.image_ = Va(), this.changed();
	}
	handleImageLoad_() {
		if (me) this.state = z.LOADED;
		else {
			let e = this.image_;
			this.state = e.naturalWidth && e.naturalHeight ? z.LOADED : z.EMPTY;
		}
		this.unlistenImage_(), this.changed();
	}
	load() {
		if (this.state == z.ERROR && (this.state = z.IDLE, this.image_ = new Image(), this.crossOrigin_ !== null && this.crossOrigin_ !== void 0 && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_)), this.state == z.IDLE) {
			this.state = z.LOADING, this.changed(), this.tileLoadFunction_(this, this.src_);
			let e = this.image_;
			e && (this.unlisten_ = La(e, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
		}
	}
	unlistenImage_() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}
	disposeInternal() {
		this.unlistenImage_(), this.image_ = null, super.disposeInternal();
	}
};
function Va() {
	let e = _e(1, 1);
	return e.fillStyle = "rgba(0,0,0,0)", e.fillRect(0, 0, 1, 1), e.canvas;
}
// node_modules/ol/TileRange.js
var Ha = class {
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
function Ua(e, t, n, r, i) {
	return i === void 0 ? new Ha(e, t, n, r) : (i.minX = e, i.maxX = t, i.minY = n, i.maxY = r, i);
}
// node_modules/ol/reproj.js
var Wa, Ga = [];
function Ka(e, t, n, r, i) {
	e.beginPath(), e.moveTo(0, 0), e.lineTo(t, n), e.lineTo(r, i), e.closePath(), e.save(), e.clip(), e.fillRect(0, 0, Math.max(t, r) + 1, Math.max(n, i)), e.restore();
}
function qa(e, t) {
	return Math.abs(e[t * 4] - 210) > 2 || Math.abs(e[t * 4 + 3] - 191.25) > 2;
}
function Ja() {
	if (Wa === void 0) {
		let e = _e(6, 6, Ga);
		e.globalCompositeOperation = "lighter", e.fillStyle = "rgba(210, 0, 0, 0.75)", Ka(e, 4, 5, 4, 0), Ka(e, 4, 5, 0, 5);
		let t = e.getImageData(0, 0, 3, 3).data;
		Wa = qa(t, 0) || qa(t, 4) || qa(t, 8), be(e), Ga.push(e.canvas);
	}
	return Wa;
}
function Ya(e, t, n, r) {
	let i = gr(n, t, e), a = sr(t, r, n), o = t.getMetersPerUnit();
	o !== void 0 && (a *= o);
	let s = e.getMetersPerUnit();
	s !== void 0 && (a /= s);
	let c = e.getExtent();
	if (!c || We(c, i)) {
		let t = sr(e, a, i) / a;
		isFinite(t) && t > 0 && (a /= t);
	}
	return a;
}
function Xa(e, t, n, r) {
	let i = Ya(e, t, ct(n), r);
	return (!isFinite(i) || i <= 0) && it(n, function(n) {
		return i = Ya(e, t, n, r), isFinite(i) && i > 0;
	}), i;
}
function Za(e, t, n, r, i, a, o, s, c, l, u, d, f, p) {
	let m = _e(Math.round(n * e), Math.round(n * t), Ga);
	if (d || (m.imageSmoothingEnabled = !1), c.length === 0) return m.canvas;
	m.scale(n, n);
	function h(e) {
		return Math.round(e * n) / n;
	}
	m.globalCompositeOperation = "lighter";
	let g = Je();
	c.forEach(function(e, t, n) {
		et(g, e.extent);
	});
	let _, v = n / r, y = (d ? 1 : 1 + 2 ** -24) / v;
	if (!f || c.length !== 1 || l !== 0) {
		let e = _e(Math.round(I(g) * v), Math.round(ft(g) * v), Ga);
		if (_ = e, d || (e.imageSmoothingEnabled = !1), i && p) {
			let t = (i[0] - g[0]) * v, n = -(i[3] - g[3]) * v, r = I(i) * v, a = ft(i) * v;
			e.rect(t, n, r, a), e.clip();
		}
		c.forEach(function(t, n, r) {
			if (t.image.width > 0 && t.image.height > 0) {
				if (t.clipExtent) {
					e.save();
					let n = (t.clipExtent[0] - g[0]) * v, r = -(t.clipExtent[3] - g[3]) * v, i = I(t.clipExtent) * v, a = ft(t.clipExtent) * v;
					e.rect(d ? n : Math.round(n), d ? r : Math.round(r), d ? i : Math.round(n + i) - Math.round(n), d ? a : Math.round(r + a) - Math.round(r)), e.clip();
				}
				let n = (t.extent[0] - g[0]) * v, r = -(t.extent[3] - g[3]) * v, i = I(t.extent) * v, a = ft(t.extent) * v;
				e.drawImage(t.image, l, l, t.image.width - 2 * l, t.image.height - 2 * l, d ? n : Math.round(n), d ? r : Math.round(r), d ? i : Math.round(n + i) - Math.round(n), d ? a : Math.round(r + a) - Math.round(r)), t.clipExtent && e.restore();
			}
		});
	}
	let b = ht(o);
	return s.getTriangles().forEach(function(e, t, n) {
		let r = e.source, i = e.target, o = r[0][0], s = r[0][1], l = r[1][0], u = r[1][1], f = r[2][0], p = r[2][1], v = h((i[0][0] - b[0]) / a), x = h(-(i[0][1] - b[1]) / a), S = h((i[1][0] - b[0]) / a), C = h(-(i[1][1] - b[1]) / a), w = h((i[2][0] - b[0]) / a), T = h(-(i[2][1] - b[1]) / a), E = o, D = s;
		o = 0, s = 0, l -= E, u -= D, f -= E, p -= D;
		let O = Mt([
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
		if (m.save(), m.beginPath(), Ja() || !d) {
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
			k = e.image, m.scale(I(t) / k.width, -ft(t) / k.height);
		}
		m.drawImage(k, 0, 0), m.restore();
	}), _ && (be(_), Ga.push(_.canvas)), u && (m.save(), m.globalCompositeOperation = "source-over", m.strokeStyle = "black", m.lineWidth = 1, s.getTriangles().forEach(function(e, t, n) {
		let r = e.target, i = (r[0][0] - b[0]) / a, o = -(r[0][1] - b[1]) / a, s = (r[1][0] - b[0]) / a, c = -(r[1][1] - b[1]) / a, l = (r[2][0] - b[0]) / a, u = -(r[2][1] - b[1]) / a;
		m.beginPath(), m.moveTo(s, c), m.lineTo(i, o), m.lineTo(l, u), m.closePath(), m.stroke();
	}), m.restore()), m.canvas;
}
// node_modules/ol/reproj/Triangulation.js
var Qa = 10, $a = .25, eo = class {
	constructor(e, t, n, r, i, a, o) {
		this.sourceProj_ = e, this.targetProj_ = t;
		let s = {}, c = o ? dr((e) => R(o, gr(e, this.targetProj_, this.sourceProj_))) : hr(this.targetProj_, this.sourceProj_);
		this.transformInv_ = function(e) {
			let t = e[0] + "/" + e[1];
			return s[t] || (s[t] = c(e)), s[t];
		}, this.maxSourceExtent_ = r, this.errorThresholdSquared_ = i * i, this.triangles_ = [], this.wrapsXInSource_ = !1, this.canWrapXInSource_ = this.sourceProj_.canWrapX() && !!r && !!this.sourceProj_.getExtent() && I(r) >= I(this.sourceProj_.getExtent()), this.sourceWorldWidth_ = this.sourceProj_.getExtent() ? I(this.sourceProj_.getExtent()) : void 0, this.targetWorldWidth_ = this.targetProj_.getExtent() ? I(this.targetProj_.getExtent()) : void 0;
		let l = ht(n), u = gt(n), d = st(n), f = ot(n), p = this.transformInv_(l), m = this.transformInv_(u), h = this.transformInv_(d), g = this.transformInv_(f), _ = Qa + (a ? Math.max(0, Math.ceil(Math.log2(at(n) / (a * a * 256 * 256)))) : 0);
		if (this.addQuad_(l, u, d, f, p, m, h, g, _), this.wrapsXInSource_) {
			let e = this.sourceWorldWidth_;
			if (!e) return;
			let t = Infinity;
			this.triangles_.forEach(function(e, n, r) {
				t = Math.min(t, e.source[0][0], e.source[1][0], e.source[2][0]);
			}), this.triangles_.forEach((n) => {
				if (Math.max(n.source[0][0], n.source[1][0], n.source[2][0]) - t > e / 2) {
					let r = [
						[n.source[0][0], n.source[0][1]],
						[n.source[1][0], n.source[1][1]],
						[n.source[2][0], n.source[2][1]]
					];
					r[0][0] - t > e / 2 && (r[0][0] -= e), r[1][0] - t > e / 2 && (r[1][0] -= e), r[2][0] - t > e / 2 && (r[2][0] -= e);
					let i = Math.min(r[0][0], r[1][0], r[2][0]);
					Math.max(r[0][0], r[1][0], r[2][0]) - i < e / 2 && (n.source = r);
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
		let l = ze([
			i,
			a,
			o,
			s
		]), u = this.sourceWorldWidth_, d = u ? I(l) / u : void 0, f = this.sourceProj_.canWrapX() && d !== void 0 && d > .5 && d < 1, p = !1;
		if (c > 0 && (this.targetProj_.isGlobal() && this.targetWorldWidth_ && (p = I(ze([
			e,
			t,
			n,
			r
		])) / this.targetWorldWidth_ > $a || p), !f && this.sourceProj_.isGlobal() && d && (p = d > $a || p)), !p && this.maxSourceExtent_ && isFinite(l[0]) && isFinite(l[1]) && isFinite(l[2]) && isFinite(l[3]) && !_t(l, this.maxSourceExtent_)) return;
		let m = 0;
		if (!p && (!isFinite(i[0]) || !isFinite(i[1]) || !isFinite(a[0]) || !isFinite(a[1]) || !isFinite(o[0]) || !isFinite(o[1]) || !isFinite(s[0]) || !isFinite(s[1]))) {
			if (c > 0) p = !0;
			else if (m = (!isFinite(i[0]) || !isFinite(i[1]) ? 8 : 0) + (!isFinite(a[0]) || !isFinite(a[1]) ? 4 : 0) + (!isFinite(o[0]) || !isFinite(o[1]) ? 2 : 0) + +(!isFinite(s[0]) || !isFinite(s[1])), m != 1 && m != 2 && m != 4 && m != 8) return;
		}
		if (c > 0) {
			if (!p) {
				let t = [(e[0] + n[0]) / 2, (e[1] + n[1]) / 2], r = this.transformInv_(t), a;
				if (f) {
					let e = u;
					a = (Ft(i[0], e) + Ft(o[0], e)) / 2 - Ft(r[0], e);
				} else a = (i[0] + o[0]) / 2 - r[0];
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
		let e = Je();
		return this.triangles_.forEach(function(t, n, r) {
			let i = t.source;
			tt(e, i[0]), tt(e, i[1]), tt(e, i[2]);
		}), e;
	}
	getTriangles() {
		return this.triangles_;
	}
}, to = .5, no = class extends ja {
	constructor(e, t, n, r, i, a, o, s, c, l, u, d) {
		super(i, z.IDLE, d), this.renderEdges_ = u !== void 0 && u, this.pixelRatio_ = o, this.gutter_ = s, this.canvas_ = null, this.sourceTileGrid_ = t, this.targetTileGrid_ = r, this.wrappedTileCoord_ = a || i, this.sourceTiles_ = [], this.sourcesListenerKeys_ = null, this.sourceZ_ = 0, this.clipExtent_ = e.canWrapX() && e.getExtent() || void 0;
		let f = r.getTileCoordExtent(this.wrappedTileCoord_), p = this.targetTileGrid_.getExtent(), m = this.sourceTileGrid_.getExtent(), h = p ? pt(f, p) : f;
		if (at(h) === 0) {
			this.state = z.EMPTY;
			return;
		}
		let g = e.getExtent();
		g && (m = m ? pt(m, g) : g);
		let _ = r.getResolution(this.wrappedTileCoord_[0]), v = Xa(e, n, h, _);
		if (!isFinite(v) || v <= 0) {
			this.state = z.EMPTY;
			return;
		}
		let y = l === void 0 ? to : l;
		if (this.triangulation_ = new eo(e, n, h, m || void 0, v * y, _), this.triangulation_.getTriangles().length === 0) {
			this.state = z.EMPTY;
			return;
		}
		this.sourceZ_ = t.getZForResolution(v);
		let b = this.triangulation_.calculateSourceExtent();
		if (m && (e.canWrapX() ? (b[1] = kt(b[1], m[1], m[3]), b[3] = kt(b[3], m[1], m[3])) : b = pt(b, m)), !at(b)) this.state = z.EMPTY;
		else {
			let n = 0, r = 0;
			e.canWrapX() && (n = I(g), r = Math.floor((b[0] - g[0]) / n)), wt(b.slice(), e, !0).forEach((e) => {
				let i = t.getTileRangeForExtentAndZ(e, this.sourceZ_);
				for (let e = i.minX; e <= i.maxX; e++) for (let t = i.minY; t <= i.maxY; t++) {
					let i = r * n;
					this.sourceTiles_.push({
						getTile: () => c(this.sourceZ_, e, t, o),
						offset: i
					});
				}
				++r;
			}), this.sourceTiles_.length === 0 && (this.state = z.EMPTY);
		}
	}
	getImage() {
		return this.canvas_;
	}
	reproject_() {
		let e = [];
		if (this.sourceTiles_.forEach((t) => {
			let n = t.tile;
			if (n && n.getState() == z.LOADED) {
				let r = this.sourceTileGrid_.getTileCoordExtent(n.tileCoord);
				r[0] += t.offset, r[2] += t.offset;
				let i = this.clipExtent_?.slice();
				i && (i[0] += t.offset, i[2] += t.offset);
				let a = n.getImage();
				if (!a) return;
				e.push({
					extent: r,
					clipExtent: i,
					image: a
				});
			}
		}), this.sourceTiles_.length = 0, e.length === 0) this.state = z.ERROR;
		else {
			let t = this.wrappedTileCoord_[0], n = this.targetTileGrid_.getTileSize(t), r = typeof n == "number" ? n : n[0], i = typeof n == "number" ? n : n[1], a = this.targetTileGrid_.getResolution(t), o = this.sourceTileGrid_.getResolution(this.sourceZ_), s = this.targetTileGrid_.getTileCoordExtent(this.wrappedTileCoord_);
			this.canvas_ = Za(r, i, this.pixelRatio_, o, this.sourceTileGrid_.getExtent() ?? s, a, s, this.triangulation_, e, this.gutter_, this.renderEdges_, this.interpolate), this.state = z.LOADED;
		}
		this.changed();
	}
	load() {
		for (let e of this.sourceTiles_) e.tile = e.getTile();
		if (this.state == z.IDLE) {
			this.state = z.LOADING, this.changed();
			let e = 0;
			this.sourcesListenerKeys_ = [], this.sourceTiles_.forEach((t) => {
				let n = t.tile;
				if (!n) return;
				let r = n.getState();
				if (r == z.IDLE || r == z.LOADING) {
					e++;
					let t = i(n, s.CHANGE, () => {
						let r = n.getState();
						(r == z.LOADED || r == z.ERROR || r == z.EMPTY) && (o(t), e--, e === 0 && (this.unlistenSources_(), this.reproject_()));
					});
					this.sourcesListenerKeys_.push(t);
				}
			}), e === 0 ? setTimeout(this.reproject_.bind(this), 0) : this.sourceTiles_.forEach(function(e) {
				let t = e.tile;
				t && t.getState() == z.IDLE && t.load();
			});
		}
	}
	unlistenSources_() {
		this.sourcesListenerKeys_ &&= (this.sourcesListenerKeys_.forEach(o), null);
	}
	release() {
		this.canvas_ &&= (be(this.canvas_.getContext("2d")), Ga.push(this.canvas_), null), this.sourceTiles_.length = 0, super.release();
	}
};
// node_modules/ol/size.js
function ro(e) {
	return e[0] > 0 && e[1] > 0;
}
function io(e, t, n) {
	return n === void 0 && (n = [0, 0]), n[0] = e[0] * t + .5 | 0, n[1] = e[1] * t + .5 | 0, n;
}
function ao(e, t) {
	return Array.isArray(e) ? e : (t === void 0 ? t = [e, e] : (t[0] = e, t[1] = e), t);
}
// node_modules/ol/structs/LRUCache.js
var oo = class {
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
		return L(n !== void 0, "Tried to get a value for a key that does not exist in the cache"), n === this.newest_ || (n === this.oldest_ ? (this.oldest_ = this.oldest_.newer, this.oldest_.older = null) : (n.newer.older = n.older, n.older.newer = n.newer), n.newer = null, n.older = this.newest_, this.newest_.newer = n, this.newest_ = n), n.value_;
	}
	remove(e) {
		let t = this.entries_[e];
		return L(t !== void 0, "Tried to get a value for a key that does not exist in the cache"), t === this.newest_ ? (this.newest_ = t.older, this.newest_ && (this.newest_.newer = null)) : t === this.oldest_ ? (this.oldest_ = t.newer, this.oldest_ && (this.oldest_.older = null)) : (t.newer.older = t.older, t.older.newer = t.newer), delete this.entries_[e], --this.count_, t.value_;
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
		L(!(e in this.entries_), "Tried to set a value for a key that is used already");
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
// node_modules/ol/tilecoord.js
function so(e, t, n, r) {
	return r === void 0 ? [
		e,
		t,
		n
	] : (r[0] = e, r[1] = t, r[2] = n, r);
}
function co(e, t, n) {
	return e + "/" + t + "/" + n;
}
function lo(e, t, n, r, i) {
	return `${y(e)},${t},${co(n, r, i)}`;
}
function uo(e) {
	return fo(e[0], e[1], e[2]);
}
function fo(e, t, n) {
	return (t << e) + n;
}
function po(e, t) {
	let n = e[0], r = e[1], i = e[2];
	if (t.getMinZoom() > n || n > t.getMaxZoom()) return !1;
	let a = t.getFullTileRange(n);
	return !a || a.containsXY(r, i);
}
// node_modules/ol/color.js
var mo = [
	NaN,
	NaN,
	NaN,
	0
], ho;
function go() {
	return ho ||= _e(1, 1, void 0, {
		willReadFrequently: !0,
		desynchronized: !0
	}), ho;
}
var _o = /^rgba?\(\s*(\d+%?)\s+(\d+%?)\s+(\d+%?)(?:\s*\/\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, vo = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, yo = /^rgba?\(\s*(\d+%)\s*,\s*(\d+%)\s*,\s*(\d+%)(?:\s*,\s*(\d+%|\d*\.\d+|[01]))?\s*\)$/i, bo = /^#([\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$/i;
function xo(e, t) {
	return e.endsWith("%") ? Number(e.substring(0, e.length - 1)) / t : Number(e);
}
function So(e) {
	throw Error("failed to parse \"" + e + "\" as color");
}
function Co(e) {
	if (e.toLowerCase().startsWith("rgb")) {
		let t = e.match(vo) || e.match(_o) || e.match(yo);
		if (t) {
			let e = t[4], n = 100 / 255;
			return [
				kt(xo(t[1], n) + .5 | 0, 0, 255),
				kt(xo(t[2], n) + .5 | 0, 0, 255),
				kt(xo(t[3], n) + .5 | 0, 0, 255),
				e === void 0 ? 1 : kt(xo(e, 100), 0, 1)
			];
		}
		So(e);
	}
	if (e.startsWith("#")) {
		if (bo.test(e)) {
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
		So(e);
	}
	let t = go();
	t.fillStyle = "#abcdef";
	let n = t.fillStyle;
	t.fillStyle = e, t.fillStyle === n && (t.fillStyle = "#fedcba", n = t.fillStyle, t.fillStyle = e, t.fillStyle === n && So(e));
	let r = t.fillStyle;
	if (r.startsWith("#") || r.startsWith("rgba")) return Co(r);
	t.clearRect(0, 0, 1, 1), t.fillRect(0, 0, 1, 1);
	let i = Array.from(t.getImageData(0, 0, 1, 1).data);
	return i[3] = Lt(i[3] / 255, 3), i;
}
function wo(e) {
	return typeof e == "string" ? e : Lo(e);
}
var To = 1024, Eo = {}, Do = 0;
function Oo(e) {
	if (e.length === 4) return e;
	let t = e.slice();
	return t[3] = 1, t;
}
function ko(e) {
	return e > .0031308 ? e ** (1 / 2.4) * 269.025 - 14.025 : e * 3294.6;
}
function Ao(e) {
	return e > .2068965 ? e ** 3 : (e - 4 / 29) * (108 / 841);
}
function jo(e) {
	return e > 10.314724 ? ((e + 14.025) / 269.025) ** 2.4 : e / 3294.6;
}
function Mo(e) {
	return e > .0088564 ? e ** (1 / 3) : e / (108 / 841) + 4 / 29;
}
function No(e) {
	let t = jo(e[0]), n = jo(e[1]), r = jo(e[2]), i = Mo(t * .222488403 + n * .716873169 + r * .06060791), a = 500 * (Mo(t * .452247074 + n * .399439023 + r * .148375274) - i), o = 200 * (i - Mo(t * .016863605 + n * .117638439 + r * .865350722)), s = 180 / Math.PI * Math.atan2(o, a);
	return [
		116 * i - 16,
		Math.sqrt(a * a + o * o),
		s < 0 ? s + 360 : s,
		e[3]
	];
}
function Po(e) {
	let t = (e[0] + 16) / 116, n = e[1], r = e[2] * Math.PI / 180, i = Ao(t), a = Ao(t + n / 500 * Math.cos(r)), o = Ao(t - n / 200 * Math.sin(r)), s = ko(a * 3.021973625 - i * 1.617392459 - o * .404875592), c = ko(a * -.943766287 + i * 1.916279586 + o * .027607165), l = ko(a * .069407491 - i * .22898585 + o * 1.159737864);
	return [
		kt(s + .5 | 0, 0, 255),
		kt(c + .5 | 0, 0, 255),
		kt(l + .5 | 0, 0, 255),
		e[3]
	];
}
function Fo(e) {
	if (e === "none") return mo;
	if (Eo.hasOwnProperty(e)) return Eo[e];
	if (Do >= To) {
		let e = 0;
		for (let t in Eo) e++ & 3 || (delete Eo[t], --Do);
	}
	let t = Co(e);
	t.length !== 4 && So(e);
	for (let n of t) isNaN(n) && So(e);
	return Eo[e] = t, ++Do, t;
}
function Io(e) {
	return Array.isArray(e) ? e : Fo(e);
}
function Lo(e) {
	let t = e[0];
	t != (t | 0) && (t = t + .5 | 0);
	let n = e[1];
	n != (n | 0) && (n = n + .5 | 0);
	let r = e[2];
	r != (r | 0) && (r = r + .5 | 0);
	let i = e[3] === void 0 ? 1 : Math.round(e[3] * 1e3) / 1e3;
	return "rgba(" + t + "," + n + "," + r + "," + i + ")";
}
// node_modules/ol/render/Event.js
var Ro = class extends E {
	constructor(e, t, n, r) {
		super(e), this.inversePixelTransform = t, this.frameState = n, this.context = r ?? void 0;
	}
}, zo = {
	PRERENDER: "prerender",
	POSTRENDER: "postrender",
	PRECOMPOSE: "precompose",
	POSTCOMPOSE: "postcompose",
	RENDERCOMPLETE: "rendercomplete"
}, Bo = class {
	constructor() {
		this.instructions_ = [], this.zIndex = 0, this.offset_ = 0, this.pendingMethod_, this.context_ = new Proxy(ye(), {
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
}, Vo = 5, Ho = class extends O {
	constructor(e) {
		super(), this.ready = !0, this.boundHandleImageChange_ = this.handleImageChange_.bind(this), this.layer_ = e, this.staleKeys_ = [], this.maxStaleKeys = Vo, this.renderedSourceKey_;
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
		return _();
	}
	getData(e) {
		return null;
	}
	prepareFrame(e) {
		return _();
	}
	renderFrame(e, t) {
		return _();
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {}
	getLayer() {
		return this.layer_;
	}
	handleFontsChanged() {}
	handleImageChange_(e) {
		let t = e.target;
		(t.getState() === B.LOADED || t.getState() === B.ERROR) && this.renderIfReadyAndVisible();
	}
	loadImage(e) {
		let t = e.getState();
		return t != B.LOADED && t != B.ERROR && e.addEventListener(s.CHANGE, this.boundHandleImageChange_), t == B.IDLE && (e.load(), t = e.getState()), t == B.LOADED;
	}
	renderIfReadyAndVisible() {
		let e = this.getLayer();
		e && e.getVisible() && e.getSourceState() === "ready" && e.changed();
	}
	renderDeferred(e) {}
	disposeInternal() {
		this.layer_ = void 0, super.disposeInternal();
	}
}, Uo = [], Wo = null;
function Go() {
	Wo = _e(1, 1, void 0, { willReadFrequently: !0 });
}
var Ko = class extends Ho {
	constructor(e) {
		super(e), this.wantRotation = !1, this.container = null, this.renderedResolution, this.tempTransform = Or(), this.pixelTransform = Or(), this.inversePixelTransform = Or(), this.context = null, this.deferredContext_ = null, this.containerReused = !1, this.frameState = null;
	}
	getImageData(e, t, n) {
		Wo || Go();
		let r = Wo;
		if (!r) return null;
		r.clearRect(0, 0, 1, 1);
		let i;
		try {
			r.drawImage(e, t, n, 1, 1, 0, 0, 1, 1), i = r.getImageData(0, 0, 1, 1).data;
		} catch {
			return Wo = null, null;
		}
		return i;
	}
	getBackground(e) {
		let t = this.getLayer().getBackground();
		return typeof t == "function" && (t = t(e.viewState.resolution)), t || void 0;
	}
	useContainer(e, t, n, r, i) {
		if (De(e) && this.pixelTransform[1] === 0 && this.pixelTransform[2] === 0 && this.pixelTransform[4] === 0 && this.pixelTransform[5] === 0 && e.width === r && e.height === i) {
			let t = e, r = t.getContext("2d");
			if (r) {
				this.container = e, this.context = r, this.containerReused = !0, n && (r.fillStyle = n, r.fillRect(0, 0, t.width, t.height));
				return;
			}
		}
		let a = this.getLayer().getClassName(), o, s;
		if (e && e.className === a && (!n || e && e.style.backgroundColor && h(Io(e.style.backgroundColor), Io(n)))) {
			let t = e.firstElementChild;
			De(t) && (s = t.getContext("2d"));
		}
		if (s && Vr(s.canvas.style.transform, t) ? (this.container = e, this.context = s, this.containerReused = !0) : this.containerReused ? (this.container = null, this.context = null, this.containerReused = !1) : this.container && (this.container.style.backgroundColor = null), !this.container) {
			o = me ? Ee() : document.createElement("div"), o.className = a;
			let e = o.style;
			e.position = "absolute", e.width = "100%", e.height = "100%", s = _e();
			let t = s.canvas;
			o.appendChild(t), e = t.style, e.position = "absolute", e.left = "0", e.transformOrigin = "top left", this.container = o, this.context = s;
		}
		!this.containerReused && n && !this.container.style.backgroundColor && (this.container.style.backgroundColor = n);
	}
	clipUnrotated(e, t, n) {
		let r = ht(n), i = gt(n), a = st(n), o = ot(n);
		R(t.coordinateToPixelTransform, r), R(t.coordinateToPixelTransform, i), R(t.coordinateToPixelTransform, a), R(t.coordinateToPixelTransform, o);
		let s = this.inversePixelTransform;
		R(s, r), R(s, i), R(s, a), R(s, o), e.save(), e.beginPath(), e.moveTo(Math.round(r[0]), Math.round(r[1])), e.lineTo(Math.round(i[0]), Math.round(i[1])), e.lineTo(Math.round(a[0]), Math.round(a[1])), e.lineTo(Math.round(o[0]), Math.round(o[1])), e.clip();
	}
	prepareContainer(e, t) {
		let n = e.extent, r = e.viewState.resolution, i = this.sourceRotates(), a = i ? 0 : e.viewState.rotation, o = e.pixelRatio, s = Math.round(i ? e.size[0] * o : I(n) / r * o), c = Math.round(i ? e.size[1] * o : ft(n) / r * o);
		Fr(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / o, 1 / o, a, -s / 2, -c / 2), Ir(this.inversePixelTransform, this.pixelTransform);
		let l = zr(this.pixelTransform), u = this.getBackground(e);
		if (this.useContainer(t, l, u, s, c), !this.containerReused) {
			let e = this.context.canvas;
			e.width != s || e.height != c ? (e.width = s, e.height = c) : this.context.clearRect(0, 0, s, c), l !== e.style.transform && (e.style.transform = l);
		}
	}
	dispatchRenderEvent_(e, t, n) {
		let r = this.getLayer();
		if (r.hasListener(e)) {
			let i = new Ro(e, this.inversePixelTransform, n, t);
			r.dispatchEvent(i);
		}
	}
	preRender(e, t) {
		this.frameState = t, t.declutter || this.dispatchRenderEvent_(zo.PRERENDER, e, t);
	}
	postRender(e, t) {
		t.declutter || this.dispatchRenderEvent_(zo.POSTRENDER, e, t);
	}
	renderDeferredInternal(e) {}
	getRenderContext(e) {
		return e.declutter && !this.deferredContext_ && (this.deferredContext_ = new Bo()), e.declutter ? this.deferredContext_.getContext() : this.context;
	}
	getCanvasContext() {
		let e = this.context;
		if (!e) throw Error("Canvas context not initialized");
		return e;
	}
	getContainerElement() {
		let e = this.container;
		if (!e) throw Error("Canvas container not initialized");
		return e;
	}
	renderDeferred(e) {
		e.declutter && (this.dispatchRenderEvent_(zo.PRERENDER, this.context, e), e.declutter && this.deferredContext_ && (this.deferredContext_.draw(this.context), this.deferredContext_.clear()), this.renderDeferredInternal(e), this.dispatchRenderEvent_(zo.POSTRENDER, this.context, e));
	}
	getRenderTransform(e, t, n, r, i, a, o) {
		let s = i / 2, c = a / 2, l = r / t, u = -l, d = -e[0] + o, f = -e[1];
		return Fr(this.tempTransform, s, c, l, u, -n, d, f);
	}
	sourceRotates() {
		return this.wantRotation && !!this.getLayer().getSource()?.canRotate;
	}
	disposeInternal() {
		delete this.frameState, super.disposeInternal();
	}
};
// node_modules/ol/renderer/canvas/TileLayer.js
function qo(e, t, n) {
	if (!(n in e)) return e[n] = /* @__PURE__ */ new Set([t]), !0;
	let r = e[n], i = r.has(t);
	return i || r.add(t), !i;
}
function Jo(e, t, n) {
	let r = e[n];
	return r ? r.delete(t) : !1;
}
function Yo(e, t) {
	let n = e.layerStatesArray[e.layerIndex];
	if (n.extent && (t = pt(t, Cr(n.extent, e.viewState.projection))), !n.layer) return t;
	let r = n.layer.getRenderSource();
	if (!r.getWrapX()) {
		let n = r.getTileGridForProjection(e.viewState.projection).getExtent();
		n && (t = pt(t, n));
	}
	return t;
}
var Xo = class extends Ko {
	constructor(e, t) {
		super(e), t ||= {}, this.extentChanged = !0, this.renderComplete = !1, this.renderedExtent_ = null, this.renderedPixelRatio, this.renderedProjection = null, this.renderedTiles = [], this.renderedSourceRevision_, this.tempExtent = Je(), this.tempTileRange_ = new Ha(0, 0, 0, 0), this.tempTileCoord_ = so(0, 0, 0);
		let n = t.cacheSize === void 0 ? 512 : t.cacheSize;
		this.tileCache_ = new oo(n), this.sourceTileCache_ = null, this.layerExtent = null, this.maxStaleKeys = n * .5;
	}
	getTileCache() {
		return this.tileCache_;
	}
	getSourceTileCache() {
		return this.sourceTileCache_ ||= new oo(512), this.sourceTileCache_;
	}
	getOrCreateTile(e, t, n, r) {
		let i = this.tileCache_, a = this.getLayer().getSource();
		if (!a) return null;
		let o = lo(a, a.getKey(), e, t, n), s;
		if (i.containsKey(o)) {
			if (s = i.get(o), !s) return null;
		} else {
			let c = r.viewState.projection, l = a.getProjection();
			if (s = a.getTile(e, t, n, r.pixelRatio, c, !l || fr(l, c) ? void 0 : this.getSourceTileCache()), !s) return null;
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
		let n = this.getLayer(), r = R(t.pixelToCoordinateTransform, e.slice()), i = n.getExtent();
		if (i && !We(i, r)) return null;
		let a = t.viewState, o = n.getRenderSource();
		if (!o) return null;
		let s = o.getTileGridForProjection(a.projection), c = o.getTilePixelRatio(t.pixelRatio);
		for (let e = s.getZForResolution(a.resolution); e >= s.getMinZoom(); --e) {
			let n = s.getTileCoordForCoordAndZ(r, e), i = this.getTile(e, n[1], n[2], t);
			if (!i || i.getState() !== z.LOADED) continue;
			let l = s.getOrigin(e), u = ao(s.getTileSize(e)), d = s.getResolution(e), f = null;
			if (i instanceof Ba || i instanceof no) f = i.getImage();
			else if (i instanceof Ia) {
				let e = i.getData();
				e && (f = Ma(e));
			}
			if (!f) continue;
			let p = Math.floor(c * ((r[0] - l[0]) / d - n[1] * u[0])), m = Math.floor(c * ((l[1] - r[1]) / d - n[2] * u[1])), h = Math.round(c * o.getGutterForProjection(a.projection));
			return this.getImageData(f, p + h, m + h);
		}
		return null;
	}
	readData(e) {
		let t = this.getLayer().getRenderSource();
		if (!t) return null;
		let n = e.viewState, r = n.projection;
		if (t.getGutterForProjection(r)) throw Error("readData() does not support sources with a gutter");
		let i = t.getTileGridForProjection(r), a = n.resolution, o = e.extent, s = i.getZForResolution(a), c = {};
		this.enqueueTiles(e, o, s, c, 0);
		let l = c[s];
		if (!l) return null;
		let u = [], d = !0;
		if (l.forEach((e) => {
			let t = e, n = t.getState();
			n === z.LOADED ? u.push(t) : n !== z.ERROR && n !== z.EMPTY && (d = !1);
		}), !d || u.length === 0) return null;
		let f = e.size[0], p = e.size[1], m = u[0], h = Na(m.getData());
		if (!h || h instanceof DataView) throw Error("ol/renderer/canvas/TileLayer#readData only supports typed array tile data");
		let g = m.getSize(), _ = Math.round(h.length / (g[0] * g[1])), v = h.constructor, y = new v(f * p * _);
		for (let e of u) {
			let t = i.getTileCoordExtent(e.tileCoord, this.tempExtent), n = e.getSize(), r = Na(e.getData()), s = (t[0] - o[0]) / a - .5, c = (o[3] - t[3]) / a - .5, l = I(t) / a, u = ft(t) / a, d = n[0] / l, m = n[1] / u, h = Math.max(0, Math.ceil(s)), g = Math.min(f, Math.ceil(s + l)), v = Math.max(0, Math.ceil(c)), b = Math.min(p, Math.ceil(c + u));
			for (let e = v; e < b; ++e) {
				let t = Math.min(Math.floor((e - c) * m), n[1] - 1);
				for (let i = h; i < g; ++i) {
					let a = Math.min(Math.floor((i - s) * d), n[0] - 1), o = (t * n[0] + a) * _, c = (e * f + i) * _;
					for (let e = 0; e < _; ++e) y[c + e] = r[o + e];
				}
			}
		}
		return this.updateCacheSize(l.size), this.tileCache_.expireCache(), this.sourceTileCache_?.expireCache(), {
			data: y,
			width: f,
			height: p,
			bandCount: _
		};
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
		let a = e.viewState, o = this.getLayer(), s = o.getRenderSource();
		if (!s) return;
		let c = s.getTileGridForProjection(a.projection), l = y(s);
		l in e.wantedTiles || (e.wantedTiles[l] = {});
		let u = e.wantedTiles[l], d = o.getMapInternal(), f = Math.max(n - i, c.getMinZoom(), c.getZForResolution(Math.min(o.getMaxResolution(), d ? d.getView().getResolutionForZoom(Math.max(o.getMinZoom(), 0)) : c.getResolution(0)), s.zDirection)), p = a.rotation, m = p ? dt(a.center, a.resolution, p, e.size) : void 0;
		for (let i = n; i >= f; --i) {
			let n = c.getTileRangeForExtentAndZ(t, i, this.tempTileRange_), a = c.getResolution(i);
			for (let t = n.minX; t <= n.maxX; ++t) for (let o = n.minY; o <= n.maxY; ++o) {
				if (p && m && !c.tileCoordIntersectsViewport([
					i,
					t,
					o
				], m)) continue;
				let n = this.getTile(i, t, o, e);
				if (!n || !qo(r, n, i)) continue;
				let s = n.getKey();
				if (u[s] = !0, n.getState() === z.IDLE && !e.tileQueue.isKeyQueued(s)) {
					let r = so(i, t, o, this.tempTileCoord_);
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
		let n = this.tileCache_, r = e[0], i = e[1], a = e[2], o = this.getStaleKeys(), s = this.getLayer().getSource();
		if (!s) return !1;
		for (let e = 0; e < o.length; ++e) {
			let c = lo(s, o[e], r, i, a);
			if (n.containsKey(c)) {
				let e = n.peek(c);
				if (e && e.getState() === z.LOADED) return e.endTransition(y(this)), qo(t, e, r), !0;
			}
		}
		return !1;
	}
	findAltTiles_(e, t, n, r) {
		let i = e.getTileRangeForTileCoordAndZ(t, n, this.tempTileRange_);
		if (!i) return !1;
		let a = !0, o = this.tileCache_, s = this.getLayer().getRenderSource();
		if (!s) return !1;
		let c = s.getKey();
		for (let e = i.minX; e <= i.maxX; ++e) for (let t = i.minY; t <= i.maxY; ++t) {
			let i = lo(s, c, n, e, t), l = !1;
			if (o.containsKey(i)) {
				let e = o.peek(i);
				e && e.getState() === z.LOADED && (qo(r, e, n), l = !0);
			}
			l || (a = !1);
		}
		return a;
	}
	renderFrame(e, t) {
		this.renderComplete = !0;
		let n = e.layerStatesArray[e.layerIndex], r = e.viewState, i = r.projection, a = r.resolution, o = r.center, s = e.pixelRatio, c = this.getLayer(), l = c.getSource();
		if (!l) return this.getContainerElement();
		let d = l.getTileGridForProjection(i), f = d.getZForResolution(a, l.zDirection), p = d.getResolution(f);
		this.updateStaleKeys(l.getKey());
		let m = e.extent, h = l.getTilePixelRatio(s);
		this.prepareContainer(e, t);
		let g = this.getCanvasContext().canvas.width, _ = this.getCanvasContext().canvas.height;
		this.layerExtent = n.extent ? Cr(n.extent, i) : null, this.layerExtent && (m = pt(m, this.layerExtent));
		let v = p * g / 2 / h, b = p * _ / 2 / h, x = [
			o[0] - v,
			o[1] - b,
			o[0] + v,
			o[1] + b
		], S = {};
		this.renderedTiles.length = 0;
		let C = c.getPreload();
		if (e.nextExtent && this.enqueueTilesForNextExtent()) {
			let t = d.getZForResolution(r.nextResolution ?? a, l.zDirection), n = Yo(e, e.nextExtent);
			this.enqueueTiles(e, n, t, S, C);
		}
		let w = Yo(e, m);
		if (this.enqueueTiles(e, w, f, S, 0), C > 0 && setTimeout(() => {
			this.enqueueTiles(e, w, f - 1, S, C - 1);
		}, 0), !(f in S)) return this.getContainerElement();
		let T = y(this), E = e.time;
		for (let t of S[f]) {
			let n = t.getState();
			if (n === z.EMPTY) continue;
			let r = t.tileCoord;
			if (n === z.LOADED && t.getAlpha(T, E) === 1) {
				t.endTransition(T);
				continue;
			}
			if (n !== z.ERROR && (this.renderComplete = !1), this.findStaleTile_(r, S)) {
				Jo(S, t, f), e.animate = !0;
				continue;
			}
			if (this.findAltTiles_(d, r, f + 1, S)) continue;
			let i = d.getMinZoom();
			for (let e = f - 1; e >= i && !this.findAltTiles_(d, r, e, S); --e);
		}
		let D = p / a * s / h, O = this.getRenderContext(e);
		Fr(this.tempTransform, g / 2, _ / 2, D, D, 0, -g / 2, -_ / 2), this.layerExtent && this.clipUnrotated(O, e, this.layerExtent), l.getInterpolate() || (O.imageSmoothingEnabled = !1), this.preRender(O, e);
		let k = Object.keys(S).map(Number);
		k.sort(u);
		let A = [], j = [], M = [];
		for (let t = k.length - 1; t >= 0; --t) {
			let n = k[t], r = l.getTilePixelSize(n, s, i), a = d.getResolution(n) / p, o = r[0] * a * D, c = r[1] * a * D, u = d.getTileCoordForCoordAndZ(ht(x), n), m = d.getTileCoordExtent(u), g = R(this.tempTransform, [h * (m[0] - x[0]) / p, h * (x[3] - m[3]) / p]), _ = h * l.getGutterForProjection(i);
			for (let t of S[n]) {
				if (t.getState() !== z.LOADED) continue;
				let r = t.tileCoord, i = u[1] - r[1], a = Math.round(g[0] - (i - 1) * o), s = u[2] - r[2], d = Math.round(g[1] - (s - 1) * c), p = Math.round(g[0] - i * o), m = Math.round(g[1] - s * c), h = a - p, v = d - m, y = n === f;
				if (y && t.inTransition(T)) {
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
				for (let e = 0, t = A.length; e < t; ++e) n < j[e] && _t(b, A[e]) && x.push(A[e]);
				let S;
				x.length > 0 && (S = Tt(b, x)), A.push(b), j.push(n), this.drawTile(t, e, p, m, h, v, _, y, S), this.renderedTiles.unshift(t), this.updateUsedTiles(e.usedTiles, l, t);
			}
		}
		for (let t = 0, n = M.length; t < n; ++t) {
			let { tile: n, x: r, y: i, w: a, h: o, gutter: s } = M[t];
			this.drawTile(n, e, r, i, a, o, s, !0, void 0);
		}
		return this.renderedResolution = p, this.extentChanged = !this.renderedExtent_ || !$e(this.renderedExtent_, x), this.renderedExtent_ = x, this.renderedPixelRatio = s, this.postRender(this.getCanvasContext(), e), this.layerExtent && O.restore(), O.imageSmoothingEnabled = !0, this.renderComplete && e.postRenderFunctions.push((e, t) => {
			let n = y(l), r = t.wantedTiles[n], i = r ? Object.keys(r).length : 0;
			this.updateCacheSize(i), this.tileCache_.expireCache(), this.sourceTileCache_?.expireCache();
		}), this.getContainerElement();
	}
	updateCacheSize(e) {
		this.tileCache_.highWaterMark = Math.max(this.tileCache_.highWaterMark, e * 2);
	}
	drawTile(e, t, n, r, i, a, o, s, c) {
		let l;
		if (e instanceof Ia) {
			let t = e.getData();
			if (!t) return;
			if (l = Ma(t), !l) throw Error("Rendering array data is not yet supported");
		} else l = this.getTileImage(e);
		if (!l) return;
		let u = this.getRenderContext(t), d = y(this), f = t.layerStatesArray[t.layerIndex], p = f.opacity * (s ? e.getAlpha(d, t.time) : 1), m = p !== u.globalAlpha;
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
		let e = this.getCanvasContext();
		return e ? e.canvas : null;
	}
	getTileImage(e) {
		return e.getImage() ?? null;
	}
	updateUsedTiles(e, t, n) {
		let r = y(t);
		r in e || (e[r] = {}), e[r][n.getKey()] = !0;
	}
}, V = {
	ANIMATING: 0,
	INTERACTING: 1
}, Zo = {
	CENTER: "center",
	RESOLUTION: "resolution",
	ROTATION: "rotation"
};
// node_modules/ol/centerconstraint.js
function Qo(e, t, n) {
	return (function(r, i, a, o, s) {
		if (!r) return;
		if (!i && !t) return r;
		let c = t ? 0 : a[0] * (i ?? 0), l = t ? 0 : a[1] * (i ?? 0), u = s ? s[0] : 0, d = s ? s[1] : 0, f = e[0] + c / 2 + u, p = e[2] - c / 2 + u, m = e[1] + l / 2 + d, h = e[3] - l / 2 + d;
		f > p && (f = (p + f) / 2, p = f), m > h && (m = (h + m) / 2, h = m);
		let g = kt(r[0], f, p), _ = kt(r[1], m, h);
		if (o && n && i) {
			let e = 30 * i;
			g += -e * Math.log(1 + Math.max(0, f - r[0]) / e) + e * Math.log(1 + Math.max(0, r[0] - p) / e), _ += -e * Math.log(1 + Math.max(0, m - r[1]) / e) + e * Math.log(1 + Math.max(0, r[1] - h) / e);
		}
		return [g, _];
	});
}
function $o(e) {
	return e;
}
// node_modules/ol/resolutionconstraint.js
function es(e, t, n, r) {
	let i = I(t) / n[0], a = ft(t) / n[1];
	return r ? Math.min(e, Math.max(i, a)) : Math.min(e, Math.min(i, a));
}
function ts(e, t, n) {
	let r = Math.min(e, t);
	return r *= Math.log(1 + 50 * Math.max(0, e / t - 1)) / 50 + 1, n && (r = Math.max(r, n), r /= Math.log(1 + 50 * Math.max(0, n / e - 1)) / 50 + 1), kt(r, n / 2, t * 2);
}
function ns(e, t, n, r) {
	return t = t === void 0 || t, (function(i, a, o, s) {
		if (i !== void 0) {
			let c = e[0], l = e[e.length - 1], u = n ? es(c, n, o, !!r) : c;
			if (s) return t ? ts(i, u, l) : kt(i, l, u);
			let d = Math.floor(f(e, Math.min(u, i), a));
			return e[d] > u && d < e.length - 1 ? e[d + 1] : e[d];
		}
	});
}
function rs(e, t, n, r, i, a) {
	return r = r === void 0 || r, n = n === void 0 ? 0 : n, (function(o, s, c, l) {
		if (o !== void 0) {
			let u = i ? es(t, i, c, !!a) : t;
			if (l) return r ? ts(o, u, n) : kt(o, n, u);
			let d = Math.ceil(Math.log(t / u) / Math.log(e) - 1e-9), f = -s * .499999999 + .5, p = Math.floor(Math.log(t / Math.min(u, o)) / Math.log(e) + f);
			return kt(t / e ** +Math.max(d, p), n, u);
		}
	});
}
function is(e, t, n, r, i) {
	return n = n === void 0 || n, (function(a, o, s, c) {
		if (a !== void 0) {
			let o = r ? es(e, r, s, !!i) : e;
			return !n || !c ? kt(a, t, o) : ts(a, o, t);
		}
	});
}
// node_modules/ol/rotationconstraint.js
function as(e) {
	if (e !== void 0) return 0;
}
function os(e) {
	if (e !== void 0) return e;
}
function ss(e) {
	let t = 2 * Math.PI / e;
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return e = Math.floor(e / t + .5) * t, e;
	});
}
function cs(e) {
	let t = e === void 0 ? Pt(5) : e;
	return (function(e, n) {
		return n || e === void 0 ? e : Math.abs(e) <= t ? 0 : e;
	});
}
// node_modules/ol/View.js
var ls = 0, us = class extends j {
	constructor(e) {
		super(), this.on, this.once, this.un, e = Object.assign({}, e), this.hints_ = [0, 0], this.animations_ = [], this.updateAnimationKey_, this.projection_ = ur(e.projection, "EPSG:3857"), this.viewportSize_ = [100, 100], this.targetCenter_ = null, this.targetResolution_, this.targetRotation_, this.nextCenter_ = null, this.nextResolution_, this.nextRotation_, this.cancelAnchor_ = void 0, e.projection && tr(), e.center && (e.center = xr(e.center, this.projection_)), e.extent && (e.extent = Cr(e.extent, this.projection_)), this.applyOptions_(e);
	}
	applyOptions_(e) {
		let t = Object.assign({}, e);
		for (let e in Zo) delete t[e];
		this.setProperties(t, !0);
		let n = ps(e);
		this.maxResolution_ = n.maxResolution, this.minResolution_ = n.minResolution, this.zoomFactor_ = n.zoomFactor, this.resolutions_ = e.resolutions, this.padding_ = e.padding, this.minZoom_ = n.minZoom;
		let r = fs(e), i = n.constraint, a = ms(e);
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
			n.center && (n = Object.assign({}, n), n.center = xr(n.center, this.getProjection())), n.anchor && (n = Object.assign({}, n), n.anchor = xr(n.anchor, this.getProjection())), t[e] = n;
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
			n && ds(n, !0);
			return;
		}
		let i = Date.now(), a = this.targetCenter_.slice(), o = this.targetResolution_, s = this.targetRotation_, c = [];
		for (; r < t; ++r) {
			let e = arguments[r], t = {
				start: i,
				complete: !1,
				anchor: e.anchor,
				duration: e.duration === void 0 ? 1e3 : e.duration,
				easing: e.easing || Ne,
				callback: n
			};
			if (e.center && (t.sourceCenter = a, t.targetCenter = e.center.slice(), a = t.targetCenter), e.zoom === void 0 ? e.resolution && (t.sourceResolution = o, t.targetResolution = e.resolution, o = t.targetResolution) : (t.sourceResolution = o, t.targetResolution = this.getResolutionForZoom(e.zoom), o = t.targetResolution), e.rotation !== void 0) {
				t.sourceRotation = s;
				let n = Ft(e.rotation - s + Math.PI, 2 * Math.PI) - Math.PI;
				t.targetRotation = s + n, s = t.targetRotation;
			}
			hs(t) ? t.complete = !0 : i += t.duration, c.push(t);
		}
		this.animations_.push(c), this.setHint(V.ANIMATING, 1), this.updateAnimations_();
	}
	getAnimating() {
		return this.hints_[V.ANIMATING] > 0;
	}
	getInteracting() {
		return this.hints_[V.INTERACTING] > 0;
	}
	cancelAnimations() {
		this.setHint(V.ANIMATING, -this.hints_[V.ANIMATING]);
		let e;
		for (let t = 0, n = this.animations_.length; t < n; ++t) {
			let n = this.animations_[t];
			if (n[0].callback && ds(n[0].callback, !1), !e) for (let t = 0, r = n.length; t < r; ++t) {
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
					let e = c === 1 ? Ft(a.targetRotation + Math.PI, 2 * Math.PI) - Math.PI : a.sourceRotation + c * (a.targetRotation - a.sourceRotation);
					if (a.anchor) {
						let t = this.constraints_.rotation(e, !0);
						this.targetCenter_ = this.calculateCenterRotate(t, a.anchor);
					}
					this.nextRotation_ = a.targetRotation, this.targetRotation_ = e;
				}
				if (this.applyTargetState_(!0), t = !0, !a.complete) break;
			}
			if (i) {
				this.animations_[n] = null, this.setHint(V.ANIMATING, -1), this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
				let e = r[0].callback;
				e && ds(e, !0);
			}
		}
		this.animations_ = this.animations_.filter(Boolean), t && this.updateAnimationKey_ === void 0 && (this.updateAnimationKey_ = requestAnimationFrame(this.updateAnimations_.bind(this)));
	}
	calculateCenterRotate(e, t) {
		let n, r = this.getCenterInternal();
		return r !== void 0 && (n = [r[0] - t[0], r[1] - t[1]], Wt(n, e - this.getRotation()), Vt(n, t)), n;
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
		return e && br(e, this.getProjection());
	}
	getCenterInternal() {
		return this.get(Zo.CENTER);
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
		return Sr(this.calculateExtentInternal(e), this.getProjection());
	}
	calculateExtentInternal(e) {
		e ||= this.getViewportSizeMinusPadding_();
		let t = this.getCenterInternal();
		L(t, "The view center is not defined");
		let n = this.getResolution();
		L(n !== void 0, "The view resolution is not defined");
		let r = this.getRotation();
		return L(r !== void 0, "The view rotation is not defined"), ut(t, n, r, e);
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
		return this.get(Zo.RESOLUTION);
	}
	getResolutions() {
		return this.resolutions_;
	}
	getResolutionForExtent(e, t) {
		return this.getResolutionForExtentInternal(Cr(e, this.getProjection()), t);
	}
	getResolutionForExtentInternal(e, t) {
		t ||= this.getViewportSizeMinusPadding_();
		let n = I(e) / t[0], r = ft(e) / t[1];
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
		return this.get(Zo.ROTATION);
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
			r = gs(r, this.getViewportSize_(), [e[0] / 2 + i[3], e[1] / 2 + i[0]], t, n);
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
			let t = kt(Math.floor(e), 0, this.resolutions_.length - 2), n = this.resolutions_[t] / this.resolutions_[t + 1];
			return this.resolutions_[t] / n ** +kt(e - t, 0, 1);
		}
		return this.maxResolution_ / this.zoomFactor_ ** +(e - this.minZoom_);
	}
	fit(e, t) {
		let n;
		if (L(Array.isArray(e) || typeof e.getSimplifiedGeometry == "function", "Invalid extent or geometry provided as `geometry`"), Array.isArray(e)) L(!vt(e), "Cannot fit empty extent provided as `geometry`"), n = $i(Cr(e, this.getProjection()));
		else if (e.getType() === "Circle") {
			let t = Cr(e.getExtent(), this.getProjection());
			n = $i(t), n.rotate(this.getRotation(), ct(t));
		} else {
			let t = yr();
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
		let c = this.getRotation(), l = Math.sin(c), u = Math.cos(c), d = ct(o);
		d[0] += (r[1] - r[3]) / 2 * s, d[1] += (r[0] - r[2]) / 2 * s;
		let f = d[0] * u - d[1] * l, p = d[1] * u + d[0] * l, m = this.getConstrainedCenter([f, p], s), h = t.callback ? t.callback : C;
		t.duration === void 0 ? (this.targetResolution_ = s, this.targetCenter_ = m, this.applyTargetState_(!1, !0), ds(h, !0)) : this.animateInternal({
			resolution: s,
			center: m,
			duration: t.duration,
			easing: t.easing
		}, h);
	}
	centerOn(e, t, n) {
		this.centerOnInternal(xr(e, this.getProjection()), t, n);
	}
	centerOnInternal(e, t, n) {
		this.setCenterInternal(gs(e, t, n, this.getResolution(), this.getRotation()));
	}
	calculateCenterShift(e, t, n, r) {
		let i, a = this.padding_;
		if (a && e) {
			let o = this.getViewportSizeMinusPadding_(-n), s = gs(e, r, [o[0] / 2 + a[3], o[1] / 2 + a[0]], t, n);
			i = [e[0] - s[0], e[1] - s[1]];
		}
		return i;
	}
	isDef() {
		return !!this.getCenterInternal() && this.getResolution() !== void 0;
	}
	adjustCenter(e) {
		let t = br(this.targetCenter_, this.getProjection());
		this.setCenter([t[0] + e[0], t[1] + e[1]]);
	}
	adjustCenterInternal(e) {
		let t = this.targetCenter_;
		this.setCenterInternal([t[0] + e[0], t[1] + e[1]]);
	}
	adjustResolution(e, t) {
		t &&= xr(t, this.getProjection()), this.adjustResolutionInternal(e, t);
	}
	adjustResolutionInternal(e, t) {
		let n = this.getAnimating() || this.getInteracting(), r = this.getViewportSize_(this.getRotation()), i = this.constraints_.resolution(this.targetResolution_ * e, 0, r, n);
		t && (this.targetCenter_ = this.calculateCenterZoom(i, t)), this.targetResolution_ *= e, this.applyTargetState_();
	}
	adjustZoom(e, t) {
		this.adjustResolution(this.zoomFactor_ ** +-e, t);
	}
	adjustRotation(e, t) {
		t &&= xr(t, this.getProjection()), this.adjustRotationInternal(e, t);
	}
	adjustRotationInternal(e, t) {
		let n = this.getAnimating() || this.getInteracting(), r = this.constraints_.rotation(this.targetRotation_ + e, n);
		t && (this.targetCenter_ = this.calculateCenterRotate(r, t)), this.targetRotation_ += e, this.applyTargetState_();
	}
	setCenter(e) {
		this.setCenterInternal(e && xr(e, this.getProjection()));
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
		this.get(Zo.ROTATION) !== r && this.set(Zo.ROTATION, r), this.get(Zo.RESOLUTION) !== a && (this.set(Zo.RESOLUTION, a), this.set("zoom", this.getZoom(), !0)), o && this.get(Zo.CENTER) && Ut(this.get(Zo.CENTER), o) || this.set(Zo.CENTER, o), this.getAnimating() && !e && this.cancelAnimations(), this.cancelAnchor_ = void 0;
	}
	resolveConstraints(e, t, n) {
		e = e === void 0 ? 200 : e;
		let r = t || 0, i = this.constraints_.rotation(this.targetRotation_), a = this.getViewportSize_(i), o = this.constraints_.resolution(this.targetResolution_, r, a), s = this.constraints_.center(this.targetCenter_, o, a, !1, this.calculateCenterShift(this.targetCenter_, o, i, a));
		e === 0 && !this.cancelAnchor_ ? (this.targetResolution_ = o, this.targetRotation_ = i, this.targetCenter_ = s, this.applyTargetState_()) : (n ||= e === 0 ? this.cancelAnchor_ : void 0, this.cancelAnchor_ = void 0, this.getResolution() === o && this.getRotation() === i && this.getCenterInternal() && Ut(this.getCenterInternal(), s) || (this.getAnimating() && this.cancelAnimations(), this.animateInternal({
			rotation: i,
			center: s,
			resolution: o,
			duration: e,
			easing: Me,
			anchor: n
		})));
	}
	beginInteraction() {
		this.resolveConstraints(0), this.setHint(V.INTERACTING, 1);
	}
	endInteraction(e, t, n) {
		n &&= xr(n, this.getProjection()), this.endInteractionInternal(e, t, n);
	}
	endInteractionInternal(e, t, n) {
		this.getInteracting() && (this.setHint(V.INTERACTING, -1), this.resolveConstraints(e, t, n));
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
function ds(e, t) {
	setTimeout(function() {
		e(t);
	}, 0);
}
function fs(e) {
	if (e.extent !== void 0) {
		let t = e.smoothExtentConstraint === void 0 || e.smoothExtentConstraint;
		return Qo(e.extent, e.constrainOnlyCenter, t);
	}
	let t = ur(e.projection, "EPSG:3857");
	if (e.multiWorld !== !0 && t.isGlobal()) {
		let e = t.getExtent().slice();
		return e[0] = -Infinity, e[2] = Infinity, Qo(e, !1, !1);
	}
	return $o;
}
function ps(e) {
	let t, n, r, i = e.minZoom === void 0 ? ls : e.minZoom, a = e.maxZoom === void 0 ? 28 : e.maxZoom, o = e.zoomFactor === void 0 ? 2 : e.zoomFactor, s = e.multiWorld !== void 0 && e.multiWorld, c = e.smoothResolutionConstraint === void 0 || e.smoothResolutionConstraint, l = e.showFullExtent !== void 0 && e.showFullExtent, u = ur(e.projection, "EPSG:3857"), d = u.getExtent(), f = e.constrainOnlyCenter, p = e.extent;
	if (!s && !p && u.isGlobal() && (f = !1, p = d), e.resolutions !== void 0) {
		let o = e.resolutions;
		n = o[i], r = o[a] === void 0 ? o[o.length - 1] : o[a], t = e.constrainResolution ? ns(o, c, !f && p, l) : is(n, r, c, !f && p, l);
	} else {
		let s = (d ? Math.max(I(d), ft(d)) : 360 * Xt.degrees / u.getMetersPerUnit()) / 256 / 2 ** ls, m = s / 2 ** 28;
		n = e.maxResolution, n === void 0 ? n = s / o ** +i : i = 0, r = e.minResolution, r === void 0 && (r = e.maxZoom === void 0 ? m : e.maxResolution === void 0 ? s / o ** +a : n / o ** +a), a = i + Math.floor(Math.log(n / r) / Math.log(o)), r = n / o ** +(a - i), t = e.constrainResolution ? rs(o, n, r, c, !f && p, l) : is(n, r, c, !f && p, l);
	}
	return {
		constraint: t,
		maxResolution: n,
		minResolution: r,
		minZoom: i,
		zoomFactor: o
	};
}
function ms(e) {
	if (e.enableRotation === void 0 || e.enableRotation) {
		let t = e.constrainRotation;
		return t === void 0 || t === !0 ? cs() : t === !1 ? os : typeof t == "number" ? ss(t) : os;
	}
	return as;
}
function hs(e) {
	return !(e.sourceCenter && e.targetCenter && !Ut(e.sourceCenter, e.targetCenter) || e.sourceResolution !== e.targetResolution || e.sourceRotation !== e.targetRotation);
}
function gs(e, t, n, r, i) {
	let a = Math.cos(-i), o = Math.sin(-i), s = e[0] * a - e[1] * o, c = e[1] * a + e[0] * o;
	return s += (t[0] / 2 - n[0]) * r, c += (n[1] - t[1] / 2) * r, o = -o, [s * a - c * o, c * a + s * o];
}
// node_modules/ol/layer/Property.js
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
}, _s = class extends j {
	constructor(e) {
		super(), this.on, this.once, this.un, this.background_ = e.background;
		let t = Object.assign({}, e);
		typeof e.properties == "object" && (delete t.properties, Object.assign(t, e.properties)), t[H.OPACITY] = e.opacity === void 0 ? 1 : e.opacity, L(typeof t[H.OPACITY] == "number", "Layer opacity must be a number"), t[H.VISIBLE] = e.visible === void 0 || e.visible, t[H.Z_INDEX] = e.zIndex, t[H.MAX_RESOLUTION] = e.maxResolution === void 0 ? Infinity : e.maxResolution, t[H.MIN_RESOLUTION] = e.minResolution === void 0 ? 0 : e.minResolution, t[H.MIN_ZOOM] = e.minZoom === void 0 ? -Infinity : e.minZoom, t[H.MAX_ZOOM] = e.maxZoom === void 0 ? Infinity : e.maxZoom, this.className_ = t.className === void 0 ? "ol-layer" : t.className, delete t.className, this.setProperties(t), this.state_ = null;
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
		return t.opacity = kt(Math.round(this.getOpacity() * 100) / 100, 0, 1), t.visible = this.getVisible(), t.extent = this.getExtent(), t.zIndex = n === void 0 && !t.managed ? Infinity : n, t.maxResolution = this.getMaxResolution(), t.minResolution = Math.max(this.getMinResolution(), 0), t.minZoom = this.getMinZoom(), t.maxZoom = this.getMaxZoom(), this.state_ = t, t;
	}
	getLayersArray(e) {
		return _();
	}
	getLayerStatesArray(e) {
		return _();
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
		return _();
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
		L(typeof e == "number", "Layer opacity must be a number"), this.set(H.OPACITY, e);
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
}, vs = class extends _s {
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
		this.changed();
		let e = this.getSource();
		e && !this.sourceReady_ && e.getState() === "ready" && (this.sourceReady_ = !0, this.dispatchEvent("sourceready"));
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
		!e && n && (e = n.getView()), t = e instanceof us ? {
			viewState: e.getState(),
			extent: e.calculateExtent()
		} : e;
		let r = t;
		!r.layerStatesArray && n && (r.layerStatesArray = n.getLayerGroup().getLayerStatesArray());
		let i;
		if (r.layerStatesArray) {
			if (i = r.layerStatesArray.find((e) => e.layer === this), !i) return !1;
		} else i = this.getLayerState();
		let a = this.getExtent();
		return ys(i, r.viewState) && (!a || _t(a, r.extent));
	}
	getAttributions(e) {
		if (!this.isVisible(e)) return [];
		let t = this.getSource()?.getAttributions();
		if (!t) return [];
		let n = t(e instanceof us ? e.getViewStateAndExtent() : e);
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
		this.mapPrecomposeKey_ &&= (o(this.mapPrecomposeKey_), null), e || this.changed(), this.mapRenderKey_ &&= (o(this.mapRenderKey_), null), e && (this.mapPrecomposeKey_ = i(e, zo.PRECOMPOSE, this.handlePrecompose_, this), this.mapRenderKey_ = i(this, s.CHANGE, e.render, e), this.changed());
	}
	handlePrecompose_(e) {
		let t = e.frameState?.layerStatesArray;
		if (!t) return;
		let n = this.getLayerState(!1);
		L(!t.some((e) => e.layer === n.layer), "A layer can only be added to the map once. Use either `layer.setMap()` or `map.addLayer()`, not both."), t.push(n);
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
function ys(e, t) {
	if (!e.visible) return !1;
	let n = t.resolution;
	if (n < e.minResolution || n >= e.maxResolution) return !1;
	let r = t.zoom;
	return r > e.minZoom && r <= e.maxZoom;
}
// node_modules/ol/layer/TileProperty.js
var bs = {
	PRELOAD: "preload",
	USE_INTERIM_TILES_ON_ERROR: "useInterimTilesOnError"
}, xs = class extends vs {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e), n = e.cacheSize;
		delete e.cacheSize, delete t.preload, delete t.useInterimTilesOnError, super(t), this.on, this.once, this.un, this.cacheSize_ = n, this.setPreload(e.preload === void 0 ? 0 : e.preload), this.setUseInterimTilesOnError(e.useInterimTilesOnError === void 0 || e.useInterimTilesOnError);
	}
	getCacheSize() {
		return this.cacheSize_;
	}
	getPreload() {
		return this.get(bs.PRELOAD);
	}
	setPreload(e) {
		this.set(bs.PRELOAD, e);
	}
	getUseInterimTilesOnError() {
		return this.get(bs.USE_INTERIM_TILES_ON_ERROR);
	}
	setUseInterimTilesOnError(e) {
		this.set(bs.USE_INTERIM_TILES_ON_ERROR, e);
	}
	getData(e) {
		return super.getData(e);
	}
}, Ss = class extends xs {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new Xo(this, { cacheSize: this.getCacheSize() });
	}
}, Cs = class {
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
		let n = ws(e, t);
		return n in this.cache_ ? this.cache_[n] : null;
	}
	getPattern(e, t) {
		let n = ws(e, t);
		return n in this.patternCache_ ? this.patternCache_[n] : null;
	}
	set(e, t, n, r) {
		let i = ws(e, t), a = i in this.cache_;
		if (n && (this.cache_[i] = n), r && n) {
			if (n.getImageState() === B.IDLE && n.load(), n.getImageState() === B.LOADING) n.ready().then(() => {
				let e = ye().createPattern(n.getImage(1), "repeat");
				e && (this.patternCache_[i] = e);
			});
			else {
				let e = ye().createPattern(n.getImage(1), "repeat");
				e && (this.patternCache_[i] = e);
			}
		}
		a || ++this.cacheSize_;
	}
	setSize(e) {
		this.maxCacheSize_ = e, this.expire();
	}
};
function ws(e, t) {
	let n = t ? Io(t) : "null";
	return e + ":" + n;
}
var Ts = new Cs(), Es = null, Ds = class extends D {
	constructor(e, t, n, r, i) {
		super(), this.hitDetectionImage_ = null, this.image_ = e, this.crossOrigin_ = n?.crossOrigin ?? null, this.referrerPolicy_ = n?.referrerPolicy, this.canvas_ = {}, this.color_ = i, this.imageState_ = r === void 0 ? B.IDLE : r, this.size_ = e && e.width && e.height ? [e.width, e.height] : null, this.src_ = t, this.tainted_, this.ready_ = null;
	}
	initializeImage_() {
		this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_), this.referrerPolicy_ !== void 0 && (this.image_.referrerPolicy = this.referrerPolicy_);
	}
	isTainted_() {
		if (this.tainted_ === void 0 && this.imageState_ === B.LOADED) {
			let e = this.image_;
			if (!e) return !1;
			Es ||= _e(1, 1, void 0, { willReadFrequently: !0 }), Es.drawImage(e, 0, 0);
			try {
				Es.getImageData(0, 0, 1, 1), this.tainted_ = !1;
			} catch {
				Es = null, this.tainted_ = !0;
			}
		}
		return this.tainted_ === !0;
	}
	dispatchChangeEvent_() {
		this.dispatchEvent(s.CHANGE);
	}
	handleImageError_() {
		this.imageState_ = B.ERROR, this.dispatchChangeEvent_();
	}
	handleImageLoad_() {
		this.imageState_ = B.LOADED;
		let e = this.image_;
		e && "width" in e && "height" in e && (this.size_ = [e.width, e.height]), this.dispatchChangeEvent_();
	}
	getImage(e) {
		this.image_ || this.initializeImage_(), this.replaceColor_(e);
		let t = this.image_;
		return this.canvas_[e] ? this.canvas_[e] : t;
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
				let e = this.size_;
				if (!e) return this.image_;
				let t = e[0], n = e[1], r = _e(t, n);
				r.fillRect(0, 0, t, n), this.hitDetectionImage_ = r.canvas;
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
		if (this.imageState_ === B.IDLE) {
			this.image_ || this.initializeImage_(), this.imageState_ = B.LOADING;
			try {
				this.src_ !== void 0 && (this.image_.src = this.src_);
			} catch {
				this.handleImageError_();
			}
			this.image_ instanceof HTMLImageElement && za(this.image_, this.src_).then((e) => {
				this.image_ = e, this.handleImageLoad_();
			}).catch(this.handleImageError_.bind(this));
		}
	}
	replaceColor_(e) {
		if (!this.color_ || this.canvas_[e] || this.imageState_ !== B.LOADED) return;
		let t = this.image_;
		if (!t) return;
		let n = _e(Math.ceil(t.width * e), Math.ceil(t.height * e)), r = n.canvas;
		n.scale(e, e), n.drawImage(t, 0, 0), n.globalCompositeOperation = "multiply", n.fillStyle = wo(this.color_), n.fillRect(0, 0, r.width / e, r.height / e), n.globalCompositeOperation = "destination-in", n.drawImage(t, 0, 0), this.canvas_[e] = r;
	}
	ready() {
		return this.ready_ ||= new Promise((e) => {
			if (this.imageState_ === B.LOADED || this.imageState_ === B.ERROR) e();
			else {
				let t = () => {
					(this.imageState_ === B.LOADED || this.imageState_ === B.ERROR) && (this.removeEventListener(s.CHANGE, t), e());
				};
				this.addEventListener(s.CHANGE, t);
			}
		}), this.ready_;
	}
};
function Os(e, t, n, r, i, a) {
	let o = t === void 0 ? void 0 : Ts.get(t, i);
	return o || (o = new Ds(e, e && "src" in e ? e.src || void 0 : t, n, r, i), t !== void 0 && Ts.set(t, i, o, a)), a && o && t !== void 0 && !Ts.getPattern(t, i) && Ts.set(t, i, o, a), o;
}
// node_modules/ol/colorlike.js
function ks(e) {
	return e ? Array.isArray(e) ? Lo(e) : typeof e == "object" && "src" in e ? As(e) : e : null;
}
function As(e) {
	if (!e.offset || !e.size) return Ts.getPattern(e.src, e.color ?? null);
	let t = e.src + ":" + e.offset, n = Ts.getPattern(t, e.color ?? null);
	if (n) return n;
	let r = Ts.get(e.src, null);
	if (!r || r.getImageState() !== B.LOADED) return null;
	let i = _e(e.size[0], e.size[1]);
	return i.drawImage(r.getImage(1), e.offset[0], e.offset[1], e.size[0], e.size[1], 0, 0, e.size[0], e.size[1]), Os(i.canvas, t, {}, B.LOADED, e.color ?? null, !0), Ts.getPattern(t, e.color ?? null);
}
// node_modules/ol/render/VectorContext.js
var js = class {
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
}, Ms = "10px sans-serif", Ns = "#000", Ps = "round", Fs = [], Is = "round", Ls = "#000", Rs = "center", zs = "middle", Bs = [
	0,
	0,
	0,
	0
], Vs = new j(), Hs = null, Us, Ws = {}, Gs = /* @__PURE__ */ new Set([
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
function Ks(e, t, n) {
	return `${e} ${t} 16px "${n}"`;
}
var qs = (function() {
	let e, t;
	async function r(e) {
		await t.ready;
		let n = ce(e);
		if (!n) return !1;
		let r = n.families[0].toLowerCase(), i = n.weight, a = [];
		return t.forEach((e) => {
			let t = e.family.replace(/^['"]|['"]$/g, "").toLowerCase(), o = se[e.weight] || e.weight;
			t === r && e.style === n.style && o == i && a.push(e);
		}), a.length !== 0 && (await Promise.all(a.map((e) => e.load().then(() => !0, () => !1)))).some((e) => e);
	}
	async function i() {
		await t.ready;
		let a = !0, o = Vs.getProperties(), s = Object.keys(o).filter((e) => o[e] < 100);
		for (let e = s.length - 1; e >= 0; --e) {
			let t = s[e], i = o[t];
			i < 100 && (await r(t) ? (n(Ws), Vs.set(t, 100)) : (i += 10, Vs.set(t, i, !0), i < 100 && (a = !1)));
		}
		e = void 0, a || (e = setTimeout(i, 100));
	}
	return async function(n) {
		t ||= me ? self.fonts : document.fonts;
		let r = ce(n);
		if (!r) return;
		let a = r.families, o = !1;
		for (let e of a) {
			if (Gs.has(e)) continue;
			let t = Ks(r.style, r.weight, e);
			Vs.get(t) === void 0 && (Vs.set(t, 0, !0), o = !0);
		}
		o && (clearTimeout(e), e = setTimeout(i, 100));
	};
})(), Js = (function() {
	let e;
	return function(t) {
		let n = Ws[t];
		if (n == null) {
			if (me) {
				let e = ce(t), r = Ys(t, "Žg");
				n = (!e || isNaN(Number(e.lineHeight)) ? 1.2 : Number(e.lineHeight)) * (r.actualBoundingBoxAscent + r.actualBoundingBoxDescent);
			} else e || (e = document.createElement("div"), e.innerHTML = "M", e.style.minHeight = "0", e.style.maxHeight = "none", e.style.height = "auto", e.style.padding = "0", e.style.border = "none", e.style.position = "absolute", e.style.display = "block", e.style.left = "-99999px"), e.style.font = t, document.body.appendChild(e), n = e.offsetHeight, document.body.removeChild(e);
			Ws[t] = n;
		}
		return n;
	};
})();
function Ys(e, t) {
	return Hs ||= _e(1, 1), e != Us && (Hs.font = e, Us = Hs.font), Hs.measureText(t);
}
function Xs(e, t) {
	return Ys(e, t).width;
}
function Zs(e, t, n) {
	if (t in n) return n[t];
	let r = t.split("\n").reduce((t, n) => Math.max(t, Xs(e, n)), 0);
	return n[t] = r, r;
}
function Qs(e, t) {
	let n = [], r = [], i = [], a = 0, o = 0, s = 0, c = 0;
	for (let l = 0, u = t.length; l <= u; l += 2) {
		let d = t[l];
		if (d === "\n" || l === u) {
			a = Math.max(a, o), i.push(o), o = 0, s += c, c = 0;
			continue;
		}
		let f = t[l + 1] || e.font, p = Xs(f, d);
		n.push(p), o += p;
		let m = Js(f);
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
function $s(e, t, n, r, i, a, o, s, c, l, u) {
	e.save(), n !== 1 && (e.globalAlpha === void 0 ? e.globalAlpha = function(e) {
		return e.globalAlpha *= n;
	} : e.globalAlpha *= n), t && e.transform.apply(e, t), r.contextInstructions ? (e.translate(c, l), e.scale(u[0], u[1]), ec(r, e)) : u[0] < 0 || u[1] < 0 ? (e.translate(c, l), e.scale(u[0], u[1]), e.drawImage(r, i, a, o, s, 0, 0, o, s)) : e.drawImage(r, i, a, o, s, c, l, o * u[0], s * u[1]), e.restore();
}
function ec(e, t) {
	let n = t, r = e.contextInstructions;
	for (let e = 0, t = r.length; e < t; e += 2) if (Array.isArray(r[e + 1])) n[r[e]].apply(n, r[e + 1]);
	else {
		let t = r[e];
		n[t] = r[e + 1];
	}
}
// node_modules/ol/render/canvas/Instruction.js
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
}, tc = [U.FILL], nc = [U.STROKE], rc = [U.BEGIN_PATH], ic = [U.CLOSE_PATH], ac = class extends js {
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
		for (let o = 0, s = e.length; o < s; o += t) r[0] = e[o], r[1] = e[o + 1], We(n, r) && (i[a++] = r[0], i[a++] = r[1]);
		return a;
	}
	appendFlatLineCoordinates(e, t, n, r, i, a) {
		let o = this.coordinates, s = o.length, c = this.getBufferedMaxExtent();
		a && (t += r);
		let l = e[t], u = e[t + 1], d = this.tmpCoordinate_, f = !0, p, m, h;
		for (p = t + r; p < n; p += r) d[0] = e[p], d[1] = e[p + 1], h = qe(c, d), h === m ? h === Re.INTERSECTING ? (o[s++] = d[0], o[s++] = d[1], f = !1) : f = !0 : (f &&= (o[s++] = l, o[s++] = u, !1), o[s++] = d[0], o[s++] = d[1]), l = d[0], u = d[1], m = h;
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
				c = e.getOrientedFlatCoordinates(), u = [];
				let t = e.getEndss();
				f = 0;
				for (let e = 0, n = t.length; e < n; ++e) {
					let n = [];
					f = this.drawCustomCoordinates_(c, f, t[e], o, n), u.push(n);
				}
				this.instructions.push([
					U.CUSTOM,
					s,
					u,
					e,
					n,
					mi,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					u,
					e,
					r || n,
					mi,
					i
				]);
				break;
			case "Polygon":
			case "MultiLineString":
				d = [], c = a == "Polygon" ? e.getOrientedFlatCoordinates() : e.getFlatCoordinates(), f = this.drawCustomCoordinates_(c, 0, e.getEnds(), o, d), this.instructions.push([
					U.CUSTOM,
					s,
					d,
					e,
					n,
					pi,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					d,
					e,
					r || n,
					pi,
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
					fi,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					l,
					e,
					r || n,
					fi,
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
					fi,
					i
				]), this.hitDetectionInstructions.push([
					U.CUSTOM,
					s,
					l,
					e,
					r || n,
					fi,
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
			t.fillPatternScale = n && typeof n == "object" && "src" in n ? this.pixelRatio : 1, t.fillStyle = ks(n || "#000") ?? void 0;
		} else t.fillStyle = void 0;
		return t;
	}
	strokeStyleToState(e, t = {}) {
		if (e) {
			t.strokeStyle = ks(e.getColor() || "#000") ?? void 0;
			let n = e.getLineCap();
			t.lineCap = n === void 0 ? Ps : n;
			let r = e.getLineDash();
			t.lineDash = r ? r.slice() : Fs, t.lineDashOffset = e.getLineDashOffset() || 0;
			let i = e.getLineJoin();
			t.lineJoin = i === void 0 ? Is : i;
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
		e.currentStrokeStyle == n && e.currentLineCap == r && (i == e.currentLineDash || i && e.currentLineDash && h(e.currentLineDash, i)) && e.currentLineDashOffset == a && e.currentLineJoin == o && e.currentLineWidth == s && e.currentMiterLimit == c && e.currentStrokeOffset == l || (t.call(this, e), e.currentStrokeStyle = n, e.currentLineCap = r, e.currentLineDash = i, e.currentLineDashOffset = a, e.currentLineJoin = o, e.currentLineWidth = s, e.currentMiterLimit = c, e.currentStrokeOffset = l);
	}
	endGeometry(e) {
		this.beginGeometryInstruction1_[2] = this.instructions.length, this.beginGeometryInstruction1_ = null, this.beginGeometryInstruction2_[2] = this.hitDetectionInstructions.length, this.beginGeometryInstruction2_ = null;
		let t = [U.END_GEOMETRY, e];
		this.instructions.push(t), this.hitDetectionInstructions.push(t);
	}
	getBufferedMaxExtent() {
		if (!this.bufferedMaxExtent_ && (this.bufferedMaxExtent_ = He(this.maxExtent), this.maxLineWidth > 0)) {
			let e = this.resolution * (this.maxLineWidth + 1) / 2;
			Ve(this.bufferedMaxExtent_, e, this.bufferedMaxExtent_);
		}
		return this.bufferedMaxExtent_;
	}
}, oc = class extends ac {
	constructor(e, t, n, r) {
		super(e, t, n, r), this.hitDetectionImage_ = null, this.image_ = null, this.imagePixelRatio_ = void 0, this.anchorX_ = void 0, this.anchorY_ = void 0, this.height_ = void 0, this.opacity_ = void 0, this.originX_ = void 0, this.originY_ = void 0, this.rotateWithView_ = void 0, this.rotation_ = void 0, this.scale_ = void 0, this.width_ = void 0, this.declutterMode_ = void 0, this.declutterImageWithText_ = void 0;
	}
	drawPoint(e, t, n) {
		if (!this.image_ || this.maxExtent && !We(this.maxExtent, e.getFlatCoordinates())) return;
		this.beginGeometry(e, t, n ?? 0);
		let r = e.getFlatCoordinates(), i = e.getStride(), a = this.coordinates.length, o = this.appendFlatPointCoordinates(r, i), s = this.imagePixelRatio_ ?? 1, c = this.anchorX_ ?? 0, l = this.anchorY_ ?? 0, u = this.height_ ?? 0, d = this.originX_ ?? 0, f = this.originY_ ?? 0, p = this.scale_ ?? [1, 1], m = this.width_ ?? 0;
		this.instructions.push([
			U.DRAW_IMAGE,
			a,
			o,
			this.image_,
			c * s,
			l * s,
			Math.ceil(u * s),
			this.opacity_,
			d * s,
			f * s,
			this.rotateWithView_,
			this.rotation_,
			[p[0] * this.pixelRatio / s, p[1] * this.pixelRatio / s],
			Math.ceil(m * s),
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
		this.beginGeometry(e, t, n ?? 0);
		let r = e.getFlatCoordinates(), i = [];
		for (let t = 0, n = r.length; t < n; t += e.getStride()) (!this.maxExtent || We(this.maxExtent, r.slice(t, t + 2))) && i.push(r[t], r[t + 1]);
		let a = this.coordinates.length, o = this.appendFlatPointCoordinates(i, 2), s = this.imagePixelRatio_ ?? 1, c = this.anchorX_ ?? 0, l = this.anchorY_ ?? 0, u = this.height_ ?? 0, d = this.originX_ ?? 0, f = this.originY_ ?? 0, p = this.scale_ ?? [1, 1], m = this.width_ ?? 0;
		this.instructions.push([
			U.DRAW_IMAGE,
			a,
			o,
			this.image_,
			c * s,
			l * s,
			Math.ceil(u * s),
			this.opacity_,
			d * s,
			f * s,
			this.rotateWithView_,
			this.rotation_,
			[p[0] * this.pixelRatio / s, p[1] * this.pixelRatio / s],
			Math.ceil(m * s),
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
		n && r && i && (this.imagePixelRatio_ = e.getPixelRatio(this.pixelRatio), this.anchorX_ = n[0], this.anchorY_ = n[1], this.hitDetectionImage_ = e.getHitDetectionImage(), this.image_ = e.getImage(this.pixelRatio), this.height_ = r[1], this.opacity_ = e.getOpacity(), this.originX_ = i[0], this.originY_ = i[1], this.rotateWithView_ = e.getRotateWithView(), this.rotation_ = e.getRotation(), this.scale_ = e.getScaleArray(), this.width_ = r[0], this.declutterMode_ = e.getDeclutterMode(), this.declutterImageWithText_ = t);
	}
}, sc = class extends ac {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	drawFlatCoordinates_(e, t, n, r, i) {
		let a = (i ?? 0) * this.pixelRatio, o = this.coordinates.length, s = this.appendFlatLineCoordinates(e, t, n, r, !1, !1);
		return this.instructions.push([
			U.MOVE_TO_LINE_TO,
			o,
			s,
			a
		]), this.hitDetectionInstructions.push([
			U.MOVE_TO_LINE_TO,
			o,
			s,
			i ?? 0
		]), n;
	}
	drawLineString(e, t, n) {
		let r = this.state;
		if (!r) return;
		let i = r.strokeStyle, a = r.lineWidth, o = r.strokeOffset;
		if (i === void 0 || a === void 0) return;
		this.updateStrokeStyle(r, this.applyStroke), this.beginGeometry(e, t, n ?? 0), this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Ls,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Fs,
			0
		], rc);
		let s = e.getFlatCoordinates(), c = e.getStride();
		this.drawFlatCoordinates_(s, 0, s.length, c, o), this.hitDetectionInstructions.push(nc), this.endGeometry(t);
	}
	drawMultiLineString(e, t, n) {
		let r = this.state;
		if (!r) return;
		let i = r.strokeStyle, a = r.lineWidth, o = r.strokeOffset;
		if (i === void 0 || a === void 0) return;
		this.updateStrokeStyle(r, this.applyStroke), this.beginGeometry(e, t, n ?? 0), this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Ls,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Fs,
			0
		], rc);
		let s = e.getEnds() ?? [], c = e.getFlatCoordinates(), l = e.getStride(), u = 0;
		for (let e = 0, t = s.length; e < t; ++e) u = this.drawFlatCoordinates_(c, u, s[e], l, o);
		this.hitDetectionInstructions.push(nc), this.endGeometry(t);
	}
	finish() {
		let e = this.state;
		return e && e.lastStroke != null && e.lastStroke != this.coordinates.length && this.instructions.push(nc), this.reverseHitDetectionInstructions(), this.state = null, super.finish();
	}
	applyStroke(e) {
		e.lastStroke != null && e.lastStroke != this.coordinates.length && (this.instructions.push(nc), e.lastStroke = this.coordinates.length), e.lastStroke = 0, super.applyStroke(e), this.instructions.push(rc);
	}
}, cc = class extends ac {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	drawFlatCoordinatess_(e, t, n, r, i) {
		let a = this.state, o = i ?? 0, s = a.fillStyle !== void 0, c = a.strokeStyle !== void 0, l = n.length;
		this.instructions.push(rc), this.hitDetectionInstructions.push(rc);
		for (let i = 0; i < l; ++i) {
			let a = n[i], s = this.coordinates.length, l = this.appendFlatLineCoordinates(e, t, a, r, !0, !c);
			this.instructions.push([
				U.MOVE_TO_LINE_TO,
				s,
				l,
				o * this.pixelRatio,
				!0
			]), this.hitDetectionInstructions.push([
				U.MOVE_TO_LINE_TO,
				s,
				l,
				o,
				!0
			]), c && (this.instructions.push(ic), this.hitDetectionInstructions.push(ic)), t = a;
		}
		return s && (this.instructions.push(tc), this.hitDetectionInstructions.push(tc)), c && (this.instructions.push(nc), this.hitDetectionInstructions.push(nc)), t;
	}
	drawCircle(e, t, n) {
		let r = this.state;
		if (!r) return;
		let i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawCircle(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n ?? 0), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([U.SET_FILL_STYLE, Ns]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Ls,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Fs,
			0
		]);
		let s = e.getFlatCoordinates(), c = e.getStride(), l = this.coordinates.length;
		this.appendFlatLineCoordinates(s, 0, s.length, c, !1, !1);
		let u = [
			U.CIRCLE,
			l,
			o
		];
		this.instructions.push(rc, u), this.hitDetectionInstructions.push(rc, u), r.fillStyle !== void 0 && (this.instructions.push(tc), this.hitDetectionInstructions.push(tc)), r.strokeStyle !== void 0 && (this.instructions.push(nc), this.hitDetectionInstructions.push(nc)), this.endGeometry(t);
	}
	drawPolygon(e, t, n) {
		let r = this.state;
		if (!r) return;
		let i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawPolygon(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n ?? 0), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([U.SET_FILL_STYLE, Ns]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Ls,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Fs,
			0
		]);
		let s = e.getEnds(), c = e.getOrientedFlatCoordinates(), l = e.getStride();
		this.drawFlatCoordinatess_(c, 0, s, l, o), this.endGeometry(t);
	}
	drawMultiPolygon(e, t, n) {
		let r = this.state;
		if (!r) return;
		let i = r.fillStyle, a = r.strokeStyle, o = r.strokeOffset;
		if (i === void 0 && a === void 0 || this.handleStrokeOffset_(() => this.drawMultiPolygon(e, t, n))) return;
		this.setFillStrokeStyles_(), this.beginGeometry(e, t, n ?? 0), r.fillStyle !== void 0 && this.hitDetectionInstructions.push([U.SET_FILL_STYLE, Ns]), r.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
			U.SET_STROKE_STYLE,
			Ls,
			r.lineWidth,
			r.lineCap,
			r.lineJoin,
			r.miterLimit,
			Fs,
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
			for (let n = 0, r = t.length; n < r; ++n) t[n] = ji(t[n], e);
		}
		return super.finish();
	}
	setFillStrokeStyles_() {
		let e = this.state;
		e && (this.updateFillStyle(e, this.createFill), this.updateStrokeStyle(e, this.applyStroke));
	}
	handleStrokeOffset_(e) {
		let t = this.state, n = t.fillStyle, r = t.strokeStyle, i = t.strokeOffset ?? 0;
		return Math.abs(i) > 0 && n !== void 0 && r !== void 0 && (t.strokeStyle = void 0, t.strokeOffset = 0, e(), t.fillStyle = void 0, t.strokeStyle = r, t.strokeOffset = i, e(), t.fillStyle = n, !0);
	}
}, lc = 0, uc = 1;
function dc(e, t, n, r, i, a, o, s) {
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
	return lc = u, uc = d, !0;
}
function fc(e, t, n, r) {
	let i = r[0], a = r[1], o = r[2], s = r[3], c = [], l = [], u = !1, d, f, p = 0;
	for (let r = 0, m = t.length; r < m; ++r) {
		let m = t[r], h = e[p], g = e[p + 1], _ = !1;
		for (let t = p + n; t < m; t += n) {
			let n = e[t], r = e[t + 1];
			if (dc(i, a, o, s, h, g, n, r)) {
				let e = n - h, t = r - g, i = h + lc * e, a = g + lc * t, o = h + uc * e, s = g + uc * t;
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
// node_modules/ol/geom/flat/linechunk.js
function pc(e, t, n, r, i) {
	let a = [], o = n, s = 0, c = t.slice(n, 2);
	for (; s < e && o + i < r;) {
		let [n, r] = c.slice(-2), l = t[o + i], u = t[o + i + 1], d = Math.sqrt((l - n) * (l - n) + (u - r) * (u - r));
		if (s += d, s >= e) {
			let t = (e - s + d) / d, f = It(n, l, t), p = It(r, u, t);
			c.push(f, p), a.push(c), c = [f, p], s == e && (o += i), s = 0;
		} else if (s < e) c.push(t[o + i], t[o + i + 1]), o += i;
		else {
			let e = d - s, t = It(n, l, e / d), f = It(r, u, e / d);
			c.push(t, f), a.push(c), c = [t, f], s = 0, o += i;
		}
	}
	return s > 0 && a.push(c), a;
}
// node_modules/ol/geom/flat/straightchunk.js
function mc(e, t, n, r, i) {
	let a = n, o = n, s = 0, c = 0, l = n, u, d, f, p, m, h, g, _, v, y;
	for (d = n; d < r; d += i) {
		let n = t[d], r = t[d + 1];
		m !== void 0 && h !== void 0 && (v = n - m, y = r - h, p = Math.sqrt(v * v + y * y), g !== void 0 && _ !== void 0 && f !== void 0 && (c += f, u = Math.acos((g * v + _ * y) / (f * p)), u > e && (c > s && (s = c, a = l, o = d), c = 0, l = d - i)), f = p, g = v, _ = y), m = n, h = r;
	}
	return c += p || 0, c > s ? [l, d] : [a, o];
}
// node_modules/ol/render/canvas/TextBuilder.js
var hc = {
	left: 0,
	center: .5,
	right: 1,
	top: 0,
	middle: .5,
	hanging: .2,
	alphabetic: .8,
	ideographic: .8,
	bottom: 1
}, gc = {
	Circle: cc,
	Default: ac,
	Image: oc,
	LineString: sc,
	Polygon: cc,
	Text: class extends ac {
		constructor(e, t, n, r) {
			super(e, t, n, r), this.labels_ = null, this.text_ = "", this.textOffsetX_ = 0, this.textOffsetY_ = 0, this.textRotateWithView_ = void 0, this.textKeepUpright_ = void 0, this.textRotation_ = 0, this.textFillState_ = null, this.fillStates = {}, this.fillStates[Ns] = { fillStyle: Ns }, this.textStrokeState_ = null, this.strokeStates = {}, this.textState_ = {}, this.textStates = {}, this.textKey_ = "", this.fillKey_ = "", this.strokeKey_ = "", this.declutterMode_ = void 0, this.declutterImageWithText_ = void 0;
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
				if (!_t(this.maxExtent, r)) return;
				let i = [];
				if (l = e.getFlatCoordinates(), c == "LineString") i = [l.length];
				else if (c == "MultiLineString") i = e.getEnds();
				else if (c == "Polygon") i = e.getEnds().slice(0, 1);
				else if (c == "MultiPolygon") {
					let t = e.getEndss();
					i = [];
					for (let e = 0, n = t.length; e < n; ++e) i.push(t[e][0]);
				}
				if (a.textAlign === void 0 && (c == "LineString" || c == "MultiLineString") && !Ge(this.getBufferedMaxExtent(), r)) {
					let e = fc(l, i, u, this.getBufferedMaxExtent());
					if (l = e.flatCoordinates, i = e.ends, u = 2, i.length === 0) return;
				}
				this.beginGeometry(e, t, n ?? 0);
				let d = a.repeat, f = d ? void 0 : a.textAlign, p = 0;
				for (let e = 0, t = i.length; e < t; ++e) {
					let t;
					t = d ? pc(d * this.resolution, l, p, i[e], u) : [l.slice(p, i[e])];
					for (let n = 0, r = t.length; n < r; ++n) {
						let r = t[n], c = 0, l = r.length;
						if (f == null) {
							let e = mc(a.maxAngle ?? Math.PI / 4, r, 0, r.length, 2);
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
						l = e.getFlatInteriorPoint(), !a.overflow && r && r.push(l[2] / this.resolution), u = 3;
						break;
					case "MultiPolygon":
						let t = e.getFlatInteriorPoints();
						l = [];
						for (let e = 0, n = t.length; e < n; e += 3) !a.overflow && r && r.push(t[e + 2] / this.resolution), l.push(t[e], t[e + 1]);
						if (l.length === 0) return;
						u = 2;
						break;
					default: return;
				}
				if (!l) return;
				let i = this.appendFlatPointCoordinates(l, u);
				if (i === s) return;
				if (r && (i - s) / 2 !== l.length / u) {
					let e = l, t = s / 2;
					r = r.filter((n, r) => {
						let i = o[(t + r) * 2] === e[r * u] && o[(t + r) * 2 + 1] === e[r * u + 1];
						return i || --t, i;
					});
				}
				this.saveTextStates_();
				let d = a.backgroundFill ? this.createFill(this.fillStyleToState(a.backgroundFill)) : null, f = a.backgroundStroke ? this.createStroke(this.strokeStyleToState(a.backgroundStroke)) : null;
				this.beginGeometry(e, t, n ?? 0);
				let p = a.scale ?? [1, 1], m = a.padding ?? Bs;
				if (m != Bs && (p[0] < 0 || p[1] < 0)) {
					let e = m[0], t = m[1], n = m[2], r = m[3];
					p[0] < 0 && (t = -t, r = -r), p[1] < 0 && (e = -e, n = -n), m = [
						e,
						t,
						n,
						r
					];
				}
				let h = this.pixelRatio;
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
					m == Bs ? Bs : m.map(function(e) {
						return e * h;
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
				let g = 1 / h, _ = d ? d.slice(0) : null;
				_ && (_[1] = Ns), this.hitDetectionInstructions.push([
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
					[g, g],
					NaN,
					this.declutterMode_,
					this.declutterImageWithText_,
					m,
					_,
					f,
					this.text_,
					this.textKey_,
					this.strokeKey_,
					this.fillKey_ ? Ns : this.fillKey_,
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
			let s = this.pixelRatio, c = hc[r.textBaseline || "middle"], l = this.textOffsetX_ * s, u = this.textOffsetY_ * s, d = this.text_, f = r.scale ?? [1, 1], p = n ? n.lineWidth * Math.abs(f[0]) / 2 : 0;
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
				p * s,
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
				o && Ns,
				r.maxAngle,
				s,
				u,
				i,
				p * s,
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
				t ? (r = this.textFillState_, r || (r = {}, this.textFillState_ = r), r.fillStyle = ks(t.getColor() || "#000") || "#000") : (r = null, this.textFillState_ = r);
				let a = e.getStroke();
				if (!a) i = null, this.textStrokeState_ = i;
				else {
					i = this.textStrokeState_, i || (i = {}, this.textStrokeState_ = i);
					let e = a.getLineDash(), t = a.getLineDashOffset(), n = a.getWidth(), r = a.getMiterLimit();
					i.lineCap = a.getLineCap() || "round", i.lineDash = e ? e.slice() : Fs, i.lineDashOffset = t === void 0 ? 0 : t, i.lineJoin = a.getLineJoin() || "round", i.lineWidth = n === void 0 ? 1 : n, i.miterLimit = r === void 0 ? 10 : r, i.strokeStyle = ks(a.getColor() || "#000") || "#000";
				}
				n = this.textState_;
				let o = e.getFont() || "10px sans-serif";
				qs(o);
				let s = e.getScaleArray();
				n.overflow = e.getOverflow(), n.font = o, n.maxAngle = e.getMaxAngle(), n.placement = e.getPlacement(), n.textAlign = e.getTextAlign(), n.repeat = e.getRepeat(), n.justify = e.getJustify(), n.textBaseline = e.getTextBaseline() || "middle", n.backgroundFill = e.getBackgroundFill() ?? void 0, n.backgroundStroke = e.getBackgroundStroke() ?? void 0, n.padding = e.getPadding() || Bs, n.scale = s === void 0 ? [1, 1] : s;
				let c = e.getOffsetX(), l = e.getOffsetY(), u = e.getRotateWithView(), d = e.getKeepUpright(), f = e.getRotation();
				this.text_ = e.getText() || "", this.textOffsetX_ = c === void 0 ? 0 : c, this.textOffsetY_ = l === void 0 ? 0 : l, this.textRotateWithView_ = u !== void 0 && u, this.textKeepUpright_ = d === void 0 || d, this.textRotation_ = f === void 0 ? 0 : f, this.strokeKey_ = i ? (typeof i.strokeStyle == "string" ? i.strokeStyle : y(i.strokeStyle)) + i.lineCap + i.lineDashOffset + "|" + i.lineWidth + i.lineJoin + i.miterLimit + "[" + i.lineDash.join() + "]" : "", this.textKey_ = n.font + n.scale + (n.textAlign || "?") + (n.repeat || "?") + (n.justify || "?") + (n.textBaseline || "?"), this.fillKey_ = r && r.fillStyle ? typeof r.fillStyle == "string" ? r.fillStyle : "|" + y(r.fillStyle) : "";
			}
			this.declutterMode_ = e ? e.getDeclutterMode() : void 0, this.declutterImageWithText_ = t;
		}
	}
}, _c = class {
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
			let e = gc[t];
			i = new e(this.tolerance_, this.maxExtent_, this.resolution_, this.pixelRatio_), r[t] = i;
		}
		return i;
	}
};
// node_modules/ol/geom/flat/lineoffset.js
function vc(e, t, n, r, i, a, o, s) {
	o ??= [], s ??= r;
	let c = e[t + r], l = e[t + r + 1], u = e[n - 2 * r], d = e[n - 2 * r + 1], f, p, m, h, g, _, v, y, b = 0;
	for (let x = t; x < n; x += r) {
		m = f, h = p, g = void 0, _ = void 0, x + r < n && (g = e[x + r], _ = e[x + r + 1]), a && x === t && (m = u, h = d), a && x === n - r && (g = c, _ = l), f = e[x], p = e[x + 1], [v, y] = yc(f, p, m, h, g, _, i), o[b++] = v, o[b++] = y;
		for (let t = 2; t < s; t++) o[b++] = e[x + t];
	}
	return o.length != b && (o.length = b), o;
}
function yc(e, t, n, r, i, a, o) {
	let s, c;
	n !== void 0 && r !== void 0 ? (s = e - n, c = t - r) : i !== void 0 && a !== void 0 ? (s = i - e, c = a - t) : (s = 1, c = 0);
	let l = Math.hypot(s, c), u = s / l, d = c / l;
	if (s = -d, c = u, n === void 0 || r === void 0 || i === void 0 || a === void 0) return [e + s * o, t + c * o];
	let f = Yt([e, t], [n, r], [i, a]);
	if (Math.cos(f) > .998) return [e + u * o, t + d * o];
	let p = Math.cos(f / 2), m = Math.sin(f / 2), h = m * s + p * c, g = -p * s + m * c, _ = 1 / m * h, v = 1 / m * g;
	return [e + _ * o, t + v * o];
}
function bc(e, t, n = !1) {
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
// node_modules/ol/geom/flat/textpath.js
var xc, Sc = /* @__PURE__ */ RegExp("[֑-ࣿיִ-﷿ﹰ-ﻼࠀ-࿿-]");
function Cc() {
	return xc ||= new Intl.Segmenter(void 0, { granularity: "grapheme" }), xc;
}
function wc(e, t, n, r, i, a, o, s, c, l, u, d, f = !0) {
	let p = e[t], m = e[t + 1], h = 0, g = 0, _ = 0, v = 0;
	function y() {
		h = p, g = m, t += r, p = e[t], m = e[t + 1], v += _, _ = Math.sqrt((p - h) * (p - h) + (m - g) * (m - g));
	}
	do
		y();
	while (t < n - r && v + _ < a);
	let b = _ === 0 ? 0 : (a - v) / _, x = It(h, p, b), S = It(g, m, b), C = t - r, w = v, T = a + s * c(l, i, u);
	for (; t < n - r && v + _ < T;) y();
	b = _ === 0 ? 0 : (T - v) / _;
	let E = It(h, p, b), D = It(g, m, b), O = !1;
	if (f) {
		if (d) {
			let e = [
				x,
				S,
				E,
				D
			];
			Ur(e, 0, 4, 2, d, e, e), O = e[0] > e[2];
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
	let N = Array.from(Cc().segment(i), (e) => e.segment), P = O !== Sc.test(i);
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
			let o = s * c(l, N[P ? i - e - 1 : e], u);
			if (t + r < n && v + _ < a + x + o / 2) break;
			x += o;
		}
		if (e === f) continue;
		let S = (P ? N.slice(i - e, i - f) : N.slice(f, e)).join("");
		b = _ === 0 ? 0 : (a + x / 2 - v) / _;
		let C = It(h, p, b), w = It(g, m, b);
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
// node_modules/ol/render/canvas/Executor.js
var Tc = Je(), Ec = [], Dc = [], Oc = [], kc = [];
function Ac(e) {
	return e[3].declutterBox;
}
function jc(e, t) {
	return t === "start" ? t = Sc.test(e) ? "right" : "left" : t === "end" && (t = Sc.test(e) ? "left" : "right"), hc[t];
}
function Mc(e, t, n) {
	return n > 0 && e.push("\n", ""), e.push(t, ""), e;
}
function Nc(e, t, n) {
	return n % 2 == 0 && (e += t), e;
}
var Pc = class {
	constructor(e, t, n, r, i) {
		this.overlaps = n, this.pixelRatio = t, this.resolution = e, this.alignAndScaleFill_, this.instructions = r.instructions, this.coordinates = r.coordinates, this.coordinateCache_ = {}, this.renderedTransform_ = Or(), this.hitDetectionInstructions = r.hitDetectionInstructions, this.pixelCoordinates_ = null, this.viewRotation_ = 0, this.fillStates = r.fillStates || {}, this.strokeStates = r.strokeStates || {}, this.textStates = r.textStates || {}, this.widths_ = {}, this.labels_ = {}, this.zIndexContext_ = i ? new Bo() : null;
	}
	getZIndexContext() {
		return this.zIndexContext_;
	}
	createLabel(e, t, n, r) {
		let i = e + t + n + r;
		if (this.labels_[i]) return this.labels_[i];
		let a = r ? this.strokeStates[r] : null, o = n ? this.fillStates[n] : null, s = this.textStates[t], c = this.pixelRatio, l = [s.scale[0] * c, s.scale[1] * c], u = s.justify ? hc[s.justify] : jc(Array.isArray(e) ? e[0] : e, s.textAlign || "center"), d = r && a.lineWidth ? a.lineWidth : 0, f = Array.isArray(e) ? e : String(e).split("\n").reduce(Mc, []), { width: p, height: m, widths: h, heights: g, lineWidths: _ } = Qs(s, f), v = p + d, y = [], b = (v + 2) * l[0], x = (m + d) * l[1], S = {
			width: b < 0 ? Math.floor(b) : Math.ceil(b),
			height: x < 0 ? Math.floor(x) : Math.ceil(x),
			contextInstructions: y
		};
		(l[0] != 1 || l[1] != 1) && y.push("scale", l), r && a && (y.push("strokeStyle", a.strokeStyle), y.push("lineWidth", d), y.push("lineCap", a.lineCap), y.push("lineJoin", a.lineJoin), y.push("miterLimit", a.miterLimit), y.push("setLineDash", [a.lineDash]), y.push("lineDashOffset", a.lineDashOffset)), n && o && y.push("fillStyle", o.fillStyle), y.push("textBaseline", "middle"), y.push("textAlign", "center");
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
		(m || u !== 0) && (Ec[0] = S, kc[0] = S, Ec[1] = C, Dc[1] = C, Dc[0] = S + b, Oc[0] = Dc[0], Oc[1] = C + x, kc[1] = Oc[1]);
		let w;
		return u === 0 ? Ye(Math.min(S, S + b), Math.min(C, C + x), Math.max(S, S + b), Math.max(C, C + x), Tc) : (w = Fr(Or(), n, r, 1, 1, u, -n, -r), R(w, Ec), R(w, Dc), R(w, Oc), R(w, kc), Ye(Math.min(Ec[0], Dc[0], Oc[0], kc[0]), Math.min(Ec[1], Dc[1], Oc[1], kc[1]), Math.max(Ec[0], Dc[0], Oc[0], kc[0]), Math.max(Ec[1], Dc[1], Oc[1], kc[1]), Tc)), f && (g = Math.round(g), _ = Math.round(_)), {
			drawImageX: g,
			drawImageY: _,
			drawImageW: v,
			drawImageH: y,
			originX: c,
			originY: l,
			declutterBox: {
				minX: Tc[0],
				minY: Tc[1],
				maxX: Tc[2],
				maxY: Tc[3],
				value: h
			},
			canvasTransform: w,
			scale: d
		};
	}
	replayImageOrLabelArgs_(e) {
		return this.replayImageOrLabel_(e[0], e[1], e[2], e[3], e[4], e[5], e[6]);
	}
	replayImageOrLabel_(e, t, n, r, i, a, o) {
		let s = !!(a || o), c = r.declutterBox, l = o ? o[2] * r.scale[0] / 2 : 0;
		return c.minX - l <= t[0] && c.maxX + l >= 0 && c.minY - l <= t[1] && c.maxY + l >= 0 && (s && this.replayTextBackground_(e, Ec, Dc, Oc, kc, a, o), $s(e, r.canvasTransform, i, n, r.originX, r.originY, r.drawImageW, r.drawImageH, r.drawImageX, r.drawImageY, r.scale)), !0;
	}
	fill_(e) {
		let t = this.alignAndScaleFill_;
		if (t) {
			let n = R(this.renderedTransform_, [0, 0]), r = 512 * this.pixelRatio;
			e.save(), e.translate(n[0] % r, n[1] % r), t !== 1 && e.scale(t, t);
		}
		e.fill(), t && e.restore();
	}
	setStrokeStyle_(e, t) {
		e.strokeStyle = t[1], t[1] && (e.lineWidth = t[2], e.lineCap = t[3], e.lineJoin = t[4], e.miterLimit = t[5], e.lineDashOffset = t[7], e.setLineDash(t[6]));
	}
	drawLabelWithPointPlacement_(e, t, n, r) {
		let i = this.textStates[t], a = this.createLabel(e, t, r, n), o = this.strokeStates[n], s = this.pixelRatio, c = jc(Array.isArray(e) ? e[0] : e, i.textAlign || "center"), l = hc[i.textBaseline || "middle"], u = o && o.lineWidth ? o.lineWidth : 0;
		return {
			label: a,
			anchorX: c * (a.width / s - 2 * i.scale[0]) + 2 * (.5 - c) * u,
			anchorY: l * a.height / s + 2 * (.5 - l) * u
		};
	}
	execute_(e, t, n, r, i, a, o, s) {
		let c = this.zIndexContext_, l;
		this.pixelCoordinates_ && h(n, this.renderedTransform_) ? l = this.pixelCoordinates_ : (this.pixelCoordinates_ ||= [], l = Hr(this.coordinates, 0, this.coordinates.length, 2, n, this.pixelCoordinates_), Mr(this.renderedTransform_, n));
		let u = 0, d = r.length, f = 0, p, m = [], g, _, v, y, b, x, S, C, w, T, E, D, O, k = 0, A = 0, j = this.coordinateCache_, M = this.viewRotation_, N = Math.round(Math.atan2(-n[1], n[0]) * 0xe8d4a51000) / 0xe8d4a51000, P = {
			context: e,
			pixelRatio: this.pixelRatio,
			resolution: this.resolution,
			rotation: M
		}, F = this.instructions != r || this.overlaps ? 0 : 200, ee, te, ne, re;
		for (; u < d;) {
			let n = r[u];
			switch (n[0]) {
				case U.BEGIN_GEOMETRY:
					ee = n[1], re = n[3], ee.getGeometry() ? o !== void 0 && !_t(o, re.getExtent()) ? u = n[2] + 1 : ++u : u = n[2], c && (c.zIndex = n[4]);
					break;
				case U.BEGIN_PATH:
					k > F && (this.fill_(e), k = 0), A > F && (e.stroke(), A = 0), !k && !A && (e.beginPath(), b = NaN, x = NaN), ++u;
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
					P.geometry = ce, P.feature = ee, u in j || (j[u] = []);
					let de = j[u];
					ue ? ue(l, f, p, 2, de) : (de[0] = l[f], de[1] = l[f + 1], de.length = 2), c && (c.zIndex = n[6]), le(de, P), ++u;
					break;
				case U.DRAW_IMAGE:
					f = n[1], p = n[2], w = n[3], g = n[4], _ = n[5];
					let fe = n[6], pe = n[7], me = n[8], he = n[9], ge = n[10], _e = n[11], ve = n[12], ye = n[13];
					y = n[14] || "declutter";
					let be = n[15];
					if (!w && n.length >= 20) {
						T = n[19], E = n[20], D = n[21], O = n[22];
						let e = this.drawLabelWithPointPlacement_(T, E, D, O);
						w = e.label, n[3] = w;
						let t = n[23];
						g = (e.anchorX - t) * this.pixelRatio, n[4] = g;
						let r = n[24];
						_ = (e.anchorY - r) * this.pixelRatio, n[5] = _, fe = w.height, n[6] = fe, ye = w.width, n[13] = ye;
					}
					let xe;
					n.length > 25 && (xe = n[25]);
					let Se, Ce, we;
					n.length > 17 ? (Se = n[16], Ce = n[17], we = n[18]) : (Se = Bs, Ce = null, we = null), ge && N ? _e += M : !ge && !N && (_e -= M);
					let Te = 0;
					for (; f < p; f += 2) {
						if (xe && xe[Te++] < ye / this.pixelRatio) continue;
						let n = this.calculateImageOrLabelDimensions_(w.width, w.height, l[f], l[f + 1], ye, fe, g, _, me, he, _e, ve, i, Se, !!Ce || !!we, ee), r = [
							e,
							t,
							w,
							n,
							pe,
							Ce,
							we
						];
						if (s) {
							let e, t, i;
							if (be) {
								let n = p - f;
								if (!be[n]) {
									be[n] = {
										args: r,
										declutterMode: y
									};
									continue;
								}
								let a = be[n];
								e = a.args, t = a.declutterMode, delete be[n], i = Ac(e);
							}
							let a, o;
							if (e && (t !== "declutter" || !s.collides(i)) && (a = !0), y === "declutter" && s.collides(n.declutterBox) || (o = !0), t === "declutter" && y === "declutter") {
								let e = a && o;
								a = e, o = e;
							}
							a && (t !== "none" && s.insert(i), this.replayImageOrLabelArgs_(e)), o && (y !== "none" && s.insert(n.declutterBox), this.replayImageOrLabelArgs_(r));
						} else this.replayImageOrLabelArgs_(r);
					}
					++u;
					break;
				case U.DRAW_CHARS:
					let Ee = n[1], De = n[2], Oe = n[3], ke = n[4];
					O = n[5];
					let Ae = n[6], je = n[7], Me = n[8];
					D = n[9];
					let Ne = n[10];
					T = n[11], Array.isArray(T) && (T = T.reduce(Nc, "")), E = n[12];
					let Pe = [n[13], n[13]];
					y = n[14] || "declutter";
					let Fe = n[15], Ie = n[16], Le = this.textStates[E], Re = Le.font, ze = [Le.scale[0] * je, Le.scale[1] * je], Be;
					Re in this.widths_ ? Be = this.widths_[Re] : (Be = {}, this.widths_[Re] = Be);
					let Ve = Oi(l, Ee, De, 2), He = Math.abs(ze[0]) * Zs(Re, T, Be);
					if (ke || He <= Ve) {
						let n = this.textStates[E].textAlign, r = (Ve - He) * jc(T, n), i = wc(l, Ee, De, 2, T, r, Ae, Math.abs(ze[0]), Zs, Re, Be, N ? 0 : this.viewRotation_, Fe);
						drawChars: if (i) {
							let n = [], r, a, o, c, l;
							if (D) for (r = 0, a = i.length; r < a; ++r) {
								l = i[r], o = l[4], c = this.createLabel(o, E, "", D), g = l[2] + (ze[0] < 0 ? -Ne : Ne) - Ie, _ = Oe * c.height + (.5 - Oe) * 2 * Ne * ze[1] / ze[0] - Me;
								let a = this.calculateImageOrLabelDimensions_(c.width, c.height, l[0], l[1], c.width, c.height, g, _, 0, 0, l[3], Pe, !1, Bs, !1, ee);
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
								l = i[r], o = l[4], c = this.createLabel(o, E, O, ""), g = l[2] - Ie, _ = Oe * c.height - Me;
								let a = this.calculateImageOrLabelDimensions_(c.width, c.height, l[0], l[1], c.width, c.height, g, _, 0, 0, l[3], Pe, !1, Bs, !1, ee);
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
							s && y !== "none" && s.load(n.map(Ac));
							for (let e = 0, t = n.length; e < t; ++e) this.replayImageOrLabelArgs_(n[e]);
						}
					}
					++u;
					break;
				case U.END_GEOMETRY:
					if (a !== void 0) {
						ee = n[1];
						let e = a(ee, re, y);
						if (e) return e;
					}
					++u;
					break;
				case U.FILL:
					F ? k++ : this.fill_(e), ++u;
					break;
				case U.MOVE_TO_LINE_TO:
					f = n[1], p = n[2], v = n[3];
					let Ue, We, Ge;
					if (v) {
						let e = (n[4] ?? !1) || Math.abs(l[f] - l[p - 2]) < 1e-6 && Math.abs(l[f + 1] - l[p - 1]) < 1e-6;
						vc(l, f, p, 2, v, e, m), bc(m, 2, e), Ue = m, We = 0, Ge = Ue.length;
					} else Ue = l, We = f, Ge = p;
					te = Ue[We], ne = Ue[We + 1], e.moveTo(te, ne), b = te + .5 | 0, x = ne + .5 | 0;
					for (let t = We + 2; t < Ge; t += 2) te = Ue[t], ne = Ue[t + 1], S = te + .5 | 0, C = ne + .5 | 0, (t == Ge - 2 || S !== b || C !== x) && (e.lineTo(te, ne), b = S, x = C);
					++u;
					break;
				case U.SET_FILL_STYLE:
					this.alignAndScaleFill_ = n[2], k ? (this.fill_(e), k = 0, A &&= (e.stroke(), 0)) : A && n[1] && (e.stroke(), A = 0), e.fillStyle = n[1], ++u;
					break;
				case U.SET_STROKE_STYLE:
					k && n[1] && (this.fill_(e), k = 0), A &&= (e.stroke(), 0), this.setStrokeStyle_(e, n), ++u;
					break;
				case U.STROKE:
					F ? A++ : e.stroke(), ++u;
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
}, Fc = [
	"Polygon",
	"Circle",
	"LineString",
	"Image",
	"Text",
	"Default"
], Ic = ["Image", "Text"], Lc = Fc.filter((e) => !Ic.includes(e)), Rc = !1, zc = !1;
function Bc() {
	let e = 0, t = (t) => {
		let n = _e(1, 1, null, { willReadFrequently: t }), r = 0, i = performance.now();
		for (; performance.now() - i < 50; ++r) n.fillStyle = `rgba(255,0,${r % 256},1)`, n.fillRect(0, 0, 1, 1), n.getImageData(0, 0, 1, 1);
		return e = r > e ? r : e, r;
	};
	Rc = {
		[t(!0)]: !0,
		[t(!1)]: !1,
		[t(void 0)]: void 0
	}[e], zc = !0;
}
var Vc = class {
	constructor(e, t, n, r, i, a, o) {
		this.maxExtent_ = e, this.overlaps_ = r, this.pixelRatio_ = n, this.resolution_ = t, this.renderBuffer_ = a, this.executorsByZIndex_ = {}, this.hitDetectionContext_ = null, this.hitDetectionTransform_ = Or(), this.renderedContext_ = null, this.deferredZIndexContexts_ = {}, this.createExecutors_(i, !!o);
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
				r[e] = new Pc(this.resolution_, this.pixelRatio_, this.overlaps_, n, t);
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
		zc === !1 && Bc(), r = Math.round(r);
		let o = r * 2 + 1, s = Fr(this.hitDetectionTransform_, r + .5, r + .5, 1 / t, -1 / t, -n, -e[0], -e[1]), c = !this.hitDetectionContext_;
		c && (this.hitDetectionContext_ = _e(o, o, null, { willReadFrequently: Rc }));
		let l = this.hitDetectionContext_;
		l.canvas.width !== o || l.canvas.height !== o ? (l.canvas.width = o, l.canvas.height = o) : c || l.clearRect(0, 0, o, o);
		let d;
		this.renderBuffer_ !== void 0 && (d = Je(), tt(d, e), Ve(d, t * (this.renderBuffer_ + r), d));
		let f = Uc(r), p, m = l;
		function h(e, t, n) {
			let s = m.getImageData(0, 0, o, o).data;
			for (let c = 0, l = f.length; c < l; c++) if (s[f[c]] > 0) {
				if (!a || n === "none" || p !== "Image" && p !== "Text" || a.includes(e)) {
					let n = (f[c] - 3) / 4, a = r - n % o, s = r - (n / o | 0), l = i(e, t, a * a + s * s);
					if (l) return l;
				}
				m.clearRect(0, 0, o, o);
				break;
			}
		}
		let g = Object.keys(this.executorsByZIndex_).map(Number);
		g.sort(u);
		let _, v, y, b, x;
		for (_ = g.length - 1; _ >= 0; --_) {
			let e = g[_].toString();
			for (y = this.executorsByZIndex_[e], v = Fc.length - 1; v >= 0; --v) if (p = Fc[v], b = y[p], b !== void 0 && (x = b.executeHitDetection(l, s, n, h, d), x)) return x;
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
		return Hr(o, 0, 8, 2, e, o), o;
	}
	isEmpty() {
		return r(this.executorsByZIndex_);
	}
	execute(e, t, n, r, i, a, o) {
		let s = Object.keys(this.executorsByZIndex_).map(Number);
		s.sort(o ? d : u), a ||= Fc;
		let c = Fc.length;
		for (let l = 0, u = s.length; l < u; ++l) {
			let u = s[l].toString(), d = this.executorsByZIndex_[u];
			for (let u = 0, f = a.length; u < f; ++u) {
				let f = a[u], p = d[f];
				if (p !== void 0) {
					let a = o === null ? void 0 : p.getZIndexContext(), u = a ? a.getContext() : e, d = this.maxExtent_ && f !== "Image" && f !== "Text";
					if (d && (u.save(), this.clip(u, n)), !a || f === "Text" || f === "Image" ? p.execute(u, t, n, r, i, o) : a.pushFunction((e) => p.execute(e, t, n, r, i, o)), d && u.restore(), a) {
						a.offset();
						let e = s[l] * c + Fc.indexOf(f);
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
}, Hc = {};
function Uc(e) {
	if (Hc[e] !== void 0) return Hc[e];
	let t = e * 2 + 1, n = e * e, r = Array(n + 1);
	for (let i = 0; i <= e; ++i) for (let a = 0; a <= e; ++a) {
		let o = i * i + a * a;
		if (o > n) break;
		let s = r[o];
		s || (s = [], r[o] = s), s.push(((e + i) * t + (e + a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e + a)) * 4 + 3), a > 0 && (s.push(((e + i) * t + (e - a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e - a)) * 4 + 3));
	}
	let i = [];
	for (let e = 0, t = r.length; e < t; ++e) r[e] && i.push(...r[e]);
	return Hc[e] = i, i;
}
// node_modules/ol/style/Image.js
var Wc = class e {
	constructor(e) {
		this.opacity_ = e.opacity, this.rotateWithView_ = e.rotateWithView, this.rotation_ = e.rotation, this.scale_ = e.scale, this.scaleArray_ = ao(e.scale), this.displacement_ = e.displacement, this.declutterMode_ = e.declutterMode;
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
		return _();
	}
	getImage(e) {
		return _();
	}
	getHitDetectionImage() {
		return _();
	}
	getPixelRatio(e) {
		return 1;
	}
	getImageState() {
		return _();
	}
	getImageSize() {
		return _();
	}
	getOrigin() {
		return _();
	}
	getSize() {
		return _();
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
		this.scale_ = e, this.scaleArray_ = ao(e);
	}
	listenImageChange(e) {
		_();
	}
	load() {
		_();
	}
	unlistenImageChange(e) {
		_();
	}
	ready() {
		return Promise.resolve();
	}
};
// node_modules/ol/style/Icon.js
function Gc(e, t, n, r) {
	return n !== void 0 && r !== void 0 ? [n / e, r / t] : n === void 0 ? r === void 0 ? 1 : r / t : n / e;
}
var Kc = class e extends Wc {
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
		L(!(o !== void 0 && a), "`image` and `src` cannot be provided at the same time"), (o === void 0 || o.length === 0) && a && (o = a.src || y(a)), L(o !== void 0 && o.length > 0, "A defined and non-empty `src` or `image` must be provided"), L(e.width === void 0 && e.height === void 0 || e.scale === void 0, "`width` or `height` cannot be provided together with `scale`");
		let s;
		if (e.src === void 0 ? a !== void 0 && (s = "complete" in a ? a.complete ? a.src ? B.LOADED : B.IDLE : B.LOADING : B.LOADED) : s = B.IDLE, this.color_ = e.color === void 0 ? null : Io(e.color), this.iconImage_ = Os(a, o, {
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
						n && this.setScale(Gc(n[0], n[1], e.width, e.height));
					};
					this.listenImageChange(t);
					return;
				}
			}
			t !== void 0 && this.setScale(Gc(t, n, e.width, e.height));
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
		let t = e ? Io(e) : null;
		if (this.color_ === t || this.color_ && t && this.color_.length === t.length && this.color_.every((e, n) => e === t[n])) return;
		this.color_ = t;
		let n = this.getSrc(), r = n === void 0 ? this.getHitDetectionImage() : null, i = n === void 0 ? this.iconImage_.getImageState() : B.IDLE;
		this.iconImage_ = Os(r, n, {
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
		this.iconImage_ = Os(null, e, {
			crossOrigin: this.crossOrigin_,
			referrerPolicy: this.referrerPolicy_
		}, B.IDLE, this.color_);
	}
	getSize() {
		return this.size_ ? this.size_ : this.iconImage_.getSize();
	}
	getWidth() {
		let e = this.getScaleArray();
		if (this.size_) return this.size_[0] * e[0];
		if (this.iconImage_.getImageState() == B.LOADED) {
			let t = this.iconImage_.getSize();
			if (t) return t[0] * e[0];
		}
	}
	getHeight() {
		let e = this.getScaleArray();
		if (this.size_) return this.size_[1] * e[1];
		if (this.iconImage_.getImageState() == B.LOADED) {
			let t = this.iconImage_.getSize();
			if (t) return t[1] * e[1];
		}
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
}, qc = class extends js {
	constructor(e, t, n, r, i, a, o) {
		super(), this.context_ = e, this.pixelRatio_ = t, this.extent_ = n, this.transform_ = r, this.transformRotation_ = r ? Lt(Math.atan2(r[1], r[0]), 10) : 0, this.viewRotation_ = i, this.squaredTolerance_ = a, this.userTransform_ = o, this.contextFillState_ = null, this.contextStrokeState_ = null, this.contextTextState_ = null, this.fillState_ = null, this.strokeState_ = null, this.image_ = null, this.imageAnchorX_ = 0, this.imageAnchorY_ = 0, this.imageHeight_ = 0, this.imageOpacity_ = 0, this.imageOriginX_ = 0, this.imageOriginY_ = 0, this.imageRotateWithView_ = !1, this.imageRotation_ = 0, this.imageScale_ = [0, 0], this.imageWidth_ = 0, this.text_ = "", this.textOffsetX_ = 0, this.textOffsetY_ = 0, this.textRotateWithView_ = !1, this.textRotation_ = 0, this.textScale_ = [0, 0], this.textFillState_ = null, this.textStrokeState_ = null, this.textState_ = null, this.pixelCoordinates_ = [], this.tmpLocalTransform_ = Or();
	}
	drawImages_(e, t, n, r) {
		if (!this.image_) return;
		let i = Hr(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.tmpLocalTransform_, s = a.globalAlpha;
		this.imageOpacity_ != 1 && (a.globalAlpha = s * this.imageOpacity_);
		let c = this.imageRotation_;
		this.transformRotation_ === 0 && (c -= this.viewRotation_), this.imageRotateWithView_ && (c += this.viewRotation_);
		for (let e = 0, t = i.length; e < t; e += 2) {
			let t = i[e] - this.imageAnchorX_, n = i[e + 1] - this.imageAnchorY_;
			if (c !== 0 || this.imageScale_[0] != 1 || this.imageScale_[1] != 1) {
				let e = t + this.imageAnchorX_, r = n + this.imageAnchorY_;
				Fr(o, e, r, 1, 1, c, -e, -r), a.save(), a.transform.apply(a, o), a.translate(e, r), a.scale(this.imageScale_[0], this.imageScale_[1]), a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, -this.imageAnchorX_, -this.imageAnchorY_, this.imageWidth_, this.imageHeight_), a.restore();
			} else a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, t, n, this.imageWidth_, this.imageHeight_);
		}
		this.imageOpacity_ != 1 && (a.globalAlpha = s);
	}
	drawText_(e, t, n, r) {
		if (!this.textState_ || this.text_ === "") return;
		this.textFillState_ && this.setContextFillState_(this.textFillState_), this.textStrokeState_ && this.setContextStrokeState_(this.textStrokeState_), this.setContextTextState_(this.textState_);
		let i = Hr(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.textRotation_;
		for (this.transformRotation_ === 0 && (o -= this.viewRotation_), this.textRotateWithView_ && (o += this.viewRotation_); t < n; t += r) {
			let e = i[t] + this.textOffsetX_, n = i[t + 1] + this.textOffsetY_;
			o !== 0 || this.textScale_[0] != 1 || this.textScale_[1] != 1 ? (a.save(), a.translate(e - this.textOffsetX_, n - this.textOffsetY_), a.rotate(o), a.translate(this.textOffsetX_, this.textOffsetY_), a.scale(this.textScale_[0], this.textScale_[1]), this.textStrokeState_ && a.strokeText(this.text_, 0, 0), this.textFillState_ && a.fillText(this.text_, 0, 0), a.restore()) : (this.textStrokeState_ && a.strokeText(this.text_, e, n), this.textFillState_ && a.fillText(this.text_, e, n));
		}
	}
	moveToLineTo_(e, t, n, r, i, a) {
		let o = this.context_, s = a ?? 0, c = Hr(e, t, n, r, this.transform_, this.pixelCoordinates_);
		if (Math.abs(s) > 0) {
			let e = c.length, t = i || Math.abs(c[0] - c[e - 2]) < 1e-6 && Math.abs(c[1] - c[e - 1]) < 1e-6;
			c = vc(c, 0, e, 2, s, t, c), bc(c, 2, t);
		}
		o.moveTo(c[0], c[1]);
		let l = c.length;
		i && (l -= 2);
		for (let e = 2; e < l; e += 2) o.lineTo(c[e], c[e + 1]);
		return i && o.closePath(), n;
	}
	drawRings_(e, t, n, r, i) {
		for (let a = 0, o = n.length; a < o; ++a) t = this.moveToLineTo_(e, t, n[a], r, !0, i);
		return t;
	}
	drawCircle(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), _t(this.extent_, e.getExtent())) {
			if (this.fillState_ || this.strokeState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				let t = Qr(e, this.transform_, this.pixelCoordinates_);
				if (!t) return;
				let n = t[2] - t[0], r = t[3] - t[1], i = Math.sqrt(n * n + r * r), a = this.context_;
				a.beginPath(), a.arc(t[0], t[1], i, 0, 2 * Math.PI), this.fillState_ && a.fill(), this.strokeState_ && a.stroke();
			}
			this.text_ !== "" && this.drawText_(e.getCenter(), 0, 2, 2);
		}
	}
	setStyle(e) {
		this.setFillStrokeStyle(e.getFill() ?? null, e.getStroke() ?? null), this.setImageStyle(e.getImage() ?? null), this.setTextStyle(e.getText() ?? null);
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), _t(this.extent_, e.getExtent())) {
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
		if (_t(this.extent_, t)) {
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), _t(this.extent_, e.getExtent())) {
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), _t(this.extent_, e.getExtent())) {
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
		let t = this.context_, n = this.contextTextState_, r = e.textAlign ? e.textAlign : Rs;
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
			this.fillState_ = { fillStyle: ks(t || "#000") || "#000" };
		}
		if (!t) this.strokeState_ = null;
		else {
			let e = t.getColor(), n = t.getLineCap(), r = t.getLineDash(), i = t.getLineDashOffset(), a = t.getLineJoin(), o = t.getWidth(), s = t.getMiterLimit(), c = r || Fs, l = t.getOffset();
			this.strokeState_ = {
				lineCap: n === void 0 ? Ps : n,
				lineDash: this.pixelRatio_ === 1 ? c : c.map((e) => e * this.pixelRatio_),
				lineDashOffset: (i || 0) * this.pixelRatio_,
				lineJoin: a === void 0 ? Is : a,
				lineWidth: (o === void 0 ? 1 : o) * this.pixelRatio_,
				miterLimit: s === void 0 ? 10 : s,
				strokeStyle: ks(e || "#000") || "#000",
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
		if (!r || !i) {
			this.image_ = null;
			return;
		}
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
				this.textFillState_ = { fillStyle: ks(e || "#000") || "#000" };
			}
			let n = e.getStroke();
			if (!n) this.textStrokeState_ = null;
			else {
				let e = n.getColor(), t = n.getLineCap(), r = n.getLineDash(), i = n.getLineDashOffset(), a = n.getLineJoin(), o = n.getWidth(), s = n.getMiterLimit();
				this.textStrokeState_ = {
					lineCap: t === void 0 ? Ps : t,
					lineDash: r || Fs,
					lineDashOffset: i || 0,
					lineJoin: a === void 0 ? Is : a,
					lineWidth: o === void 0 ? 1 : o,
					miterLimit: s === void 0 ? 10 : s,
					strokeStyle: ks(e || "#000") || "#000"
				};
			}
			let r = e.getFont(), i = e.getOffsetX(), a = e.getOffsetY(), o = e.getRotateWithView(), s = e.getRotation(), c = e.getScaleArray(), l = e.getText(), u = e.getTextAlign(), d = e.getTextBaseline();
			this.textState_ = {
				font: r === void 0 ? Ms : r,
				textAlign: u === void 0 ? Rs : u,
				textBaseline: d === void 0 ? zs : d
			}, this.text_ = l === void 0 ? "" : Array.isArray(l) ? l.reduce((e, t, n) => e += n % 2 ? " " : t, "") : l, this.textOffsetX_ = i === void 0 ? 0 : this.pixelRatio_ * i, this.textOffsetY_ = a === void 0 ? 0 : this.pixelRatio_ * a, this.textRotateWithView_ = o !== void 0 && o, this.textRotation_ = s === void 0 ? 0 : s, this.textScale_ = [this.pixelRatio_ * c[0], this.pixelRatio_ * c[1]];
		}
	}
}, Jc = .5;
function Yc(e, t, n, r, i, a, o, s, c) {
	let l = c ? Sr(i, c) : i, d = _e(e[0] * Jc, e[1] * Jc);
	d.imageSmoothingEnabled = !1;
	let f = d.canvas, p = new qc(d, Jc, i, [
		1,
		0,
		0,
		1,
		0,
		0
	], o, s ?? 0, c ? pr(yr(), c) ?? void 0 : void 0), m = n.length, h = Math.floor(16777215 / m), g = {};
	for (let e = 1; e <= m; ++e) {
		let t = n[e - 1], i = t.getStyleFunction() || r;
		if (!i) continue;
		let o = i(t, a);
		if (!o) continue;
		Array.isArray(o) || (o = [o]);
		let s = (e * h).toString(16).padStart(7, "#00000");
		for (let e = 0, n = o.length; e < n; ++e) {
			let n = o[e], r = n.getGeometryFunction()(t);
			if (!r || !_t(l, r.getExtent())) continue;
			let i = n.clone(), a = i.getFill();
			a && a.setColor(s);
			let c = i.getStroke();
			c && (c.setColor(s), c.setLineDash(null)), i.setText(null);
			let u = n.getImage();
			if (u) {
				let e = u.getImageSize();
				if (!e) continue;
				let t = _e(e[0], e[1], void 0, { alpha: !1 }), n = t.canvas;
				t.fillStyle = s, t.fillRect(0, 0, n.width, n.height), i.setImage(new Kc({
					img: n,
					anchor: u.getAnchor() ?? void 0,
					anchorXUnits: "pixels",
					anchorYUnits: "pixels",
					offset: u.getOrigin() ?? void 0,
					opacity: 1,
					size: u.getSize() ?? void 0,
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
function Xc(e, t, n) {
	let r = [];
	if (n) {
		let i = Math.floor(Math.round(e[0]) * Jc), a = Math.floor(Math.round(e[1]) * Jc), o = (kt(i, 0, n.width - 1) + kt(a, 0, n.height - 1) * n.width) * 4, s = n.data[o], c = n.data[o + 1], l = n.data[o + 2] + 256 * (c + 256 * s), u = Math.floor(16777215 / t.length);
		l && l % u === 0 && r.push(t[l / u - 1]);
	}
	return r;
}
// node_modules/ol/renderer/vector.js
var Zc = .5, Qc = {
	Point: ul,
	LineString: sl,
	Polygon: fl,
	MultiPoint: dl,
	MultiLineString: cl,
	MultiPolygon: ll,
	GeometryCollection: ol,
	Circle: nl
};
function $c(e, t) {
	return parseInt(y(e), 10) - parseInt(y(t), 10);
}
function el(e, t) {
	let n = tl(e, t);
	return n * n;
}
function tl(e, t) {
	return Zc * e / t;
}
function nl(e, t, n, r, i) {
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
function rl(e, t, n, r, i, a, o, s) {
	let c = [], l = n.getImage();
	if (l) {
		let e = !0, t = l.getImageState();
		t == B.LOADED || t == B.ERROR ? e = !1 : t == B.IDLE && l.load(), e && c.push(l.ready());
	}
	let u = n.getFill();
	u && u.loading() && c.push(u.ready());
	let d = c.length > 0;
	return d && Promise.all(c).then(() => i(null)), il(e, t, n, r, a, o, s), d;
}
function il(e, t, n, r, i, a, o) {
	let s = n.getGeometryFunction()(t);
	if (!s) return;
	let c = s.simplifyTransformed(r, i);
	if (n.getRenderer()) al(e, c, n, t, o);
	else {
		let r = Qc[c.getType()];
		r(e, c, n, t, o, a);
	}
}
function al(e, t, n, r, i) {
	if (t.getType() == "GeometryCollection") {
		let a = t.getGeometries();
		for (let t = 0, o = a.length; t < o; ++t) al(e, a[t], n, r, i);
	} else e.getBuilder(n.getZIndex(), "Default").drawCustom(t, r, n.getRenderer(), n.getHitDetectionRenderer(), i);
}
function ol(e, t, n, r, i, a) {
	let o = t.getGeometriesArray(), s, c;
	for (s = 0, c = o.length; s < c; ++s) {
		let t = Qc[o[s].getType()];
		t(e, o[s], n, r, i, a);
	}
}
function sl(e, t, n, r, i) {
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
function cl(e, t, n, r, i) {
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
function ll(e, t, n, r, i) {
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
function ul(e, t, n, r, i, a) {
	let o = n.getImage(), s = n.getText(), c = s && s.getText(), l = a && o && c ? {} : void 0;
	if (o) {
		if (o.getImageState() != B.LOADED) return;
		let a = e.getBuilder(n.getZIndex(), "Image");
		a.setImageStyle(o, l), a.drawPoint(t, r, i);
	}
	if (c) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(s, l), a.drawText(t, r, i);
	}
}
function dl(e, t, n, r, i, a) {
	let o = n.getImage(), s = o && o.getOpacity() !== 0, c = n.getText(), l = c && c.getText(), u = a && s && l ? {} : void 0;
	if (s) {
		if (o.getImageState() != B.LOADED) return;
		let a = e.getBuilder(n.getZIndex(), "Image");
		a.setImageStyle(o, u), a.drawMultiPoint(t, r, i);
	}
	if (l) {
		let a = e.getBuilder(n.getZIndex(), "Text");
		a.setTextStyle(c, u), a.drawText(t, r, i);
	}
}
function fl(e, t, n, r, i) {
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
// node_modules/ol/renderer/canvas/VectorLayer.js
var pl = class extends Ko {
	constructor(e, t) {
		super(e), this.boundHandleStyleImageChange_ = this.handleStyleImageChange_.bind(this), this.animatingOrInteracting_, this.hitDetectionImageData_ = null, this.clipExtent_ = null, this.extendX_ = !1, this.renderedFeatures_ = null, this.renderedRevision_ = -1, this.renderedResolution_ = NaN, this.renderedExtent_ = Je(), this.wrappedRenderedExtent_ = Je(), this.renderedRotation_, this.renderedCenter_ = null, this.renderedProjection_ = null, this.renderedPixelRatio_ = 1, this.renderedRenderOrder_ = null, this.renderedFrameDeclutter_, this.replayGroup_ = null, this.replayGroupChanged = !0, this.clipping = !0, this.targetContext_ = null, this.opacity_ = 1, t?.wantRotation !== void 0 && (this.wantRotation = t.wantRotation);
	}
	renderWorlds(e, t, n) {
		let r = t.extent, i = t.viewState, a = i.center, o = i.resolution, s = i.projection, c = i.rotation, l = s.getExtent(), u = this.getLayer().getSource(), d = this.getLayer().getDeclutter(), f = t.pixelRatio, p = t.viewHints, m = !(p[V.ANIMATING] || p[V.INTERACTING]), h = this.context, g = h.canvas.width, _ = h.canvas.height, v = u.getWrapX() && s.canWrapX(), y = v ? I(l) : null, b = r, x = v ? Math.ceil((b[2] - l[2]) / y) + (this.extendX_ ? 2 : 1) : 1, S = v ? Math.floor((b[0] - l[0]) / y) - +!!this.extendX_ : 0;
		do {
			let r = this.getRenderTransform(a, o, this.sourceRotates() ? c : 0, f, g, _, S * y);
			t.declutter && (r = r.slice(0)), e.execute(h, [h.canvas.width, h.canvas.height], r, c, m, n === void 0 ? Fc : n ? Ic : Lc, n ? d && t.declutter[d] : void 0);
		} while (++S < x);
	}
	setDrawContext_() {
		this.opacity_ !== 1 && (this.targetContext_ = this.context, this.context = _e(this.context.canvas.width, this.context.canvas.height, Uo));
	}
	resetDrawContext_() {
		if (this.opacity_ !== 1 && this.targetContext_) {
			let e = this.targetContext_.globalAlpha;
			this.targetContext_.globalAlpha = this.opacity_, this.targetContext_.drawImage(this.context.canvas, 0, 0), this.targetContext_.globalAlpha = e, be(this.context), Uo.push(this.context.canvas), this.context = this.targetContext_, this.targetContext_ = null;
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
		if (!o && !(this.getLayer().hasListener(zo.PRERENDER) || this.getLayer().hasListener(zo.POSTRENDER))) return this.container;
		this.setDrawContext_(), this.preRender(i, e);
		let s = r.projection;
		this.clipExtent_ = null;
		let c = !1;
		if (o && n.extent && this.clipping) {
			let t = Cr(n.extent, s);
			o = _t(t, e.extent), o && !Ge(t, e.extent) && (e.declutter ? this.clipExtent_ = t : (this.clipUnrotated(i, e, t), c = !0));
		}
		return o && this.renderWorlds(a, e, !this.getLayer().getDeclutter() && void 0), c && i.restore(), this.postRender(i, e), this.renderedRotation_ !== r.rotation && (this.renderedRotation_ = r.rotation, this.hitDetectionImageData_ = null), e.declutter || this.resetDrawContext_(), this.container;
	}
	getFeatures(e) {
		return new Promise((t) => {
			if (this.frameState && !this.hitDetectionImageData_ && !this.animatingOrInteracting_) {
				let e = this.frameState.size.slice(), t = this.renderedCenter_, n = this.renderedResolution_, r = this.renderedRotation_, i = this.renderedProjection_, a = this.wrappedRenderedExtent_, o = this.getLayer(), s = [], c = e[0] * Jc, l = e[1] * Jc;
				s.push(this.getRenderTransform(t, n, r, Jc, c, l, 0).slice());
				let u = o.getSource(), d = i.getExtent();
				if (u.getWrapX() && i.canWrapX() && !Ge(d, a)) {
					let e = a[0], i = I(d), o = 0, u;
					for (; e < d[0];) --o, u = i * o, s.push(this.getRenderTransform(t, n, r, Jc, c, l, u).slice()), e += i;
					for (o = 0, e = a[2]; e > d[2];) ++o, u = i * o, s.push(this.getRenderTransform(t, n, r, Jc, c, l, u).slice()), e -= i;
				}
				let f = yr();
				this.hitDetectionImageData_ = Yc(e, s, this.renderedFeatures_, o.getStyleFunction(), a, n, r, el(n, this.renderedPixelRatio_), f ? i : null);
			}
			t(Xc(e, this.renderedFeatures_, this.hitDetectionImageData_));
		});
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		if (!this.replayGroup_) return;
		let a = t.viewState.resolution, o = t.viewState.rotation, s = this.getLayer(), c = {}, l = function(e, t, n) {
			let a = y(e), o = c[a];
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
		let r = e.viewHints[V.ANIMATING], i = e.viewHints[V.INTERACTING], a = t.getUpdateWhileAnimating(), o = t.getUpdateWhileInteracting();
		if (this.ready && !a && r || !o && i) return this.animatingOrInteracting_ = !0, !0;
		this.animatingOrInteracting_ = !1;
		let s = e.extent, c = e.viewState, l = c.projection, u = c.resolution, d = e.pixelRatio, f = t.getRevision(), p = t.getRenderBuffer(), m = t.getRenderOrder();
		m === void 0 && (m = $c);
		let g = c.center.slice(), _ = Ve(s, p * u), v = _.slice(), y = [_.slice()], b = l.getExtent(), x = n.getWrapX() && l.canWrapX();
		if (this.extendX_ = !1, x) {
			let e = n.getExtent();
			e && !vt(e) && (this.extendX_ = e[0] < b[0] || e[2] > b[2]);
		}
		if (x && (!Ge(b, e.extent) || this.extendX_)) {
			let e = I(b), t = Math.max(I(_) / 2, e), n = b[0], r = b[2];
			this.extendX_ && (n -= e, r += e), _[0] = n - t, _[2] = r + t, Kt(g, l);
			let i = Ct(y[0], l);
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
		if (this.ready && this.renderedResolution_ == u && this.renderedPixelRatio_ === d && this.renderedRevision_ == f && this.renderedRenderOrder_ == m && this.renderedFrameDeclutter_ === !!e.declutter && Ge(this.wrappedRenderedExtent_, _)) return h(this.renderedExtent_, v) || (this.hitDetectionImageData_ = null, this.renderedExtent_ = v), this.renderedCenter_ = g, this.replayGroupChanged = !1, !0;
		this.replayGroup_ = null;
		let S = new _c(tl(u, d), _, u, d), C = yr(), w;
		if (C) {
			for (let e = 0, t = y.length; e < t; ++e) {
				let t = y[e], r = Sr(t, l);
				n.loadFeatures(r, wr(u, l), C);
			}
			w = pr(C, l);
		} else for (let e = 0, t = y.length; e < t; ++e) n.loadFeatures(y[e], u, l);
		let T = el(u, d), E = !0, D = (e, n) => {
			let r, i = e.getStyleFunction() || t.getStyleFunction();
			if (i && (r = i(e, u)), r) {
				let t = this.renderFeature(e, T, r, S, w, this.getLayer().getDeclutter(), n);
				E &&= !t;
			}
		}, O = Sr(_, l), k = n.getFeaturesInExtent(O);
		m && k.sort(m);
		for (let e = 0, t = k.length; e < t; ++e) D(k[e], e);
		this.renderedFeatures_ = k, this.ready = E;
		let A = S.finish(), j = new Vc(_, u, d, n.getOverlaps(), A, t.getRenderBuffer(), !!e.declutter);
		return this.renderedResolution_ = u, this.renderedRevision_ = f, this.renderedRenderOrder_ = m, this.renderedFrameDeclutter_ = !!e.declutter, this.renderedExtent_ = v, this.wrappedRenderedExtent_ = _, this.renderedCenter_ = g, this.renderedProjection_ = l, this.renderedPixelRatio_ = d, this.replayGroup_ = j, this.hitDetectionImageData_ = null, this.replayGroupChanged = !0, !0;
	}
	renderFeature(e, t, n, r, i, a, o) {
		if (!n) return !1;
		let s = !1;
		if (Array.isArray(n)) for (let c = 0, l = n.length; c < l; ++c) s = rl(r, e, n[c], t, this.boundHandleStyleImageChange_, i, a, o) || s;
		else s = rl(r, e, n, t, this.boundHandleStyleImageChange_, i, a, o);
		return s;
	}
};
// node_modules/quickselect/index.js
function ml(e, t, n = 0, r = e.length - 1, i = gl) {
	for (; r > n;) {
		if (r - n > 600) {
			let a = r - n + 1, o = t - n + 1, s = Math.log(a), c = .5 * Math.exp(2 * s / 3), l = .5 * Math.sqrt(s * c * (a - c) / a) * (o - a / 2 < 0 ? -1 : 1);
			ml(e, t, Math.max(n, Math.floor(t - o * c / a + l)), Math.min(r, Math.floor(t + (a - o) * c / a + l)), i);
		}
		let a = e[t], o = n, s = r;
		for (hl(e, n, t), i(e[r], a) > 0 && hl(e, n, r); o < s;) {
			for (hl(e, o, s), o++, s--; i(e[o], a) < 0;) o++;
			for (; i(e[s], a) > 0;) s--;
		}
		i(e[n], a) === 0 ? hl(e, n, s) : (s++, hl(e, s, r)), s <= t && (n = s + 1), t <= s && (r = s - 1);
	}
}
function hl(e, t, n) {
	let r = e[t];
	e[t] = e[n], e[n] = r;
}
function gl(e, t) {
	return e < t ? -1 : +(e > t);
}
// node_modules/rbush/index.js
var _l = class {
	constructor(e = 9) {
		this._maxEntries = Math.max(4, e), this._minEntries = Math.max(2, Math.ceil(this._maxEntries * .4)), this.clear();
	}
	all() {
		return this._all(this.data, []);
	}
	search(e) {
		let t = this.data, n = [];
		if (!kl(e, t)) return n;
		let r = this.toBBox, i = [];
		for (; t;) {
			for (let a = 0; a < t.children.length; a++) {
				let o = t.children[a], s = t.leaf ? r(o) : o;
				kl(e, s) && (t.leaf ? n.push(o) : Ol(e, s) ? this._all(o, n) : i.push(o));
			}
			t = i.pop();
		}
		return n;
	}
	collides(e) {
		let t = this.data;
		if (!kl(e, t)) return !1;
		let n = [];
		for (; t;) {
			for (let r = 0; r < t.children.length; r++) {
				let i = t.children[r], a = t.leaf ? this.toBBox(i) : i;
				if (kl(e, a)) {
					if (t.leaf || Ol(e, a)) return !0;
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
		return this.data = Al([]), this;
	}
	remove(e, t) {
		if (!e) return this;
		let n = this.data, r = this.toBBox(e), i = [], a = [], o, s, c;
		for (; n || i.length;) {
			if (n || (n = i.pop(), s = i[i.length - 1], o = a.pop(), c = !0), n.leaf) {
				let r = vl(e, n.children, t);
				if (r !== -1) return n.children.splice(r, 1), i.push(n), this._condense(i), this;
			}
			!c && !n.leaf && Ol(n, r) ? (i.push(n), a.push(o), o = 0, s = n, n = n.children[0]) : s ? (o++, n = s.children[o], c = !1) : n = null;
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
		if (i <= a) return o = Al(e.slice(t, n + 1)), yl(o, this.toBBox), o;
		r || (r = Math.ceil(Math.log(i) / Math.log(a)), a = Math.ceil(i / a ** (r - 1))), o = Al([]), o.leaf = !1, o.height = r;
		let s = Math.ceil(i / a), c = s * Math.ceil(Math.sqrt(a));
		jl(e, t, n, c, this.compareMinX);
		for (let i = t; i <= n; i += c) {
			let t = Math.min(i + c - 1, n);
			jl(e, i, t, s, this.compareMinY);
			for (let n = i; n <= t; n += s) {
				let i = Math.min(n + s - 1, t);
				o.children.push(this._build(e, n, i, r - 1));
			}
		}
		return yl(o, this.toBBox), o;
	}
	_chooseSubtree(e, t, n, r) {
		for (; r.push(t), !(t.leaf || r.length - 1 === n);) {
			let n = Infinity, r = Infinity, i;
			for (let a = 0; a < t.children.length; a++) {
				let o = t.children[a], s = wl(o), c = El(e, o) - s;
				c < r ? (r = c, n = s < n ? s : n, i = o) : c === r && s < n && (n = s, i = o);
			}
			t = i || t.children[0];
		}
		return t;
	}
	_insert(e, t, n) {
		let r = n ? e : this.toBBox(e), i = [], a = this._chooseSubtree(r, this.data, t, i);
		for (a.children.push(e), xl(a, r); t >= 0 && i[t].children.length > this._maxEntries;) this._split(i, t), t--;
		this._adjustParentBBoxes(r, i, t);
	}
	_split(e, t) {
		let n = e[t], r = n.children.length, i = this._minEntries;
		this._chooseSplitAxis(n, i, r);
		let a = this._chooseSplitIndex(n, i, r), o = Al(n.children.splice(a, n.children.length - a));
		o.height = n.height, o.leaf = n.leaf, yl(n, this.toBBox), yl(o, this.toBBox), t ? e[t - 1].children.push(o) : this._splitRoot(n, o);
	}
	_splitRoot(e, t) {
		this.data = Al([e, t]), this.data.height = e.height + 1, this.data.leaf = !1, yl(this.data, this.toBBox);
	}
	_chooseSplitIndex(e, t, n) {
		let r, i = Infinity, a = Infinity;
		for (let o = t; o <= n - t; o++) {
			let t = bl(e, 0, o, this.toBBox), s = bl(e, o, n, this.toBBox), c = Dl(t, s), l = wl(t) + wl(s);
			c < i ? (i = c, r = o, a = l < a ? l : a) : c === i && l < a && (a = l, r = o);
		}
		return r || n - t;
	}
	_chooseSplitAxis(e, t, n) {
		let r = e.leaf ? this.compareMinX : Sl, i = e.leaf ? this.compareMinY : Cl;
		this._allDistMargin(e, t, n, r) < this._allDistMargin(e, t, n, i) && e.children.sort(r);
	}
	_allDistMargin(e, t, n, r) {
		e.children.sort(r);
		let i = this.toBBox, a = bl(e, 0, t, i), o = bl(e, n - t, n, i), s = Tl(a) + Tl(o);
		for (let r = t; r < n - t; r++) {
			let t = e.children[r];
			xl(a, e.leaf ? i(t) : t), s += Tl(a);
		}
		for (let r = n - t - 1; r >= t; r--) {
			let t = e.children[r];
			xl(o, e.leaf ? i(t) : t), s += Tl(o);
		}
		return s;
	}
	_adjustParentBBoxes(e, t, n) {
		for (let r = n; r >= 0; r--) xl(t[r], e);
	}
	_condense(e) {
		for (let t = e.length - 1, n; t >= 0; t--) e[t].children.length === 0 ? t > 0 ? (n = e[t - 1].children, n.splice(n.indexOf(e[t]), 1)) : this.clear() : yl(e[t], this.toBBox);
	}
};
function vl(e, t, n) {
	if (!n) return t.indexOf(e);
	for (let r = 0; r < t.length; r++) if (n(e, t[r])) return r;
	return -1;
}
function yl(e, t) {
	bl(e, 0, e.children.length, t, e);
}
function bl(e, t, n, r, i) {
	i ||= Al(null), i.minX = Infinity, i.minY = Infinity, i.maxX = -Infinity, i.maxY = -Infinity;
	for (let a = t; a < n; a++) {
		let t = e.children[a];
		xl(i, e.leaf ? r(t) : t);
	}
	return i;
}
function xl(e, t) {
	return e.minX = Math.min(e.minX, t.minX), e.minY = Math.min(e.minY, t.minY), e.maxX = Math.max(e.maxX, t.maxX), e.maxY = Math.max(e.maxY, t.maxY), e;
}
function Sl(e, t) {
	return e.minX - t.minX;
}
function Cl(e, t) {
	return e.minY - t.minY;
}
function wl(e) {
	return (e.maxX - e.minX) * (e.maxY - e.minY);
}
function Tl(e) {
	return e.maxX - e.minX + (e.maxY - e.minY);
}
function El(e, t) {
	return (Math.max(t.maxX, e.maxX) - Math.min(t.minX, e.minX)) * (Math.max(t.maxY, e.maxY) - Math.min(t.minY, e.minY));
}
function Dl(e, t) {
	let n = Math.max(e.minX, t.minX), r = Math.max(e.minY, t.minY), i = Math.min(e.maxX, t.maxX), a = Math.min(e.maxY, t.maxY);
	return Math.max(0, i - n) * Math.max(0, a - r);
}
function Ol(e, t) {
	return e.minX <= t.minX && e.minY <= t.minY && t.maxX <= e.maxX && t.maxY <= e.maxY;
}
function kl(e, t) {
	return t.minX <= e.maxX && t.minY <= e.maxY && t.maxX >= e.minX && t.maxY >= e.minY;
}
function Al(e) {
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
function jl(e, t, n, r, i) {
	let a = [t, n];
	for (; a.length;) {
		if (n = a.pop(), t = a.pop(), n - t <= r) continue;
		let o = t + Math.ceil((n - t) / r / 2) * r;
		ml(e, o, t, n, i), a.push(t, o, o, n);
	}
}
// node_modules/ol/expr/expression.js
var Ml = 0, Nl = 1 << Ml++, W = 1 << Ml++, G = 1 << Ml++, K = 1 << Ml++, Pl = 1 << Ml++, Fl = 1 << Ml++, Il = 2 ** Ml - 1, Ll = {
	[Nl]: "boolean",
	[W]: "number",
	[G]: "string",
	[K]: "color",
	[Pl]: "number[]",
	[Fl]: "size"
}, Rl = Object.keys(Ll).map(Number).sort(u);
function zl(e) {
	return e in Ll;
}
function Bl(e) {
	let t = [];
	for (let n of Rl) Vl(e, n) && t.push(Ll[n]);
	return t.length === 0 ? "untyped" : t.length < 3 ? t.join(" or ") : t.slice(0, -1).join(", ") + ", or " + t[t.length - 1];
}
function Vl(e, t) {
	return (e & t) === t;
}
function Hl(e, t) {
	return !!(e & t);
}
function Ul(e, t) {
	return e === t;
}
function Wl(e) {
	return e instanceof Error ? e.message : String(e);
}
var Gl = class {
	constructor(e, t) {
		if (!zl(e)) throw Error(`literal expressions must have a specific type, got ${Bl(e)}`);
		this.type = e, this.value = t;
	}
}, Kl = class {
	constructor(e, t, ...n) {
		this.type = e, this.operator = t, this.args = n;
	}
};
function ql(e) {
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
function Jl(e, t, n) {
	switch (typeof e) {
		case "boolean":
			if (Ul(t, G)) return new Gl(G, e ? "true" : "false");
			if (!Vl(t, Nl)) throw Error(`got a boolean, but expected ${Bl(t)}`);
			return new Gl(Nl, e);
		case "number":
			if (Ul(t, Fl)) return new Gl(Fl, ao(e));
			if (Ul(t, Nl)) return new Gl(Nl, !!e);
			if (Ul(t, G)) return new Gl(G, e.toString());
			if (!Vl(t, W)) throw Error(`got a number, but expected ${Bl(t)}`);
			return new Gl(W, e);
		case "string":
			if (Ul(t, K)) return new Gl(K, Fo(e));
			if (Ul(t, Nl)) return new Gl(Nl, !!e);
			if (!Vl(t, G)) throw Error(`got a string, but expected ${Bl(t)}`);
			return new Gl(G, e);
	}
	if (!Array.isArray(e)) throw Error("expression must be an array or a primitive value");
	if (e.length === 0) throw Error("empty expression");
	if (typeof e[0] == "string") return pu(e, t, n);
	for (let t of e) if (typeof t != "number") throw Error("expected an array of numbers");
	if (Ul(t, Fl)) {
		if (e.length !== 2) throw Error(`expected an array of two values for a size, got ${e.length}`);
		return new Gl(Fl, e);
	}
	if (Ul(t, K)) {
		if (e.length === 3) return new Gl(K, [...e, 1]);
		if (e.length === 4) return new Gl(K, e);
		throw Error(`expected an array of 3 or 4 values for a color, got ${e.length}`);
	}
	if (!Vl(t, Pl)) throw Error(`got an array of numbers, but expected ${Bl(t)}`);
	return new Gl(Pl, e);
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
	InterpolateHcl: "interpolate-hcl",
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
}, Yl = {
	[q.Get]: X(J(1, Infinity), Xl),
	[q.Var]: Zl(),
	[q.Has]: X(J(1, Infinity), Xl),
	[q.Id]: X(Ql, nu),
	[q.Concat]: X(J(2, Infinity), Y(G)),
	[q.GeometryType]: X($l, nu),
	[q.LineMetric]: X(eu, nu),
	[q.Resolution]: X(tu, nu),
	[q.Zoom]: X(tu, nu),
	[q.Time]: X(tu, nu),
	[q.Any]: X(J(2, Infinity), Y(Nl)),
	[q.All]: X(J(2, Infinity), Y(Nl)),
	[q.Not]: X(J(1, 1), Y(Nl)),
	[q.Equal]: X(J(2, 2), iu()),
	[q.NotEqual]: X(J(2, 2), iu()),
	[q.GreaterThan]: X(J(2, 2), Y(W)),
	[q.GreaterThanOrEqualTo]: X(J(2, 2), Y(W)),
	[q.LessThan]: X(J(2, 2), Y(W)),
	[q.LessThanOrEqualTo]: X(J(2, 2), Y(W)),
	[q.Multiply]: X(J(2, Infinity), ru),
	[q.Coalesce]: X(J(2, Infinity), ru),
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
	[q.Match]: X(J(4, Infinity), ou, su),
	[q.Between]: X(J(3, 3), Y(W)),
	[q.Interpolate]: X(J(6, Infinity), ou, lu),
	[q.InterpolateHcl]: X(J(6, Infinity), ou, cu, lu),
	[q.Case]: X(J(3, Infinity), au, uu),
	[q.In]: X(J(2, 2), du),
	[q.Number]: X(J(1, Infinity), Y(Il)),
	[q.String]: X(J(1, Infinity), Y(Il)),
	[q.Array]: X(J(1, Infinity), Y(W)),
	[q.Color]: X(J(1, 4), Y(W)),
	[q.Band]: X(J(1, 3), Y(W)),
	[q.Palette]: X(J(2, 2), fu),
	[q.ToString]: X(J(1, 1), Y(Nl | W | G | K))
};
function Xl(e, t, n) {
	let r = e.length - 1, i = Array(r);
	for (let a = 0; a < r; ++a) {
		let r = e[a + 1];
		switch (typeof r) {
			case "number":
				i[a] = new Gl(W, r);
				break;
			case "string":
				i[a] = new Gl(G, r);
				break;
			default: throw Error(`expected a string key or numeric array index for a get operation, got ${r}`);
		}
		a === 0 && n.properties.set(String(r), t);
	}
	return i;
}
function Zl() {
	return function(e, t, n) {
		let r = e[1];
		if (typeof r != "string") throw Error("expected a string argument for var operation");
		let i = t, a = n.inputVariables?.[r];
		if (a !== void 0) {
			let e = Jl(a, Il, n);
			if (!(e instanceof Gl)) throw Error(`style variables should only be literal values (no expressions!), variable name: ${r}`);
			let o = e.type;
			if (typeof a == "string" && Hl(i, K) && !Hl(i, G) ? o = K : Array.isArray(a) && a.length === 2 && Hl(i, Fl) && !Hl(i, Pl) && (o = Fl), i &= o, i === 0) throw Error(`the type expected from the var operator (${Bl(t)}) did not have any overlap with the type of the corresponding style variables (${Bl(o)}), variable name: ${r}`);
		}
		if (n.variables.has(r)) {
			let e = n.variables.get(r);
			if (i &= e, i === 0) throw Error(`a new type expected from the var operator (${Bl(t)}) did not have any overlap with the previous type expected for it (${Bl(e)}), variable name: ${r}`);
		}
		return n.variables.set(r, i), new Kl(i, "var", new Gl(G, r));
	};
}
function Ql(e, t, n) {
	n.featureId = !0;
}
function $l(e, t, n) {
	n.geometryType = !0;
}
function eu(e, t, n) {
	n.mCoordinate = !0;
}
function tu(e, t, n) {
	n.mapState = !0;
}
function nu(e, t, n) {
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
function ru(e, t, n) {
	let r = e.length - 1, i = Array(r);
	for (let a = 0; a < r; ++a) {
		let r = Jl(e[a + 1], t, n);
		i[a] = r;
	}
	return i;
}
function Y(e) {
	return function(t, n, r) {
		let i = t.length - 1, a = Array(i);
		for (let n = 0; n < i; ++n) {
			let i = Jl(t[n + 1], e, r);
			a[n] = i;
		}
		return a;
	};
}
function iu() {
	return function(e, t, n) {
		let r = e[0], i = e.length - 1, a = Array(i), o = Il;
		for (let t = 0; t < i; ++t) {
			let r = Jl(e[t + 1], o, n);
			o &= r.type;
		}
		if (o === 0) throw Error(`no common type was found among the arguments of ${r}`);
		for (let t = 0; t < i; ++t) {
			let r = Jl(e[t + 1], o, n);
			a[t] = r;
		}
		return a;
	};
}
function au(e, t, n) {
	let r = e[0], i = e.length - 1;
	if (i % 2 == 0) throw Error(`expected an odd number of arguments for ${r}, got ${i} instead`);
}
function ou(e, t, n) {
	let r = e[0], i = e.length - 1;
	if (i % 2 == 1) throw Error(`expected an even number of arguments for operation ${r}, got ${i} instead`);
}
function su(e, t, n) {
	let r = e.length - 1, i = Jl(e[e.length - 1], t, n), a = G | W | Nl, o = Array(r - 2);
	for (let t = 0; t < r - 2; t += 2) {
		try {
			let r = Jl(e[t + 2], a, n);
			a &= r.type;
		} catch (e) {
			let n = Wl(e);
			throw Error(`failed to parse argument ${t + 1} of match expression: ${n}`);
		}
		if (a === 0) throw Error("no common type was found among the arguments of match expression");
	}
	for (let t = 0; t < r - 2; t += 2) {
		try {
			let r = Jl(e[t + 2], a, n);
			o[t] = r;
		} catch (e) {
			let n = Wl(e);
			throw Error(`failed to parse argument ${t + 1} of match expression: ${n}`);
		}
		try {
			let r = Jl(e[t + 3], i.type, n);
			o[t + 1] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 2} of match expression: ${Wl(e)}`);
		}
	}
	return [
		Jl(e[1], a, n),
		...o,
		i
	];
}
function cu(e, t, n) {
	if (!Vl(t, K)) throw Error(`the ${e[0]} operator can only return a color, got ${Bl(t)}`);
}
function lu(e, t, n) {
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
	let a = new Gl(W, i), o;
	try {
		o = Jl(e[2], W, n);
	} catch (e) {
		throw Error(`failed to parse argument 1 in interpolate expression: ${Wl(e)}`);
	}
	let s = Array(e.length - 3);
	for (let r = 0; r < s.length; r += 2) {
		try {
			let t = Jl(e[r + 3], W, n);
			s[r] = t;
		} catch (e) {
			throw Error(`failed to parse argument ${r + 2} for interpolate expression: ${Wl(e)}`);
		}
		try {
			let i = Jl(e[r + 4], t, n);
			s[r + 1] = i;
		} catch (e) {
			throw Error(`failed to parse argument ${r + 3} for interpolate expression: ${Wl(e)}`);
		}
	}
	return [
		a,
		o,
		...s
	];
}
function uu(e, t, n) {
	let r = Jl(e[e.length - 1], t, n), i = Array(e.length - 1);
	for (let t = 0; t < i.length - 1; t += 2) {
		try {
			let r = Jl(e[t + 1], Nl, n);
			i[t] = r;
		} catch (e) {
			throw Error(`failed to parse argument ${t} of case expression: ${Wl(e)}`);
		}
		try {
			let a = Jl(e[t + 2], r.type, n);
			i[t + 1] = a;
		} catch (e) {
			throw Error(`failed to parse argument ${t + 1} of case expression: ${Wl(e)}`);
		}
	}
	return i[i.length - 1] = r, i;
}
function du(e, t, n) {
	let r = e[2];
	if (!Array.isArray(r)) throw Error("the second argument for the \"in\" operator must be an array");
	let i;
	if (r[0] === "literal") {
		if (r = r[1], !Array.isArray(r)) throw Error("failed to parse \"in\" expression: the literal operator must be followed by an array");
	} else if (typeof r[0] == "string") throw Error("for the \"in\" operator, a string array should be wrapped in a \"literal\" operator to disambiguate from expressions");
	i = typeof r[0] == "string" ? G : W;
	let a = Array(r.length);
	for (let e = 0; e < a.length; e++) try {
		let t = Jl(r[e], i, n);
		a[e] = t;
	} catch (t) {
		throw Error(`failed to parse haystack item ${e} for "in" expression: ${Wl(t)}`);
	}
	return [Jl(e[1], i, n), ...a];
}
function fu(e, t, n) {
	let r;
	try {
		r = Jl(e[1], W, n);
	} catch (e) {
		throw Error(`failed to parse first argument in palette expression: ${Wl(e)}`);
	}
	let i = e[2];
	if (!Array.isArray(i)) throw Error("the second argument of palette must be an array");
	let a = Array(i.length);
	for (let e = 0; e < a.length; e++) {
		let t;
		try {
			t = Jl(i[e], K, n);
		} catch (t) {
			throw Error(`failed to parse color at index ${e} in palette expression: ${Wl(t)}`);
		}
		if (!(t instanceof Gl)) throw Error(`the palette color at index ${e} must be a literal value`);
		a[e] = t;
	}
	return [r, ...a];
}
function X(...e) {
	return function(t, n, r) {
		let i = t[0], a = [];
		for (let i = 0; i < e.length; i++) {
			let o = e[i](t, n, r);
			if (i == e.length - 1) {
				if (!o) throw Error("expected last argument validator to return the parsed args");
				a = o;
			}
		}
		return new Kl(n, i, ...a);
	};
}
function pu(e, t, n) {
	let r = e[0], i = Yl[r];
	if (!i) throw Error(`unknown operator: ${r}`);
	return i(e, t, n);
}
function mu(e) {
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
		case "GeometryCollection": return mu(e.getGeometries()[0]);
		default: return "";
	}
}
// node_modules/ol/expr/cpu.js
function hu() {
	return {
		variables: {},
		properties: {},
		resolution: NaN,
		featureId: null,
		geometryType: ""
	};
}
function gu(e, t, n) {
	return _u(Jl(e, t, n), n);
}
function _u(e, t) {
	if (e instanceof Gl) {
		if (e.type === K && typeof e.value == "string") {
			let t = Fo(e.value);
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
		case q.Coalesce: return bu(e, t);
		case q.Get:
		case q.Var:
		case q.Has: return xu(e, t);
		case q.Id: return (e) => e.featureId ?? "";
		case q.GeometryType: return (e) => e.geometryType;
		case q.Concat: {
			let n = e.args.map((e) => _u(e, t));
			return (e) => "".concat(...n.map((t) => (t(e) ?? "").toString()));
		}
		case q.Resolution: return (e) => e.resolution;
		case q.Any:
		case q.All:
		case q.Between:
		case q.In:
		case q.Not: return Cu(e, t);
		case q.Equal:
		case q.NotEqual:
		case q.LessThan:
		case q.LessThanOrEqualTo:
		case q.GreaterThan:
		case q.GreaterThanOrEqualTo: return Su(e, t);
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
		case q.Sqrt: return wu(e, t);
		case q.Case: return Tu(e, t);
		case q.Match: return Eu(e, t);
		case q.Interpolate:
		case q.InterpolateHcl: return Du(e, t);
		case q.ToString: return Ou(e, t);
		case q.Array: return vu(e, t);
		case q.Color: return yu(e, t);
		default: throw Error(`Unsupported operator ${n}`);
	}
}
function vu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = _u(e.args[i], t);
	if (e.type === K && (n === 3 || n === 4)) {
		let e = n === 4 ? r[3] : null;
		return (t) => [
			r[0](t) * 255,
			r[1](t) * 255,
			r[2](t) * 255,
			e ? e(t) : 1
		];
	}
	return (e) => {
		let t = Array(n);
		for (let i = 0; i < n; ++i) t[i] = r[i](e);
		return t;
	};
}
function yu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = _u(e.args[i], t);
	if (n < 3) {
		let e = n === 2 ? r[1] : null;
		return (t) => {
			let n = r[0](t);
			return [
				n,
				n,
				n,
				e ? e(t) : 1
			];
		};
	}
	let i = n === 4 ? r[3] : null;
	return (e) => [
		r[0](e),
		r[1](e),
		r[2](e),
		i ? i(e) : 1
	];
}
function bu(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = _u(e.args[n], t);
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
function xu(e, t) {
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
function Su(e, t) {
	let n = e.operator, r = _u(e.args[0], t), i = _u(e.args[1], t);
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
function Cu(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = _u(e.args[n], t);
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
function wu(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = _u(e.args[n], t);
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
function Tu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = _u(e.args[i], t);
	return (e) => {
		for (let t = 0; t < n - 1; t += 2) if (r[t](e)) return r[t + 1](e);
		return r[n - 1](e);
	};
}
function Eu(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = _u(e.args[i], t);
	return (e) => {
		let t = r[0](e);
		for (let i = 1; i < n - 1; i += 2) if (t === r[i](e)) return r[i + 1](e);
		return r[n - 1](e);
	};
}
function Du(e, t) {
	let n = e.args.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = _u(e.args[i], t);
	let i = e.operator === q.InterpolateHcl ? ju : Au;
	return (e) => {
		let t = r[0](e), a = r[1](e), o, s;
		for (let c = 2; c < n; c += 2) {
			let n = r[c](e), l = r[c + 1](e), u = Array.isArray(l);
			if (u && (l = Oo(l)), n >= a) return c === 2 ? l : u ? i(t, a, o, s, n, l) : ku(t, a, o, s, n, l);
			o = n, s = l;
		}
		return s;
	};
}
function Ou(e, t) {
	let n = e.operator, r = e.args.length, i = Array(r);
	for (let n = 0; n < r; ++n) i[n] = _u(e.args[n], t);
	switch (n) {
		case q.ToString: return (t) => {
			let n = i[0](t);
			return e.args[0].type === K ? Lo(n) : n.toString();
		};
		default: throw Error(`Unsupported convert operator ${n}`);
	}
}
function ku(e, t, n, r, i, a) {
	let o = i - n;
	if (o === 0) return r;
	let s = t - n;
	return r + (e === 1 ? s / o : (e ** +s - 1) / (e ** +o - 1)) * (a - r);
}
function Au(e, t, n, r, i, a) {
	return i - n === 0 ? r : [
		ku(e, t, n, r[0], i, a[0]),
		ku(e, t, n, r[1], i, a[1]),
		ku(e, t, n, r[2], i, a[2]),
		ku(e, t, n, r[3], i, a[3])
	];
}
function ju(e, t, n, r, i, a) {
	if (i - n === 0) return r;
	let o = No(r), s = No(a), c = s[2] - o[2];
	return c > 180 ? c -= 360 : c < -180 && (c += 360), Po([
		ku(e, t, n, o[0], i, s[0]),
		ku(e, t, n, o[1], i, s[1]),
		o[2] + ku(e, t, n, 0, i, c),
		ku(e, t, n, r[3], i, a[3])
	]);
}
// node_modules/ol/style/RegularShape.js
var Mu = class e extends Wc {
	constructor(e) {
		super({
			opacity: 1,
			rotateWithView: e.rotateWithView !== void 0 && e.rotateWithView,
			rotation: e.rotation === void 0 ? 0 : e.rotation,
			scale: e.scale === void 0 ? 1 : e.scale,
			displacement: e.displacement === void 0 ? [0, 0] : e.displacement,
			declutterMode: e.declutterMode
		}), this.hitDetectionCanvas_ = null, this.fill_ = e.fill === void 0 ? null : e.fill, this.origin_ = [0, 0], this.points_ = e.points, this.radius = e.radius, this.radius2_ = e.radius2, this.angle_ = e.angle === void 0 ? 0 : e.angle, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.size_, this.renderOptions_, this.imageState_ = this.fill_ && this.fill_.loading() ? B.LOADING : B.LOADED, this.imageState_ === B.LOADING && this.ready().then(() => this.imageState_ = B.LOADED), this.render();
	}
	clone() {
		let t = this.getScale(), n = new e({
			fill: this.getFill()?.clone(),
			points: this.getPoints(),
			radius: this.getRadius(),
			radius2: this.getRadius2(),
			angle: this.getAngle(),
			stroke: this.getStroke()?.clone(),
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
		let t = this.fill_?.getKey(), n = `${e},${this.angle_},${this.radius},${this.radius2_},${this.points_},${t}` + Object.values(this.renderOptions_).join(","), r = Ts.get(n, null)?.getImage(1);
		if (!r) {
			let t = this.renderOptions_, i = Math.ceil(t.size * e), a = _e(i, i);
			this.draw_(t, a, e), r = a.canvas;
			let o = new Ds(r, void 0, {}, B.LOADED, null);
			Ts.set(n, null, o), createImageBitmap(r).then((e) => {
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
		let e = Ps, t = Is, n = 0, r = null, i = 0, a, o = 0;
		this.stroke_ && (a = ks(this.stroke_.getColor() ?? "#000"), o = this.stroke_.getWidth() ?? 1, r = this.stroke_.getLineDash(), i = this.stroke_.getLineDashOffset() ?? 0, t = this.stroke_.getLineJoin() ?? "round", e = this.stroke_.getLineCap() ?? "round", n = this.stroke_.getMiterLimit() ?? 10);
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
			e === null && (e = Ns), t.fillStyle = ks(e), t.fill();
		}
		e.strokeStyle && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineCap = e.lineCap, t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}
	createHitDetectionCanvas_(e) {
		let t;
		if (this.fill_) {
			let n = this.fill_.getColor(), r = 0;
			typeof n == "string" && (n = Io(n)), n === null ? r = 1 : Array.isArray(n) && (r = n.length === 4 ? n[3] : 1), r === 0 && (t = _e(e.size, e.size), this.drawHitDetectionCanvas_(e, t));
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
		t.translate(e.size / 2, e.size / 2), this.createPath_(t), t.fillStyle = Ns, t.fill(), e.strokeStyle && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}
	ready() {
		return this.fill_ ? this.fill_.ready() : Promise.resolve();
	}
}, Nu = class e extends Mu {
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
			fill: this.getFill()?.clone(),
			stroke: this.getStroke()?.clone(),
			radius: this.getRadius(),
			scale: Array.isArray(t) ? t.slice() : t,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			displacement: this.getDisplacement().slice(),
			declutterMode: this.getDeclutterMode()
		});
		return n.setOpacity(this.getOpacity()), n;
	}
}, Pu = class e {
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
			let t = Os(null, e.src, { crossOrigin: "anonymous" }, void 0, e.offset ? null : e.color ? e.color : null, !(e.offset && e.size));
			t.ready().then(() => {
				this.patternImage_ = null;
			}), t.getImageState() === B.IDLE && t.load(), t.getImageState() === B.LOADING && (this.patternImage_ = t);
		}
		this.color_ = e;
	}
	getKey() {
		let e = this.getColor();
		return e ? e instanceof CanvasPattern || e instanceof CanvasGradient ? y(e) : typeof e == "object" && "src" in e ? e.src + ":" + e.offset : Io(e).toString() : "";
	}
	loading() {
		return !!this.patternImage_;
	}
	ready() {
		return this.patternImage_ ? this.patternImage_.ready() : Promise.resolve();
	}
}, Fu = class e {
	constructor(e) {
		e ||= {}, this.color_ = e.color === void 0 ? null : e.color, this.lineCap_ = e.lineCap, this.lineDash_ = e.lineDash === void 0 ? null : e.lineDash, this.lineDashOffset_ = e.lineDashOffset, this.lineJoin_ = e.lineJoin, this.miterLimit_ = e.miterLimit, this.offset_ = e.offset, this.width_ = e.width;
	}
	clone() {
		let t = this.getColor();
		return new e({
			color: Array.isArray(t) ? t.slice() : t || void 0,
			lineCap: this.getLineCap(),
			lineDash: this.getLineDash()?.slice(),
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
}, Iu = class e {
	constructor(e) {
		e ||= {}, this.geometry_ = null, this.geometryFunction_ = Bu, e.geometry !== void 0 && this.setGeometry(e.geometry), this.fill_ = e.fill === void 0 ? null : e.fill, this.image_ = e.image === void 0 ? null : e.image, this.renderer_ = e.renderer === void 0 ? null : e.renderer, this.hitDetectionRenderer_ = e.hitDetectionRenderer === void 0 ? null : e.hitDetectionRenderer, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.text_ = e.text === void 0 ? null : e.text, this.zIndex_ = e.zIndex;
	}
	clone() {
		let t = this.getGeometry();
		return t && typeof t == "object" && (t = t.clone()), new e({
			geometry: t ?? void 0,
			fill: this.getFill()?.clone(),
			image: this.getImage()?.clone(),
			renderer: this.getRenderer() ?? void 0,
			stroke: this.getStroke()?.clone(),
			text: this.getText()?.clone(),
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
		}) : this.geometryFunction_ = Bu, this.geometry_ = e;
	}
	setZIndex(e) {
		this.zIndex_ = e;
	}
};
function Lu(e) {
	let t;
	if (typeof e == "function") t = e;
	else {
		let n;
		Array.isArray(e) ? n = e : (L(typeof e.getZIndex == "function", "Expected an `Style` or an array of `Style`"), n = [e]), t = function() {
			return n;
		};
	}
	return t;
}
var Ru = null;
function zu(e, t) {
	if (!Ru) {
		let e = new Pu({ color: "rgba(255,255,255,0.4)" }), t = new Fu({
			color: "#3399CC",
			width: 1.25
		});
		Ru = [new Iu({
			image: new Nu({
				fill: e,
				stroke: t,
				radius: 5
			}),
			fill: e,
			stroke: t
		})];
	}
	return Ru;
}
function Bu(e) {
	return e.getGeometry();
}
// node_modules/ol/style/Text.js
var Vu = "#333", Hu = class e {
	constructor(e) {
		e ||= {}, this.font_ = e.font, this.rotation_ = e.rotation, this.rotateWithView_ = e.rotateWithView, this.keepUpright_ = e.keepUpright, this.scale_ = e.scale, this.scaleArray_ = ao(e.scale === void 0 ? 1 : e.scale), this.text_ = e.text, this.textAlign_ = e.textAlign, this.justify_ = e.justify, this.repeat_ = e.repeat, this.textBaseline_ = e.textBaseline, this.fill_ = e.fill === void 0 ? new Pu({ color: Vu }) : e.fill, this.maxAngle_ = e.maxAngle === void 0 ? Math.PI / 4 : e.maxAngle, this.placement_ = e.placement === void 0 ? "point" : e.placement, this.overflow_ = !!e.overflow, this.stroke_ = e.stroke === void 0 ? null : e.stroke, this.offsetX_ = e.offsetX === void 0 ? 0 : e.offsetX, this.offsetY_ = e.offsetY === void 0 ? 0 : e.offsetY, this.backgroundFill_ = e.backgroundFill ? e.backgroundFill : null, this.backgroundStroke_ = e.backgroundStroke ? e.backgroundStroke : null, this.padding_ = e.padding === void 0 ? null : e.padding, this.declutterMode_ = e.declutterMode;
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
			fill: this.getFill() instanceof Pu ? this.getFill().clone() : this.getFill(),
			stroke: this.getStroke()?.clone(),
			offsetX: this.getOffsetX(),
			offsetY: this.getOffsetY(),
			backgroundFill: this.getBackgroundFill()?.clone(),
			backgroundStroke: this.getBackgroundStroke()?.clone(),
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
		this.scale_ = e, this.scaleArray_ = ao(e === void 0 ? 1 : e);
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
// node_modules/ol/render/canvas/style.js
function Uu(e) {
	return !0;
}
function Wu(e, t) {
	t ??= ql();
	let n = qu(e, t), r = hu();
	return function(e, i) {
		if (r.properties = e.getPropertiesInternal() ?? {}, r.resolution = i, t.featureId) {
			let t = e.getId();
			t === void 0 ? r.featureId = null : r.featureId = t;
		}
		if (t.geometryType) {
			let t = e.getGeometry();
			t && (r.geometryType = mu(t));
		}
		return n(r);
	};
}
function Gu(e, t) {
	t ??= ql();
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = Ju(e[i], t);
	let i = hu(), a = Array(n);
	return function(e, o) {
		if (i.properties = e.getPropertiesInternal() ?? {}, i.resolution = o, t.featureId) {
			let t = e.getId();
			t === void 0 ? i.featureId = null : i.featureId = t;
		}
		if (t.geometryType) {
			let t = e.getGeometry();
			t && (i.geometryType = mu(t));
		}
		let s = 0;
		for (let e = 0; e < n; ++e) {
			let t = r[e](i);
			t && (a[s] = t, s += 1);
		}
		return a.length = s, a;
	};
}
function Ku(e, t) {
	if (t ??= ql(), !Array.isArray(e)) return Gu([e], t);
	let n = e.length;
	if ("style" in e[0]) {
		let r = Array(n);
		for (let t = 0; t < n; ++t) {
			let n = e[t];
			if (!("style" in n)) throw Error("Expected a list of rules with a style property");
			r[t] = n;
		}
		return Wu(r, t);
	}
	return Gu(e, t);
}
function qu(e, t) {
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) {
		let n = e[i], a = "filter" in n && n.filter !== void 0 ? gu(n.filter, Nl, t) : Uu, o;
		if (Array.isArray(n.style)) {
			let e = n.style.length;
			o = Array(e);
			for (let r = 0; r < e; ++r) o[r] = Ju(n.style[r], t);
		} else o = [Ju(n.style, t)];
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
function Ju(e, t) {
	let n = Yu(e, "", t), i = Xu(e, "", t), a = Zu(e, t), o = Qu(e, t), s = rd(e, "z-index", t);
	if (!n && !i && !a && !o && !r(e)) throw Error("No fill, stroke, point, or text symbolizer properties in style: " + JSON.stringify(e));
	let c = new Iu();
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
function Yu(e, t, n) {
	let r;
	if (t + "fill-pattern-src" in e) r = ad(e, t + "fill-", n);
	else {
		if (nd(e, t + "fill-color") === "none") return (e) => null;
		r = sd(e, t + "fill-color", n);
	}
	if (!r) return null;
	let i = new Pu();
	return function(e) {
		let t = r(e);
		return t === mo ? null : (i.setColor(t), i);
	};
}
function Xu(e, t, n) {
	let r = rd(e, t + "stroke-width", n), i = sd(e, t + "stroke-color", n);
	if (!r && !i) return null;
	let a = id(e, t + "stroke-line-cap", n), o = id(e, t + "stroke-line-join", n), s = cd(e, t + "stroke-line-dash", n), c = rd(e, t + "stroke-line-dash-offset", n), l = rd(e, t + "stroke-miter-limit", n), u = rd(e, t + "stroke-offset", n), d = new Fu();
	return function(e) {
		if (i) {
			let t = i(e);
			if (t === mo) return null;
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
function Zu(e, t) {
	let n = "text-", r = id(e, "text-value", t);
	if (!r) return null;
	let i = Yu(e, n, t), a = Yu(e, "text-background-", t), o = Xu(e, n, t), s = Xu(e, "text-background-", t), c = id(e, "text-font", t), l = rd(e, "text-max-angle", t), u = rd(e, "text-offset-x", t), d = rd(e, "text-offset-y", t), f = od(e, "text-overflow", t), p = id(e, "text-placement", t), m = rd(e, "text-repeat", t), h = dd(e, "text-scale", t), g = od(e, "text-rotate-with-view", t), _ = rd(e, "text-rotation", t), v = id(e, "text-align", t), y = id(e, "text-justify", t), b = id(e, "text-baseline", t), x = od(e, "text-keep-upright", t), S = cd(e, "text-padding", t), C = new Hu({ declutterMode: vd(e, "text-declutter-mode") });
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
function Qu(e, t) {
	return "icon-src" in e ? $u(e, t) : "shape-points" in e ? ed(e, t) : "circle-radius" in e ? td(e, t) : null;
}
function $u(e, t) {
	let n = "icon-src", r = bd(nd(e, n), n), i = ld(e, "icon-anchor", t), a = dd(e, "icon-scale", t), o = rd(e, "icon-opacity", t), s = ld(e, "icon-displacement", t), c = rd(e, "icon-rotation", t), l = od(e, "icon-rotate-with-view", t), u = hd(e, "icon-anchor-origin"), d = gd(e, "icon-anchor-x-units"), f = gd(e, "icon-anchor-y-units"), p = nd(e, "icon-color"), m, h = null;
	p !== void 0 && (Array.isArray(p) && p.length > 0 && typeof p[0] == "string" ? h = sd(e, "icon-color", t) : m = Sd(p, "icon-color"));
	let g = md(e, "icon-cross-origin"), _ = _d(e, "icon-offset"), v = hd(e, "icon-offset-origin"), y = fd(e, "icon-width"), b = {
		src: r,
		anchorOrigin: u,
		anchorXUnits: d,
		anchorYUnits: f,
		crossOrigin: g,
		offset: _,
		offsetOrigin: v,
		height: fd(e, "icon-height"),
		width: y,
		size: pd(e, "icon-size"),
		declutterMode: vd(e, "icon-declutter-mode")
	}, x = null;
	return function(e) {
		if (x) h && x.setColor(h(e));
		else {
			let t = h ? h(e) : m;
			x = new Kc(t === void 0 ? Object.assign({}, b) : Object.assign({}, b, { color: t }));
		}
		return o && x.setOpacity(o(e)), s && x.setDisplacement(s(e)), c && x.setRotation(c(e)), l && x.setRotateWithView(l(e)), a && x.setScale(a(e)), i && x.setAnchor(i(e)), x;
	};
}
function ed(e, t) {
	let n = "shape-", r = "shape-points", i = "shape-radius", a = xd(nd(e, r), r);
	if (!(i in e)) throw Error(`Expected a number for ${i}`);
	let o = rd(e, i, t), s = nd(e, i), c = typeof s == "number" ? s : 5, l = "shape-radius2", u = rd(e, l, t), d = nd(e, l), f = typeof d == "number" ? d : void 0, p = Yu(e, n, t), m = Xu(e, n, t), h = dd(e, "shape-scale", t), g = ld(e, "shape-displacement", t), _ = rd(e, "shape-rotation", t), v = od(e, "shape-rotate-with-view", t), y = new Mu({
		points: a,
		radius: c,
		radius2: f,
		angle: fd(e, "shape-angle"),
		declutterMode: vd(e, "shape-declutter-mode")
	});
	return function(e) {
		return o && y.setRadius(o(e)), u && y.setRadius2(u(e)), p && y.setFill(p(e)), m && y.setStroke(m(e)), g && y.setDisplacement(g(e)), _ && y.setRotation(_(e)), v && y.setRotateWithView(v(e)), h && y.setScale(h(e)), y;
	};
}
function td(e, t) {
	let n = "circle-", r = Yu(e, n, t), i = Xu(e, n, t), a = rd(e, "circle-radius", t), o = dd(e, "circle-scale", t), s = ld(e, "circle-displacement", t), c = rd(e, "circle-rotation", t), l = od(e, "circle-rotate-with-view", t), u = new Nu({
		radius: 5,
		declutterMode: vd(e, "circle-declutter-mode")
	});
	return function(e) {
		return a && u.setRadius(a(e)), r && u.setFill(r(e)), i && u.setStroke(i(e)), s && u.setDisplacement(s(e)), c && u.setRotation(c(e)), l && u.setRotateWithView(l(e)), o && u.setScale(o(e)), u;
	};
}
function nd(e, t) {
	if (!(t in e)) return;
	let n = e[t];
	return n === void 0 ? void 0 : n;
}
function rd(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return;
	let i = gu(r, W, n);
	return function(e) {
		return xd(i(e), t);
	};
}
function id(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	let i = gu(r, G, n);
	return function(e) {
		return bd(i(e), t);
	};
}
function ad(e, t, n) {
	let r = id(e, t + "pattern-src", n), i = ud(e, t + "pattern-offset", n), a = ud(e, t + "pattern-size", n), o = sd(e, t + "color", n);
	return function(e) {
		return r ? {
			src: r(e),
			offset: i ? i(e) : null,
			size: a ? a(e) : null,
			color: o ? o(e) : null
		} : null;
	};
}
function od(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	let i = gu(r, Nl, n);
	return function(e) {
		let n = i(e);
		if (typeof n != "boolean") throw Error(`Expected a boolean for ${t}`);
		return n;
	};
}
function sd(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	let i = gu(r, K, n);
	return function(e) {
		return Sd(i(e), t);
	};
}
function cd(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	if (Array.isArray(r) && (r.length === 0 || typeof r[0] != "string")) {
		let e = r.map((e, r) => {
			if (typeof e == "number") return () => e;
			let i = gu(e, W, n);
			return function(e) {
				return xd(i(e), `${t}[${r}]`);
			};
		});
		return function(t) {
			let n = Array(e.length);
			for (let r = 0; r < e.length; ++r) n[r] = e[r](t);
			return n;
		};
	}
	let i = gu(r, Pl, n);
	return function(e) {
		return yd(i(e), t);
	};
}
function ld(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	let i = gu(r, Pl, n);
	return function(e) {
		let n = yd(i(e), t);
		if (n.length !== 2) throw Error(`Expected two numbers for ${t}`);
		return n;
	};
}
function ud(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	let i = gu(r, Pl, n);
	return function(e) {
		return Cd(i(e), t);
	};
}
function dd(e, t, n) {
	let r = nd(e, t);
	if (r === void 0) return null;
	let i = gu(r, Pl | W, n);
	return function(e) {
		return wd(i(e), t);
	};
}
function fd(e, t) {
	let n = nd(e, t);
	if (n !== void 0) {
		if (typeof n != "number") throw Error(`Expected a number for ${t}`);
		return n;
	}
}
function pd(e, t) {
	let n = nd(e, t);
	if (n !== void 0) {
		if (typeof n == "number") return ao(n);
		if (!Array.isArray(n) || n.length !== 2 || typeof n[0] != "number" || typeof n[1] != "number") throw Error(`Expected a number or size array for ${t}`);
		return n;
	}
}
function md(e, t) {
	let n = nd(e, t);
	if (n !== void 0) {
		if (typeof n != "string") throw Error(`Expected a string for ${t}`);
		return n;
	}
}
function hd(e, t) {
	let n = nd(e, t);
	if (n !== void 0) {
		if (n !== "bottom-left" && n !== "bottom-right" && n !== "top-left" && n !== "top-right") throw Error(`Expected bottom-left, bottom-right, top-left, or top-right for ${t}`);
		return n;
	}
}
function gd(e, t) {
	let n = nd(e, t);
	if (n !== void 0) {
		if (n !== "pixels" && n !== "fraction") throw Error(`Expected pixels or fraction for ${t}`);
		return n;
	}
}
function _d(e, t) {
	let n = nd(e, t);
	if (n !== void 0) return yd(n, t);
}
function vd(e, t) {
	let n = nd(e, t);
	if (n !== void 0) {
		if (typeof n != "string") throw Error(`Expected a string for ${t}`);
		if (n !== "declutter" && n !== "obstacle" && n !== "none") throw Error(`Expected declutter, obstacle, or none for ${t}`);
		return n;
	}
}
function yd(e, t) {
	if (!Array.isArray(e)) throw Error(`Expected an array for ${t}`);
	let n = e.length;
	for (let r = 0; r < n; ++r) if (typeof e[r] != "number") throw Error(`Expected an array of numbers for ${t}`);
	return e;
}
function bd(e, t) {
	if (typeof e != "string") throw Error(`Expected a string for ${t}`);
	return e;
}
function xd(e, t) {
	if (typeof e != "number") throw Error(`Expected a number for ${t}`);
	return e;
}
function Sd(e, t) {
	if (typeof e == "string") return e;
	let n = yd(e, t), r = n.length;
	if (r < 3 || r > 4) throw Error(`Expected a color with 3 or 4 values for ${t}`);
	return n;
}
function Cd(e, t) {
	let n = yd(e, t);
	if (n.length !== 2) throw Error(`Expected an array of two numbers for ${t}`);
	return n;
}
function wd(e, t) {
	return typeof e == "number" ? e : Cd(e, t);
}
// node_modules/ol/layer/BaseVector.js
var Td = { RENDER_ORDER: "renderOrder" }, Ed = class extends vs {
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
		return this.get(Td.RENDER_ORDER);
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
		if (!n) return;
		e.declutter ||= {}, n in e.declutter || (e.declutter[n] = new _l(9));
		let r = this.getRenderer();
		r && r.renderDeclutter(e, t);
	}
	setRenderOrder(e) {
		this.set(Td.RENDER_ORDER, e);
	}
	setStyle(e) {
		this.style_ = e === void 0 ? zu : e ?? null;
		let t = Dd(e);
		this.styleFunction_ = e === null || t === null ? void 0 : Lu(t), this.changed();
	}
	setDeclutter(e) {
		this.declutter_ = e ? String(e) : void 0, this.changed();
	}
};
function Dd(e) {
	if (e === void 0) return zu;
	if (!e) return null;
	if (typeof e == "function" || e instanceof Iu) return e;
	if (Array.isArray(e) && e.length === 0) return [];
	if (Array.isArray(e) && e[0] instanceof Iu) {
		let t = e.length, n = Array(t);
		for (let r = 0; r < t; ++r) {
			let t = e[r];
			if (!(t instanceof Iu)) throw Error("Expected a list of style instances");
			n[r] = t;
		}
		return n;
	}
	return Ku(e);
}
// node_modules/ol/layer/Vector.js
var Od = class extends Ed {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new pl(this);
	}
}, kd = class extends E {
	constructor(e, t, n) {
		super(e), this.map = t, this.frameState = n === void 0 ? null : n;
	}
}, Ad = class extends kd {
	constructor(e, t, n, r, i, a) {
		super(e, t, i), this.originalEvent = n, this.pixel_ = null, this.coordinate_ = null, this.dragging = r !== void 0 && r, this.activePointers = a, this.ctrlKey = !1;
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
}, jd = {
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
}, Md = {
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, Nd = class extends D {
	constructor(e, t) {
		super(e), this.map_ = e, this.clickTimeoutId_, this.emulateClicks_ = !1, this.dragging_ = !1, this.dragListenerKeys_ = [], this.moveTolerance_ = t === void 0 ? 1 : t, this.down_ = null;
		let n = this.map_.getViewport();
		if (!n) throw Error("Map viewport is required");
		this.activePointers_ = [], this.trackedTouches_ = {}, this.element_ = n, this.pointerdownListenerKey_ = i(n, Md.POINTERDOWN, this.handlePointerDown_, this), this.originalPointerMoveEvent_, this.relayedListenerKey_ = i(n, Md.POINTERMOVE, this.relayMoveEvent_, this), this.boundHandleTouchMove_ = this.handleTouchMove_.bind(this), this.element_.addEventListener(s.TOUCHMOVE, this.boundHandleTouchMove_, ge ? { passive: !1 } : !1);
	}
	emulateClick_(e) {
		let t = new Ad(jd.CLICK, this.map_, e);
		this.dispatchEvent(t), this.clickTimeoutId_ === void 0 ? this.clickTimeoutId_ = setTimeout(() => {
			this.clickTimeoutId_ = void 0;
			let t = new Ad(jd.SINGLECLICK, this.map_, e);
			this.dispatchEvent(t);
		}, 250) : (clearTimeout(this.clickTimeoutId_), this.clickTimeoutId_ = void 0, t = new Ad(jd.DBLCLICK, this.map_, e), this.dispatchEvent(t));
	}
	updateActivePointers_(e) {
		let t = e, n = t.pointerId;
		if (t.type == jd.POINTERUP || t.type == jd.POINTERCANCEL) {
			delete this.trackedTouches_[n];
			for (let e in this.trackedTouches_) if (this.trackedTouches_[e].target !== t.target) {
				delete this.trackedTouches_[e];
				break;
			}
		} else (t.type == jd.POINTERDOWN || t.type == jd.POINTERMOVE) && (this.trackedTouches_[n] = t);
		this.activePointers_ = Object.values(this.trackedTouches_);
	}
	handlePointerUp_(e) {
		this.updateActivePointers_(e);
		let t = new Ad(jd.POINTERUP, this.map_, e, void 0, void 0, this.activePointers_);
		this.dispatchEvent(t), this.emulateClicks_ && !t.defaultPrevented && !this.dragging_ && this.isMouseActionButton_(e) && this.down_ && this.emulateClick_(this.down_), this.activePointers_.length === 0 && (this.dragListenerKeys_.forEach(o), this.dragListenerKeys_.length = 0, this.dragging_ = !1, this.down_ = null);
	}
	isMouseActionButton_(e) {
		return e.button === 0;
	}
	handlePointerDown_(e) {
		this.emulateClicks_ = this.activePointers_.length === 0, this.updateActivePointers_(e);
		let t = new Ad(jd.POINTERDOWN, this.map_, e, void 0, void 0, this.activePointers_);
		if (this.dispatchEvent(t), this.down_ = new PointerEvent(e.type, e), Object.defineProperty(this.down_, "target", {
			writable: !1,
			value: e.target
		}), this.dragListenerKeys_.length === 0) {
			let e = this.map_.getOwnerDocument(), t = this.element_;
			if (!t) return;
			if (this.dragListenerKeys_.push(i(e, jd.POINTERMOVE, this.handlePointerMove_, this), i(e, jd.POINTERUP, this.handlePointerUp_, this), i(t, jd.POINTERCANCEL, this.handlePointerUp_, this)), t.getRootNode && t.getRootNode() !== e) {
				let e = t.getRootNode();
				this.dragListenerKeys_.push(i(e, jd.POINTERUP, this.handlePointerUp_, this));
			}
		}
	}
	handlePointerMove_(e) {
		if (this.isMoving_(e)) {
			this.updateActivePointers_(e), this.dragging_ = !0;
			let t = new Ad(jd.POINTERDRAG, this.map_, e, this.dragging_, void 0, this.activePointers_);
			this.dispatchEvent(t);
		}
	}
	relayMoveEvent_(e) {
		this.originalPointerMoveEvent_ = e;
		let t = !!(this.down_ && this.isMoving_(e));
		this.dispatchEvent(new Ad(jd.POINTERMOVE, this.map_, e, t));
	}
	handleTouchMove_(e) {
		let t = this.originalPointerMoveEvent_;
		t && !t.defaultPrevented || typeof e.cancelable == "boolean" && e.cancelable !== !0 || e.preventDefault();
	}
	isMoving_(e) {
		let t = this.down_;
		return this.dragging_ || t !== null && (Math.abs(e.clientX - t.clientX) > this.moveTolerance_ || Math.abs(e.clientY - t.clientY) > this.moveTolerance_);
	}
	disposeInternal() {
		let e = this.element_;
		this.relayedListenerKey_ &&= (o(this.relayedListenerKey_), void 0), e && e.removeEventListener(s.TOUCHMOVE, this.boundHandleTouchMove_), this.pointerdownListenerKey_ &&= (o(this.pointerdownListenerKey_), void 0), this.dragListenerKeys_.forEach(o), this.dragListenerKeys_.length = 0, this.element_ = void 0, super.disposeInternal();
	}
}, Pd = {
	LAYERGROUP: "layergroup",
	SIZE: "size",
	TARGET: "target",
	VIEW: "view"
}, Fd = Infinity, Id = class {
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
		L(!(this.keyFunction_(e) in this.queuedElements_), "Tried to enqueue an `element` that was already added to the queue");
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
}, Ld = class extends Id {
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
		if (n === z.LOADED || n === z.ERROR || n === z.EMPTY) {
			n !== z.ERROR && t.removeEventListener(s.CHANGE, this.boundHandleTileChange_);
			let e = t.getKey();
			e in this.tilesLoadingKeys_ && (delete this.tilesLoadingKeys_[e], --this.tilesLoading_), this.tileChangeCallback_();
		}
	}
	loadMoreTiles(e, t) {
		let n = 0;
		for (; this.tilesLoading_ < e && n < t && this.getCount() > 0;) {
			let e = this.dequeue()[0], t = e.getKey();
			e.getState() === z.IDLE && !(t in this.tilesLoadingKeys_) && (this.tilesLoadingKeys_[t] = !0, ++this.tilesLoading_, ++n, e.load());
		}
	}
};
function Rd(e, t, n, r, i) {
	if (!e || !(n in e.wantedTiles) || !e.wantedTiles[n][t.getKey()]) return Fd;
	let a = e.viewState.center, o = r[0] - a[0], s = r[1] - a[1];
	return 65536 * Math.log(i) + Math.sqrt(o * o + s * s) / i;
}
// node_modules/ol/Kinetic.js
var zd = class {
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
}, Bd = { ACTIVE: "active" }, Vd = class extends j {
	constructor(e) {
		super(), this.on, this.once, this.un, e && e.handleEvent && (this.handleEvent = e.handleEvent), this.map_ = null, this.setActive(!0);
	}
	getActive() {
		return this.get(Bd.ACTIVE);
	}
	getMap() {
		return this.map_;
	}
	handleEvent(e) {
		return !0;
	}
	setActive(e) {
		this.set(Bd.ACTIVE, e);
	}
	setMap(e) {
		this.map_ = e;
	}
};
function Hd(e, t, n) {
	let r = e.getCenterInternal();
	if (r) {
		let i = [r[0] + t[0], r[1] + t[1]];
		e.animateInternal({
			duration: n === void 0 ? 250 : n,
			easing: Pe,
			center: e.getConstrainedCenter(i)
		});
	}
}
function Ud(e, t, n, r) {
	let i = e.getZoom();
	if (i === void 0) return;
	let a = e.getConstrainedZoom(i + t);
	if (a === void 0) return;
	let o = e.getResolutionForZoom(a);
	o !== void 0 && (e.getAnimating() && e.cancelAnimations(), e.animate({
		resolution: o,
		anchor: n,
		duration: r === void 0 ? 250 : r,
		easing: Me
	}));
}
// node_modules/ol/interaction/DoubleClickZoom.js
var Wd = class extends Vd {
	constructor(e) {
		super(), e ||= {}, this.delta_ = e.delta ? e.delta : 1, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == jd.DBLCLICK) {
			let n = e.originalEvent, r = e.map, i = e.coordinate, a = n.shiftKey ? -this.delta_ : this.delta_;
			Ud(r.getView(), a, i, this.duration_), n.preventDefault(), t = !0;
		}
		return !t;
	}
};
// node_modules/ol/events/condition.js
function Gd(e) {
	let t = arguments;
	return function(e) {
		let n = !0;
		for (let r = 0, i = t.length; r < i && (n &&= t[r](e), n); ++r);
		return n;
	};
}
var Kd = function(e) {
	let t = e.originalEvent;
	return t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, qd = function(e) {
	let t = e.map.getTargetElement();
	if (!t) return !1;
	let n = t.getRootNode(), r = e.map.getOwnerDocument().activeElement;
	return n instanceof ShadowRoot ? n.host.contains(r) : t.contains(r);
}, Jd = function(e) {
	let t = e.map.getTargetElement();
	if (!t) return !0;
	let n = t.getRootNode();
	return !(n instanceof ShadowRoot ? n.host : t).hasAttribute("tabindex") || qd(e);
}, Yd = x, Xd = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.button == 0 && !(de && fe && t.ctrlKey);
}, Zd = function(e) {
	let t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && !t.shiftKey;
}, Qd = function(e) {
	let t = e.originalEvent;
	return fe ? t.metaKey : t.ctrlKey;
}, $d = function(e) {
	let t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, ef = function(e) {
	let t = e.originalEvent, n = t.target.tagName;
	return n !== "INPUT" && n !== "SELECT" && n !== "TEXTAREA" && !t.target.isContentEditable;
}, tf = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.pointerType == "mouse";
}, nf = function(e) {
	let t = e.originalEvent;
	return "pointerId" in t && t.isPrimary && t.button === 0;
}, rf = class extends Vd {
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
			if (e.type == jd.POINTERDRAG) this.handleDragEvent(e), e.originalEvent.preventDefault();
			else if (e.type == jd.POINTERUP) {
				let t = this.handleUpEvent(e);
				this.handlingDownUpSequence = t && this.targetPointers.length > 0;
			}
		} else if (e.type == jd.POINTERDOWN) {
			let n = this.handleDownEvent(e);
			this.handlingDownUpSequence = n, t = this.stopDown(n);
		} else e.type == jd.POINTERMOVE && this.handleMoveEvent(e);
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
function af(e) {
	let t = e.length, n = 0, r = 0;
	for (let i = 0; i < t; i++) n += e[i].clientX, r += e[i].clientY;
	return {
		clientX: n / t,
		clientY: r / t
	};
}
// node_modules/ol/interaction/DragPan.js
var of = class extends rf {
	constructor(e) {
		super({ stopDown: S }), e ||= {}, this.kinetic_ = e.kinetic, this.lastCentroid = null, this.lastPointersCount_, this.panning_ = !1;
		let t = e.condition ? e.condition : Gd(Zd, nf);
		this.condition_ = e.onFocusOnly ? Gd(Jd, t) : t, this.noKinetic_ = !1;
	}
	handleDragEvent(e) {
		let t = e.map;
		this.panning_ || (this.panning_ = !0, t.getView().beginInteraction());
		let n = this.targetPointers, r = t.getEventPixel(af(n));
		if (n.length == this.lastPointersCount_) {
			if (this.kinetic_ && this.kinetic_.update(r[0], r[1]), this.lastCentroid) {
				let t = [this.lastCentroid[0] - r[0], r[1] - this.lastCentroid[1]], n = e.map.getView(), i = n.getResolution();
				i !== void 0 && (Gt(t, i), Wt(t, n.getRotation()), n.adjustCenterInternal(t));
			}
		} else this.kinetic_ && this.kinetic_.begin();
		this.lastCentroid = r, this.lastPointersCount_ = n.length, e.originalEvent.preventDefault();
	}
	handleUpEvent(e) {
		let t = e.map, n = t.getView();
		if (this.targetPointers.length === 0) {
			if (!this.noKinetic_ && this.kinetic_ && this.kinetic_.end()) {
				let e = this.kinetic_.getDistance(), r = this.kinetic_.getAngle(), i = n.getCenterInternal();
				if (i) {
					let a = t.getPixelFromCoordinateInternal(i), o = t.getCoordinateFromPixelInternal([a[0] - e * Math.cos(r), a[1] - e * Math.sin(r)]);
					n.animateInternal({
						center: n.getConstrainedCenter(o),
						duration: 500,
						easing: Me
					});
				}
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
}, sf = class extends rf {
	constructor(e) {
		e ||= {}, super({ stopDown: S }), this.condition_ = e.condition ? e.condition : Kd, this.lastAngle_ = void 0, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleDragEvent(e) {
		if (!tf(e)) return;
		let t = e.map, n = t.getView();
		if (n.getConstraints().rotation === as) return;
		let r = t.getSize();
		if (!r) return;
		let i = e.pixel, a = Math.atan2(r[1] / 2 - i[1], i[0] - r[0] / 2);
		if (this.lastAngle_ !== void 0) {
			let e = a - this.lastAngle_;
			n.adjustRotationInternal(-e);
		}
		this.lastAngle_ = a;
	}
	handleUpEvent(e) {
		return !tf(e) || (e.map.getView().endInteraction(this.duration_), !1);
	}
	handleDownEvent(e) {
		return tf(e) && Xd(e) && this.condition_(e) ? (e.map.getView().beginInteraction(), this.lastAngle_ = void 0, !0) : !1;
	}
}, cf = class extends c {
	constructor(e) {
		super(), this.geometry_ = null, this.element_ = document.createElement("div"), this.element_.style.position = "absolute", this.element_.style.pointerEvents = "auto", this.element_.className = "ol-box " + e, this.map_ = null, this.startPixel_ = null, this.endPixel_ = null;
	}
	disposeInternal() {
		this.setMap(null);
	}
	render_() {
		let e = this.startPixel_, t = this.endPixel_;
		if (!e || !t) return;
		let n = this.element_.style;
		n.left = Math.min(e[0], t[0]) + "px", n.top = Math.min(e[1], t[1]) + "px", n.width = Math.abs(t[0] - e[0]) + "px", n.height = Math.abs(t[1] - e[1]) + "px";
	}
	setMap(e) {
		if (this.map_) {
			let e = this.map_.getOverlayContainer();
			e && e.removeChild(this.element_);
			let t = this.element_.style;
			t.left = "inherit", t.top = "inherit", t.width = "inherit", t.height = "inherit";
		}
		if (this.map_ = e, this.map_) {
			let e = this.map_.getOverlayContainer();
			e && e.appendChild(this.element_);
		}
	}
	setPixels(e, t) {
		this.startPixel_ = e, this.endPixel_ = t, this.createOrUpdateGeometry(), this.render_();
	}
	createOrUpdateGeometry() {
		if (!this.map_) return;
		let e = this.startPixel_, t = this.endPixel_;
		if (!e || !t) return;
		let n = [
			e,
			[e[0], t[1]],
			t,
			[t[0], e[1]]
		], r = this.map_, i = n.map((e) => r.getCoordinateFromPixelInternal(e)), a = i[0];
		if (!a) return;
		i[4] = a.slice();
		let o = i;
		this.geometry_ ? this.geometry_.setCoordinates([o]) : this.geometry_ = new Qi([o]);
	}
	getGeometry() {
		return this.geometry_;
	}
}, lf = {
	BOXSTART: "boxstart",
	BOXDRAG: "boxdrag",
	BOXEND: "boxend",
	BOXCANCEL: "boxcancel"
}, uf = class extends E {
	constructor(e, t, n) {
		super(e), this.coordinate = t, this.mapBrowserEvent = n;
	}
}, df = class extends rf {
	constructor(e) {
		super(), this.on, this.once, this.un, e ??= {}, this.box_ = new cf(e.className || "ol-dragbox"), this.minArea_ = e.minArea ?? 64, e.onBoxEnd && (this.onBoxEnd = e.onBoxEnd), this.startPixel_ = null, this.condition_ = e.condition ?? Xd, this.boxEndCondition_ = e.boxEndCondition ?? this.defaultBoxEndCondition;
	}
	defaultBoxEndCondition(e, t, n) {
		let r = n[0] - t[0], i = n[1] - t[1];
		return r * r + i * i >= this.minArea_;
	}
	getGeometry() {
		return this.box_.getGeometry();
	}
	handleDragEvent(e) {
		this.startPixel_ && (this.box_.setPixels(this.startPixel_, e.pixel), this.dispatchEvent(new uf(lf.BOXDRAG, e.coordinate, e)));
	}
	handleUpEvent(e) {
		if (!this.startPixel_) return !1;
		let t = this.boxEndCondition_(e, this.startPixel_, e.pixel);
		return t && this.onBoxEnd(e), this.dispatchEvent(new uf(t ? lf.BOXEND : lf.BOXCANCEL, e.coordinate, e)), this.box_.setMap(null), this.startPixel_ = null, !1;
	}
	handleDownEvent(e) {
		return this.condition_(e) ? (this.startPixel_ = e.pixel, this.box_.setMap(e.map), this.box_.setPixels(this.startPixel_, this.startPixel_), this.dispatchEvent(new uf(lf.BOXSTART, e.coordinate, e)), !0) : !1;
	}
	onBoxEnd(e) {}
	setActive(e) {
		e || (this.box_.setMap(null), this.startPixel_ &&= (this.dispatchEvent(new uf(lf.BOXCANCEL, this.startPixel_, null)), null)), super.setActive(e);
	}
	setMap(e) {
		this.getMap() && (this.box_.setMap(null), this.startPixel_ &&= (this.dispatchEvent(new uf(lf.BOXCANCEL, this.startPixel_, null)), null)), super.setMap(e);
	}
}, ff = class extends df {
	constructor(e) {
		e ||= {};
		let t = e.condition ? e.condition : $d;
		super({
			condition: t,
			className: e.className || "ol-dragzoom",
			minArea: e.minArea
		}), this.duration_ = e.duration === void 0 ? 200 : e.duration, this.out_ = e.out !== void 0 && e.out;
	}
	onBoxEnd(e) {
		let t = this.getMap();
		if (!t) return;
		let n = t.getView(), r = this.getGeometry();
		if (this.out_) {
			let e = n.rotatedExtentForGeometry(r), t = n.getResolutionForExtentInternal(e), i = n.getResolution();
			if (t === void 0 || i === void 0) return;
			let a = i / t;
			r = r.clone(), r.scale(a * a);
		}
		n.fitInternal(r, {
			duration: this.duration_,
			easing: Me
		});
	}
}, pf = {
	LEFT: "ArrowLeft",
	UP: "ArrowUp",
	RIGHT: "ArrowRight",
	DOWN: "ArrowDown"
}, mf = class extends Vd {
	constructor(e) {
		super(), e ||= {}, this.defaultCondition_ = function(e) {
			return Zd(e) && ef(e);
		}, this.condition_ = e.condition === void 0 ? this.defaultCondition_ : e.condition, this.duration_ = e.duration === void 0 ? 100 : e.duration, this.pixelDelta_ = e.pixelDelta === void 0 ? 128 : e.pixelDelta;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == s.KEYDOWN) {
			let n = e.originalEvent, r = n.key;
			if (this.condition_(e) && (r == pf.DOWN || r == pf.LEFT || r == pf.RIGHT || r == pf.UP)) {
				let i = e.map.getView(), a = i.getResolution();
				if (a === void 0) return !0;
				let o = a * this.pixelDelta_, s = 0, c = 0;
				r == pf.DOWN ? c = -o : r == pf.LEFT ? s = -o : r == pf.RIGHT ? s = o : c = o;
				let l = [s, c];
				Wt(l, i.getRotation()), Hd(i, l, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}
}, hf = class extends Vd {
	constructor(e) {
		super(), e ||= {}, this.condition_ = e.condition ? e.condition : function(e) {
			return !Qd(e) && ef(e);
		}, this.delta_ = e.delta ? e.delta : 1, this.duration_ = e.duration === void 0 ? 100 : e.duration;
	}
	handleEvent(e) {
		let t = !1;
		if (e.type == s.KEYDOWN || e.type == s.KEYPRESS) {
			let n = e.originalEvent, r = n.key;
			if (this.condition_(e) && (r === "+" || r === "-")) {
				let i = e.map, a = r === "+" ? this.delta_ : -this.delta_;
				Ud(i.getView(), a, void 0, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}
}, gf = 40, _f = 300, vf = 3, yf = class extends Vd {
	constructor(e) {
		e ||= {}, super(e), this.totalDelta_ = 0, this.lastDelta_ = 0, this.maxDelta_ = e.maxDelta === void 0 ? 1 : e.maxDelta, this.duration_ = e.duration === void 0 ? 250 : e.duration, this.timeout_ = e.timeout === void 0 ? 80 : e.timeout, this.useAnchor_ = e.useAnchor === void 0 || e.useAnchor, this.constrainResolution_ = e.constrainResolution !== void 0 && e.constrainResolution;
		let t = e.condition ? e.condition : Yd;
		this.condition_ = e.onFocusOnly ? Gd(Jd, t) : t, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_, this.mode_ = void 0, this.trackpadEventGap_ = 400, this.trackpadTimeoutId_, this.deltaPerZoom_ = 300;
	}
	endInteraction_() {
		this.trackpadTimeoutId_ = void 0;
		let e = this.getMap();
		if (!e) return;
		let t = e.getView(), n = this.lastDelta_ ? this.lastDelta_ > 0 ? 1 : -1 : 0;
		t.endInteraction(this.constrainResolution_ || t.getConstrainResolution() ? 100 : void 0, n, this.lastAnchor_ ? e.getCoordinateFromPixel(this.lastAnchor_) : void 0);
	}
	handleEvent(e) {
		if (!this.condition_(e) || e.type !== s.WHEEL) return !0;
		let t = e.map, n = e.originalEvent;
		n.preventDefault();
		let r = n.ctrlKey && !e.ctrlKey;
		this.useAnchor_ && (this.lastAnchor_ = e.pixel);
		let i = n.deltaY;
		switch (n.deltaMode) {
			case WheelEvent.DOM_DELTA_LINE:
				i *= gf;
				break;
			case WheelEvent.DOM_DELTA_PAGE: i *= _f;
		}
		if (i === 0) return !1;
		this.lastDelta_ = i;
		let a = Date.now();
		this.startTime_ === void 0 && (this.startTime_ = a), (!this.mode_ || a - this.startTime_ > this.trackpadEventGap_) && (this.mode_ = Math.abs(i) < 4 ? "trackpad" : "wheel");
		let o = t.getView();
		if (this.mode_ === "trackpad") return this.trackpadTimeoutId_ ? clearTimeout(this.trackpadTimeoutId_) : (o.getAnimating() && o.cancelAnimations(), o.beginInteraction()), this.trackpadTimeoutId_ = setTimeout(this.endInteraction_.bind(this), this.timeout_), r && (i *= vf), o.adjustZoom(-i / this.deltaPerZoom_, this.lastAnchor_ ? t.getCoordinateFromPixel(this.lastAnchor_) : void 0), this.startTime_ = a, !1;
		this.totalDelta_ += i;
		let c = Math.max(this.timeout_ - (a - this.startTime_), 0);
		return clearTimeout(this.timeoutId_), this.timeoutId_ = setTimeout(this.handleWheelZoom_.bind(this, t), c), !1;
	}
	handleWheelZoom_(e) {
		let t = e.getView();
		t.getAnimating() && t.cancelAnimations();
		let n = -kt(this.totalDelta_, -this.maxDelta_ * this.deltaPerZoom_, this.maxDelta_ * this.deltaPerZoom_) / this.deltaPerZoom_;
		(t.getConstrainResolution() || this.constrainResolution_) && (n = n ? n > 0 ? 1 : -1 : 0), Ud(t, n, this.lastAnchor_ ? e.getCoordinateFromPixel(this.lastAnchor_) : void 0, this.duration_), this.mode_ = void 0, this.totalDelta_ = 0, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_ = void 0;
	}
	setMouseAnchor(e) {
		this.useAnchor_ = e, e || (this.lastAnchor_ = null);
	}
}, bf = class extends rf {
	constructor(e) {
		e ||= {};
		let t = e;
		t.stopDown ||= S, super(t), this.anchor_ = null, this.lastAngle_ = void 0, this.rotating_ = !1, this.rotationDelta_ = 0, this.threshold_ = e.threshold === void 0 ? .3 : e.threshold, this.duration_ = e.duration === void 0 ? 250 : e.duration;
	}
	handleDragEvent(e) {
		let t = 0, n = this.targetPointers[0], r = this.targetPointers[1], i = Math.atan2(r.clientY - n.clientY, r.clientX - n.clientX);
		if (this.lastAngle_ !== void 0) {
			let e = i - this.lastAngle_;
			this.rotationDelta_ += e, !this.rotating_ && Math.abs(this.rotationDelta_) > this.threshold_ && (this.rotating_ = !0), t = e;
		}
		this.lastAngle_ = i;
		let a = e.map, o = a.getView();
		o.getConstraints().rotation !== as && (this.anchor_ = a.getCoordinateFromPixelInternal(a.getEventPixel(af(this.targetPointers))) ?? null, this.rotating_ && (a.render(), o.adjustRotationInternal(t, this.anchor_ ?? void 0)));
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
}, xf = class extends rf {
	constructor(e) {
		e ||= {};
		let t = e;
		t.stopDown ||= S, super(t), this.anchor_ = null, this.duration_ = e.duration === void 0 ? 400 : e.duration, this.lastDistance_ = void 0, this.lastScaleDelta_ = 1;
	}
	handleDragEvent(e) {
		let t = 1, n = this.targetPointers[0], r = this.targetPointers[1], i = n.clientX - r.clientX, a = n.clientY - r.clientY, o = Math.sqrt(i * i + a * a);
		this.lastDistance_ !== void 0 && (t = this.lastDistance_ / o), this.lastDistance_ = o;
		let s = e.map, c = s.getView();
		t != 1 && (this.lastScaleDelta_ = t), this.anchor_ = s.getCoordinateFromPixelInternal(s.getEventPixel(af(this.targetPointers))) ?? null, s.render(), c.adjustResolutionInternal(t, this.anchor_ ?? void 0);
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
// node_modules/ol/interaction/defaults.js
function Sf(e) {
	e ||= {};
	let t = new P(), n = new zd(-.005, .05, 100);
	return (e.altShiftDragRotate === void 0 || e.altShiftDragRotate) && t.push(new sf()), (e.doubleClickZoom === void 0 || e.doubleClickZoom) && t.push(new Wd({
		delta: e.zoomDelta,
		duration: e.zoomDuration
	})), (e.dragPan === void 0 || e.dragPan) && t.push(new of({
		onFocusOnly: e.onFocusOnly,
		kinetic: n
	})), (e.pinchRotate === void 0 || e.pinchRotate) && t.push(new bf()), (e.pinchZoom === void 0 || e.pinchZoom) && t.push(new xf({ duration: e.zoomDuration })), (e.keyboard === void 0 || e.keyboard) && (t.push(new mf()), t.push(new hf({
		delta: e.zoomDelta,
		duration: e.zoomDuration
	}))), (e.mouseWheelZoom === void 0 || e.mouseWheelZoom) && t.push(new yf({
		onFocusOnly: e.onFocusOnly,
		duration: e.zoomDuration
	})), (e.shiftDragZoom === void 0 || e.shiftDragZoom) && t.push(new ff({ duration: e.zoomDuration })), t;
}
// node_modules/ol/layer/Group.js
var Cf = {
	ADDLAYER: "addlayer",
	REMOVELAYER: "removelayer"
}, wf = class extends E {
	constructor(e, t) {
		super(e), this.layer = t;
	}
}, Tf = { LAYERS: "layers" }, Ef = class r extends _s {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.layers;
		let n = e.layers;
		super(t), this.on, this.once, this.un, this.layersListenerKeys_ = [], this.listenerKeys_ = {}, this.addChangeListener(Tf.LAYERS, this.handleLayersChanged_), n ? Array.isArray(n) ? n = new P(n.slice(), { unique: !0 }) : L(typeof n.getArray == "function", "Expected `layers` to be an array or a `Collection`") : n = new P(void 0, { unique: !0 }), this.setLayers(n);
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
			this.registerLayerListeners_(t), this.dispatchEvent(new wf(Cf.ADDLAYER, t));
		}
		this.changed();
	}
	registerLayerListeners_(e) {
		let n = [i(e, t.PROPERTYCHANGE, this.handleLayerChange_, this), i(e, s.CHANGE, this.handleLayerChange_, this)];
		e instanceof r && n.push(i(e, Cf.ADDLAYER, this.handleLayerGroupAdd_, this), i(e, Cf.REMOVELAYER, this.handleLayerGroupRemove_, this)), this.listenerKeys_[y(e)] = n;
	}
	handleLayerGroupAdd_(e) {
		this.dispatchEvent(new wf(Cf.ADDLAYER, e.layer));
	}
	handleLayerGroupRemove_(e) {
		this.dispatchEvent(new wf(Cf.REMOVELAYER, e.layer));
	}
	handleLayersAdd_(e) {
		let t = e.element;
		this.registerLayerListeners_(t), this.dispatchEvent(new wf(Cf.ADDLAYER, t)), this.changed();
	}
	handleLayersRemove_(e) {
		let t = e.element, n = y(t);
		this.listenerKeys_[n].forEach(o), delete this.listenerKeys_[n], this.dispatchEvent(new wf(Cf.REMOVELAYER, t)), this.changed();
	}
	getLayers() {
		return this.get(Tf.LAYERS);
	}
	setLayers(e) {
		let t = this.getLayers();
		if (t) {
			let e = t.getArray();
			for (let t = 0, n = e.length; t < n; ++t) this.dispatchEvent(new wf(Cf.REMOVELAYER, e[t]));
		}
		this.set(Tf.LAYERS, e);
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
			n.opacity *= r.opacity, n.visible = n.visible && r.visible, n.maxResolution = Math.min(n.maxResolution, r.maxResolution), n.minResolution = Math.max(n.minResolution, r.minResolution), n.minZoom = Math.max(n.minZoom, r.minZoom), n.maxZoom = Math.min(n.maxZoom, r.maxZoom), r.extent !== void 0 && (n.extent = n.extent === void 0 ? r.extent : pt(n.extent, r.extent)), n.zIndex === void 0 && (n.zIndex = i);
		}
		return t;
	}
	getSourceState() {
		return "ready";
	}
}, Df = class extends c {
	constructor(e) {
		super(), this.map_ = e;
	}
	dispatchRenderEvent(e, t) {
		_();
	}
	calculateMatrices2D(e) {
		let t = e.viewState, n = e.coordinateToPixelTransform, r = e.pixelToCoordinateTransform;
		Fr(n, e.size[0] / 2, e.size[1] / 2, 1 / t.resolution, -1 / t.resolution, -t.rotation, -t.center[0], -t.center[1]), Ir(r, n);
	}
	forEachFeatureAtCoordinate(e, t, n, r, i, a, o, s) {
		let c, l = t.viewState;
		function u(e, t, n, r) {
			return i.call(a, t, e ? n : null, r);
		}
		let d = l.projection, f = Kt(e.slice(), d), p = [[0, 0]];
		if (d.canWrapX() && r) {
			let e = I(d.getExtent());
			p.push([-e, 0], [e, 0]);
		}
		let m = t.layerStatesArray, h = m.length, g = [], _ = [];
		for (let r = 0; r < p.length; r++) for (let i = h - 1; i >= 0; --i) {
			let a = m[i], d = a.layer;
			if (d.hasRenderer() && ys(a, l) && o.call(s, d)) {
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
		return this.forEachFeatureAtCoordinate(e, t, n, r, x, this, i, a) !== void 0;
	}
	getMap() {
		return this.map_;
	}
	renderFrame(e) {
		_();
	}
	scheduleExpireIconCache(e) {
		Ts.canExpireCache() && e.postRenderFunctions.push(Of);
	}
};
function Of(e, t) {
	Ts.expire();
}
// node_modules/ol/renderer/Composite.js
var kf = class extends Df {
	constructor(e) {
		super(e), this.fontChangeListenerKey_ = i(Vs, t.PROPERTYCHANGE, e.redrawText, e), this.element_ = me ? Ee() : document.createElement("div");
		let n = this.element_.style;
		n.position = "absolute", n.width = "100%", n.height = "100%", n.zIndex = "0", this.element_.className = te + " ol-layers";
		let r = e.getViewport();
		r && r.insertBefore(this.element_, r.firstChild || null), this.children_ = [], this.renderedVisible_ = !0;
	}
	dispatchRenderEvent(e, t) {
		let n = this.getMap();
		if (n.hasListener(e)) {
			let r = new Ro(e, void 0, t);
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
		this.calculateMatrices2D(e), this.dispatchRenderEvent(zo.PRECOMPOSE, e);
		let t = e.layerStatesArray.sort((e, t) => e.zIndex - t.zIndex);
		t.some((e) => e.layer instanceof Ed && e.layer.getDeclutter()) && (e.declutter = {});
		let n = e.viewState;
		this.children_.length = 0;
		let r = this.getMap().getTargetElement(), i;
		De(r) && (i = r.getContext("2d"), i.setTransform(1, 0, 0, 1, 0, 0), i.clearRect(0, 0, r.width, r.height));
		let a = [], o = i ? r : null;
		for (let r = 0, i = t.length; r < i; ++r) {
			let i = t[r];
			e.layerIndex = r;
			let s = i.layer, c = s.getSourceState();
			if (!ys(i, n) || c != "ready" && c != "undefined") {
				s.unrender();
				continue;
			}
			let l = s.render(e, o);
			l && (l !== o && (this.children_.push(l), o = l), a.push(i));
		}
		if (this.declutter(e, a), Te(this.element_, this.children_), i) for (let e of this.children_) {
			let t = e.firstElementChild || e, n = e.style.backgroundColor;
			if (n && (!De(t) || t.width > 0) && (i.fillStyle = n, i.fillRect(0, 0, i.canvas.width, i.canvas.height)), !De(t) || t.width === 0) continue;
			i.save();
			let r = e.style.opacity || t.style.opacity;
			i.globalAlpha = r === "" ? 1 : Number(r);
			let a = t.style.transform;
			if (a) i.transform(...Br(a));
			else {
				let e = parseFloat(t.style.width) / t.width, n = parseFloat(t.style.height) / t.height;
				i.transform(e, 0, 0, n, 0, 0);
			}
			i.drawImage(t, 0, 0), i.restore();
		}
		this.dispatchRenderEvent(zo.POSTCOMPOSE, e), this.renderedVisible_ ||= (this.element_.style.display = "", !0), this.scheduleExpireIconCache(e);
	}
	declutter(e, t) {
		if (e.declutter) {
			for (let n = t.length - 1; n >= 0; --n) {
				let r = t[n], i = r.layer;
				i.getDeclutter() && i.renderDeclutter(e, r);
			}
			t.forEach((t) => t.layer?.renderDeferred(e));
		}
	}
};
// node_modules/ol/Map.js
function Af(e) {
	e instanceof vs ? e.setMapInternal(null) : e instanceof Ef && e.getLayers().forEach(Af);
}
function jf(e, t) {
	if (e instanceof vs) e.setMapInternal(t);
	else if (e instanceof Ef) {
		let n = e.getLayers().getArray();
		for (let e = 0, r = n.length; e < r; ++e) jf(n[e], t);
	}
}
var Mf = class extends j {
	constructor(t) {
		super(), t ||= {}, this.on, this.once, this.un;
		let n = Nf(t);
		this.renderComplete_ = !1, this.loaded_ = !0, this.boundHandleBrowserEvent_ = this.handleBrowserEvent.bind(this), this.ctrlKey_ = !1, this.boundHandleCtrlKey_ = this.handleCtrlKey_.bind(this), this.maxTilesLoading_ = t.maxTilesLoading === void 0 ? 16 : t.maxTilesLoading, this.pixelRatio_ = t.pixelRatio === void 0 ? pe : t.pixelRatio, this.postRenderTimeoutHandle_, this.animationDelayKey_, this.animationDelay_ = this.animationDelay_.bind(this), this.coordinateToPixelTransform_ = Or(), this.pixelToCoordinateTransform_ = Or(), this.frameIndex_ = 0, this.frameState_ = null, this.previousExtent_ = null, this.viewPropertyListenerKey_ = null, this.viewChangeListenerKey_ = null, this.layerGroupPropertyListenerKeys_ = null, this.viewport_, this.overlayContainer_, this.overlayContainerStopEvent_, me || (this.viewport_ = document.createElement("div"), this.viewport_.className = "ol-viewport" + ("ontouchstart" in window ? " ol-touch" : ""), this.viewport_.style.position = "relative", this.viewport_.style.overflow = "hidden", this.viewport_.style.width = "100%", this.viewport_.style.height = "100%", this.overlayContainer_ = document.createElement("div"), this.overlayContainer_.style.position = "absolute", this.overlayContainer_.style.zIndex = "0", this.overlayContainer_.style.width = "100%", this.overlayContainer_.style.height = "100%", this.overlayContainer_.style.pointerEvents = "none", this.overlayContainer_.className = "ol-overlaycontainer", this.viewport_.appendChild(this.overlayContainer_), this.overlayContainerStopEvent_ = document.createElement("div"), this.overlayContainerStopEvent_.style.position = "absolute", this.overlayContainerStopEvent_.style.zIndex = "0", this.overlayContainerStopEvent_.style.width = "100%", this.overlayContainerStopEvent_.style.height = "100%", this.overlayContainerStopEvent_.style.pointerEvents = "none", this.overlayContainerStopEvent_.className = "ol-overlaycontainer-stopevent", this.viewport_.appendChild(this.overlayContainerStopEvent_)), this.mapBrowserEventHandler_ = null, this.moveTolerance_ = t.moveTolerance, this.keyboardEventTarget_ = n.keyboardEventTarget, this.targetChangeHandlerKeys_ = null, this.targetElement_ = null, me || (this.resizeObserver_ = new ResizeObserver(() => this.updateSize())), this.controls = n.controls || (me ? new P() : Le()), this.interactions = n.interactions || (me ? new P() : Sf({ onFocusOnly: !0 })), this.overlays_ = n.overlays, this.overlayIdIndex_ = {}, this.renderer_ = null, this.postRenderFunctions_ = [], this.tileQueue_ = new Ld(this.getTilePriority.bind(this), this.handleTileChange_.bind(this)), this.addChangeListener(Pd.LAYERGROUP, this.handleLayerGroupChanged_), this.addChangeListener(Pd.VIEW, this.handleViewChanged_), this.addChangeListener(Pd.SIZE, this.handleSizeChanged_), this.addChangeListener(Pd.TARGET, this.handleTargetChanged_), this.setProperties(n.values);
		let r = this;
		t.view && !(t.view instanceof us) && t.view.then(function(e) {
			r.setView(new us(e));
		}), this.controls.addEventListener(e.ADD, ((e) => {
			e.element.setMap(this);
		})), this.controls.addEventListener(e.REMOVE, ((e) => {
			e.element.setMap(null);
		})), this.interactions.addEventListener(e.ADD, ((e) => {
			e.element.setMap(this);
		})), this.interactions.addEventListener(e.REMOVE, ((e) => {
			e.element.setMap(null);
		})), this.overlays_.addEventListener(e.ADD, ((e) => {
			this.addOverlayInternal_(e.element);
		})), this.overlays_.addEventListener(e.REMOVE, ((e) => {
			let t = e.element.getId();
			t !== void 0 && delete this.overlayIdIndex_[t.toString()], e.element.setMap(null);
		})), this.controls.forEach((e) => {
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
		jf(e.layer, this);
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
		let i = n.hitTolerance === void 0 ? 0 : n.hitTolerance, a = n.layerFilter === void 0 ? x : n.layerFilter, o = n.checkWrapped !== !1;
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
				n instanceof Ef ? t(n.getLayers()) : e.push(n);
			});
		}
		return t(this.getLayers()), e;
	}
	hasFeatureAtPixel(e, t) {
		if (!this.frameState_ || !this.renderer_) return !1;
		let n = this.getCoordinateFromPixelInternal(e);
		t = t === void 0 ? {} : t;
		let r = t.layerFilter === void 0 ? x : t.layerFilter, i = t.hitTolerance === void 0 ? 0 : t.hitTolerance, a = t.checkWrapped !== !1;
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
		return this.get(Pd.TARGET);
	}
	getTargetElement() {
		return this.targetElement_;
	}
	getCoordinateFromPixel(e, t) {
		let n = br(this.getCoordinateFromPixelInternal(e), this.getView().getProjection());
		return t?.wrapX && n && Kt(n, yr() || this.getView().getProjection()), n;
	}
	getCoordinateFromPixelInternal(e) {
		let t = this.frameState_;
		return t ? R(t.pixelToCoordinateTransform, e.slice()) : null;
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
		return this.get(Pd.LAYERGROUP);
	}
	setLayers(e) {
		let t = this.getLayerGroup();
		if (e instanceof P) {
			t.setLayers(e);
			return;
		}
		let n = t.getLayers();
		n.clear(), n.extend(e);
	}
	getLayers() {
		return this.getLayerGroup().getLayers();
	}
	updateRenderComplete_() {
		this.renderComplete_ = (this.hasListener(Oe.LOADSTART) || this.hasListener(Oe.LOADEND) || this.hasListener(zo.RENDERCOMPLETE)) && !this.tileQueue_.getTilesLoading() && !this.tileQueue_.getCount() && !this.getLoadingOrNotReady();
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
	getPixelFromCoordinate(e, t) {
		let n = this.getView().getProjection(), r = xr(e, n);
		return t?.wrapX && this.frameState_ && (r = qt(r.slice(), n, this.frameState_.viewState.center[0])), this.getPixelFromCoordinateInternal(r);
	}
	getPixelFromCoordinateInternal(e) {
		let t = this.frameState_;
		return t ? R(t.coordinateToPixelTransform, e.slice(0, 2)) : null;
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
		return this.get(Pd.SIZE);
	}
	getView() {
		return this.get(Pd.VIEW);
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
		return Rd(this.frameState_, e, t, n, r);
	}
	handleBrowserEvent(e, t) {
		t ||= e.type;
		let n = new Ad(t, this, e);
		this.handleMapBrowserEvent(n);
	}
	handleCtrlKey_(e) {
		let t = e;
		this.ctrlKey_ = t.key === "Control" ? t.type === s.KEYDOWN : t.ctrlKey;
	}
	handleMapBrowserEvent(e) {
		if (e.originalEvent.ctrlKey === !1 && (this.ctrlKey_ = !1), e.ctrlKey = this.ctrlKey_, !this.frameState_) return;
		let t = e.originalEvent, n = t.type;
		if (n === Md.POINTERDOWN || n === s.WHEEL || n === s.KEYDOWN) {
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
			let n = this.maxTilesLoading_, r = n, i = e ? e.viewHints : void 0;
			if (i && (i[V.ANIMATING] || i[V.INTERACTING])) {
				let t = Date.now() - e.time > 8;
				n = t ? 0 : 8, r = t ? 0 : 2;
			}
			if (t.getTilesLoading() < n) {
				let e = t.getCount();
				t.reprioritize(), t.getCount() < e && this.updateRenderComplete_(), t.loadMoreTiles(n, r);
			}
		}
		e && this.renderer_ && !e.animate && (this.renderComplete_ ? (this.hasListener(zo.RENDERCOMPLETE) && this.renderer_.dispatchRenderEvent(zo.RENDERCOMPLETE, e), this.loaded_ === !1 && (this.loaded_ = !0, this.dispatchEvent(new kd(Oe.LOADEND, this, e)))) : this.loaded_ === !0 && (this.loaded_ = !1, this.dispatchEvent(new kd(Oe.LOADSTART, this, e))));
		let n = this.postRenderFunctions_;
		if (e) for (let t = 0, r = n.length; t < r; ++t) n[t](this, e);
		n.length = 0;
	}
	handleSizeChanged_() {
		this.getView() && !this.getView().getAnimating() && this.getView().resolveConstraints(0), this.render();
	}
	handleTargetChanged_() {
		if (this.mapBrowserEventHandler_) {
			let e = this.targetChangeHandlerKeys_;
			for (let t = 0, n = e.length; t < n; ++t) o(e[t]);
			this.targetChangeHandlerKeys_ = null;
			let t = this.getOwnerDocument();
			t.removeEventListener(s.KEYDOWN, this.boundHandleCtrlKey_, !0), t.removeEventListener(s.KEYUP, this.boundHandleCtrlKey_, !0), this.ctrlKey_ = !1, this.viewport_.removeEventListener(s.CONTEXTMENU, this.boundHandleBrowserEvent_), this.viewport_.removeEventListener(s.WHEEL, this.boundHandleBrowserEvent_), this.mapBrowserEventHandler_.dispose(), this.mapBrowserEventHandler_ = null, this.viewport_.remove();
		}
		if (this.targetElement_ && !De(this.targetElement_)) {
			this.resizeObserver_?.unobserve(this.targetElement_);
			let e = this.targetElement_.getRootNode();
			e instanceof ShadowRoot && this.resizeObserver_.unobserve(e.host), this.setSize(void 0);
		}
		let e = this.getTarget(), t = typeof e == "string" ? document.getElementById(e) : e;
		if (this.targetElement_ = t, !t) this.renderer_ &&= (clearTimeout(this.postRenderTimeoutHandle_), this.postRenderTimeoutHandle_ = void 0, this.postRenderFunctions_.length = 0, this.renderer_.dispose(), null), this.animationDelayKey_ &&= (cancelAnimationFrame(this.animationDelayKey_), void 0);
		else {
			if (De(t) || t.appendChild(this.viewport_), this.renderer_ ||= new kf(this), !De(t)) {
				this.mapBrowserEventHandler_ = new Nd(this, this.moveTolerance_);
				for (let e in jd) this.mapBrowserEventHandler_.addEventListener(jd[e], this.handleMapBrowserEvent.bind(this));
				this.viewport_.addEventListener(s.CONTEXTMENU, this.boundHandleBrowserEvent_, !1), this.viewport_.addEventListener(s.WHEEL, this.boundHandleBrowserEvent_, ge ? { passive: !1 } : !1);
				let e = this.getOwnerDocument();
				e.addEventListener(s.KEYDOWN, this.boundHandleCtrlKey_, !0), e.addEventListener(s.KEYUP, this.boundHandleCtrlKey_, !0);
				let n;
				if (this.keyboardEventTarget_) n = this.keyboardEventTarget_;
				else {
					let e = t.getRootNode();
					n = e instanceof ShadowRoot ? e.host : t;
				}
				if (this.targetChangeHandlerKeys_ = [i(n, s.KEYDOWN, this.handleBrowserEvent, this), i(n, s.KEYPRESS, this.handleBrowserEvent, this)], !De(t)) {
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
		e && (this.handleLayerAdd_(new wf("addlayer", e)), this.layerGroupPropertyListenerKeys_ = [
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
		Af(e.layer);
	}
	removeOverlay(e) {
		return this.getOverlays().remove(e);
	}
	renderFrame_(e) {
		let t = this.getSize(), n = this.getView(), r = this.frameState_, i = null;
		if (t !== void 0 && ro(t) && n && n.isDef()) {
			let r = n.getHints(this.frameState_ ? this.frameState_.viewHints : void 0), a = n.getState();
			if (i = {
				animate: !1,
				coordinateToPixelTransform: this.coordinateToPixelTransform_,
				declutter: null,
				extent: ut(a.center, a.resolution, a.rotation, t),
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
				mapId: y(this),
				renderTargets: {}
			}, a.nextCenter && a.nextResolution) {
				let e = isNaN(a.nextRotation) ? a.rotation : a.nextRotation;
				i.nextExtent = ut(a.nextCenter, a.nextResolution, e, t);
			}
		}
		this.frameState_ = i, this.renderer_.renderFrame(i), i && (i.animate && this.render(), Array.prototype.push.apply(this.postRenderFunctions_, i.postRenderFunctions), r && (this.previousExtent_ && (vt(this.previousExtent_) || $e(i.extent, this.previousExtent_)) || (this.dispatchEvent(new kd(Oe.MOVESTART, this, r)), this.previousExtent_ = Xe(this.previousExtent_))), this.previousExtent_ && !i.viewHints[V.ANIMATING] && !i.viewHints[V.INTERACTING] && !$e(i.extent, this.previousExtent_) && (this.dispatchEvent(new kd(Oe.MOVEEND, this, i)), He(i.extent, this.previousExtent_))), this.dispatchEvent(new kd(Oe.POSTRENDER, this, i)), this.updateRenderComplete_(), this.postRenderTimeoutHandle_ ||= setTimeout(() => {
			this.postRenderTimeoutHandle_ = void 0, this.handlePostRender();
		}, 0);
	}
	setLayerGroup(e) {
		let t = this.getLayerGroup();
		t && this.handleLayerRemove_(new wf("removelayer", t)), this.set(Pd.LAYERGROUP, e);
	}
	setSize(e) {
		this.set(Pd.SIZE, e);
	}
	setTarget(e) {
		this.set(Pd.TARGET, e);
	}
	setView(e) {
		if (!e || e instanceof us) {
			this.set(Pd.VIEW, e);
			return;
		}
		this.set(Pd.VIEW, new us());
		let t = this;
		e.then(function(e) {
			t.setView(new us(e));
		});
	}
	updateSize() {
		let e = this.getTargetElement(), t;
		if (e) {
			let n, r;
			if (De(e)) {
				let t = e.getContext("2d").getTransform();
				n = e.width / t.a, r = e.height / t.d;
			} else {
				let t = getComputedStyle(e);
				n = e.offsetWidth - parseFloat(t.borderLeftWidth) - parseFloat(t.paddingLeft) - parseFloat(t.paddingRight) - parseFloat(t.borderRightWidth), r = e.offsetHeight - parseFloat(t.borderTopWidth) - parseFloat(t.paddingTop) - parseFloat(t.paddingBottom) - parseFloat(t.borderBottomWidth);
			}
			!isNaN(n) && !isNaN(r) && (t = [Math.max(0, n), Math.max(0, r)], !ro(t) && (e.offsetWidth || e.offsetHeight || e.getClientRects().length) && Ot("No map visible because the map container's width or height are 0."));
		}
		let n = this.getSize();
		!t || n && h(t, n) || (this.updateViewportSize_(t), this.setSize(t));
	}
	updateViewportSize_(e) {
		let t = this.getView();
		t && t.setViewportSize(e);
	}
};
function Nf(e) {
	let t = null;
	e.keyboardEventTarget !== void 0 && (t = typeof e.keyboardEventTarget == "string" ? document.getElementById(e.keyboardEventTarget) : e.keyboardEventTarget);
	let n = {}, r = e.layers && typeof e.layers.getLayers == "function" ? e.layers : new Ef({ layers: e.layers });
	n[Pd.LAYERGROUP] = r, n[Pd.TARGET] = e.target, n[Pd.VIEW] = e.view instanceof us ? e.view : new us();
	let i;
	e.controls !== void 0 && (Array.isArray(e.controls) ? i = new P(e.controls.slice()) : (L(typeof e.controls.getArray == "function", "Expected `controls` to be an array or an `ol/Collection.js`"), i = e.controls));
	let a;
	e.interactions !== void 0 && (Array.isArray(e.interactions) ? a = new P(e.interactions.slice()) : (L(typeof e.interactions.getArray == "function", "Expected `interactions` to be an array or an `ol/Collection.js`"), a = e.interactions));
	let o;
	return e.overlays === void 0 ? o = new P() : Array.isArray(e.overlays) ? o = new P(e.overlays.slice()) : (L(typeof e.overlays.getArray == "function", "Expected `overlays` to be an array or an `ol/Collection.js`"), o = e.overlays), {
		controls: i,
		interactions: a,
		keyboardEventTarget: t,
		overlays: o,
		values: n
	};
}
// node_modules/ol/featureloader.js
var Pf = !1;
function Ff(e, t, n, r, i, a, o) {
	let s = new XMLHttpRequest();
	s.open("GET", typeof e == "function" ? e(n, r, i) : e, !0), t.getType() == "arraybuffer" && (s.responseType = "arraybuffer"), s.withCredentials = Pf, s.onload = function(e) {
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
function If(e, t) {
	return function(n, r, i, a, o) {
		Ff(e, t, n, r, i, (e, t) => {
			this.addFeatures(e), a !== void 0 && a(e);
		}, () => {
			this.changed(), o !== void 0 && o();
		});
	};
}
// node_modules/ol/loadingstrategy.js
function Lf(e, t) {
	return [[
		-Infinity,
		-Infinity,
		Infinity,
		Infinity
	]];
}
// node_modules/ol/structs/RBush.js
var Rf = class {
	constructor(e) {
		this.rbush_ = new _l(e), this.items_ = {};
	}
	insert(e, t) {
		let n = {
			minX: e[0],
			minY: e[1],
			maxX: e[2],
			maxY: e[3],
			value: t
		};
		this.rbush_.insert(n), this.items_[y(t)] = n;
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
			n[r] = o, this.items_[y(a)] = o;
		}
		this.rbush_.load(n);
	}
	remove(e) {
		let t = y(e), n = this.items_[t];
		return delete this.items_[t], this.rbush_.remove(n) !== null;
	}
	update(e, t) {
		let n = this.items_[y(t)];
		$e([
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
		return Ye(t.minX, t.minY, t.maxX, t.maxY, e);
	}
	concat(e) {
		this.rbush_.load(e.rbush_.all());
		for (let t in e.items_) this.items_[t] = e.items_[t];
	}
}, zf = class extends j {
	constructor(e) {
		super(), this.projection = or(e.projection), this.attributions_ = Bf(e.attributions), this.attributionsCollapsible_ = e.attributionsCollapsible ?? !0, this.loading = !1, this.canRotate = !1, this.state_ = e.state === void 0 ? "ready" : e.state, this.wrapX_ = e.wrapX !== void 0 && e.wrapX, this.interpolate_ = !!e.interpolate, this.viewResolver = null, this.viewRejector = null;
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
		this.attributions_ = Bf(e), this.changed();
	}
	setState(e) {
		this.state_ = e, this.changed();
	}
};
function Bf(e) {
	return e ? typeof e == "function" ? e : (Array.isArray(e) || (e = [e]), (t) => e) : null;
}
// node_modules/ol/source/VectorEventType.js
var Vf = {
	ADDFEATURE: "addfeature",
	CHANGEFEATURE: "changefeature",
	CLEAR: "clear",
	REMOVEFEATURE: "removefeature",
	FEATURESLOADSTART: "featuresloadstart",
	FEATURESLOADEND: "featuresloadend",
	FEATURESLOADERROR: "featuresloaderror"
}, Hf = class extends E {
	constructor(e, t, n) {
		super(e), this.feature = t, this.features = n;
	}
}, Uf = class extends zf {
	constructor(e) {
		e ||= {}, super({
			attributions: e.attributions,
			interpolate: !0,
			projection: void 0,
			state: "ready",
			wrapX: e.wrapX === void 0 || e.wrapX
		}), this.canRotate = !0, this.on, this.once, this.un, this.loader_ = C, this.format_ = e.format || null, this.overlaps_ = e.overlaps === void 0 || e.overlaps, this.url_ = e.url, e.loader === void 0 ? this.url_ !== void 0 && (L(this.format_, "`format` must be set when `url` is set"), this.loader_ = If(this.url_, this.format_)) : this.loader_ = e.loader, this.strategy_ = e.strategy === void 0 ? Lf : e.strategy;
		let t = e.useSpatialIndex === void 0 || e.useSpatialIndex;
		this.featuresRtree_ = t ? new Rf() : null, this.loadedExtentsRtree_ = new Rf(), this.nullGeometryFeatures_ = {}, this.idIndex_ = {}, this.uidIndex_ = {}, this.featureChangeKeys_ = {}, this.featuresCollection_ = null;
		let n, r;
		Array.isArray(e.features) ? r = e.features : e.features && (n = e.features, r = n.getArray()), !t && n === void 0 && (n = new P(r)), r !== void 0 && this.addFeaturesInternal(r), n !== void 0 && this.bindFeaturesCollection_(n);
	}
	addFeature(e) {
		this.addFeatureInternal(e), this.changed();
	}
	addFeatureInternal(e) {
		let t = y(e);
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
		this.dispatchEvent(new Hf(Vf.ADDFEATURE, e));
	}
	setupChangeEvents_(e, n) {
		n instanceof ra || (this.featureChangeKeys_[e] = [i(n, s.CHANGE, this.handleFeatureChange_, this), i(n, t.PROPERTYCHANGE, this.handleFeatureChange_, this)]);
	}
	addToIndex_(e, t) {
		let n = !0;
		if (t.getId() !== void 0) {
			let e = String(t.getId());
			if (!(e in this.idIndex_)) this.idIndex_[e] = t;
			else if (t instanceof ra) {
				let r = this.idIndex_[e];
				r instanceof ra ? Array.isArray(r) ? r.push(t) : this.idIndex_[e] = [r, t] : n = !1;
			} else n = !1;
		}
		return n && (L(!(e in this.uidIndex_), "The passed `feature` was already added to the source"), this.uidIndex_[e] = t), n;
	}
	addFeatures(e) {
		this.addFeaturesInternal(e), this.changed();
	}
	addFeaturesInternal(e) {
		let t = [], n = [], r = [];
		for (let t = 0, r = e.length; t < r; t++) {
			let r = e[t], i = y(r);
			this.addToIndex_(i, r) && n.push(r);
		}
		for (let e = 0, i = n.length; e < i; e++) {
			let i = n[e], a = y(i);
			this.setupChangeEvents_(a, i);
			let o = i.getGeometry();
			if (o) {
				let e = o.getExtent();
				t.push(e), r.push(i);
			} else this.nullGeometryFeatures_[a] = i;
		}
		if (this.featuresRtree_ && this.featuresRtree_.load(t, r), this.hasListener(Vf.ADDFEATURE)) for (let e = 0, t = n.length; e < t; e++) this.dispatchEvent(new Hf(Vf.ADDFEATURE, n[e]));
	}
	bindFeaturesCollection_(t) {
		let n = !1;
		this.addEventListener(Vf.ADDFEATURE, (function(e) {
			!n && e.feature && (n = !0, t.push(e.feature), n = !1);
		})), this.addEventListener(Vf.REMOVEFEATURE, (function(e) {
			!n && e.feature && (n = !0, t.remove(e.feature), n = !1);
		})), t.addEventListener(e.ADD, ((e) => {
			n ||= (n = !0, this.addFeature(e.element), !1);
		})), t.addEventListener(e.REMOVE, ((e) => {
			n ||= (n = !0, this.removeFeature(e.element), !1);
		})), this.featuresCollection_ = t;
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
		let t = new Hf(Vf.CLEAR);
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
			if (r instanceof ra || r && r.intersectsCoordinate(e)) return t(n);
		});
	}
	forEachFeatureInExtent(e, t) {
		if (this.featuresRtree_) return this.featuresRtree_.forEachInExtent(e, t);
		this.featuresCollection_ && this.featuresCollection_.forEach(t);
	}
	forEachFeatureIntersectingExtent(e, t) {
		return this.forEachFeatureInExtent(e, function(n) {
			let r = n.getGeometry();
			if (r instanceof ra || r && r.intersectsExtent(e)) {
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
		return this.featuresCollection_ ? e = this.featuresCollection_.getArray().slice(0) : this.featuresRtree_ && (e = this.featuresRtree_.getAll(), r(this.nullGeometryFeatures_) || m(e, Object.values(this.nullGeometryFeatures_))), e || [];
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
			let n = this.featuresRtree_;
			return wt(e, t).flatMap((e) => n.getInExtent(e));
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
		t ||= x;
		let c = this.featuresRtree_;
		return c ? (c.forEachInExtent(s, function(e) {
			if (t(e)) {
				let t = e.getGeometry();
				if (!t) return;
				let c = o;
				if (o = t instanceof ra ? 0 : t.closestPointXY(n, r, a, o), o < c) {
					i = e;
					let t = Math.sqrt(o);
					s[0] = n - t, s[1] = r - t, s[2] = n + t, s[3] = r + t;
				}
			}
		}), i) : null;
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
		let t = e.target, n = y(t), r = t.getGeometry();
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
		this.changed(), this.dispatchEvent(new Hf(Vf.CHANGEFEATURE, t));
	}
	hasFeature(e) {
		let t = e.getId();
		if (t !== void 0) {
			let n = this.idIndex_[String(t)];
			return Array.isArray(n) ? n.includes(e) : n === e;
		}
		return y(e) in this.uidIndex_;
	}
	isEmpty() {
		return this.featuresRtree_ ? this.featuresRtree_.isEmpty() && r(this.nullGeometryFeatures_) : !this.featuresCollection_ || this.featuresCollection_.getLength() === 0;
	}
	loadFeatures(e, t, n) {
		let r = this.loadedExtentsRtree_, i = this.strategy_(e, t, n);
		for (let e = 0, a = i.length; e < a; ++e) {
			let a = i[e];
			if (!r.forEachInExtent(a, function(e) {
				return Ge(e.extent, a);
			})) {
				this.loading = Number(this.loading) + 1, this.dispatchEvent(new Hf(Vf.FEATURESLOADSTART));
				let e = (e) => {
					this.loading = Number(this.loading) - 1, this.dispatchEvent(new Hf(Vf.FEATURESLOADEND, void 0, e));
				}, i = () => {
					this.changed(), this.loading = Number(this.loading) - 1, this.dispatchEvent(new Hf(Vf.FEATURESLOADERROR));
				}, o = !1, s = this.loader_.call(this, a, t, n, (t) => o || e(t), () => o || i());
				b(s) ? (o = !0, s.then((t) => {
					let n = t;
					this.addFeatures(n), e(n);
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
			let r = mt(n.extent, e);
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
		let t = y(e);
		if (!(t in this.uidIndex_)) return !1;
		t in this.nullGeometryFeatures_ ? delete this.nullGeometryFeatures_[t] : this.featuresRtree_ && this.featuresRtree_.remove(e), this.featureChangeKeys_[t]?.forEach(o), delete this.featureChangeKeys_[t];
		let n = e.getId();
		if (n !== void 0) {
			let t = n.toString(), r = this.idIndex_[t];
			r === e ? delete this.idIndex_[t] : Array.isArray(r) && (r.splice(r.indexOf(e), 1), r.length === 1 && (this.idIndex_[t] = r[0]));
		}
		return delete this.uidIndex_[t], this.hasListener(Vf.REMOVEFEATURE) && this.dispatchEvent(new Hf(Vf.REMOVEFEATURE, e)), !0;
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
		L(this.format_, "`format` must be set when `url` is set"), this.url_ = e, this.setLoader(If(e, this.format_));
	}
	setOverlaps(e) {
		this.overlaps_ = e, this.changed();
	}
}, Wf = [
	0,
	0,
	0
], Gf = 5, Kf = class {
	constructor(e) {
		let t = e.minZoom, n = e.resolutions;
		t === void 0 && n && (t = n.findIndex((e) => e !== void 0)), this.minZoom = t === void 0 ? 0 : t, this.resolutions_ = n, L(g(this.resolutions_, (e, t) => t - e, !0), "`resolutions` must be sorted in descending order");
		let r;
		if (!e.origins) {
			for (let e = 0, t = this.resolutions_.length - 1; e < t; ++e) if (!r) r = this.resolutions_[e] / this.resolutions_[e + 1];
			else if (this.resolutions_[e] / this.resolutions_[e + 1] !== r) {
				r = void 0;
				break;
			}
		}
		this.zoomFactor_ = r, this.maxZoom = this.resolutions_.length - 1, this.origin_ = e.origin === void 0 ? null : e.origin, this.origins_ = null, e.origins !== void 0 && (this.origins_ = e.origins, L(this.origins_.length == this.resolutions_.length, "Number of `origins` and `resolutions` must be equal"));
		let i = e.extent;
		i !== void 0 && !this.origin_ && !this.origins_ && (this.origin_ = ht(i)), L(!this.origin_ && this.origins_ || this.origin_ && !this.origins_, "Either `origin` or `origins` must be configured, never both"), this.tileSizes_ = null, e.tileSizes !== void 0 && (this.tileSizes_ = e.tileSizes, L(this.tileSizes_.length == this.resolutions_.length, "Number of `tileSizes` and `resolutions` must be equal")), this.tileSize_ = e.tileSize === void 0 ? this.tileSizes_ ? null : 256 : e.tileSize, L(!this.tileSize_ && this.tileSizes_ || this.tileSize_ && !this.tileSizes_, "Either `tileSize` or `tileSizes` must be configured, never both"), this.extent_ = i === void 0 ? null : i, this.fullTileRanges_ = null, this.tmpSize_ = [0, 0], this.tmpExtent_ = [
			0,
			0,
			0,
			0
		], e.tileRanges === void 0 ? e.sizes === void 0 ? i && this.calculateTileRanges_(i) : this.fullTileRanges_ = e.sizes.map((e, t) => {
			let n = new Ha(Math.min(0, e[0]), Math.max(e[0] - 1, -1), Math.min(0, e[1]), Math.max(e[1] - 1, -1));
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
			if (a !== void 0 && o !== void 0 ? (a = Math.floor(a / 2), o = Math.floor(o / 2), i = Ua(a, a, o, o, n)) : i = this.getTileRangeForExtentAndZ(s, c, n), t(c, i)) return !0;
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
		return this.origin_ ? this.origin_ : this.origins_ ? this.origins_[e] : [0, 0];
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
				return Ua(n, n + 1, r, r + 1, t);
			}
			let r = this.getTileCoordExtent(e, n || this.tmpExtent_);
			return this.getTileRangeForExtentAndZ(r, e[0] + 1, t);
		}
		return null;
	}
	getTileRangeForTileCoordAndZ(e, t, n) {
		if (t > this.maxZoom || t < this.minZoom) return null;
		let r = e[0], i = e[1], a = e[2];
		if (t === r) return Ua(i, i, a, a, n);
		if (this.zoomFactor_) {
			let e = this.zoomFactor_ ** +(t - r), o = Math.floor(i * e), s = Math.floor(a * e);
			return t < r ? Ua(o, o, s, s, n) : Ua(o, Math.floor(e * (i + 1)) - 1, s, Math.floor(e * (a + 1)) - 1, n);
		}
		let o = this.getTileCoordExtent(e, this.tmpExtent_);
		return this.getTileRangeForExtentAndZ(o, t, n);
	}
	getTileRangeForExtentAndZ(e, t, n) {
		this.getTileCoordForXYAndZ_(e[0], e[3], t, !1, Wf);
		let r = Wf[1], i = Wf[2];
		this.getTileCoordForXYAndZ_(e[2], e[1], t, !0, Wf);
		let a = Wf[1], o = Wf[2];
		return Ua(r, a, i, o, n);
	}
	getTileCoordCenter(e) {
		let t = this.getOrigin(e[0]), n = this.getResolution(e[0]), r = ao(this.getTileSize(e[0]), this.tmpSize_);
		return [t[0] + (e[1] + .5) * r[0] * n, t[1] - (e[2] + .5) * r[1] * n];
	}
	getTileCoordExtent(e, t) {
		let n = this.getOrigin(e[0]), r = this.getResolution(e[0]), i = ao(this.getTileSize(e[0]), this.tmpSize_), a = n[0] + e[1] * i[0] * r, o = n[1] - (e[2] + 1) * i[1] * r;
		return Ye(a, o, a + i[0] * r, o + i[1] * r, t);
	}
	getTileCoordForCoordAndResolution(e, t, n) {
		return this.getTileCoordForXYAndResolution_(e[0], e[1], t, !1, n);
	}
	getTileCoordForXYAndResolution_(e, t, n, r, i) {
		let a = this.getZForResolution(n), o = n / this.getResolution(a), s = this.getOrigin(a), c = ao(this.getTileSize(a), this.tmpSize_), l = o * (e - s[0]) / n / c[0], u = o * (s[1] - t) / n / c[1];
		return r ? (l = zt(l, Gf) - 1, u = zt(u, Gf) - 1) : (l = Rt(l, Gf), u = Rt(u, Gf)), so(a, l, u, i);
	}
	getTileCoordForXYAndZ_(e, t, n, r, i) {
		let a = this.getOrigin(n), o = this.getResolution(n), s = ao(this.getTileSize(n), this.tmpSize_), c = (e - a[0]) / o / s[0], l = (a[1] - t) / o / s[1];
		return r ? (c = zt(c, Gf) - 1, l = zt(l, Gf) - 1) : (c = Rt(c, Gf), l = Rt(l, Gf)), so(n, c, l, i);
	}
	getTileCoordForCoordAndZ(e, t, n) {
		return this.getTileCoordForXYAndZ_(e[0], e[1], t, !1, n);
	}
	getTileCoordResolution(e) {
		return this.resolutions_[e[0]];
	}
	getTileSize(e) {
		return this.tileSize_ ? this.tileSize_ : this.tileSizes_ ? this.tileSizes_[e] : 256;
	}
	getFullTileRange(e) {
		return this.fullTileRanges_ ? this.fullTileRanges_[e] : this.extent_ ? this.getTileRangeForExtentAndZ(this.extent_, e) : null;
	}
	getZForResolution(e, t) {
		return kt(f(this.resolutions_, e, t || 0), this.minZoom, this.maxZoom);
	}
	tileCoordIntersectsViewport(e, t) {
		return Ti(t, 0, t.length, 2, this.getTileCoordExtent(e));
	}
	calculateTileRanges_(e) {
		let t = this.resolutions_.length, n = Array(t);
		for (let r = this.minZoom; r < t; ++r) n[r] = this.getTileRangeForExtentAndZ(e, r);
		this.fullTileRanges_ = n;
	}
}, qf = /\{z\}/g, Jf = /\{x\}/g, Yf = /\{y\}/g, Xf = /\{-y\}/g;
function Zf(e, t, n, r, i) {
	return e.replace(qf, t.toString()).replace(Jf, n.toString()).replace(Yf, r.toString()).replace(Xf, function() {
		if (i === void 0) throw Error("If the URL template has a {-y} placeholder, the grid extent must be known");
		return (i - r).toString();
	});
}
function Qf(e) {
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
// node_modules/ol/tileurlfunction.js
function $f(e, t) {
	return (function(n, r, i) {
		if (!n) return;
		let a, o = n[0];
		if (t) {
			let e = t.getFullTileRange(o);
			e && (a = e.getHeight() - 1);
		}
		return Zf(e, o, n[1], n[2], a);
	});
}
function ep(e, t) {
	let n = e.length, r = Array(n);
	for (let i = 0; i < n; ++i) r[i] = $f(e[i], t);
	return tp(r);
}
function tp(e) {
	return e.length === 1 ? e[0] : (function(t, n, r) {
		return t ? e[Ft(uo(t), e.length)](t, n, r) : void 0;
	});
}
// node_modules/ol/tilegrid.js
function np(e) {
	let t = e.getDefaultTileGrid();
	return t || (t = op(e), e.setDefaultTileGrid(t)), t;
}
function rp(e, t, n) {
	let r = t[0], i = e.getTileCoordCenter(t), a = sp(n);
	if (!We(a, i)) {
		let t = I(a), n = Math.ceil((a[0] - i[0]) / t);
		return i[0] += t * n, e.getTileCoordForCoordAndZ(i, r);
	}
	return t;
}
function ip(e, t, n, r) {
	r = r === void 0 ? "top-left" : r;
	let i = ap(e, t, n);
	return new Kf({
		extent: e,
		origin: lt(e, r),
		resolutions: i,
		tileSize: n
	});
}
function ap(e, t, n, r) {
	t = t === void 0 ? 42 : t, n = ao(n === void 0 ? 256 : n);
	let i = ft(e), a = I(e);
	r = r !== void 0 && r > 0 ? r : Math.max(a / n[0], i / n[1]);
	let o = t + 1, s = Array(o);
	for (let e = 0; e < o; ++e) s[e] = r / 2 ** e;
	return s;
}
function op(e, t, n, r) {
	return ip(sp(e), t, n, r);
}
function sp(e) {
	let t = or(e);
	if (!t) throw Error("Unable to determine extent from unknown projection");
	let n = t.getExtent();
	if (!n) {
		let e = t.getMetersPerUnit();
		if (e === void 0) throw Error("Unable to determine extent from projection without units");
		let r = 180 * Xt.degrees / e;
		n = Ye(-r, -r, r, r);
	}
	return n;
}
// node_modules/ol/source/Tile.js
var cp = class extends zf {
	constructor(e) {
		super({
			attributions: e.attributions,
			attributionsCollapsible: e.attributionsCollapsible,
			projection: e.projection,
			state: e.state,
			wrapX: e.wrapX,
			interpolate: e.interpolate
		}), this.on, this.once, this.un, this.tilePixelRatio_ = e.tilePixelRatio === void 0 ? 1 : e.tilePixelRatio, this.tileGrid = e.tileGrid === void 0 ? null : e.tileGrid, this.tileGrid && ao(this.tileGrid.getTileSize(this.tileGrid.getMinZoom()), [256, 256]), this.tmpSize = [0, 0], this.key_ = e.key || y(this), this.tileOptions = {
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
		return _();
	}
	getTileGrid() {
		return this.tileGrid;
	}
	getTileGridForProjection(e) {
		return this.tileGrid ? this.tileGrid : np(e);
	}
	getTilePixelRatio(e) {
		return this.tilePixelRatio_;
	}
	getTilePixelSize(e, t, n) {
		let r = this.getTileGridForProjection(n), i = this.getTilePixelRatio(t), a = ao(r.getTileSize(e), this.tmpSize);
		return i == 1 ? a : io(a, i, this.tmpSize);
	}
	getTileCoordForTileUrlFunction(e, t) {
		let n = t === void 0 ? this.getProjection() : t, r = t === void 0 && this.tileGrid || this.getTileGridForProjection(n);
		return this.getWrapX() && n.isGlobal() && (e = rp(r, e, n)), po(e, r) ? e : null;
	}
	clear() {}
	refresh() {
		this.clear(), super.refresh();
	}
}, lp = class extends E {
	constructor(e, t) {
		super(e), this.tile = t;
	}
}, up = {
	TILELOADSTART: "tileloadstart",
	TILELOADEND: "tileloadend",
	TILELOADERROR: "tileloaderror"
}, dp = class e extends cp {
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
		let t = e.target, n = y(t), r = t.getState(), i;
		r == z.LOADING ? (this.tileLoadingKeys_[n] = !0, i = up.TILELOADSTART) : n in this.tileLoadingKeys_ && (delete this.tileLoadingKeys_[n], i = r == z.ERROR ? up.TILELOADERROR : r == z.LOADED ? up.TILELOADEND : void 0), i != null && this.dispatchEvent(new lp(i, t));
	}
	setTileLoadFunction(e) {
		this.tileLoadFunction = e, this.changed();
	}
	setTileUrlFunction(e, t) {
		this.tileUrlFunction = e, t === void 0 ? this.changed() : this.setKey(t);
	}
	setUrl(e) {
		let t = Qf(e);
		this.urls = t, this.setUrls(t);
	}
	setUrls(e) {
		this.urls = e;
		let t = e.join("\n");
		this.generateTileUrlFunction_ ? this.setTileUrlFunction(ep(e, this.tileGrid), t) : this.setKey(t);
	}
	tileUrlFunction(e, t, n) {}
}, fp = class extends dp {
	constructor(e) {
		super({
			attributions: e.attributions,
			cacheSize: e.cacheSize,
			projection: e.projection,
			state: e.state,
			tileGrid: e.tileGrid,
			tileLoadFunction: e.tileLoadFunction ? e.tileLoadFunction : pp,
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
		}), this.crossOrigin = e.crossOrigin === void 0 ? null : e.crossOrigin, this.referrerPolicy = e.referrerPolicy, this.tileClass = e.tileClass === void 0 ? Ba : e.tileClass, this.tileGridForProjection = {}, this.reprojectionErrorThreshold_ = e.reprojectionErrorThreshold, this.renderReprojectionEdges_ = !1;
	}
	getGutterForProjection(e) {
		let t = this.getProjection();
		return t && e && !fr(t, e) ? 0 : this.getGutter();
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
		if (this.tileGrid && (!t || fr(t, e))) return this.tileGrid;
		let n = y(e);
		return n in this.tileGridForProjection || (this.tileGridForProjection[n] = np(e)), this.tileGridForProjection[n];
	}
	createTile_(e, t, n, r, i, a) {
		let o = [
			e,
			t,
			n
		], c = this.getTileCoordForTileUrlFunction(o, i), l = c ? this.tileUrlFunction(c, r, i) : void 0, u = new this.tileClass(o, l === void 0 ? z.EMPTY : z.IDLE, l === void 0 ? "" : l, {
			crossOrigin: this.crossOrigin,
			referrerPolicy: this.referrerPolicy
		}, this.tileLoadFunction, this.tileOptions);
		return u.key = a, u.addEventListener(s.CHANGE, this.handleTileChange.bind(this)), u;
	}
	getTile(e, t, n, r, i, a) {
		let o = this.getProjection();
		if (!o || !i || fr(o, i)) return this.getTileInternal(e, t, n, r, o || i);
		let s = [
			e,
			t,
			n
		], c = this.getKey(), l = new no(o, this.getTileGridForProjection(o), i, this.getTileGridForProjection(i), s, this.getTileCoordForTileUrlFunction(s, i) || s, this.getTilePixelRatio(r), this.getGutter(), (e, t, n, r) => this.getTileInternal(e, t, n, r, o, a), this.reprojectionErrorThreshold_, this.renderReprojectionEdges_, this.tileOptions);
		return l.key = c, l;
	}
	getTileInternal(e, t, n, r, i, a) {
		let o = this.getKey(), s = lo(this, o, e, t, n);
		if (a && a.containsKey(s)) return a.get(s);
		let c = this.createTile_(e, t, n, r, i, o);
		return a?.set(s, c), c;
	}
	setRenderReprojectionEdges(e) {
		this.renderReprojectionEdges_ != e && (this.renderReprojectionEdges_ = e, this.changed());
	}
	setTileGridForProjection(e, t) {
		let n = or(e);
		if (n) {
			let e = y(n);
			e in this.tileGridForProjection || (this.tileGridForProjection[e] = t);
		}
	}
};
function pp(e, t) {
	if (me) {
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
			if (!n) return;
			n.width = t.width, n.height = t.height;
			let r = n.getContext("2d");
			r && (r.drawImage(t, 0, 0), t.close?.(), n.dispatchEvent(new Event("load")));
		}).catch(() => {
			e.getImage()?.dispatchEvent(new Event("error"));
		});
	} else e.getImage().src = t;
}
// node_modules/ol/source/Zoomify.js
var mp = class extends Ba {
	constructor(e, t, n, r, i, a, o) {
		super(t, n, r, i, a, o), this.zoomifyImage_ = void 0, this.tileSize_ = e;
	}
	getImage() {
		if (this.zoomifyImage_) return this.zoomifyImage_;
		let e = super.getImage();
		if (!e) return null;
		if (this.state == z.LOADED) {
			let t = this.tileSize_;
			if (e.width == t[0] && e.height == t[1]) return this.zoomifyImage_ = e, e;
			let n = _e(t[0], t[1]);
			return n.drawImage(e, 0, 0), this.zoomifyImage_ = n.canvas, n.canvas;
		}
		return e;
	}
}, hp = class extends fp {
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
		let d = e.extent || [
			0,
			-a,
			i,
			0
		], f = new Kf({
			tileSize: s,
			extent: d,
			resolutions: l
		}), p = e.url;
		p && !p.includes("{TileGroup}") && !p.includes("{tileIndex}") && (p += "{TileGroup}/{z}-{x}-{y}.jpg");
		let m = Qf(p), h = s * r;
		function g(e) {
			return (function(t, n, r) {
				if (!t) return;
				let i = t[0], a = t[1], s = t[2], c = a + s * o[i][0], l = {
					z: i,
					x: a,
					y: s,
					tileIndex: c,
					TileGroup: "TileGroup" + ((c + u[i]) / h | 0)
				};
				return e.replace(/\{(\w+?)\}/g, function(e, t) {
					return String(l[t]);
				});
			});
		}
		let _ = tp(m.map(g)), v = mp.bind(null, ao(s * r));
		super({
			attributions: e.attributions,
			cacheSize: e.cacheSize,
			crossOrigin: e.crossOrigin,
			referrerPolicy: e.referrerPolicy,
			interpolate: e.interpolate,
			projection: e.projection,
			tilePixelRatio: r,
			reprojectionErrorThreshold: e.reprojectionErrorThreshold,
			tileClass: v,
			tileGrid: f,
			tileUrlFunction: _,
			transition: e.transition
		}), this.zDirection = e.zDirection ?? 0;
		let y = _(f.getTileCoordForCoordAndResolution(ct(d), l[l.length - 1]), 1, null);
		if (y) {
			let e = new Image();
			e.addEventListener("error", () => {
				h = s, this.changed();
			}), e.src = y;
		}
	}
}, gp = {
	image: [
		"Polygon",
		"Circle",
		"LineString",
		"Image",
		"Text"
	],
	hybrid: ["Polygon", "LineString"],
	vector: []
}, _p = {
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
}, vp = class extends Xo {
	constructor(e, t) {
		super(e, t), this.boundHandleStyleImageChange_ = this.handleStyleImageChange_.bind(this), this.renderedLayerRevision_, this.renderedPixelToCoordinateTransform_ = null, this.renderedRotation_, this.renderedOpacity_ = 1, this.tmpTransform_ = Or(), this.tileClipContexts_ = null;
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
		let a = r.viewState, o = a.resolution, s = r.viewHints, c = this.getLayer().getSource();
		if (!c) return null;
		let l = c.getTileGridForProjection(a.projection), u = !(s[V.ANIMATING] || s[V.INTERACTING]), d = l.getZForResolution(o, c.zDirection) === e;
		return u && d ? i.wantedResolution = o : i.wantedResolution ||= l.getResolution(e), i;
	}
	prepareFrame(e) {
		let t = this.getLayer().getRevision();
		return this.renderedLayerRevision_ !== t && (this.renderedLayerRevision_ = t, this.renderedTiles.length = 0), super.prepareFrame(e);
	}
	updateExecutorGroup_(e, t, n) {
		let r = this.getLayer(), i = r.getRevision(), a = r.getRenderOrder() || null, o = e.wantedResolution, s = e.getReplayState(r);
		if (!s.dirty && s.renderedResolution === o && s.renderedRevision == i && s.renderedPixelRatio === t && s.renderedRenderOrder == a) return;
		let c = r.getSource();
		if (!c) return;
		let l = !!r.getDeclutter(), u = c.getTileGrid();
		if (!u) return;
		let d = c.getTileGridForProjection(n).getTileCoordExtent(e.wrappedTileCoord), f = c.getSourceTiles(t, n, e), p = y(r);
		delete e.hitDetectionImageData[p], e.executorGroups[p] = [], s.dirty = !1;
		for (let i = 0, m = f.length; i < m; ++i) {
			let m = f[i];
			if (m.getState() != z.LOADED) continue;
			let h = c.getProjection(), g = m.tileCoord, _ = u.getTileCoordExtent(g);
			n && h && !fr(n, h) && (_ = _r(_, h, n, 32));
			let v = pt(d, _), y = Ve(v, (r.getRenderBuffer() ?? 0) * o, this.tempExtent), b = $e(_, v) ? null : y, x = new _c(0, v, o, t), S = el(o, t), C = function(e, t) {
				let n, i = e.getStyleFunction() || r.getStyleFunction();
				if (i && (n = i(e, o)), n) {
					let r = this.renderFeature(e, S, n, x, l, t);
					s.dirty = s.dirty || r;
				}
			}, w = m.getFeatures() ?? [];
			a && a !== s.renderedRenderOrder && w.sort(a);
			for (let e = 0, t = w.length; e < t; ++e) {
				let t = w[e];
				n && m.projection && !fr(n, m.projection) && (t = t.clone(), t.getGeometry().applyTransform(hr(m.projection, n))), (!b || _t(b, t.getGeometry()?.getExtent())) && C.call(this, t, e);
			}
			let T = x.finish(), E = new Vc(r.getRenderMode() !== "vector" && l && f.length === 1 ? null : v, o, t, c.getOverlaps(), T, r.getRenderBuffer() ?? 0, !0);
			e.executorGroups[p].push(E);
		}
		s.renderedRevision = i, s.renderedPixelRatio = t, s.renderedRenderOrder = a, s.renderedResolution = o;
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		let a = t.viewState.resolution, o = t.viewState.rotation;
		n ??= 0;
		let s = this.getLayer(), c = s.getSource();
		if (!c) return;
		let l = c.getTileGridForProjection(t.viewState.projection), u = s.getRenderBuffer() ?? 0, d = ze([e]);
		Ve(d, a * (u + n), d);
		let f = {}, p = function(e, t, n) {
			let a = e.getId();
			a === void 0 && (a = y(e));
			let o = f[a];
			if (!o) {
				if (n === 0) return f[a] = !0, r(e, s, t);
				i.push(f[a] = {
					feature: e,
					layer: s,
					geometry: t,
					distanceSq: n,
					callback: r
				});
			} else if (o !== !0 && n < o.distanceSq) {
				if (n === 0) return f[a] = !0, i.splice(i.lastIndexOf(o), 1), r(e, s, t);
				o.geometry = t, o.distanceSq = n;
			}
		}, m = this.renderedTiles, h = y(s), g = s.getDeclutter(), _ = g ? t.declutter?.[g]?.all().map((e) => e.value) ?? [] : [], v;
		foundFeature: for (let t = m.length - 1; t >= 0; --t) {
			let r = m[t];
			if (!_t(l.getTileCoordExtent(r.wrappedTileCoord), d)) continue;
			let i = r.executorGroups[h];
			for (let t = 0, r = i.length; t < r; ++t) if (v = i[t].forEachFeatureAtCoordinate(e, a, o, n, p, _), v) break foundFeature;
		}
		return v;
	}
	getFeatures(e) {
		return this.renderedTiles.length === 0 ? Promise.resolve([]) : new Promise((t, n) => {
			let r = this.getLayer(), i = r.getSource(), a = this.renderedProjection;
			if (!i || !a || !this.renderedPixelToCoordinateTransform_) {
				t([]);
				return;
			}
			let o = a.getExtent(), s = this.renderedResolution, c = i.getTileGridForProjection(a), l = R(this.renderedPixelToCoordinateTransform_, e.slice()), u = c.getTileCoordForCoordAndResolution(l, s).toString(), d = this.renderedTiles.find((e) => e.tileCoord.toString() === u && e.getState() === z.LOADED);
			if (!d || d.loadingSourceTiles > 0) {
				t([]);
				return;
			}
			i.getWrapX() && a.canWrapX() && !Ge(o, c.getTileCoordExtent(d.tileCoord)) && Kt(l, a);
			let f = y(r), p = ht(c.getTileCoordExtent(d.wrappedTileCoord)), m = [(l[0] - p[0]) / s, (p[1] - l[1]) / s], h = d.getSourceTiles().reduce((e, t) => e.concat(t.getFeatures() ?? []), []), g = d.hitDetectionImageData[f];
			if (!g) {
				let e = ao(c.getTileSize(c.getZForResolution(s, i.zDirection))), t = this.renderedRotation_;
				g = Yc(e, [this.getRenderTransform(c.getTileCoordCenter(d.wrappedTileCoord), s, 0, Jc, e[0] * Jc, e[1] * Jc, 0)], h, r.getStyleFunction(), c.getTileCoordExtent(d.wrappedTileCoord), d.getReplayState(r).renderedResolution, t), d.hitDetectionImageData[f] = g;
			}
			t(Xc(m, h, g));
		});
	}
	getFeaturesInExtent(e) {
		let t = [], n = this.getTileCache();
		if (n.getCount() === 0) return t;
		let r = this.getLayer().getSource();
		if (!r || !this.frameState) return t;
		let i = r.getTileGridForProjection(this.frameState.viewState.projection), a = i.getZForResolution(this.renderedResolution), o = {};
		return n.forEach((n) => {
			if (n.tileCoord[0] !== a || n.getState() !== z.LOADED) return;
			let r = n.getSourceTiles();
			for (let n = 0, a = r.length; n < a; ++n) {
				let a = r[n], s = a.getKey();
				if (s in o) continue;
				o[s] = !0;
				let c = a.tileCoord;
				if (_t(e, i.getTileCoordExtent(c))) {
					let n = a.getFeatures();
					if (n) for (let r = 0, i = n.length; r < i; ++r) {
						let i = n[r], a = i.getGeometry();
						a && _t(e, a.getExtent()) && t.push(i);
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
		let n = this.getCanvasContext(), r = n.globalAlpha;
		n.globalAlpha = t.opacity;
		let i = e.viewHints, a = !(i[V.ANIMATING] || i[V.INTERACTING]), o = [this.getCanvasContext().canvas.width, this.getCanvasContext().canvas.height], s = this.getLayer().getDeclutter(), c = s ? e.declutter?.[s] : void 0, l = y(this.getLayer()), u = this.renderedTiles;
		for (let t = 0, n = u.length; t < n; ++t) {
			let n = u[t], r = n.executorGroups[l];
			if (r) for (let t = r.length - 1; t >= 0; --t) r[t].execute(this.getCanvasContext(), o, this.getTileRenderTransform(n, e), e.viewState.rotation, a, Ic, c);
		}
		n.globalAlpha = r;
	}
	renderDeferredInternal(e) {
		let t = this.renderedTiles, n = y(this.getLayer()), r = t.reduce((e, t, r) => (t.executorGroups[n].forEach((t) => e.push({
			executorGroup: t,
			index: r
		})), e), []), i = r.map(({ executorGroup: e }) => e.getDeferredZIndexContexts()), a = {};
		for (let e = 0, t = r.length; e < t; ++e) {
			let t = r[e].executorGroup.getDeferredZIndexContexts();
			for (let e in t) a[e] = !0;
		}
		let o = Object.keys(a).map(Number).sort(u);
		this.layerExtent && this.clipUnrotated(this.getCanvasContext(), e, this.layerExtent), o.forEach((e) => {
			i.forEach((t, n) => {
				t[e] && (t[e].forEach((e) => {
					let { executorGroup: t, index: i } = r[n], a = t.getRenderedContext();
					if (!a) return;
					let o = a.globalAlpha;
					a.globalAlpha = this.renderedOpacity_;
					let s = this.tileClipContexts_?.[i];
					s && s.draw(a), e.draw(a), s && a.restore(), a.globalAlpha = o, e.clear();
				}), t[e].length = 0);
			});
		}), this.layerExtent && this.getCanvasContext().restore();
	}
	getTileRenderTransform(e, t) {
		let n = t.pixelRatio, r = t.viewState, i = r.center, a = r.resolution, o = r.rotation, s = t.size, c = Math.round(s[0] * n), l = Math.round(s[1] * n), u = this.getLayer().getSource();
		if (!u) return Or();
		let d = u.getTileGridForProjection(t.viewState.projection), f = e.tileCoord, p = d.getTileCoordExtent(e.wrappedTileCoord), m = d.getTileCoordExtent(f, this.tempExtent)[0] - p[0];
		return Ar(Nr(this.inversePixelTransform.slice(), 1 / n, 1 / n), this.getRenderTransform(i, a, o, n, c, l, m));
	}
	clipTileContext_(e, t, n, r, i, a) {
		let o = [];
		for (let e = 0, a = n.length; e < a; ++e) i < r[e] && _t(t, n[e]) && o.push(n[e]);
		if (o.length === 0) return !1;
		let s = Tt(t, o);
		e.save(), e.beginPath();
		for (let t = 0, n = s.length; t < n; ++t) {
			let n = s[t], r = R(a, [n[0], n[1]]), i = R(a, [n[0], n[3]]), o = R(a, [n[2], n[3]]), c = R(a, [n[2], n[1]]);
			e.moveTo(r[0], r[1]), e.lineTo(i[0], i[1]), e.lineTo(o[0], o[1]), e.lineTo(c[0], c[1]), e.closePath();
		}
		return e.clip(), !0;
	}
	postRender(e, t) {
		let n = t.viewHints, r = !(n[V.ANIMATING] || n[V.INTERACTING]);
		this.renderedPixelToCoordinateTransform_ = t.pixelToCoordinateTransform.slice(), this.renderedRotation_ = t.viewState.rotation, this.renderedOpacity_ = t.layerStatesArray[t.layerIndex].opacity;
		let i = this.getLayer(), a = i.getRenderMode(), o = e.globalAlpha;
		e.globalAlpha = this.renderedOpacity_;
		let s = i.getDeclutter(), c = s ? _p[a].filter((e) => !Ic.includes(e)) : _p[a], l = t.viewState, u = l.rotation;
		this.layerExtent && this.clipUnrotated(e, t, this.layerExtent);
		let d = i.getSource();
		if (!d) return;
		let f = d.getTileGridForProjection(l.projection).getZForResolution(l.resolution, d.zDirection), p = this.renderedTiles, m = [], h = [], g = [], _ = y(i), v = !0;
		for (let n = p.length - 1; n >= 0; --n) {
			let a = p[n];
			v &&= !a.getReplayState(i).dirty;
			let o = a.executorGroups[_].filter((e) => e.hasExecutors(c));
			if (o.length === 0) continue;
			let l = this.getTileRenderTransform(a, t), d = a.tileCoord[0], y = !1, b = o[0].getClipCoords(Er), x = e, S;
			if (b) {
				let e = [
					b[0],
					b[1],
					b[4],
					b[5]
				];
				S = new Bo(), x = S.getContext(), f !== d && (y = this.clipTileContext_(x, e, m, h, d, l)), m.push(e), h.push(d);
			}
			for (let n = 0, i = o.length; n < i; ++n) o[n].execute(e, [e.canvas.width, e.canvas.height], l, u, r, c, s ? t.declutter?.[s] : void 0);
			y && (x === e ? x.restore() : g[n] = S ?? null);
		}
		this.layerExtent && e.restore(), e.globalAlpha = o, this.ready = v, this.tileClipContexts_ = g, t.declutter || this.renderDeferredInternal(t), super.postRender(e, t);
	}
	renderFeature(e, t, n, r, i, a) {
		if (!n) return !1;
		let o = !1;
		if (Array.isArray(n)) for (let s = 0, c = n.length; s < c; ++s) o = rl(r, e, n[s], t, this.boundHandleStyleImageChange_, void 0, i, a) || o;
		else o = rl(r, e, n, t, this.boundHandleStyleImageChange_, void 0, i, a);
		return o;
	}
	tileImageNeedsRender_(e) {
		let t = this.getLayer();
		if (t.getRenderMode() === "vector") return !1;
		let n = e.getReplayState(t), r = t.getRevision(), i = e.wantedResolution;
		return n.renderedTileResolution !== i || n.renderedTileRevision !== r;
	}
	renderTileImage_(e, t) {
		let n = this.getLayer(), r = e.getReplayState(n), i = n.getRevision(), a = e.executorGroups[y(n)];
		r.renderedTileRevision = i;
		let o = e.wrappedTileCoord, s = o[0], c = n.getSource();
		if (!c) return;
		let l = t.pixelRatio, u = t.viewState.projection, d = c.getTileGridForProjection(u), f = d.getResolution(e.tileCoord[0]), p = t.pixelRatio / e.wantedResolution * f, m = d.getResolution(s), h = e.getContext();
		l = Math.round(Math.max(l, p / l));
		let g = c.getTilePixelSize(s, l, u);
		h.canvas.width = g[0], h.canvas.height = g[1];
		let _ = l / p;
		if (_ !== 1) {
			let e = kr(this.tmpTransform_);
			Nr(e, _, _), h.setTransform(e[0], e[1], e[2], e[3], e[4], e[5]);
		}
		let v = d.getTileCoordExtent(o, this.tempExtent), b = p / m, x = kr(this.tmpTransform_);
		Nr(x, b, -b), Pr(x, -v[0], -v[3]);
		for (let e = 0, t = a.length; e < t; ++e) a[e].execute(h, [h.canvas.width * _, h.canvas.height * _], x, 0, !0, gp[n.getRenderMode()], null);
		r.renderedTileResolution = e.wantedResolution;
	}
}, yp = class extends Ed {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.preload;
		let n = e.cacheSize === void 0 ? 0 : e.cacheSize;
		delete e.cacheSize, delete t.useInterimTilesOnError, super(t), this.on, this.once, this.un, this.cacheSize_ = n;
		let r = e.renderMode || "hybrid";
		L(r == "hybrid" || r == "vector", "`renderMode` must be `'hybrid'` or `'vector'`"), this.renderMode_ = r, this.setPreload(e.preload ? e.preload : 0), this.setUseInterimTilesOnError(e.useInterimTilesOnError === void 0 || e.useInterimTilesOnError), this.getBackground, this.setBackground;
	}
	createRenderer() {
		return new vp(this, { cacheSize: this.cacheSize_ });
	}
	getFeatures(e) {
		return super.getFeatures(e);
	}
	getFeaturesInExtent(e) {
		let t = this.getRenderer();
		return t ? t.getFeaturesInExtent(e) : [];
	}
	getRenderMode() {
		return this.renderMode_;
	}
	getPreload() {
		return this.get(bs.PRELOAD);
	}
	getUseInterimTilesOnError() {
		return this.get(bs.USE_INTERIM_TILES_ON_ERROR);
	}
	setPreload(e) {
		this.set(bs.PRELOAD, e);
	}
	setUseInterimTilesOnError(e) {
		this.set(bs.USE_INTERIM_TILES_ON_ERROR, e);
	}
}, bp = class extends Ko {
	constructor(e) {
		super(e), this.image = null, this.renderedSourceRevision_ = 0, this.renderedRotation = 0;
	}
	getImage() {
		return this.image ? this.image.getImage() : null;
	}
	prepareFrame(e) {
		let t = e.layerStatesArray[e.layerIndex], n = e.pixelRatio, r = e.viewState, i = r.resolution, a = this.getLayer().getSource(), o = e.viewHints, s = e.extent;
		if (t.extent !== void 0 && (s = pt(s, Cr(t.extent, r.projection))), !o[V.ANIMATING] && !o[V.INTERACTING] && !vt(s)) {
			if (a) {
				!this.getLayer().rendered && this.renderedSourceRevision_ !== a.getRevision() && (this.image = null), this.renderedSourceRevision_ = a.getRevision();
				let e = r.projection, t = a.getImage(s, i, n, e);
				t && (this.loadImage(t) ? (this.image = t, this.renderedRotation = this.sourceRotates() ? r.rotation : 0) : t.getState() === B.EMPTY && (this.image = null));
			} else this.image = null;
		}
		return !!this.image;
	}
	getData(e) {
		let t = this.frameState;
		if (!t || !this.image) return null;
		let n = this.getLayer(), r = R(t.pixelToCoordinateTransform, e.slice()), i = n.getExtent();
		if (i && !We(i, r)) return null;
		let a = this.image.getExtent(), o = this.image.getImage();
		if (!o) return null;
		let s = this.renderedRotation;
		if (s !== 0) {
			let e = ct(a), t = Math.cos(-s), n = Math.sin(-s), i = r[0] - e[0], o = r[1] - e[1];
			r[0] = e[0] + t * i - n * o, r[1] = e[1] + n * i + t * o;
		}
		let c = I(a), l = Math.floor(o.width * ((r[0] - a[0]) / c));
		if (l < 0 || l >= o.width) return null;
		let u = ft(a), d = Math.floor(o.height * ((a[3] - r[1]) / u));
		return d < 0 || d >= o.height ? null : this.getImageData(o, l, d);
	}
	renderFrame(e, t) {
		let n = this.image;
		if (!n) return this.getContainerElement();
		let r = n.getExtent(), i = n.getResolution(), [a, o] = Array.isArray(i) ? i : [i, i], s = n.getPixelRatio(), c = this.renderedRotation, l = this.getLayer().getSource(), u = this.sourceRotates(), d = e.layerStatesArray[e.layerIndex], f = e.pixelRatio, p = e.viewState, m = p.center, h = p.resolution, g = p.rotation, _ = f * a / (h * s), v = f * o / (h * s);
		this.prepareContainer(e, t);
		let y = this.getCanvasContext().canvas.width, b = this.getCanvasContext().canvas.height, x = this.getRenderContext(e), S = !1, C = !0;
		if (d.extent) {
			let t = Cr(d.extent, p.projection), n = e.extent;
			C = _t(t, n), S = C && !Ge(t, n), S && this.clipUnrotated(x, e, t);
		}
		let w = n.getImage();
		if (!w) return this.getContainerElement();
		let T, E, D;
		if (!u) T = Fr(this.tempTransform, y / 2, b / 2, _, v, 0, s * (r[0] - m[0]) / a, s * (m[1] - r[3]) / o), E = w.width * T[0], D = w.height * T[3];
		else {
			let e = f / h, t = Math.cos(g), n = Math.sin(g), i = ct(r), a = i[0] - m[0], o = i[1] - m[1];
			T = Fr(this.tempTransform, y / 2 + e * (t * a + n * o), b / 2 + e * (n * a - t * o), _, v, g - c, -w.width / 2, -w.height / 2), E = w.width * _, D = w.height * v;
		}
		if (this.renderedResolution = o * f / s, l?.getInterpolate() || u && g !== c || (x.imageSmoothingEnabled = !1), this.preRender(x, e), C && E >= .5 && D >= .5) {
			let e = d.opacity;
			if (u) x.save(), e !== 1 && (x.globalAlpha = e), x.setTransform(T[0], T[1], T[2], T[3], T[4], T[5]), x.drawImage(w, 0, 0), x.restore();
			else {
				let t = T[4], n = T[5];
				e !== 1 && (x.save(), x.globalAlpha = e), x.drawImage(w, 0, 0, w.width, w.height, t, n, E, D), e !== 1 && x.restore();
			}
		}
		return this.postRender(this.getCanvasContext(), e), S && x.restore(), x.imageSmoothingEnabled = !0, this.getContainerElement();
	}
}, xp = class extends vs {
	constructor(e) {
		e ||= {}, super(e);
	}
}, Sp = class extends xp {
	constructor(e) {
		super(e);
	}
	createRenderer() {
		return new bp(this);
	}
	getData(e) {
		return super.getData(e);
	}
};
// node_modules/ol/vec/mat4.js
function Cp() {
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
function wp(e) {
	return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = 1, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = 1, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, e;
}
function Tp(e, t) {
	return e[0] = t[0], e[1] = t[1], e[4] = t[2], e[5] = t[3], e[12] = t[4], e[13] = t[5], e;
}
function Ep(e, t, n, r, i) {
	return i ??= Cp(), i[0] = e[0] * t, i[1] = e[1] * t, i[2] = e[2] * t, i[3] = e[3] * t, i[4] = e[4] * n, i[5] = e[5] * n, i[6] = e[6] * n, i[7] = e[7] * n, i[8] = e[8] * r, i[9] = e[9] * r, i[10] = e[10] * r, i[11] = e[11] * r, i[12] = e[12], i[13] = e[13], i[14] = e[14], i[15] = e[15], i;
}
function Dp(e, t, n, r, i) {
	i ??= Cp();
	let a, o, s, c, l, u, d, f, p, m, h, g;
	return e === i ? (i[12] = e[0] * t + e[4] * n + e[8] * r + e[12], i[13] = e[1] * t + e[5] * n + e[9] * r + e[13], i[14] = e[2] * t + e[6] * n + e[10] * r + e[14], i[15] = e[3] * t + e[7] * n + e[11] * r + e[15]) : (a = e[0], o = e[1], s = e[2], c = e[3], l = e[4], u = e[5], d = e[6], f = e[7], p = e[8], m = e[9], h = e[10], g = e[11], i[0] = a, i[1] = o, i[2] = s, i[3] = c, i[4] = l, i[5] = u, i[6] = d, i[7] = f, i[8] = p, i[9] = m, i[10] = h, i[11] = g, i[12] = a * t + l * n + p * r + e[12], i[13] = o * t + u * n + m * r + e[13], i[14] = s * t + d * n + h * r + e[14], i[15] = c * t + f * n + g * r + e[15]), i;
}
function Op(e, t, n) {
	n ??= Cp();
	let r = Math.cos(t), i = Math.sin(t), a = r, o = -i, s = i, c = r, l = e[0], u = e[1], d = e[2], f = e[3], p = e[4], m = e[5], h = e[6], g = e[7];
	return n[0] = a * l + o * p, n[1] = a * u + o * m, n[2] = a * d + o * h, n[3] = a * f + o * g, n[4] = s * l + c * p, n[5] = s * u + c * m, n[6] = s * d + c * h, n[7] = s * f + c * g, n !== e && (n[8] = e[8], n[9] = e[9], n[10] = e[10], n[11] = e[11], n[12] = e[12], n[13] = e[13], n[14] = e[14], n[15] = e[15]), n;
}
// node_modules/ol/webgl.js
var kp = 34962, Ap = 34963, jp = 35040, Mp = 35044, Np = 35048, Pp = 5121, Fp = 5123, Ip = 5125, Lp = 5126, Rp = [
	"experimental-webgl",
	"webgl",
	"webkit-3d",
	"moz-webgl"
];
function zp(e, t) {
	t = Object.assign({
		preserveDrawingBuffer: !0,
		antialias: !ue
	}, t);
	let n = Rp.length;
	for (let r = 0; r < n; ++r) try {
		let n = e.getContext(Rp[r], t);
		if (n) return n;
	} catch {}
	return null;
}
// node_modules/ol/webgl/Buffer.js
var Bp = {
	STATIC_DRAW: Mp,
	STREAM_DRAW: jp,
	DYNAMIC_DRAW: Np
}, Vp = class {
	constructor(e, t) {
		this.array_ = null, this.type_ = e, L(e === 34962 || e === 34963, "A `WebGLArrayBuffer` must either be of type `ELEMENT_ARRAY_BUFFER` or `ARRAY_BUFFER`"), this.usage_ = t === void 0 ? Bp.STATIC_DRAW : t;
	}
	ofSize(e) {
		return this.array_ = new (Hp(this.type_))(e), this;
	}
	fromArray(e) {
		return this.array_ = Hp(this.type_).from(e), this;
	}
	fromArrayBuffer(e) {
		return this.array_ = new (Hp(this.type_))(e), this;
	}
	getType() {
		return this.type_;
	}
	getArray() {
		return this.array_;
	}
	setArray(e) {
		let t = Hp(this.type_);
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
function Hp(e) {
	switch (e) {
		case kp: return Float32Array;
		case Ap: return Uint32Array;
		default: return Float32Array;
	}
}
// node_modules/ol/webgl/ContextEventType.js
var Up = {
	LOST: "webglcontextlost",
	RESTORED: "webglcontextrestored"
}, Wp = "\n  precision mediump float;\n\n  attribute vec2 a_position;\n  varying vec2 v_texCoord;\n  varying vec2 v_screenCoord;\n\n  uniform vec2 u_screenSize;\n\n  void main() {\n    v_texCoord = a_position * 0.5 + 0.5;\n    v_screenCoord = v_texCoord * u_screenSize;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n  }\n", Gp = "\n  precision mediump float;\n\n  uniform sampler2D u_image;\n  uniform float u_opacity;\n\n  varying vec2 v_texCoord;\n\n  void main() {\n    gl_FragColor = texture2D(u_image, v_texCoord) * u_opacity;\n  }\n", Kp = class {
	constructor(e) {
		this.gl_ = e.webGlContext;
		let t = this.gl_;
		this.scaleRatio_ = e.scaleRatio || 1, this.renderTargetTexture_ = t.createTexture(), this.renderTargetTextureSize_ = null, this.frameBuffer_ = t.createFramebuffer(), this.depthBuffer_ = t.createRenderbuffer();
		let n = t.createShader(t.VERTEX_SHADER);
		t.shaderSource(n, e.vertexShader || Wp), t.compileShader(n);
		let r = t.createShader(t.FRAGMENT_SHADER);
		if (t.shaderSource(r, e.fragmentShader || Gp), t.compileShader(r), !t.getShaderParameter(r, t.COMPILE_STATUS)) {
			let e = `Fragment shader compilation failed: ${t.getShaderInfoLog(r)}`;
			throw Error(e);
		}
		if (this.renderTargetProgram_ = t.createProgram(), t.attachShader(this.renderTargetProgram_, n), t.attachShader(this.renderTargetProgram_, r), t.linkProgram(this.renderTargetProgram_), this.renderTargetVerticesBuffer_ = t.createBuffer(), t.bindBuffer(t.ARRAY_BUFFER, this.renderTargetVerticesBuffer_), t.bufferData(t.ARRAY_BUFFER, new Float32Array([
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
		]), t.STATIC_DRAW), this.renderTargetAttribLocation_ = t.getAttribLocation(this.renderTargetProgram_, "a_position"), this.renderTargetUniformLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_screenSize"), this.renderTargetOpacityLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_opacity"), this.renderTargetTextureLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_image"), this.uniforms_ = [], e.uniforms) {
			let n = e.uniforms;
			Object.keys(n).forEach((e) => {
				this.uniforms_.push({
					value: n[e],
					location: t.getUniformLocation(this.renderTargetProgram_, e)
				});
			});
		}
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
			let t = y(i.canvas);
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
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof ImageData) i.texture ||= t.createTexture(), t.activeTexture(t.TEXTURE0 + r), t.bindTexture(t.TEXTURE_2D, i.texture), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), n instanceof ImageData ? t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, n.width, n.height, 0, t.UNSIGNED_BYTE, new Uint8Array(n.data)) : t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n), t.uniform1i(i.location, r++);
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
}, qp = {
	PROJECTION_MATRIX: "u_projectionMatrix",
	INVERT_PROJECTION_MATRIX: "u_invertProjectionMatrix",
	TIME: "u_time",
	ZOOM: "u_zoom",
	RESOLUTION: "u_resolution",
	ROTATION: "u_rotation",
	VIEWPORT_SIZE_PX: "u_viewportSizePx",
	PIXEL_RATIO: "u_pixelRatio",
	HIT_DETECTION: "u_hitDetection"
}, Jp = {
	UNSIGNED_BYTE: Pp,
	UNSIGNED_SHORT: Fp,
	UNSIGNED_INT: Ip,
	FLOAT: Lp
}, Yp = {};
function Xp(e) {
	return "shared/" + e;
}
var Zp = 0;
function Qp() {
	let e = "unique/" + Zp;
	return Zp += 1, e;
}
function $p(e) {
	let t = Yp[e];
	if (!t) {
		let n = document.createElement("canvas");
		n.width = 1, n.height = 1, n.style.position = "absolute", n.style.left = "0", t = {
			users: 0,
			context: zp(n)
		}, Yp[e] = t;
	}
	return t.users += 1, t.context;
}
function em(e) {
	let t = Yp[e];
	if (!t || (--t.users, t.users > 0)) return;
	let n = t.context, r = n.getExtension("WEBGL_lose_context");
	r && r.loseContext();
	let i = n.canvas;
	i.width = 1, i.height = 1, delete Yp[e];
}
var tm = class extends c {
	constructor(e) {
		super(), e ||= {}, this.boundHandleWebGLContextLost_ = this.handleWebGLContextLost.bind(this), this.boundHandleWebGLContextRestored_ = this.handleWebGLContextRestored.bind(this), this.canvasCacheKey_ = e.canvasCacheKey ? Xp(e.canvasCacheKey) : Qp(), this.gl_ = $p(this.canvasCacheKey_), this.bufferCache_ = {}, this.extensionCache_ = {}, this.currentProgram_ = null, this.needsToBeRecreated_ = !1;
		let t = this.gl_.canvas;
		t.addEventListener(Up.LOST, this.boundHandleWebGLContextLost_), t.addEventListener(Up.RESTORED, this.boundHandleWebGLContextRestored_), this.tmpMat4_ = Cp(), this.uniformLocationsByProgram_ = {}, this.attribLocationsByProgram_ = {}, this.uniforms_ = [], e.uniforms && this.setUniforms(e.uniforms), this.postProcessPasses_ = e.postProcesses?.length ? e.postProcesses.map((e) => new Kp({
			webGlContext: this.gl_,
			scaleRatio: e.scaleRatio,
			vertexShader: e.vertexShader,
			fragmentShader: e.fragmentShader,
			uniforms: e.uniforms
		})) : [new Kp({ webGlContext: this.gl_ })], this.shaderCompileErrors_ = null, this.startTime_ = Date.now(), this.maxAttributeCount_ = this.gl_.getParameter(this.gl_.MAX_VERTEX_ATTRIBS);
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
		return this.canvasCacheKey_ === Xp(e);
	}
	getExtension(e) {
		if (e in this.extensionCache_) return this.extensionCache_[e];
		let t = this.gl_.getExtension(e);
		return this.extensionCache_[e] = t, t;
	}
	getInstancedRenderingExtension_() {
		let e = this.getExtension("ANGLE_instanced_arrays");
		return L(!!e, "WebGL extension 'ANGLE_instanced_arrays' is required for vector rendering"), e;
	}
	bindBuffer(e) {
		let t = this.gl_, n = y(e), r = this.bufferCache_[n];
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
		let t = y(e);
		delete this.bufferCache_[t];
	}
	disposeInternal() {
		let e = this.gl_.canvas;
		e.removeEventListener(Up.LOST, this.boundHandleWebGLContextLost_), e.removeEventListener(Up.RESTORED, this.boundHandleWebGLContextRestored_), em(this.canvasCacheKey_);
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
		for (let r = 0, i = this.postProcessPasses_.length; r < i; r++) r === i - 1 ? this.postProcessPasses_[r].apply(e, void 0, t, n) : this.postProcessPasses_[r].apply(e, this.postProcessPasses_[r + 1]);
	}
	getCanvas() {
		return this.gl_.canvas;
	}
	getGL() {
		return this.gl_;
	}
	applyFrameState(e) {
		let t = e.size, n = e.viewState.rotation, r = e.pixelRatio;
		this.setUniformFloatValue(qp.TIME, (Date.now() - this.startTime_) * .001), this.setUniformFloatValue(qp.ZOOM, e.viewState.zoom), this.setUniformFloatValue(qp.RESOLUTION, e.viewState.resolution), this.setUniformFloatValue(qp.PIXEL_RATIO, r), this.setUniformFloatVec2(qp.VIEWPORT_SIZE_PX, [t[0], t[1]]), this.setUniformFloatValue(qp.ROTATION, n);
	}
	applyHitDetectionUniform(e) {
		let t = this.getUniformLocation(qp.HIT_DETECTION);
		t && this.getGL().uniform1i(t, +!!e), e && this.setUniformFloatValue(qp.PIXEL_RATIO, .5);
	}
	applyUniforms(e) {
		let t = this.gl_, n, r = 0;
		this.uniforms_.forEach((i) => {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof HTMLImageElement || n instanceof ImageData || n instanceof WebGLTexture) {
				n instanceof WebGLTexture && !i.texture ? (i.prevValue = void 0, i.texture = n) : i.texture ||= (i.prevValue = void 0, t.createTexture()), this.bindTexture(i.texture, r, i.name), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE);
				let e = !(n instanceof HTMLImageElement) || n.complete;
				!(n instanceof WebGLTexture) && e && i.prevValue !== n && (i.prevValue = n, t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n)), r++;
			} else if (Array.isArray(n) && n.length === 6) this.setUniformMatrixValue(i.name, Tp(this.tmpMat4_, n));
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
		let t = y(this.currentProgram_);
		return this.uniformLocationsByProgram_[t] === void 0 && (this.uniformLocationsByProgram_[t] = {}), this.uniformLocationsByProgram_[t][e] === void 0 && (this.uniformLocationsByProgram_[t][e] = this.currentProgram_ && this.gl_.getUniformLocation(this.currentProgram_, e)), this.uniformLocationsByProgram_[t][e];
	}
	getAttributeLocation(e) {
		let t = y(this.currentProgram_);
		return this.attribLocationsByProgram_[t] === void 0 && (this.attribLocationsByProgram_[t] = {}), this.attribLocationsByProgram_[t][e] === void 0 && (this.attribLocationsByProgram_[t][e] = this.currentProgram_ ? this.gl_.getAttribLocation(this.currentProgram_, e) : -1), this.attribLocationsByProgram_[t][e];
	}
	makeProjectionTransform(e, t, n) {
		let r = e.size, i = n ? 0 : e.viewState.rotation, a = e.viewState.resolution, o = e.viewState.center;
		return Fr(t, 0, 0, 2 / (a * r[0]), 2 / (a * r[1]), -i, -o[0], -o[1]), t;
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
		let n = nm(e), r = 0;
		for (let i = 0; i < e.length; i++) {
			let a = e[i];
			a.name && this.enableAttributeArray_(a.name, a.size, a.type || 5126, n, r, t), r += a.size * rm(a.type || 5126);
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
function nm(e) {
	let t = 0;
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		t += r.size * rm(r.type || 5126);
	}
	return t;
}
function rm(e) {
	switch (e) {
		case Jp.UNSIGNED_BYTE: return Uint8Array.BYTES_PER_ELEMENT;
		case Jp.UNSIGNED_SHORT: return Uint16Array.BYTES_PER_ELEMENT;
		case Jp.UNSIGNED_INT: return Uint32Array.BYTES_PER_ELEMENT;
		case Jp.FLOAT:
		default: return Float32Array.BYTES_PER_ELEMENT;
	}
}
// node_modules/ol/renderer/webgl/Layer.js
var im = class e extends Ho {
	constructor(e, t) {
		super(e), t ||= {}, this.inversePixelTransform_ = Or(), this.postProcesses_ = t.postProcesses, this.uniforms_ = t.uniforms, this.helper, this.onMapChanged_ = () => {
			this.clearCache(), this.removeHelper();
		}, e.addChangeListener(H.MAP, this.onMapChanged_), this.dispatchPreComposeEvent = this.dispatchPreComposeEvent.bind(this), this.dispatchPostComposeEvent = this.dispatchPostComposeEvent.bind(this);
	}
	dispatchPreComposeEvent(e, t) {
		let n = this.getLayer();
		if (n.hasListener(zo.PRECOMPOSE)) {
			let r = new Ro(zo.PRECOMPOSE, void 0, t, e);
			n.dispatchEvent(r);
		}
	}
	dispatchPostComposeEvent(e, t) {
		let n = this.getLayer();
		if (n.hasListener(zo.POSTCOMPOSE)) {
			let r = new Ro(zo.POSTCOMPOSE, void 0, t, e);
			n.dispatchEvent(r);
		}
	}
	reset(e) {
		this.uniforms_ = e.uniforms ?? {}, this.helper && this.helper.setUniforms(this.uniforms_);
	}
	removeHelper() {
		this.helper &&= (this.helper.dispose(), void 0);
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
			this.helper && this.helper.canvasCacheKeyMatches(a) && !this.helper.needsToBeRecreated() || (this.removeHelper(), this.helper = new tm({
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
		return this.postProcesses_ ?? [];
	}
	disposeInternal() {
		this.clearCache(), this.removeHelper(), this.getLayer()?.removeChangeListener(H.MAP, this.onMapChanged_), super.disposeInternal();
	}
	dispatchRenderEvent_(e, t, n) {
		let r = this.getLayer();
		if (r.hasListener(e)) {
			Fr(this.inversePixelTransform_, 0, 0, n.pixelRatio, -n.pixelRatio, 0, 0, -n.size[1]);
			let i = new Ro(e, this.inversePixelTransform_, n, t);
			r.dispatchEvent(i);
		}
	}
	preRender(e, t) {
		this.dispatchRenderEvent_(zo.PRERENDER, e, t);
	}
	postRender(e, t) {
		this.dispatchRenderEvent_(zo.POSTRENDER, e, t);
	}
}, am = {
	...qp,
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
({ TEXTURE_COORD: "a_textureCoord" }).TEXTURE_COORD, Jp.FLOAT;
// node_modules/ol/webgl/PaletteTexture.js
var om = class {
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
// node_modules/ol/expr/gpu.js
function sm(e, t) {
	return `operator_${e}_${Object.keys(t.functions).length}`;
}
function cm(e) {
	let t = e.toString();
	return t.includes(".") ? t : t + ".0";
}
function lm(e) {
	if (e.length < 2 || e.length > 4) throw Error("`formatArray` can only output `vec2`, `vec3` or `vec4` arrays.");
	return `vec${e.length}(${e.map(cm).join(", ")})`;
}
function um(e) {
	let t = Io(e), n = t.length > 3 ? t[3] : 1;
	return lm([
		t[0] / 255,
		t[1] / 255,
		t[2] / 255,
		n
	]);
}
function dm(e) {
	return lm(ao(e));
}
var fm = {}, pm = 0;
function mm(e) {
	return e in fm || (fm[e] = pm++), fm[e];
}
function hm(e) {
	return cm(mm(e));
}
function gm(e) {
	return "u_var_" + e;
}
function _m(e) {
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
var vm = "getBandValue", ym = "u_paletteTextures", bm = "featureId", xm = "geometryType", Sm = -9999999;
function Cm(e, t, n, r) {
	let i = Jl(e, t, n);
	return r.properties = new Map([...r.properties, ...n.properties]), r.variables = new Map([...r.variables, ...n.variables]), Tm(i, t, r);
}
function Z(e) {
	return (t, n, r) => {
		let i = n.args.length, a = Array(i);
		for (let e = 0; e < i; ++e) a[e] = Tm(n.args[e], r, t);
		return e(a, t);
	};
}
var wm = {
	[q.Get]: (e, t) => {
		let n = "a_prop_" + t.args[0].value;
		return Ul(t.type, Nl) && (n = `(${n} > 0.0)`), n;
	},
	[q.Id]: (e) => (e.featureId = !0, "a_featureId"),
	[q.GeometryType]: (e) => (e.geometryType = !0, "a_geometryType"),
	[q.LineMetric]: () => "currentLineMetric",
	[q.Var]: (e, t) => {
		let n = t.args[0].value, r = gm(n);
		return Ul(t.type, Nl) && (r = `(${r} > 0.0)`), r;
	},
	[q.Has]: (e, t) => `(a_prop_${t.args[0].value} != ${cm(Sm)})`,
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
		let t = e[0], n = e[e.length - 1];
		for (let r = e.length - 3; r >= 1; r -= 2) n = `(${t} == ${e[r]} ? ${e[r + 1]} : ${n})`;
		return n;
	}),
	[q.Between]: Z(([e, t, n]) => `(${e} >= ${t} && ${e} <= ${n})`),
	[q.Interpolate]: Z(([e, t, ...n]) => {
		let r = "";
		for (let i = 0; i < n.length - 2; i += 2) {
			let a = n[i], o = r || n[i + 1], s = n[i + 2], c = n[i + 3], l;
			l = e === cm(1) ? `(${t} - ${a}) / (${s} - ${a})` : `(pow(${e}, (${t} - ${a})) - 1.0) / (pow(${e}, (${s} - ${a})) - 1.0)`, r = `mix(${o}, ${c}, clamp(${l}, 0.0, 1.0))`;
		}
		return r;
	}),
	[q.Case]: Z((e) => {
		let t = e[e.length - 1];
		for (let n = e.length - 3; n >= 0; n -= 2) t = `(${e[n]} ? ${e[n + 1]} : ${t})`;
		return t;
	}),
	[q.In]: Z(([e, ...t], n) => {
		let r = sm("in", n), i = [];
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
		if (!(vm in r.functions)) {
			let e = "", t = r.bandCount || 1;
			for (let n = 0; n < t; n++) {
				let r = Math.floor(n / 4), i = n % 4;
				n === t - 1 && i === 1 && (i = 3);
				let a = `${am.TILE_TEXTURE_ARRAY}[${r}]`;
				e += `  if (band == ${n + 1}.0) {
    return texture2D(${a}, v_textureCoord + vec2(dx, dy))[${i}];
  }
`;
			}
			r.functions[vm] = `float getBandValue(float band, float xOffset, float yOffset) {
  float dx = xOffset / ${am.TEXTURE_PIXEL_WIDTH};
  float dy = yOffset / ${am.TEXTURE_PIXEL_HEIGHT};
${e}
}`;
		}
		return `${vm}(${e}, ${t ?? "0.0"}, ${n ?? "0.0"})`;
	}),
	[q.Palette]: (e, t) => {
		let [n, ...r] = t.args, i = r.length, a = new Uint8Array(i * 4);
		for (let e = 0; e < r.length; e++) {
			let t = r[e].value, n = Io(t), i = e * 4;
			a[i] = n[0], a[i + 1] = n[1], a[i + 2] = n[2], a[i + 3] = n[3] * 255;
		}
		e.paletteTextures ||= [];
		let o = `${ym}[${e.paletteTextures.length}]`, s = new om(o, a);
		return e.paletteTextures.push(s), `texture2D(${o}, vec2((${Tm(n, W, e)} + 0.5) / ${i}.0, 0.5))`;
	}
};
function Tm(e, t, n) {
	if (e instanceof Kl) {
		let r = wm[e.operator];
		if (r === void 0) throw Error(`No compiler defined for this operator: ${JSON.stringify(e.operator)}`);
		return r(n, e, t);
	}
	if ((e.type & W) > 0) return cm(e.value);
	if ((e.type & Nl) > 0) return e.value.toString();
	if ((e.type & G) > 0) return hm(e.value.toString());
	if ((e.type & K) > 0) return um(e.value);
	if ((e.type & Pl) > 0) return lm(e.value);
	if ((e.type & Fl) > 0) return dm(e.value);
	throw Error(`Unexpected expression ${e.value} (expected type ${Bl(t)})`);
}
// node_modules/ol/style/flat.js
function Em() {
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
// node_modules/ol/render/webgl/bufferUtil.js
var Dm = .985;
// node_modules/ol/render/webgl/compileUtil.js
function Om(e, t, n, r) {
	return Cm(t, n ?? 0, r ?? ql(e.inputVariables), e);
}
function km(e) {
	let t = Io(e), n = t[0] * 256, r = t[1], i = t[2] * 256, a = Math.round(t[3] * 255);
	return [n + r, i + a];
}
var Am = "vec4 unpackColor(vec2 packedColor) {\n  return vec4(\n    min(floor(packedColor[0] / 256.0) / 255.0, 1.0),\n    min(mod(packedColor[0], 256.0) / 255.0, 1.0),\n    min(floor(packedColor[1] / 256.0) / 255.0, 1.0),\n    min(mod(packedColor[1], 256.0) / 255.0, 1.0)\n  );\n}";
function jm(e) {
	return e === K || e === Fl ? 2 : e === Pl ? 4 : e === G ? 3 : 1;
}
function Mm(e) {
	if (e === G) return "float";
	let t = jm(e);
	return t > 1 ? `vec${t}` : "float";
}
function Nm(e, t) {
	for (let n of t.variables.entries()) {
		let [t, r] = n, i = gm(t), a = Mm(r);
		r === K && (a = "vec4"), e.addUniform(i, a);
	}
	for (let n of t.properties.entries()) {
		let [t, r] = n, i = Mm(r), a = `a_prop_${t}`;
		r === K ? e.addAttribute(a, i, `unpackColor(${a})`, "vec4") : e.addAttribute(a, i);
	}
	for (let n in t.functions) e.addVertexShaderFunction(t.functions[n]), e.addFragmentShaderFunction(t.functions[n]);
}
function Pm(e, t) {
	let n = {};
	for (let r of e.variables.entries()) {
		let [e, i] = r, a = gm(e);
		n[a] = () => {
			let n = t?.[e];
			if (i === Nl) return +!!n;
			if (i === K) {
				let e = [...Io(n || "#eee")];
				return e[0] /= 255, e[1] /= 255, e[2] /= 255, e[3] ??= 1, e;
			}
			return i === G ? mm(n) : n;
		};
	}
	return n;
}
function Fm(e) {
	let t = {};
	for (let n of e.properties.entries()) {
		let [e, r] = n, i = (t) => {
			let n = t.get(e);
			return r === K ? km([...Io(n || "#eee")]) : r === Nl ? +!!n : n;
		};
		t[`prop_${e}`] = {
			size: jm(r),
			callback: i
		};
	}
	return t;
}
// node_modules/ol/render/webgl/float64Util.js
function Im(e) {
	return e - Lm(e);
}
function Lm(e) {
	return Math.fround(e);
}
// node_modules/ol/render/webgl/ShaderBuilder.js
var Rm = `#ifdef GL_FRAGMENT_PRECISION_HIGH
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
${Am}

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

`, zm = Em(), Bm = class {
	constructor() {
		this.uniforms_ = [], this.attributes_ = [], this.hasSymbol_ = !1, this.symbolSizeExpression_ = `vec2(${cm(zm["circle-radius"])} + ${cm(zm["circle-stroke-width"] * .5)})`, this.symbolRotationExpression_ = "0.0", this.symbolOffsetExpression_ = "vec2(0.0)", this.symbolColorExpression_ = um(zm["circle-fill-color"]), this.texCoordExpression_ = "vec4(0.0, 0.0, 1.0, 1.0)", this.fragmentDiscardExpression_ = null, this.shapeDiscardExpression_ = null, this.symbolRotateWithView_ = !1, this.hasStroke_ = !1, this.strokeWidthExpression_ = cm(zm["stroke-width"]), this.strokeColorExpression_ = um(zm["stroke-color"]), this.strokeOffsetExpression_ = "0.", this.strokeCapExpression_ = hm("round"), this.strokeJoinExpression_ = hm("round"), this.strokeMiterLimitExpression_ = "10.", this.strokeDistanceFieldExpression_ = "-1000.", this.strokePatternLengthExpression_ = null, this.hasFill_ = !1, this.fillColorExpression_ = um(zm["fill-color"]), this.fillPatternSizeExpression_ = null, this.vertexShaderFunctions_ = [], this.fragmentShaderFunctions_ = [];
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
		return this.hasSymbol_ ? `${Rm}
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
		return this.hasSymbol_ ? `${Rm}
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
		return this.hasStroke_ ? `${Rm}
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
  if (cos(angle) > ${Dm} || isCap(angle)) {
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
		return this.hasStroke_ ? `${Rm}
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
  if (cos(joinAngle) > ${Dm}) { // avoid risking a division by zero
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
   if (capType == ${hm("butt")}) {
    return buttCapDistanceField(point, start, end);
  } else if (capType == ${hm("square")}) {
    return squareCapDistanceField(point, start, end, width);
  }
  return roundCapDistanceField(point, start, end, width);
}

float joinDistanceField(vec2 point, vec2 start, vec2 end, float width, float joinAngle, float joinType) {
  if (joinType == ${hm("bevel")}) {
    return bevelJoinField(point, start, end, width, joinAngle);
  } else if (joinType == ${hm("miter")}) {
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
		return this.hasFill_ ? `${Rm}
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
		return this.hasFill_ ? `${Rm}
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
}, Vm = class e {
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
		let t = y(e), n = this.pointBatch.entries[t];
		if (n) return this.pointBatch.geometriesCount -= n.flatCoordss.length, delete this.pointBatch.entries[t], n;
	}
	clearFeatureEntryInLineStringBatch_(e) {
		let t = y(e), n = this.lineStringBatch.entries[t];
		if (n) return this.lineStringBatch.verticesCount -= n.verticesCount ?? 0, this.lineStringBatch.geometriesCount -= n.flatCoordss.length, delete this.lineStringBatch.entries[t], n;
	}
	clearFeatureEntryInPolygonBatch_(e) {
		let t = y(e), n = this.polygonBatch.entries[t];
		if (n) return this.polygonBatch.verticesCount -= n.verticesCount ?? 0, this.polygonBatch.ringsCount -= n.ringsCount ?? 0, this.polygonBatch.geometriesCount -= n.flatCoordss.length, delete this.polygonBatch.entries[t], n;
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
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEndss(), t, y(t), r.getStride());
				break;
			}
			case "MultiLineString": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEnds(), t, y(t), r.getStride());
				break;
			}
			case "MultiPoint": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, y(t), r.getStride());
				break;
			}
			case "Polygon": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), r.getEnds(), t, y(t), r.getStride());
				break;
			}
			case "Point": {
				let r = e;
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, y(t), r.getStride());
				break;
			}
			case "LineString":
			case "LinearRing": {
				let r = e, i = r.getStride();
				this.addCoordinates_(n, r.getFlatCoordinates(), null, t, y(t), i, r.getLayout?.());
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
			case "MultiPoint": {
				let e = a ?? 2;
				for (let n = 0, a = t.length; n < a; n += e) this.addCoordinates_("Point", t.slice(n, n + 2), null, r, i, 2, void 0);
				break;
			}
			case "Polygon": {
				if (!n) break;
				let e = n, c = a ?? 2;
				if (r instanceof ra) {
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
				})), s = t.length / c;
				let l = e.length, u = e.map((e, t, n) => t > 0 ? (e - n[t - 1]) / c : e / c);
				this.polygonBatch.verticesCount += s, this.polygonBatch.ringsCount += l, this.polygonBatch.geometriesCount++;
				let d = this.polygonBatch.entries[i];
				if (!d || !d.ringsVerticesCounts) break;
				d.flatCoordss.push(Hm(t, c)), d.ringsVerticesCounts.push(u), d.verticesCount = (d.verticesCount ?? 0) + s, d.ringsCount = (d.ringsCount ?? 0) + l;
				for (let n = 0, a = e.length; n < a; n++) {
					let a = n > 0 ? e[n - 1] : 0;
					this.addCoordinates_("LinearRing", t.slice(a, e[n]), null, r, i, c, o ?? void 0);
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
			case "LinearRing": {
				let e = a ?? 2;
				this.lineStringBatch.entries[i] || (this.lineStringBatch.entries[i] = this.addRefToEntry_(i, {
					feature: r,
					flatCoordss: [],
					verticesCount: 0
				})), s = t.length / e, this.lineStringBatch.verticesCount += s, this.lineStringBatch.geometriesCount++;
				let n = this.lineStringBatch.entries[i];
				if (!n) break;
				n.flatCoordss.push(Um(t, e, o ?? "")), n.verticesCount = (n.verticesCount ?? 0) + s;
				break;
			}
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
		if (!this.uidToRef_.get(y(e))) return;
		this.removeFeature(e);
		let n = e.getGeometry();
		n && (t && (n = n.clone(), n.applyTransform(t)), this.addGeometry_(n, e));
	}
	removeFeature(e) {
		let t = this.clearFeatureEntryInPointBatch_(e);
		t = this.clearFeatureEntryInPolygonBatch_(e) || t, t = this.clearFeatureEntryInLineStringBatch_(e) || t, t && t.ref !== void 0 && this.removeRef_(t.ref, y(t.feature));
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
function Hm(e, t) {
	return t === 2 ? e : e.filter((e, n) => n % t < 2);
}
function Um(e, t, n) {
	return t === 3 && n === "XYM" ? e : t === 4 ? e.filter((e, n) => n % t !== 2) : t === 3 ? e.map((e, n) => n % t === 2 ? 0 : e) : Array(e.length * 1.5).fill(0).map((t, n) => n % 3 == 2 ? 0 : e[Math.round(n / 1.5)]);
}
// node_modules/ol/webgl/LabelsArray.js
var Wm = new TextEncoder(), Gm = 1e5, Km = class {
	constructor() {
		this.array_ = new Uint8Array(Gm), this.actualSize_ = 0, this.labelPositionMap_ = /* @__PURE__ */ new Map();
	}
	push(e) {
		if (e === "") return [0, 0];
		if (this.labelPositionMap_.has(e)) return this.labelPositionMap_.get(e);
		let t = Wm.encode(e);
		if (this.actualSize_ + t.length > this.array_.length) {
			let e = new Uint8Array(this.array_.length + Gm);
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
// node_modules/ol/worker/webgl.js
function qm() {
	let e = "const t=new Set;let e=!1;function n(e,n,i=2){const f=n&&n.length,u=f?n[0]*i:e.length;t.size&&t.clear();let s=r(e,0,u,i,!0);const v=[];if(!s||s.next===s.prev)return v;let b=0,A=0,m=0;if(f&&(s=function(e,n,x,i){const f=[];for(let o=0,x=n.length;o<x;o++){const u=r(e,n[o]*i,o<x-1?n[o+1]*i:e.length,i,!1);u===u.next&&t.add(u),f.push(B(u))}f.sort(c),function(t,e){const n=Math.ceil((t+2*e)/y)+e+2;h.length<4*n&&(h=new Float64Array(4*n));p=0}(e.length/i,n.length),M(x,x),l=!0;for(let t=0;t<f.length;t++)x=a(f[t],x);return l=!1,o(x)}(e,n,s,i)),e.length>80*i){b=e[0],A=e[1];let t=b,n=A;for(let r=i;r<u;r+=i){const o=e[r],x=e[r+1];o<b&&(b=o),x<A&&(A=x),o>t&&(t=o),x>n&&(n=x)}m=Math.max(t-b,n-A),m=0!==m?32767/m:0}return x(s,v,b,A,m),v}function r(t,e,n,r,o){let x=null;if(o===function(t,e,n,r){let o=0;for(let x=e,i=n-r;x<n;x+=r)o+=(t[i]-t[x])*(t[x+1]+t[i+1]),i=x;return o}(t,e,n,r)>0)for(let o=e;o<n;o+=r)x=G(o/r|0,t[o],t[o+1],x);else for(let o=n-r;o>=e;o-=r)x=G(o/r|0,t[o],t[o+1],x);return x&&N(x,x.next)&&(k(x),x=x.next),x}function o(n,r=n){const o=r===n;let x,i=n;do{x=!1,i===i.next||0!==t.size&&t.has(i)||!N(i,i.next)&&0!==S(i.prev,i,i.next)?(o||i!==r)&&(i=i.next,x=!o):((o||i===r)&&(r=i.prev),e=!0,k(i),i=i.prev,x=!0)}while(x||i!==r);return r}function x(t,n,r,x,l){l&&function(t,e,n,r){let o=t,x=0;do{o.z=I(o.x,o.y,e,n,r),w[x++]=o,o=o.next}while(o!==t);!function(t){if(t<=32){for(let e=1;e<t;e++){const t=w[e],n=t.z;let r=e-1;for(;r>=0&&w[r].z>n;)w[r+1]=w[r],r--;w[r+1]=t}return}F.length<t&&(F=new Uint32Array(t),Z=new Uint32Array(t),d=new Array(t));for(let e=0;e<t;e++)F[e]=w[e].z;E(t,w,F,d,Z,0),E(t,d,Z,w,F,8),E(t,w,F,d,Z,16),E(t,d,Z,w,F,24)}(x);let i=null;for(let t=0;t<x;t++){const e=w[t];e.prevZ=i,i&&(i.nextZ=e),i=e}i.nextZ=null,w.fill(null,0,x),d.fill(null,0,x)}(t,r,x,l);let c=t,a=!1;for(;t.prev!==t.next;){const y=t.prev,h=t.next;if(S(y,t,h)<0&&(l?f(t,r,x,l):i(t)))n.push(y.i,t.i,h.i),k(t),t=h,c=h;else if((t=h)===c){if(e=!1,t=o(t),e){c=t;continue}if(!a){c=t=u(t,n),a=!0;continue}s(t,n,r,x,l);break}}}function i(t){const e=t.prev,n=t,r=t.next,o=e.x,x=n.x,i=r.x,f=e.y,u=n.y,s=r.y,l=Math.min(o,x,i),c=Math.min(f,u,s),a=Math.max(o,x,i),y=Math.max(f,u,s);let h=r.next;for(;h!==e;){if(h.x>=l&&h.x<=a&&h.y>=c&&h.y<=y&&(o!==h.x||f!==h.y)&&U(o,f,x,u,i,s,h.x,h.y)&&S(h.prev,h,h.next)>=0)return!1;h=h.next}return!0}function f(t,e,n,r){const o=t.prev,x=t,i=t.next,f=o.x,u=x.x,s=i.x,l=o.y,c=x.y,a=i.y,y=Math.min(f,u,s),h=Math.min(l,c,a),p=Math.max(f,u,s),v=Math.max(l,c,a),b=I(y,h,e,n,r),M=I(p,v,e,n,r);let A=t.prevZ;for(;A&&A.z>=b;){if(A.x>=y&&A.x<=p&&A.y>=h&&A.y<=v&&A!==i&&(f!==A.x||l!==A.y)&&U(f,l,u,c,s,a,A.x,A.y)&&S(A.prev,A,A.next)>=0)return!1;A=A.prevZ}let m=t.nextZ;for(;m&&m.z<=M;){if(m.x>=y&&m.x<=p&&m.y>=h&&m.y<=v&&m!==i&&(f!==m.x||l!==m.y)&&U(f,l,u,c,s,a,m.x,m.y)&&S(m.prev,m,m.next)>=0)return!1;m=m.nextZ}return!0}function u(t,e){let n=t,r=!1;do{const o=n.prev,x=n.next.next;R(o,n,n.next,x,!1)&&_(o,x)&&_(x,o)&&(e.push(o.i,n.i,x.i),k(n),k(n.next),n=t=x,r=!0),n=n.next}while(n!==t);return r?o(n):n}function s(t,e,n,r,i){let f=t;do{let t=f.next.next;for(;t!==f.prev;){if(f.i!==t.i&&P(f,t)){let u=O(f,t);return f=o(f,f.next),u=o(u,u.next),x(f,e,n,r,i),void x(u,e,n,r,i)}t=t.next}f=f.next}while(f!==t)}let l=!1;function c(t,e){return t.x-e.x||t.y-e.y||(t.next.y-t.y)/(t.next.x-t.x)-(e.next.y-e.y)/(e.next.x-e.x)}function a(t,e){const n=function(t,e){let n=e;const r=t.x,o=t.y;let x,i=-1/0;if(N(t,n))return n;for(let e=0,f=0;e<p;e++,f+=4){if(o<h[f+1]||o>h[f+3]||h[f]>r||h[f+2]<=i)continue;const u=A(e);n=m(e);do{if(n.prev.next===n){if(N(t,n.next))return n.next;if(o<=n.y&&o>=n.next.y&&n.next.y!==n.y){const t=n.x+(o-n.y)*(n.next.x-n.x)/(n.next.y-n.y);if(t<=r&&t>i&&(i=t,x=n.x<n.next.x?n:n.next,t===r))return x}}n=n.next}while(n!==u)}if(!x)return null;const f=x.x,u=x.y,s=Math.min(o,u),l=Math.max(o,u);let c=1/0;for(let e=0,a=0;e<p;e++,a+=4){if(h[a+2]<f||h[a]>r||h[a+3]<s||h[a+1]>l)continue;const y=A(e);n=m(e);do{if(n.prev.next===n&&r>=n.x&&n.x>=f&&r!==n.x&&U(o<u?r:i,o,f,u,o<u?i:r,o,n.x,n.y)){const e=Math.abs(o-n.y)/(r-n.x);(_(n,t)||n.y===o&&n.next.y===o&&n.next.x>r)&&(e<c||e===c&&(n.x>x.x||n.x===x.x&&g(x,n)))&&(x=n,c=e)}n=n.next}while(n!==y)}return x}(t,e);if(!n)return e;const r=O(n,t);return M(n,r.next.next),o(r,r.next),o(n,n.next)}const y=16;let h=new Float64Array(0),p=0;const v=[],b=[];function M(t,e){let n=t;do{const t=p++;v[t]=n;let r=n.x,o=n.y,x=n.x,i=n.y,f=0;do{const e=n.next;n.z=t,e.x<r&&(r=e.x),e.x>x&&(x=e.x),e.y<o&&(o=e.y),e.y>i&&(i=e.y),n=e}while(++f<y&&n!==e);b[t]=n;const u=4*t;h[u]=r,h[u+1]=o,h[u+2]=x,h[u+3]=i}while(n!==e)}function A(t){let e=b[t];for(;e.prev.next!==e;)e=e.next;return b[t]=e,e}function m(t){let e=v[t];for(;e.prev.next!==e;)e=e.next;return v[t]=e,e}function g(t,e){return S(t.prev,t,e.prev)<0&&S(e.next,t,t.next)<0}const w=[];let d=[],F=new Uint32Array(0),Z=new Uint32Array(0);const z=new Uint32Array(256);function E(t,e,n,r,o,x){z.fill(0);for(let e=0;e<t;e++)z[n[e]>>>x&255]++;let i=0;for(let t=0;t<256;t++){const e=z[t];z[t]=i,i+=e}for(let i=0;i<t;i++){const t=n[i],f=z[t>>>x&255]++;r[f]=e[i],o[f]=t}}function I(t,e,n,r,o){return(t=1431655765&((t=858993459&((t=252645135&((t=16711935&((t=(t-n)*o|0)|t<<8))|t<<4))|t<<2))|t<<1))|(e=1431655765&((e=858993459&((e=252645135&((e=16711935&((e=(e-r)*o|0)|e<<8))|e<<4))|e<<2))|e<<1))<<1}function B(t){let e=t,n=t;do{(e.x<n.x||e.x===n.x&&e.y<n.y)&&(n=e),e=e.next}while(e!==t);return n}function U(t,e,n,r,o,x,i,f){return(o-i)*(e-f)>=(t-i)*(x-f)&&(t-i)*(r-f)>=(n-i)*(e-f)&&(n-i)*(x-f)>=(o-i)*(r-f)}function P(t,e){const n=N(t,e)&&S(t.prev,t,t.next)>0&&S(e.prev,e,e.next)>0;return t.next.i!==e.i&&(n||_(t,e)&&_(e,t)&&(0!==S(t.prev,t,e.prev)||0!==S(t,e.prev,e)))&&!function(t,e){const n=Math.min(t.x,e.x),r=Math.max(t.x,e.x),o=Math.min(t.y,e.y),x=Math.max(t.y,e.y);let i=t;do{const f=i.next;if(i.x>r&&f.x>r||i.x<n&&f.x<n||i.y>x&&f.y>x||i.y<o&&f.y<o)i=f;else{if(i.i!==t.i&&f.i!==t.i&&i.i!==e.i&&f.i!==e.i&&R(i,f,t,e))return!0;i=f}}while(i!==t);return!1}(t,e)&&(n||function(t,e){let n=t,r=!1;const o=(t.x+e.x)/2,x=(t.y+e.y)/2;do{const t=n.next;n.y>x!=t.y>x&&o<(t.x-n.x)*(x-n.y)/(t.y-n.y)+n.x&&(r=!r),n=t}while(n!==t);return r}(t,e))}function S(t,e,n){return(e.y-t.y)*(n.x-e.x)-(e.x-t.x)*(n.y-e.y)}function N(t,e){return t.x===e.x&&t.y===e.y}function R(t,e,n,r,o=!0){const x=S(t,e,n),i=S(t,e,r),f=S(n,r,t),u=S(n,r,e);return(x>0&&i<0||x<0&&i>0)&&(f>0&&u<0||f<0&&u>0)||!!o&&(!(0!==x||!T(t,n,e))||(!(0!==i||!T(t,r,e))||(!(0!==f||!T(n,t,r))||!(0!==u||!T(n,e,r)))))}function T(t,e,n){return e.x<=Math.max(t.x,n.x)&&e.x>=Math.min(t.x,n.x)&&e.y<=Math.max(t.y,n.y)&&e.y>=Math.min(t.y,n.y)}function _(t,e){return S(t.prev,t,t.next)<0?S(t,e,t.next)>=0&&S(t,t.prev,e)>=0:S(t,e,t.prev)<0||S(t,t.next,e)<0}function O(t,e){const n=j(t.i,t.x,t.y),r=j(e.i,e.x,e.y),o=t.next,x=e.prev;return t.next=e,e.prev=t,n.next=o,o.prev=n,r.next=n,n.prev=r,x.next=r,r.prev=x,r}function G(t,e,n,r){const o=j(t,e,n);return r?(o.next=r.next,o.prev=r,r.next.prev=o,r.next=o):(o.prev=o,o.next=o),o}function k(t){t.next.prev=t.prev,t.prev.next=t.next,t.prevZ&&(t.prevZ.nextZ=t.nextZ),t.nextZ&&(t.nextZ.prevZ=t.prevZ),l&&function(t,e){const n=4*t.z;e.x<h[n]&&(h[n]=e.x),e.y<h[n+1]&&(h[n+1]=e.y),e.x>h[n+2]&&(h[n+2]=e.x),e.y>h[n+3]&&(h[n+3]=e.y)}(t.prev,t.next)}function j(t,e,n){return{i:t,x:e,y:n,prev:null,next:null,z:0,prevZ:null,nextZ:null}}function q(t,e,n){const r=Math.sqrt((e[0]-t[0])*(e[0]-t[0])+(e[1]-t[1])*(e[1]-t[1])),o=[(e[0]-t[0])/r,(e[1]-t[1])/r],x=[-o[1],o[0]],i=Math.sqrt((n[0]-t[0])*(n[0]-t[0])+(n[1]-t[1])*(n[1]-t[1])),f=[(n[0]-t[0])/i,(n[1]-t[1])/i];let u=0===r||0===i?0:Math.acos((s=f[0]*o[0]+f[1]*o[1],l=-1,c=1,Math.min(Math.max(s,l),c)));var s,l,c;u=Math.max(u,1e-5);return f[0]*x[0]+f[1]*x[1]>0?u:2*Math.PI-u}const L=[1,0,0,1,0,0];function Y(t,e){const n=e[0],r=e[1];return e[0]=t[0]*n+t[2]*r+t[4],e[1]=t[1]*n+t[3]*r+t[5],e}function C(t,e){const n=(r=e)[0]*r[3]-r[1]*r[2];var r;!function(t,e){if(!t)throw new Error(e)}(0!==n,\"Transformation matrix cannot be inverted\");const o=e[0],x=e[1],i=e[2],f=e[3],u=e[4],s=e[5];return t[0]=f/n,t[1]=-x/n,t[2]=-i/n,t[3]=o/n,t[4]=(i*s-f*u)/n,t[5]=-(o*s-x*u)/n,t}new Array(6);const D=[],H={vertexAttributesPosition:0,instanceAttributesPosition:0,indicesPosition:0};function J(t,e,n,r,o){const x=t[e++],i=t[e++],f=D;f.length=r;for(let n=0;n<f.length;n++)f[n]=t[e+n];let u=o?o.instanceAttributesPosition:0;return n[u++]=x,n[u++]=i,f.length&&(n.set(f,u),u+=f.length),H.instanceAttributesPosition=u,H}function K(t,e,n,r,o,x,i,f,u,s){const l=[t[e],t[e+1]],c=[t[n],t[n+1]],a=t[e+2],y=t[n+2],h=Y(f,[...l]),p=Y(f,[...c]);let v=-1,b=-1,M=s;const A=null!==o;if(null!==r){v=q(h,p,Y(f,[...[t[r],t[r+1]]])),Math.cos(v)<=.985&&(M+=Math.tan((v-Math.PI)/2))}if(A){b=q(p,h,Y(f,[...[t[o],t[o+1]]])),Math.cos(b)<=.985&&(M+=Math.tan((Math.PI-b)/2))}const m=Math.pow(2,24),g=u%m,w=Math.floor(u/m)*m;return x.push(l[0],l[1],a,c[0],c[1],y,v,b,g,w,s),x.push(...i),{length:u+Math.sqrt((p[0]-h[0])*(p[0]-h[0])+(p[1]-h[1])*(p[1]-h[1])),angle:M}}function Q(t,e,r,o,x){const i=2+x;let f=e;const u=t.slice(f,f+x);f+=x;const s=t[f++];let l=0;const c=new Array(s-1);for(let e=0;e<s;e++)l+=t[f++],e<s-1&&(c[e]=l);const a=t.slice(f,f+2*l),y=n(a,c,2);for(let t=0;t<y.length;t++)o.push(y[t]+r.length/i);for(let t=0;t<a.length;t+=2)r.push(a[t],a[t+1],...u);return f+2*l}const V=\"GENERATE_POLYGON_BUFFERS\",W=\"GENERATE_POINT_BUFFERS\",X=\"GENERATE_LINE_STRING_BUFFERS\",$=self;$.onmessage=t=>{const e=t.data;switch(e.type){case W:{const t=2,n=2,r=e.customAttributesSize,o=n+r,x=new Float32Array(e.renderInstructions),i=x.length/o*(t+r),f=Uint32Array.from([0,1,3,1,2,3]),u=Float32Array.from([-1,-1,1,-1,1,1,-1,1]),s=new Float32Array(i);let l;for(let t=0;t<x.length;t+=o)l=J(x,t,s,r,l);const c=Object.assign({indicesBuffer:f.buffer,vertexAttributesBuffer:u.buffer,instanceAttributesBuffer:s.buffer,renderInstructions:x.buffer},e);$.postMessage(c,[u.buffer,s.buffer,f.buffer,x.buffer]);break}case X:{const t=[],n=e.customAttributesSize,r=3,o=new Float32Array(e.renderInstructions);let x=0;const i=e.renderInstructionsTransform,f=L.slice(0);let u,s;for(C(f,i);x<o.length;){s=Array.from(o.slice(x,x+n)),x+=n,u=o[x++];const e=x,i=x+(u-1)*r,l=o[e]===o[i]&&o[e+1]===o[i+1];let c=0,a=0;for(let n=0;n<u-1;n++){let y=null;n>0?y=x+(n-1)*r:l&&(y=i-r);let h=null;n<u-2?h=x+(n+2)*r:l&&(h=e+r);const p=K(o,x+n*r,x+(n+1)*r,y,h,t,s,f,c,a);c=p.length,a=p.angle}x+=u*r}const l=Uint32Array.from([0,1,3,1,2,3]),c=Float32Array.from([-1,-1,1,-1,1,1,-1,1]),a=Float32Array.from(t),y=Object.assign({indicesBuffer:l.buffer,vertexAttributesBuffer:c.buffer,instanceAttributesBuffer:a.buffer,renderInstructions:o.buffer},e);$.postMessage(y,[c.buffer,a.buffer,l.buffer,o.buffer]);break}case V:{const t=[],n=[],r=e.customAttributesSize,o=new Float32Array(e.renderInstructions);let x=0;for(;x<o.length;)x=Q(o,x,t,n,r);const i=Uint32Array.from(n),f=Float32Array.from(t),u=Float32Array.from([]),s=Object.assign({indicesBuffer:i.buffer,vertexAttributesBuffer:f.buffer,instanceAttributesBuffer:u.buffer,renderInstructions:o.buffer},e);$.postMessage(s,[f.buffer,u.buffer,i.buffer,o.buffer]);break}}};";
	return new Worker(typeof Blob > "u" ? "data:application/javascript;base64," + Buffer.from(e, "binary").toString("base64") : URL.createObjectURL(new Blob([e], { type: "application/javascript" })));
}
// node_modules/ol/render/webgl/constants.js
var Jm = {
	GENERATE_POLYGON_BUFFERS: "GENERATE_POLYGON_BUFFERS",
	GENERATE_POINT_BUFFERS: "GENERATE_POINT_BUFFERS",
	GENERATE_LINE_STRING_BUFFERS: "GENERATE_LINE_STRING_BUFFERS"
}, Ym = {
	BUILD_INSTRUCTIONS: "BUILD_INSTRUCTIONS",
	DISPOSE_INSTRUCTIONS: "DISPOSE_INSTRUCTIONS",
	RENDER: "RENDER"
};
// node_modules/ol/render/webgl/encodeUtil.js
function Xm(e, t) {
	t ||= [];
	let n = Math.floor(e / 256 / 256 / 256) / 255, r = Math.floor(e / 256 / 256) % 256 / 255, i = Math.floor(e / 256) % 256 / 255, a = e % 256 / 255;
	return t[0] = n * 256 * 255 + r * 255, t[1] = i * 256 * 255 + a * 255, t;
}
function Zm(e) {
	let t = 0;
	return t += Math.round(e[0] * 256 * 256 * 256 * 255), t += Math.round(e[1] * 256 * 256 * 255), t += Math.round(e[2] * 256 * 255), t += Math.round(e[3] * 255), t;
}
// node_modules/ol/render/webgl/renderinstructions.js
function Qm(e, t, n, r, i) {
	let a = 0;
	for (let o in n) {
		let s = n[o], c = s.callback.call(r, r.feature);
		if (typeof c == "string") {
			let [n, r] = t.push(c);
			e[i + a++] = mm(c), e[i + a++] = n, e[i + a++] = r;
			continue;
		}
		let l = Array.isArray(c) ? c : void 0, u = l ? l[0] : c;
		u === -9999999 && console.warn("The \"has\" operator might return false positives."), u === void 0 ? u = Sm : u === null && (u = 0), e[i + a++] = u, s.size && s.size !== 1 && (e[i + a++] = l?.[1] ?? -9999999, s.size < 3 || (e[i + a++] = l?.[2] ?? -9999999, s.size < 4 || (e[i + a++] = l?.[3] ?? -9999999)));
	}
	return a;
}
function $m(e) {
	return Object.keys(e).reduce((t, n) => t + (e[n].size || 1), 0);
}
function eh(e, t, n, r, i) {
	let a = (2 + $m(r)) * e.geometriesCount;
	t && t.length === a || (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) o[0] = c.flatCoordss[e][0], o[1] = c.flatCoordss[e][1], R(i, o), t[s++] = o[0], t[s++] = o[1], s += Qm(t, n, r, c, s);
	}
	return t;
}
function th(e, t, n, r, i) {
	let a = 3 * e.verticesCount + (1 + $m(r)) * e.geometriesCount;
	t && t.length === a || (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) {
			o.length = c.flatCoordss[e].length, Hr(c.flatCoordss[e], 0, o.length, 3, i, o, 3), s += Qm(t, n, r, c, s), t[s++] = o.length / 3;
			for (let e = 0, n = o.length; e < n; e += 3) t[s++] = o[e], t[s++] = o[e + 1], t[s++] = o[e + 2];
		}
	}
	return t;
}
function nh(e, t, n, r, i) {
	let a = 2 * e.verticesCount + (1 + $m(r)) * e.geometriesCount + e.ringsCount;
	t && t.length === a || (t = new Float32Array(a));
	let o = [], s = 0;
	for (let a in e.entries) {
		let c = e.entries[a];
		for (let e = 0, a = c.flatCoordss.length; e < a; e++) {
			let a = c.ringsVerticesCounts?.[e];
			if (a) {
				o.length = c.flatCoordss[e].length, Hr(c.flatCoordss[e], 0, o.length, 2, i, o), s += Qm(t, n, r, c, s), t[s++] = a.length;
				for (let e = 0, n = a.length; e < n; e++) t[s++] = a[e];
				for (let e = 0, n = o.length; e < n; e += 2) t[s++] = o[e], t[s++] = o[e + 1];
			}
		}
	}
	return t;
}
// node_modules/ol/render/webgl/serialize.js
function rh(e) {
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
// node_modules/ol/render/webgl/style.js
function ih(e, t) {
	return e[t];
}
function Q(e, t, n, r) {
	return Om(e, t, n, r);
}
function ah(e) {
	return (JSON.stringify(e).split("").reduce((e, t) => (e << 5) - e + t.charCodeAt(0), 0) >>> 0).toString();
}
function oh(e, t, n, r) {
	if (`${r}radius` in e && r !== "icon-") {
		let i = Q(n, e[`${r}radius`], W);
		if (`${r}radius2` in e) {
			let t = Q(n, ih(e, `${r}radius2`), W);
			i = `max(${i}, ${t})`;
		}
		`${r}stroke-width` in e && (i = `(${i} + ${Q(n, e[`${r}stroke-width`], W)} * 0.5)`), t.setSymbolSizeExpression(`vec2(${i} * 2. + 0.5)`);
	}
	if (`${r}scale` in e) {
		let i = Q(n, e[`${r}scale`], Fl);
		t.setSymbolSizeExpression(`${t.getSymbolSizeExpression()} * ${i}`);
	}
	`${r}displacement` in e && t.setSymbolOffsetExpression(Q(n, e[`${r}displacement`], Pl)), `${r}rotation` in e && t.setSymbolRotationExpression(Q(n, e[`${r}rotation`], W)), `${r}rotate-with-view` in e && t.setSymbolRotateWithView(!!e[`${r}rotate-with-view`]);
}
function sh(e, t, n, r, i) {
	let a = "vec4(0.)";
	if (t !== null && (a = t), n !== null && r !== null) {
		let t = `smoothstep(-${r} + 0.63, -${r} - 0.58, ${e})`;
		a = `mix(${n}, ${a}, ${t})`;
	}
	let o = `(1.0 - smoothstep(-0.63, 0.58, ${e}))`, s = `${a} * vec4(1.0, 1.0, 1.0, ${o})`;
	return i !== null && (s = `${s} * vec4(1.0, 1.0, 1.0, ${i})`), s;
}
function ch(e, t, n, r, i) {
	let a = new Image();
	a.crossOrigin = ih(e, `${r}cross-origin`) === void 0 ? "anonymous" : ih(e, `${r}cross-origin`), L(typeof ih(e, `${r}src`) == "string", `WebGL layers do not support expressions for the ${r}src style property`), a.src = ih(e, `${r}src`), n[`u_texture${i}_size`] = () => a.complete ? [a.width, a.height] : [0, 0], t.addUniform(`u_texture${i}_size`, "vec2");
	let o = `u_texture${i}_size`;
	return n[`u_texture${i}`] = a, t.addUniform(`u_texture${i}`, "sampler2D"), o;
}
function lh(e, t, n, r, i) {
	let a = Q(n, e[`${t}offset`], Fl);
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
function uh(e, t, n, r) {
	r.functions.circleDistanceField = "float circleDistanceField(vec2 point, float radius) {\n  return length(point) - radius;\n}", oh(e, t, r, "circle-");
	let i = null;
	"circle-opacity" in e && (i = Q(r, e["circle-opacity"], W));
	let a = "coordsPx";
	"circle-scale" in e && (a = `coordsPx / ${Q(r, e["circle-scale"], Fl)}`);
	let o = null;
	"circle-fill-color" in e && (o = Q(r, e["circle-fill-color"], K));
	let s = null;
	"circle-stroke-color" in e && (s = Q(r, e["circle-stroke-color"], K));
	let c = Q(r, e["circle-radius"], W), l = null;
	"circle-stroke-width" in e && (l = Q(r, e["circle-stroke-width"], W), c = `(${c} + ${l} * 0.5)`);
	let u = sh(`circleDistanceField(${a}, ${c})`, o, s, l, i);
	t.setSymbolColorExpression(u);
}
function dh(e, t, n, r) {
	r.functions.round = "float round(float v) {\n  return sign(v) * floor(abs(v) + 0.5);\n}", r.functions.starDistanceField = "float starDistanceField(vec2 point, float numPoints, float radius, float radius2, float angle) {\n  float startAngle = -PI * 0.5 + angle; // tip starts upwards and rotates clockwise with angle\n  float c = cos(startAngle);\n  float s = sin(startAngle);\n  vec2 pointRotated = vec2(c * point.x - s * point.y, s * point.x + c * point.y);\n  float alpha = TWO_PI / numPoints; // the angle of one sector\n  float beta = atan(pointRotated.y, pointRotated.x);\n  float gamma = round(beta / alpha) * alpha; // angle in sector\n  c = cos(-gamma);\n  s = sin(-gamma);\n  vec2 inSector = vec2(c * pointRotated.x - s * pointRotated.y, abs(s * pointRotated.x + c * pointRotated.y));\n  vec2 tipToPoint = inSector + vec2(-radius, 0.);\n  vec2 edgeNormal = vec2(radius2 * sin(alpha * 0.5), -radius2 * cos(alpha * 0.5) + radius);\n  return dot(normalize(edgeNormal), tipToPoint);\n}", r.functions.regularDistanceField = "float regularDistanceField(vec2 point, float numPoints, float radius, float angle) {\n  float startAngle = -PI * 0.5 + angle; // tip starts upwards and rotates clockwise with angle\n  float c = cos(startAngle);\n  float s = sin(startAngle);\n  vec2 pointRotated = vec2(c * point.x - s * point.y, s * point.x + c * point.y);\n  float alpha = TWO_PI / numPoints; // the angle of one sector\n  float radiusIn = radius * cos(PI / numPoints);\n  float beta = atan(pointRotated.y, pointRotated.x);\n  float gamma = round((beta - alpha * 0.5) / alpha) * alpha + alpha * 0.5; // angle in sector from mid\n  c = cos(-gamma);\n  s = sin(-gamma);\n  vec2 inSector = vec2(c * pointRotated.x - s * pointRotated.y, abs(s * pointRotated.x + c * pointRotated.y));\n  return inSector.x - radiusIn;\n}", oh(e, t, r, "shape-");
	let i = null;
	"shape-opacity" in e && (i = Q(r, e["shape-opacity"], W));
	let a = "coordsPx";
	"shape-scale" in e && (a = `coordsPx / ${Q(r, e["shape-scale"], Fl)}`);
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
	let p = sh(d, o, s, c, i);
	t.setSymbolColorExpression(p);
}
function fh(e, t, n, r) {
	let i = "vec4(1.0)";
	"icon-color" in e && (i = Q(r, e["icon-color"], K)), "icon-opacity" in e && (i = `${i} * vec4(1.0, 1.0, 1.0, ${Q(r, e["icon-opacity"], W)})`);
	let a = ah(e["icon-src"]), o = ch(e, t, n, "icon-", a);
	if (t.setSymbolColorExpression(`${i} * texture2D(u_texture${a}, v_texCoord)`).setSymbolSizeExpression(o), "icon-width" in e && "icon-height" in e && t.setSymbolSizeExpression(`vec2(${Q(r, e["icon-width"], W)}, ${Q(r, e["icon-height"], W)})`), "icon-offset" in e && "icon-size" in e) {
		let n = Q(r, e["icon-size"], Pl), i = t.getSymbolSizeExpression();
		t.setSymbolSizeExpression(n);
		let a = lh(e, "icon-", r, "v_quadSizePx", n);
		t.setTextureCoordinateExpression(`(vec4((${a}).xyxy) + vec4(0., 0., ${n})) / (${i}).xyxy`);
	}
	if (oh(e, t, r, "icon-"), "icon-anchor" in e) {
		let n = Q(r, e["icon-anchor"], Pl), i = "1.0";
		"icon-scale" in e && (i = Q(r, e["icon-scale"], Fl));
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
function ph(e, t, n, r) {
	if ("stroke-color" in e && t.setStrokeColorExpression(Q(r, e["stroke-color"], K)), "stroke-pattern-src" in e) {
		let i = ah(e["stroke-pattern-src"]), a = ch(e, t, n, "stroke-pattern-", i), o = a, s = "vec2(0.)";
		"stroke-pattern-offset" in e && "stroke-pattern-size" in e && (o = Q(r, e["stroke-pattern-size"], Pl), s = lh(e, "stroke-pattern-", r, a, o));
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
  if (capType == ${hm("square")}) {
    distanceSegment -= lineWidth * 0.5;
  } else if (capType == ${hm("round")}) {
    distanceSegment = min(distanceSegment, sqrt(distanceSegment * distanceSegment + radius * radius) - lineWidth * 0.5);
  }
  return distanceSegment;
}`;
		let n = e["stroke-line-dash"].map((e) => Q(r, e, W));
		n.length % 2 == 1 && (n = [...n, ...n]);
		let i = "0.";
		"stroke-line-dash-offset" in e && (i = Q(r, e["stroke-line-dash-offset"], W));
		let a = `dashDistanceField_${ah(e["stroke-line-dash"])}`, o = n.map((e, t) => `float dashLength${t}`).join(", "), s = n.map((e, t) => `dashLength${t}`).join(" + "), c = "0.", l = `getSingleDashDistance(distance, radius, ${c}, dashLength0, totalDashLength, capType, lineWidth)`;
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
function mh(e, t, n, r) {
	if ("fill-color" in e && t.setFillColorExpression(Q(r, e["fill-color"], K)), "fill-pattern-src" in e) {
		let i = ah(e["fill-pattern-src"]), a = ch(e, t, n, "fill-pattern-", i);
		t.setFillPatternSizeExpression(a);
		let o = "vec2(0.)";
		if ("fill-pattern-offset" in e && "fill-pattern-size" in e) {
			let n = Q(r, e["fill-pattern-size"], Pl);
			t.setFillPatternSizeExpression(n), o = lh(e, "fill-pattern-", r, a, "v_patternSizePx");
		}
		r.functions.sampleFillPattern = "vec4 sampleFillPattern(sampler2D texture, vec2 textureSize, vec2 textureOffset, vec2 sampleSize, vec2 patternOriginPx, vec2 pxPosition, float sampleScaleRatio) {\n  vec2 pxRelativePos = pxPosition - patternOriginPx;\n\n  // rotate the relative position from origin by the current view rotation\n  pxRelativePos = vec2(pxRelativePos.x * cos(u_rotation) - pxRelativePos.y * sin(u_rotation), pxRelativePos.x * sin(u_rotation) + pxRelativePos.y * cos(u_rotation));\n  // sample position is computed according to the sample offset & size\n  vec2 samplePos = mod(pxRelativePos / sampleScaleRatio, sampleSize);\n  // also make sure that we're not sampling too close to the borders to avoid interpolation with outside pixels\n  samplePos = clamp(samplePos, vec2(0.5), sampleSize - vec2(0.5));\n  samplePos.y = sampleSize.y - samplePos.y; // invert y axis so that images appear upright\n  return texture2D(texture, (samplePos + textureOffset) / textureSize);\n}";
		let s = `u_texture${i}`, c = "1.";
		"fill-color" in e && (c = t.getFillColorExpression()), t.setFillColorExpression(`${c} * sampleFillPattern(${s}, ${a}, ${o}, v_patternSizePx, v_patternOriginPx, pxPos, df_float(u_df_patternScaleRatio))`);
	}
}
function hh(e, t, n, r) {
	function i(e, t, n) {
		try {
			Q(e, t, n);
		} catch {}
	}
	"text-value" in e && i(r, e["text-value"], G), "text-font" in e && i(r, e["text-font"], G), "text-max-angle" in e && i(r, e["text-max-angle"], W), "text-offset-x" in e && i(r, e["text-offset-x"], W), "text-offset-y" in e && i(r, e["text-offset-y"], W), "text-overflow" in e && i(r, e["text-overflow"], Nl), "text-placement" in e && i(r, e["text-placement"], G), "text-repeat" in e && i(r, e["text-repeat"], W), "text-scale" in e && i(r, e["text-scale"], Fl), "text-rotate-with-view" in e && i(r, e["text-rotate-with-view"], Nl), "text-rotation" in e && i(r, e["text-rotation"], W), "text-align" in e && i(r, e["text-align"], G), "text-justify" in e && i(r, e["text-justify"], G), "text-baseline" in e && i(r, e["text-baseline"], G), "text-padding" in e && i(r, e["text-padding"], Pl), "text-fill-color" in e && i(r, e["text-fill-color"], K), "text-stroke-color" in e && i(r, e["text-stroke-color"], K), "text-stroke-line-cap" in e && i(r, e["text-stroke-line-cap"], G), "text-stroke-line-join" in e && i(r, e["text-stroke-line-join"], G), "text-stroke-line-dash" in e && i(r, e["text-stroke-line-dash"], Pl), "text-stroke-line-dash-offset" in e && i(r, e["text-stroke-line-dash-offset"], W), "text-stroke-miter-limit" in e && i(r, e["text-stroke-miter-limit"], W), "text-stroke-width" in e && i(r, e["text-stroke-width"], W), "text-background-fill-color" in e && i(r, e["text-background-fill-color"], K), "text-background-stroke-color" in e && i(r, e["text-background-stroke-color"], K), "text-background-stroke-line-cap" in e && i(r, e["text-background-stroke-line-cap"], G), "text-background-stroke-line-join" in e && i(r, e["text-background-stroke-line-join"], G), "text-background-stroke-line-dash" in e && i(r, e["text-background-stroke-line-dash"], Pl), "text-background-stroke-line-dash-offset" in e && i(r, e["text-background-stroke-line-dash-offset"], W), "text-background-stroke-miter-limit" in e && i(r, e["text-background-stroke-miter-limit"], W), "text-background-stroke-width" in e && i(r, e["text-background-stroke-width"], W), "z-index" in e && i(r, e["z-index"], W);
}
function gh(e, t, n) {
	let r = _m(t), i = new Bm(), a = {};
	if ("icon-src" in e ? fh(e, i, a, r) : "shape-points" in e ? dh(e, i, a, r) : "circle-radius" in e && uh(e, i, a, r), ph(e, i, a, r), mh(e, i, a, r), hh(e, i, a, r), n) {
		let e = ql(t), a = Q(r, n, Nl, e);
		e.mCoordinate ? i.setFragmentDiscardExpression(`!${a}`) : i.setShapeDiscardExpression(`!${a}`);
	}
	let o = {};
	function s(e, t, n, a) {
		if (!r[e]) return;
		let s = Mm(n), c = jm(n);
		i.addAttribute(`a_${t}`, s), o[t] = {
			size: c,
			callback: a
		};
	}
	return s("geometryType", xm, G, (e) => {
		let t = e.getGeometry();
		return t ? mm(mu(t)) : 0;
	}), s("featureId", bm, G | W, (e) => {
		let t = e.getId() ?? null;
		return typeof t == "string" ? mm(t) : t;
	}), Nm(i, r), {
		builder: i,
		attributes: {
			...o,
			...Fm(r)
		},
		uniforms: {
			...a,
			...Pm(r, t)
		}
	};
}
// node_modules/ol/render/webgl/textUtil.js
var _h = {
	TEXT_OVERLAY_TEXTURE: "u_textOverlay",
	TEXT_OVERLAY_MATRIX: "u_textOverlayMatrix"
};
function vh(e) {
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
function yh(e, t) {
	let n = Cp();
	return {
		fragmentShader: `
      precision mediump float;

      uniform sampler2D u_image;
      uniform sampler2D ${_h.TEXT_OVERLAY_TEXTURE};
      uniform mat4 ${_h.TEXT_OVERLAY_MATRIX};

      varying vec2 v_texCoord;

      void main() {
        vec4 color = texture2D(u_image, v_texCoord);

        vec2 coords = v_texCoord * 2. - vec2(1.);
        coords = (${_h.TEXT_OVERLAY_MATRIX} * vec4(coords.xy, 0., 1.)).xy;
        coords = coords * 0.5 + vec2(0.5);
        float outOfBounds = clamp(step(1., coords.x) + step(1., coords.y) + step(0., -coords.x) + step(0., -coords.y), 0., 1.);

        vec4 textColor = texture2D(${_h.TEXT_OVERLAY_TEXTURE}, vec2(coords.x, 1. - coords.y));
        textColor.a *= 1. - outOfBounds; // if we're sampling out of the text overlay, make alpha 0 to avoid drawing anything

        gl_FragColor = textColor.a * textColor + (1. - textColor.a) * color;
      }`,
		uniforms: {
			[_h.TEXT_OVERLAY_TEXTURE]: e,
			[_h.TEXT_OVERLAY_MATRIX]: (r) => {
				let i = e(), a = t();
				if (!i || !a) return n;
				let o = a.viewState, s = r.viewState, c = s.center, l = s.resolution, u = s.rotation, d = r.size, f = o.center, p = o.resolution, m = o.rotation, h = i.width, g = i.height;
				return wp(n), Ep(n, 1 / p / (h / 2), 1 / p / (g / 2), 1, n), Op(n, m, n), Dp(n, c[0] - f[0], c[1] - f[1], 0, n), Op(n, -u, n), Ep(n, l * d[0] / 2, l * d[1] / 2, 1, n), n;
			}
		}
	};
}
new ra("Point", [0, 0], [], 2, {}, "dummy"), new TextDecoder();
// node_modules/ol/render/webgl/VectorStyleRenderer.js
var bh = [], xh;
function Sh() {
	return xh ||= qm(), xh;
}
var Ch = 0;
function wh(e, t, n) {
	let r = Ch++;
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
var Th = {
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
}, Eh = class extends c {
	constructor(e, t, n, r) {
		super(), this.helper_, this.hitDetectionEnabled_ = !!r, this.flatStyle = Dh(e), this.styleShaders = Oh(e, t), this.customAttributes_ = {}, this.uniforms_ = {}, this.hitDetectionEnabled_ && (this.customAttributes_.hitColor = {
			callback() {
				return Xm(this.ref ?? 0, bh);
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
				type: Jp.FLOAT
			}));
			return e.builder.getFillVertexShader() && (t.fillRenderPass = {
				vertexShader: e.builder.getFillVertexShader(),
				fragmentShader: e.builder.getFillFragmentShader(),
				attributesDesc: [{
					name: Th.POSITION,
					size: 2,
					type: Jp.FLOAT
				}, ...n],
				instancedAttributesDesc: [],
				instancePrimitiveVertexCount: 3
			}), e.builder.getStrokeVertexShader() && (t.strokeRenderPass = {
				vertexShader: e.builder.getStrokeVertexShader(),
				fragmentShader: e.builder.getStrokeFragmentShader(),
				attributesDesc: [{
					name: Th.LOCAL_POSITION,
					size: 2,
					type: Jp.FLOAT
				}],
				instancedAttributesDesc: [
					{
						name: Th.SEGMENT_START,
						size: 2,
						type: Jp.FLOAT
					},
					{
						name: Th.MEASURE_START,
						size: 1,
						type: Jp.FLOAT
					},
					{
						name: Th.SEGMENT_END,
						size: 2,
						type: Jp.FLOAT
					},
					{
						name: Th.MEASURE_END,
						size: 1,
						type: Jp.FLOAT
					},
					{
						name: Th.JOIN_ANGLES,
						size: 2,
						type: Jp.FLOAT
					},
					{
						name: Th.DISTANCE_LOW,
						size: 1,
						type: Jp.FLOAT
					},
					{
						name: Th.DISTANCE_HIGH,
						size: 1,
						type: Jp.FLOAT
					},
					{
						name: Th.ANGLE_TANGENT_SUM,
						size: 1,
						type: Jp.FLOAT
					},
					...n
				],
				instancePrimitiveVertexCount: 6
			}), e.builder.getSymbolVertexShader() && (t.symbolRenderPass = {
				vertexShader: e.builder.getSymbolVertexShader(),
				fragmentShader: e.builder.getSymbolFragmentShader(),
				attributesDesc: [{
					name: Th.LOCAL_POSITION,
					size: 2,
					type: Jp.FLOAT
				}],
				instancedAttributesDesc: [{
					name: Th.POSITION,
					size: 2,
					type: Jp.FLOAT
				}, ...n],
				instancePrimitiveVertexCount: 6
			}), t;
		}), this.hasFill_ = this.renderPasses_.some((e) => e.fillRenderPass), this.hasStroke_ = this.renderPasses_.some((e) => e.strokeRenderPass), this.hasSymbol_ = this.renderPasses_.some((e) => e.symbolRenderPass), this.hasText_ = this.flatStyle && vh(this.flatStyle), this.hasText_ && (this.textOverlayCanvas_ = _e().canvas, this.textOverlayContext_ = this.textOverlayCanvas_.getContext("2d"), this.textOverlayRenderFrameState_ = null, this.textOverlayWorker_ = import("./textOverlay-2eG4Edfr.js").then(({ create: e }) => e()), this.textOverlayRenderList_ = /* @__PURE__ */ new Set()), this.setHelper(n);
	}
	async generateBuffers(e, t, n) {
		let r = Ir(Or(), t);
		if (e.isEmpty()) return {
			polygonBuffers: null,
			lineStringBuffers: null,
			pointBuffers: null,
			invertVerticesTransform: r,
			textInstructionsKey: null
		};
		let i = new Km(), a = this.generateRenderInstructions_(e, i, t), [o, s, c, l] = await Promise.all([
			this.hasText_ ? this.generateTextInstructions_(a, i, t, n) : null,
			this.hasFill_ ? this.generateBuffersForType_(a.polygonInstructions, "Polygon", t) : null,
			this.hasStroke_ ? this.generateBuffersForType_(a.lineStringInstructions, "LineString", t) : null,
			this.hasSymbol_ ? this.generateBuffersForType_(a.pointInstructions, "Point", t) : null
		]);
		return {
			polygonBuffers: s ?? null,
			lineStringBuffers: c ?? null,
			pointBuffers: l ?? null,
			invertVerticesTransform: r,
			textInstructionsKey: o
		};
	}
	generateRenderInstructions_(e, t, n) {
		return {
			polygonInstructions: this.hasFill_ || this.hasText_ ? nh(e.polygonBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null,
			lineStringInstructions: this.hasStroke_ || this.hasText_ ? th(e.lineStringBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null,
			pointInstructions: this.hasSymbol_ || this.hasText_ ? eh(e.pointBatch, /* @__PURE__ */ new Float32Array(), t, this.customAttributes_, n) : null
		};
	}
	generateBuffersForType_(e, t, n) {
		if (e === null) return null;
		let r;
		switch (t) {
			case "Polygon":
				r = Jm.GENERATE_POLYGON_BUFFERS;
				break;
			case "LineString":
				r = Jm.GENERATE_LINE_STRING_BUFFERS;
				break;
			case "Point":
				r = Jm.GENERATE_POINT_BUFFERS;
				break;
			default: return null;
		}
		let i = {
			type: r,
			renderInstructions: e.buffer,
			renderInstructionsTransform: n,
			customAttributesSize: $m(this.customAttributes_)
		};
		return wh(Sh(), i, [e.buffer]).then((e) => {
			if (!this.helper_.getGL()) return;
			let t = e;
			if (!t.indicesBuffer || !t.vertexAttributesBuffer || !t.instanceAttributesBuffer) return;
			let n = new Vp(Ap, Np).fromArrayBuffer(t.indicesBuffer), r = new Vp(kp, Np).fromArrayBuffer(t.vertexAttributesBuffer), i = new Vp(kp, Np).fromArrayBuffer(t.instanceAttributesBuffer);
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
			type: Ym.BUILD_INSTRUCTIONS,
			polygonRenderInstructions: a ?? void 0,
			lineStringRenderInstructions: o ?? void 0,
			pointRenderInstructions: s ?? void 0,
			labelsArray: t.getArray(),
			style: this.flatStyle ?? void 0,
			customAttributesSizes: c,
			renderInstructionsTransform: n,
			resolution: r
		};
		return this.textOverlayWorker_.then((e) => wh(e, l, i)).then((e) => e.instructionsSetKey ?? null);
	}
	render(e, t, n) {
		for (let r of this.renderPasses_) r.fillRenderPass && e.polygonBuffers && this.renderInternal_(e.polygonBuffers[0], e.polygonBuffers[1], e.polygonBuffers[2], r.fillRenderPass, t, n), r.strokeRenderPass && e.lineStringBuffers && this.renderInternal_(e.lineStringBuffers[0], e.lineStringBuffers[1], e.lineStringBuffers[2], r.strokeRenderPass, t, n), r.symbolRenderPass && e.pointBuffers && this.renderInternal_(e.pointBuffers[0], e.pointBuffers[1], e.pointBuffers[2], r.symbolRenderPass, t, n);
		e.textInstructionsKey && this.renderText_(e);
	}
	renderInternal_(e, t, n, r, i, a) {
		let o = e.getSize();
		if (o === 0) return;
		let s = r.instancedAttributesDesc.length, c = r.program;
		if (c) {
			if (this.helper_.useProgram(c, i), this.helper_.bindBuffer(t), this.helper_.bindBuffer(e), this.helper_.enableAttributes(r.attributesDesc), this.helper_.bindBuffer(n), this.helper_.enableAttributesInstanced(r.instancedAttributesDesc), a(), s) {
				let e = r.instancedAttributesDesc.reduce((e, t) => e + (t.size || 1), 0), t = n.getSize() / e;
				this.helper_.drawElementsInstanced(0, o, t);
			} else this.helper_.drawElements(0, o);
		}
	}
	renderText_(e) {
		let t = e.textInstructionsKey;
		t && this.textOverlayRenderList_.add(t);
	}
	finalizeTextRender(e) {
		if (!this.hasText_) return Promise.resolve();
		let t = this.textOverlayCanvas_, n = this.textOverlayContext_, r = {
			type: Ym.RENDER,
			frameState: rh(e),
			batchesToRender: this.textOverlayRenderList_
		};
		return this.textOverlayWorker_.then((e) => wh(e, r)).then((e) => {
			let r = e;
			if (r.imageData) {
				this.textOverlayRenderFrameState_ = r.frameState === void 0 ? null : r.frameState;
				let e = r.imageData;
				e.width !== t.width || e.height !== t.height ? (t.width = e.width, t.height = e.height) : n.clearRect(0, 0, t.width, t.height), n.drawImage(e, 0, 0), e.close();
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
		this.textOverlayWorker_?.then((t) => t.postMessage({
			type: Ym.DISPOSE_INSTRUCTIONS,
			instructionsSetKey: e
		}));
	}
	disposeInternal() {
		this.textOverlayWorker_?.then((e) => e.terminate()), super.disposeInternal();
	}
};
function Dh(e) {
	if (Array.isArray(e)) return e.some((e) => "builder" in e && !("sourceRule" in e)) ? null : e.some((e) => "builder" in e) ? e.flatMap((e) => {
		let t = e.sourceRule;
		return t ? [t] : [];
	}) : e;
	if ("builder" in e) {
		if (!("sourceRule" in e)) return null;
		let t = e.sourceRule;
		return t ? [t] : null;
	}
	return e;
}
function Oh(e, t) {
	let n = Array.isArray(e) ? e : [e];
	if ("style" in n[0]) {
		let e = [], r = n, i = [];
		for (let n of r) {
			let r = Array.isArray(n.style) ? n.style : [n.style], a = n.filter;
			n.else && i.length && (a = ["all", ...i.map((e) => ["!", e])], n.filter && a.push(n.filter), a.length < 3 && (a = a[1])), n.filter && i.push(n.filter);
			let o = r.map((e) => ({
				...gh(e, t, a),
				sourceRule: n
			}));
			e.push(...o);
		}
		return e;
	}
	return "builder" in n[0] ? n : n.map((e) => ({
		...gh(e, t, void 0),
		sourceRule: { style: e }
	}));
}
// node_modules/ol/webgl/RenderTarget.js
var kh = /* @__PURE__ */ new Uint8Array(4), Ah = class {
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
		if (e < 0 || t < 0 || e > this.size_[0] || t >= this.size_[1]) return kh[0] = 0, kh[1] = 0, kh[2] = 0, kh[3] = 0, kh;
		this.readAll();
		let n = Math.floor(e) + (this.size_[1] - Math.floor(t) - 1) * this.size_[0];
		return kh[0] = this.data_[n * 4], kh[1] = this.data_[n * 4 + 1], kh[2] = this.data_[n * 4 + 2], kh[3] = this.data_[n * 4 + 3], kh;
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
}, jh = {
	PATTERN_ORIGIN_X_DOUBLE: "u_df_patternOriginX",
	PATTERN_ORIGIN_Y_DOUBLE: "u_df_patternOriginY",
	PATTERN_SCALE_RATIO_DOUBLE: "u_df_patternScaleRatio",
	ONE: "u_one"
}, Mh = [0, 0], Nh = [0, 0], Ph = Or(), Fh = Cp();
function Ih(e, t, n, r) {
	Mr(Ph, t), Ar(Ph, n), e.setUniformMatrixValue(qp.PROJECTION_MATRIX, Tp(Fh, Ph)), Ir(Ph, Ph), e.setUniformMatrixValue(qp.INVERT_PROJECTION_MATRIX, Tp(Fh, Ph)), Mh[0] = 0, Mh[1] = 0;
	let i = r.size, a = r.viewState.resolution, o = r.viewState.center;
	Fr(Ph, i[0] / 2, i[1] / 2, 1 / a, 1 / a, 0, -o[0], -o[1]), R(Ph, Mh), Nh[0] = Lm(Mh[0]), Nh[1] = Im(Mh[0]), e.setUniformFloatVec2(jh.PATTERN_ORIGIN_X_DOUBLE, Nh), Nh[0] = Lm(Mh[1]), Nh[1] = Im(Mh[1]), e.setUniformFloatVec2(jh.PATTERN_ORIGIN_Y_DOUBLE, Nh);
	let s = 2 ** ((r.viewState.zoom + .5) % 1 - .5);
	Mh[0] = Lm(s), Mh[1] = Im(s), e.setUniformFloatVec2(jh.PATTERN_SCALE_RATIO_DOUBLE, Mh);
}
// node_modules/ol/renderer/webgl/worldUtil.js
function Lh(e, t) {
	let n = e.viewState.projection, r = t.getSource().getWrapX() && n.canWrapX(), i = n.getExtent(), a = e.extent, o = r ? I(i) : 0, s = r ? Math.ceil((a[2] - i[2]) / o) + 1 : 1;
	return [
		r ? Math.floor((a[0] - i[0]) / o) : 0,
		s,
		r ? o : 0
	];
}
// node_modules/ol/renderer/webgl/VectorLayer.js
var Rh = {
	...qp,
	...jh,
	..._h,
	RENDER_EXTENT: "u_renderExtent",
	GLOBAL_ALPHA: "u_globalAlpha"
}, zh = class extends im {
	constructor(e, t) {
		let n = {
			[Rh.RENDER_EXTENT]: [
				0,
				0,
				0,
				0
			],
			[Rh.GLOBAL_ALPHA]: 1,
			[Rh.ONE]: 1
		};
		super(e, {
			uniforms: n,
			postProcesses: t.postProcesses ?? []
		}), this.hitDetectionEnabled_ = !t.disableHitDetection, this.hitRenderTarget_, this.sourceRevision_ = -1, this.layerRevision_ = -1, this.skipNextTextRender_ = !1, this.previousExtent_ = Je(), this.currentTransform_ = Or(), this.currentFrameStateTransform_ = Or(), this.styleVariables_ = {}, this.style_ = [], this.hasText_ = !1, this.styleRenderer_ = null, this.buffers_ = null, this.batch_ = new Vm(), this.initialFeaturesAdded_ = !1, this.sourceListenKeys_ = null, this.applyOptions_(t);
	}
	addInitialFeatures_(e) {
		let t = this.getLayer().getSource();
		if (!t) return;
		let n = yr(), r;
		n && (r = pr(n, e.viewState.projection) ?? void 0), this.batch_.addFeatures(t.getFeatures(), r), this.sourceListenKeys_ = [
			i(t, Vf.ADDFEATURE, this.handleSourceFeatureAdded_.bind(this, r)),
			i(t, Vf.CHANGEFEATURE, this.handleSourceFeatureChanged_.bind(this, r), this),
			i(t, Vf.REMOVEFEATURE, this.handleSourceFeatureDelete_, this),
			i(t, Vf.CLEAR, this.handleSourceFeatureClear_, this)
		];
	}
	applyOptions_(e) {
		this.styleVariables_ = e.variables, this.style_ = e.style;
		let t = Dh(this.style_), n = !!t && vh(t);
		n && !this.hasText_ ? this.setPostProcesses([yh(() => this.styleRenderer_.getTextOverlayCanvas(), () => this.styleRenderer_.getTextOverlayFrameState()), ...this.getPostProcesses()]) : !n && this.hasText_ && this.setPostProcesses(this.getPostProcesses().slice(1)), this.hasText_ = n;
	}
	createRenderers_() {
		this.buffers_ = null, this.styleRenderer_ = new Eh(this.style_, this.styleVariables_, this.helper, this.hitDetectionEnabled_);
	}
	reset(e) {
		this.applyOptions_(e), this.helper && this.createRenderers_(), super.reset(e);
	}
	afterHelperCreated() {
		this.styleRenderer_ ? this.styleRenderer_.setHelper(this.helper, this.buffers_) : this.createRenderers_(), this.hitDetectionEnabled_ && (this.hitRenderTarget_ = new Ah(this.helper));
	}
	handleSourceFeatureAdded_(e, t) {
		let n = t.feature;
		n && this.batch_.addFeature(n, e);
	}
	handleSourceFeatureChanged_(e, t) {
		let n = t.feature;
		n && this.batch_.changeFeature(n, e);
	}
	handleSourceFeatureDelete_(e) {
		let t = e.feature;
		t && this.batch_.removeFeature(t);
	}
	handleSourceFeatureClear_() {
		this.batch_.clear();
	}
	applyUniforms_(e, t) {
		Ih(this.helper, this.currentFrameStateTransform_, e, t);
	}
	renderFrame(e) {
		let t = this.helper.getGL();
		this.preRender(t, e);
		let n = this.getLayer(), [r, i, a] = Lh(e, n);
		this.helper.prepareDraw(e), this.renderWorlds(e, !1, r, i, a), this.hasText_ && this.styleRenderer_?.finalizeTextRender(e).then(() => {
			this.skipNextTextRender_ ? this.skipNextTextRender_ = !1 : (this.skipNextTextRender_ = !0, this.layerRevision_++, n.changed());
		}), this.helper.finalizeDraw(e, this.dispatchPreComposeEvent, this.dispatchPostComposeEvent);
		let o = this.helper.getCanvas();
		return this.hitDetectionEnabled_ && this.hitRenderTarget_ && (this.renderWorlds(e, !0, r, i, a), this.hitRenderTarget_.clearCachedData()), this.postRender(t, e), o;
	}
	prepareFrameInternal(e) {
		this.initialFeaturesAdded_ ||= (this.addInitialFeatures_(e), !0);
		let t = this.getLayer(), n = t.getSource();
		if (!n) return !0;
		let r = e.viewState, i = !e.viewHints[V.ANIMATING] && !e.viewHints[V.INTERACTING], a = e.extent;
		if (!a) return !0;
		let o = !$e(this.previousExtent_, a), s = this.sourceRevision_ < n.getRevision(), c = this.layerRevision_ < t.getRevision();
		if (this.sourceRevision_ = n.getRevision(), this.layerRevision_ = t.getRevision(), (c || o || s) && (this.skipNextTextRender_ = !1), i && (o || s)) {
			let i = r.projection, o = r.resolution, s = Ve(a, ((t instanceof Ed ? t.getRenderBuffer() : 0) ?? 0) * o), c = yr();
			c ? n.loadFeatures(Sr(s, c), wr(o, i), c) : n.loadFeatures(s, o, i), this.ready = !1;
			let l = this.styleRenderer_;
			if (!l) return !0;
			let u = this.helper.makeProjectionTransform(e, Or(), !0);
			l.generateBuffers(this.batch_, u, e.viewState.resolution).then((e) => {
				this.buffers_ && this.disposeBuffers(this.buffers_), this.buffers_ = e, this.ready = !0, this.getLayer()?.changed();
			}), this.previousExtent_ = a.slice();
		}
		return !0;
	}
	renderWorlds(e, t, n, r, i) {
		let a = n;
		if (t) {
			let t = this.hitRenderTarget_;
			if (!t) return;
			t.setSize([Math.floor(e.size[0] / 2), Math.floor(e.size[1] / 2)]), this.helper.prepareDrawToRenderTarget(e, t, !0);
		}
		do {
			this.helper.makeProjectionTransform(e, this.currentFrameStateTransform_), Pr(this.currentFrameStateTransform_, a * i, 0);
			let n = this.buffers_;
			if (!n) continue;
			let r = this.styleRenderer_;
			r && r.render(n, e, () => {
				this.applyUniforms_(n.invertVerticesTransform, e), this.helper.applyHitDetectionUniform(t);
			});
		} while (++a < r);
	}
	forEachFeatureAtCoordinate(e, t, n, r, i) {
		if (L(this.hitDetectionEnabled_, "`forEachFeatureAtCoordinate` cannot be used on a WebGL layer if the hit detection logic has been disabled using the `disableHitDetection: true` option."), !this.styleRenderer_ || !this.hitDetectionEnabled_) return;
		let a = R(t.coordinateToPixelTransform, e.slice()), o = this.hitRenderTarget_?.readPixel(a[0] / 2, a[1] / 2);
		if (!o) return;
		let s = Zm([
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
		e.pointBuffers && t(e.pointBuffers), e.lineStringBuffers && t(e.lineStringBuffers), e.polygonBuffers && t(e.polygonBuffers), e.textInstructionsKey && this.styleRenderer_?.disposeTextInstructions(e.textInstructionsKey);
	}
	disposeInternal() {
		this.buffers_ && this.disposeBuffers(this.buffers_), this.sourceListenKeys_ &&= (this.sourceListenKeys_.forEach(function(e) {
			e && o(e);
		}), null), this.styleRenderer_ && this.styleRenderer_.dispose(), super.disposeInternal();
	}
	renderDeclutter() {}
}, Bh = {
	BLUR: "blur",
	GRADIENT: "gradient",
	RADIUS: "radius"
}, Vh = [
	"#00f",
	"#0ff",
	"#0f0",
	"#ff0",
	"#f00"
], Hh = class extends Ed {
	constructor(e) {
		e ||= {};
		let t = Object.assign({}, e);
		delete t.gradient, delete t.radius, delete t.blur, delete t.weight, super(t), this.on, this.once, this.un, this.filter_ = e.filter ?? !0, this.styleVariables_ = e.variables || {}, this.gradient_ = null, this.addChangeListener(Bh.GRADIENT, this.handleGradientChanged_), this.setGradient(e.gradient ? e.gradient : Vh), this.setBlur(e.blur === void 0 ? 15 : e.blur), this.setRadius(e.radius === void 0 ? 8 : e.radius);
		let n = e.weight ? e.weight : "weight";
		this.weight_ = n, this.setRenderOrder(null);
	}
	getBlur() {
		return this.get(Bh.BLUR);
	}
	getGradient() {
		return this.get(Bh.GRADIENT);
	}
	getRadius() {
		return this.get(Bh.RADIUS);
	}
	handleGradientChanged_() {
		this.gradient_ = Uh(this.getGradient());
	}
	setBlur(e) {
		let t = this.get(Bh.BLUR);
		this.set(Bh.BLUR, e), typeof e == "number" && typeof t == "number" ? this.changed() : this.clearRenderer();
	}
	setGradient(e) {
		this.set(Bh.GRADIENT, e);
	}
	setRadius(e) {
		let t = this.get(Bh.RADIUS);
		this.set(Bh.RADIUS, e), typeof e == "number" && typeof t == "number" ? this.changed() : this.clearRenderer();
	}
	setFilter(e) {
		this.filter_ = e, this.changed(), this.clearRenderer();
	}
	setWeight(e) {
		this.weight_ = e, this.changed(), this.clearRenderer();
	}
	createRenderer() {
		let e = new Bm(), t = _m(this.styleVariables_), n = ql(this.styleVariables_), r = Om(t, this.filter_, Nl, n), i = Om(t, this.getRadius(), W), a = Om(t, this.getBlur(), W), o = {};
		typeof this.getBlur() == "number" && (a = "a_blur", o.a_blur = () => this.getBlur(), e.addUniform("a_blur", "float")), typeof this.getRadius() == "number" && (i = "a_radius", o.a_radius = () => this.getRadius(), e.addUniform("a_radius", "float"));
		let s = {}, c = null;
		if (typeof this.weight_ == "string" || typeof this.weight_ == "function") {
			let t = typeof this.weight_ == "string" ? ((e) => e.get(this.weight_)) : this.weight_;
			s.prop_weight = {
				size: 1,
				callback: (e) => {
					let n = t(e);
					return n === void 0 ? 1 : kt(n, 0, 1);
				}
			}, c = "a_prop_weight", e.addAttribute("a_prop_weight", "float");
		} else c = Om(t, [
			"clamp",
			this.weight_,
			0,
			1
		], W);
		let l = `(${i} / max(1., ${a}))`;
		e.setSymbolSizeExpression(`vec2(${i} + ${a}) * 2.`).setSymbolColorExpression(`vec4(smoothstep(0., 1., (1. - length(coordsPx * 2. / v_quadSizePx)) * ${l}) * ${c})`).setStrokeColorExpression(`vec4(smoothstep(0., 1., (1. - length(currentRadiusPx * 2. / v_width)) * ${l}) * ${c})`).setStrokeWidthExpression(`(${i} + ${a}) * 2.`).setFillColorExpression(`vec4(${c})`), n.mCoordinate ? e.setFragmentDiscardExpression(`!${r}`) : e.setShapeDiscardExpression(`!${r}`), Nm(e, t);
		let u = Fm(t), d = Pm(t, this.styleVariables_);
		return new zh(this, {
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
function Uh(e) {
	let t = _e(1, 256), n = t.createLinearGradient(0, 0, 1, 256), r = 1 / (e.length - 1);
	for (let t = 0, i = e.length; t < i; ++t) n.addColorStop(t * r, e[t]);
	return t.fillStyle = n, t.fillRect(0, 0, 1, 256), t.canvas;
}
// node_modules/ol-ext/util/input/Base.js
var Wh = class extends j {
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
}, Gh = class extends Wh {
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
}, Kh = class extends Gh {
	constructor(e) {
		e ||= {}, super(e), this.element.className = ("ol-ext-toggle-switch " + (e.className || "")).trim();
	}
}, qh = class extends Gh {
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
	return new Kh(Object.assign({ input: t }, e || {})), t;
}, $.createCheck = function(e) {
	var t = $.create("INPUT", {
		name: e.name,
		type: e.type === "radio" ? "radio" : "checkbox",
		on: e.on,
		parent: e.parent
	}), n = Object.assign({ input: t }, e || {});
	return e.type === "radio" ? new qh(n) : new Gh(n), t;
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
	e instanceof Mf && (e = e.getTargetElement()), !("ontouchstart" in window) && e instanceof Element && (e.style.cursor = t);
};
// node_modules/ol-ext/control/LayerSwitcher.js
var Jh = class extends ke {
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
		if (super.setMap(e), this.drawPanel(), this._listener) for (var t in this._listener) k(this._listener[t]);
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
				k(e);
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
		return !r || _t(this.getMap().getView().calculateExtent(this.getMap().getSize()), r);
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
				d && d.classList.contains("dropover") || n.parentNode.querySelectorAll("li").forEach(function(e) {
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
			if (!this.displayInLayerSwitcher(c)) this._setLayerForLI(null, c);
			else {
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
				if (this.show_progress && c instanceof Ss) {
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
		}
		for (var l = r.length - 1; l >= 0; l--) c.call(this, r[l]);
		this.viewChange(), e === this.panel_ && (this.overflow(""), this._focus = null);
	}
	getLayerClass(e) {
		return e ? e.getLayers ? "ol-layer-group" : e instanceof Od ? "ol-layer-vector" : e instanceof yp ? "ol-layer-vectortile" : e instanceof Ss ? "ol-layer-tile" : e instanceof Sp ? "ol-layer-image" : e instanceof Hh ? "ol-layer-heatmap" : e.getFeatures ? "ol-layer-vectorimage" : "unknown" : "none";
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
Jh.prototype.tip = {
	up: "up/down",
	down: "down",
	info: "informations...",
	extent: "zoom to extent",
	trash: "remove layer",
	plus: "expand/shrink"
};
// src/api/tileserver.js
async function Yh() {
	let e = await fetch("/tileserver/session_id");
	if (!e.ok) throw Error("Failed to create TileServer session.");
	return (await e.json()).session_id;
}
async function Xh(e) {
	let t = new FormData();
	if (t.append("slide_path", e), !(await fetch("/tileserver/slide", {
		method: "PUT",
		body: t
	})).ok) throw Error(`Failed to load slide: ${e}`);
	let n = await fetch("/tileserver/slide");
	if (!n.ok) throw Error("Failed to retrieve slide metadata.");
	return n.json();
}
async function Zh(e) {
	let t = await fetch(`/tileserver/files/${e}`);
	if (!t.ok) throw Error(`Failed to get configured ${e} files.`);
	return t.json();
}
async function Qh() {
	if (!(await fetch("/tileserver/clear_overlays", { method: "PUT" })).ok) throw Error("Failed to clear overlays.");
}
async function $h() {
	if (!(await fetch("/tileserver/slide", { method: "DELETE" })).ok) throw Error("Failed to remove the current slide.");
}
async function eg(e, t) {
	let n = new FormData();
	n.append("overlay_path", e), n.append("layer_name", t);
	let r = await fetch("/tileserver/overlay", {
		method: "PUT",
		body: n
	});
	if (!r.ok) throw Error(`Failed to load overlay: ${e}`);
	return r.json();
}
async function tg(e) {
	if (!(await fetch(`/tileserver/overlay/${encodeURIComponent(e)}`, { method: "DELETE" })).ok) throw Error(`Failed to remove overlay: ${e}`);
}
function ng(e, t) {
	return t === null ? e : `${e}?${new URLSearchParams({ layer: t })}`;
}
async function rg(e, t = null) {
	let n = new FormData();
	if (n.append("val", JSON.stringify(e)), !(await fetch(ng("/tileserver/renderer/where", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation visibility.");
}
async function ig(e, t = null) {
	let n = e instanceof Map ? [...e.entries()] : Object.entries(e), r = new FormData();
	if (r.append("cmap", JSON.stringify({
		keys: n.map(([e]) => e),
		values: n.map(([, e]) => e)
	})), !(await fetch(ng("/tileserver/cmap", t), {
		method: "PUT",
		body: r
	})).ok) throw Error("Failed to update annotation colours.");
}
async function ag(e, t = null) {
	let n = e instanceof Map ? [...e.entries()] : Object.entries(e), r = new FormData();
	if (r.append("opacities", JSON.stringify({
		keys: n.map(([e]) => e),
		values: n.map(([, e]) => e)
	})), !(await fetch(ng("/tileserver/annotation_opacities", t), {
		method: "PUT",
		body: r
	})).ok) throw Error("Failed to update annotation opacity.");
}
async function og(e, t = null) {
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
async function sg(e) {
	let t = new URLSearchParams({ layer: e }), n = await fetch(`/tileserver/prop_names/all?${t}`);
	if (!n.ok) throw Error("Failed to get annotation properties.");
	return n.json();
}
async function cg(e, t) {
	let n = new URLSearchParams({ layer: e }), r = await fetch(`/tileserver/prop_values/${encodeURIComponent(t)}/all?${n}`);
	if (!r.ok) throw Error("Failed to get annotation property values.");
	return r.json();
}
async function lg(e, t, n, { details: r = !1 } = {}) {
	let i = new URLSearchParams({ layer: e });
	r && i.set("details", "1");
	let a = await fetch(`/tileserver/tap_query/${t}/${n}?${i}`);
	if (!a.ok) throw Error("Failed to inspect annotation.");
	return a.json();
}
async function ug(e, t = null) {
	let n = new FormData();
	if (n.append("val", JSON.stringify(e)), !(await fetch(ng("/tileserver/renderer/score_prop", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation property.");
}
async function dg(e, t = null) {
	let n = new FormData();
	if (n.append("cmap", JSON.stringify(e)), !(await fetch(ng("/tileserver/cmap", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation colour map.");
}
async function fg(e, t = null) {
	let n = new FormData();
	if (n.append("range", JSON.stringify(e)), !(await fetch(ng("/tileserver/prop_range", t), {
		method: "PUT",
		body: n
	})).ok) throw Error("Failed to update annotation property range.");
}
async function pg(e, t, n, r, i = null) {
	let a = new FormData();
	if (a.append("type_id", JSON.stringify(e)), a.append("prop", t), a.append("cmap", JSON.stringify(n)), a.append("range", JSON.stringify(r)), !(await fetch(ng("/tileserver/secondary_cmap", i), {
		method: "PUT",
		body: a
	})).ok) throw Error("Failed to update secondary annotation colour map.");
}
async function mg(e = null) {
	let t = new FormData();
	if (t.append("val", JSON.stringify(null)), !(await fetch(ng("/tileserver/renderer/secondary_cmap", e), {
		method: "PUT",
		body: t
	})).ok) throw Error("Failed to clear secondary annotation colour map.");
}
// node_modules/ol/control/FullScreen.js
var hg = ["fullscreenchange", "webkitfullscreenchange"], gg = {
	ENTERFULLSCREEN: "enterfullscreen",
	LEAVEFULLSCREEN: "leavefullscreen"
}, _g = class extends ke {
	constructor(e) {
		e ||= {}, super({
			element: document.createElement("div"),
			target: e.target
		}), this.on, this.once, this.un, this.keys_ = e.keys !== void 0 && e.keys, this.source_ = e.source, this.isInFullscreen_ = !1, this.boundHandleMapTargetChange_ = this.handleMapTargetChange_.bind(this), this.cssClassName_ = e.className === void 0 ? "ol-full-screen" : e.className, this.documentListeners_ = [], this.activeClassName_ = e.activeClassName === void 0 ? [this.cssClassName_ + "-true"] : e.activeClassName.split(" "), this.inactiveClassName_ = e.inactiveClassName === void 0 ? [this.cssClassName_ + "-false"] : e.inactiveClassName.split(" ");
		let t = e.label === void 0 ? "⛶" : e.label;
		this.labelNode_ = typeof t == "string" ? document.createTextNode(t) : t;
		let n = e.labelActive === void 0 ? "×" : e.labelActive;
		this.labelActiveNode_ = typeof n == "string" ? document.createTextNode(n) : n;
		let r = e.tipLabel ? e.tipLabel : "Toggle full-screen";
		this.button_ = document.createElement("button"), this.button_.title = r, this.button_.setAttribute("type", "button"), this.button_.appendChild(this.labelNode_), this.button_.addEventListener(s.CLICK, this.handleClick_.bind(this), !1), this.setClassName_(this.button_, this.isInFullscreen_);
		let i = this.element;
		i.className = `${this.cssClassName_} ${te} ${re}`, i.appendChild(this.button_);
	}
	handleClick_(e) {
		e.preventDefault(), this.handleFullScreen_();
	}
	handleFullScreen_() {
		let e = this.getMap();
		if (!e) return;
		let t = e.getOwnerDocument();
		if (vg(t)) {
			if (yg(t)) Sg(t);
			else {
				let n;
				if (n = this.source_ ? typeof this.source_ == "string" ? t.getElementById(this.source_) : this.source_ : e.getTargetElement(), !n) return;
				this.keys_ ? xg(n) : bg(n);
			}
		}
	}
	handleFullScreenChange_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.isInFullscreen_;
		this.isInFullscreen_ = yg(e.getOwnerDocument()), t !== this.isInFullscreen_ && (this.setClassName_(this.button_, this.isInFullscreen_), this.isInFullscreen_ ? (Ce(this.labelActiveNode_, this.labelNode_), this.dispatchEvent(gg.ENTERFULLSCREEN)) : (Ce(this.labelNode_, this.labelActiveNode_), this.dispatchEvent(gg.LEAVEFULLSCREEN)), e.updateSize());
	}
	setClassName_(e, t) {
		t ? (e.classList.remove(...this.inactiveClassName_), e.classList.add(...this.activeClassName_)) : (e.classList.remove(...this.activeClassName_), e.classList.add(...this.inactiveClassName_));
	}
	setMap(e) {
		let t = this.getMap();
		t && t.removeChangeListener(Pd.TARGET, this.boundHandleMapTargetChange_), super.setMap(e), this.handleMapTargetChange_(), e && e.addChangeListener(Pd.TARGET, this.boundHandleMapTargetChange_);
	}
	handleMapTargetChange_() {
		let e = this.documentListeners_;
		for (let t = 0, n = e.length; t < n; ++t) o(e[t]);
		e.length = 0;
		let t = this.getMap();
		if (t) {
			let n = t.getOwnerDocument();
			vg(n) ? this.element.classList.remove(ne) : this.element.classList.add(ne);
			for (let t = 0, r = hg.length; t < r; ++t) e.push(i(n, hg[t], this.handleFullScreenChange_, this));
			this.handleFullScreenChange_();
		}
	}
};
function vg(e) {
	let t = e.body;
	return !!(t.webkitRequestFullscreen || "requestFullscreen" in t && e.fullscreenEnabled);
}
function yg(e) {
	return !!(e.webkitIsFullScreen || e.fullscreenElement);
}
function bg(e) {
	let t = e;
	e.requestFullscreen ? e.requestFullscreen() : t.webkitRequestFullscreen && t.webkitRequestFullscreen();
}
function xg(e) {
	let t = e;
	t.webkitRequestFullscreen ? t.webkitRequestFullscreen() : bg(e);
}
function Sg(e) {
	let t = e;
	e.exitFullscreen ? e.exitFullscreen() : t.webkitExitFullscreen && t.webkitExitFullscreen();
}
// node_modules/ol/control/MousePosition.js
var Cg = "projection", wg = "coordinateFormat", Tg = class extends ke {
	constructor(e) {
		e ||= {};
		let t = document.createElement("div");
		t.className = e.className === void 0 ? "ol-mouse-position" : e.className, super({
			element: t,
			render: e.render,
			target: e.target
		}), this.on, this.once, this.un, this.addChangeListener(Cg, this.handleProjectionChanged_), e.coordinateFormat && this.setCoordinateFormat(e.coordinateFormat), e.projection && this.setProjection(e.projection), this.renderOnMouseOut_ = e.placeholder !== void 0, this.placeholder_ = this.renderOnMouseOut_ ? e.placeholder : "&#160;", this.renderedHTML_ = t.innerHTML, this.mapProjection_ = null, this.transform_ = null, this.wrapX_ = e.wrapX !== !1;
	}
	handleProjectionChanged_() {
		this.transform_ = null;
	}
	getCoordinateFormat() {
		return this.get(wg);
	}
	getProjection() {
		return this.get(Cg);
	}
	handleMouseMove(e) {
		let t = this.getMap();
		t && this.updateHTML_(t.getEventPixel(e));
	}
	handleMouseOut(e) {
		this.updateHTML_(null);
	}
	setMap(e) {
		if (super.setMap(e), e) {
			let t = e.getViewport();
			if (!t) return;
			this.listenerKeys.push(i(t, Md.POINTERMOVE, this.handleMouseMove, this)), this.renderOnMouseOut_ && this.listenerKeys.push(i(t, Md.POINTEROUT, this.handleMouseOut, this)), this.updateHTML_(null);
		}
	}
	setCoordinateFormat(e) {
		this.set(wg, e);
	}
	setProjection(e) {
		this.set(Cg, or(e));
	}
	updateHTML_(e) {
		let t = this.placeholder_;
		if (e && this.mapProjection_) {
			if (!this.transform_) {
				let e = this.getProjection();
				this.transform_ = e ? pr(this.mapProjection_, e) : rr;
			}
			let n = this.getMap();
			if (!n) return;
			let r = n.getCoordinateFromPixelInternal(e);
			if (r) {
				let e = yr();
				e && (this.transform_ = pr(this.mapProjection_, e));
				let n = this.transform_;
				n && n(r, r), this.wrapX_ && Kt(r, e || this.getProjection() || this.mapProjection_);
				let i = this.getCoordinateFormat();
				t = i ? i(r) : r.toString();
			}
		}
		if (!this.renderedHTML_ || t !== this.renderedHTML_) {
			let e = this.element;
			e && (e.innerHTML = t, this.renderedHTML_ = t);
		}
	}
	render(e) {
		let t = e.frameState;
		t ? this.mapProjection_ != t.viewState.projection && (this.mapProjection_ = t.viewState.projection, this.transform_ = null) : this.mapProjection_ = null;
	}
};
// src/controls/map-controls.js
function Eg({ map: e, viewerApp: t, getSlideSource: n, zoomVisibleInput: r, zoomLevelVisibleInput: i, rotationVisibleInput: a, resetViewButton: o, resetViewControl: s, resetViewVisibleInput: c, fullscreenVisibleInput: l, mousePositionVisibleInput: u, mouseWheelZoomSensitivitySelect: d, zoomButtonStepSelect: f }) {
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
		let e = p[d.value] ?? p.default, t = new yf({ maxDelta: e.maxDelta });
		return t.deltaPerZoom_ = e.deltaPerZoom, t.setActive(n() !== null), t;
	}
	let h = m();
	e.addInteraction(h);
	function g() {
		return new Ie({ delta: Number(f.value) });
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
		t === void 0 ? b.value = "" : b.value = Number.isInteger(t) ? t.toString() : t.toFixed(1);
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
		e.key === "Enter" ? (e.preventDefault(), b.blur()) : e.key === "Escape" && (e.preventDefault(), x(), b.blur());
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
	let w = new Tg({
		coordinateFormat: (e) => Ht([e[0], -e[1]], "{x}, {y}", 0),
		className: "ol-mouse-position",
		placeholder: "\xA0"
	});
	e.addControl(w);
	let T = new Fe({
		autoHide: !1,
		className: "ol-rotate"
	});
	e.addControl(T);
	let E = new _g({ source: t });
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
// node_modules/ol-ext/util/getMapCanvas.js
var Dg = function(e) {
	if (!e) return null;
	var t = e.getViewport().getElementsByClassName("ol-fixedoverlay")[0];
	return t || (e.getViewport().querySelector(".ol-layers") ? (t = document.createElement("canvas"), t.className = "ol-fixedoverlay", e.getViewport().querySelector(".ol-layers").after(t), e.on("precompose", function(n) {
		t.width = e.getSize()[0] * n.frameState.pixelRatio, t.height = e.getSize()[1] * n.frameState.pixelRatio;
	})) : t = e.getViewport().querySelector("canvas")), t;
}, Og = class extends ke {
	constructor(e) {
		e ||= {}, super(e), this.setStyle(e.style);
	}
	setMap(e) {
		this.getCanvas(e);
		var t = this.getMap();
		if (this._listener &&= (k(this._listener), null), super.setMap(e), t) try {
			t.renderSync();
		} catch {}
		e && (this._listener = e.on("postcompose", this._draw.bind(this)));
	}
	getCanvas(e) {
		return Dg(e);
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
		this._style = e || new Iu({});
	}
	getStyle() {
		return this._style;
	}
	getStroke() {
		return this._style.getStroke() || this._style.setStroke(new Fu({
			color: "#000",
			width: 1.25
		})), this._style.getStroke();
	}
	getFill() {
		return this._style.getFill() || this._style.setFill(new Pu({ color: "#fff" })), this._style.getFill();
	}
	getTextStroke() {
		var e = this._style.getText();
		return e ||= new Hu({}), e.getStroke() || e.setStroke(new Fu({
			color: "#fff",
			width: 3
		})), e.getStroke();
	}
	getTextFill() {
		var e = this._style.getText();
		return e ||= new Hu({}), e.getFill() || e.setFill(new Pu({ color: "#fff" })), e.getFill();
	}
	getTextFont() {
		var e = this._style.getText();
		return e ||= new Hu({}), e.getFont() || e.setFont("12px sans-serif"), e.getFont();
	}
	_draw() {
		console.warn("[CanvasBase] draw function not implemented.");
	}
}, kg = class extends Og {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = "ol-graticule ol-unselectable ol-hidden", super({ element: t }), this.set("projection", e.projection || "EPSG:4326");
		var n = new Zt({ code: this.get("projection") }).getMetersPerUnit();
		for (this.fac = 1; n / this.fac > 10;) this.fac *= 10;
		this.fac = 1e4 / this.fac, this.set("maxResolution", e.maxResolution || Infinity), this.set("step", e.step || .1), this.set("stepCoord", e.stepCoord || 1), this.set("spacing", e.spacing || 40), this.set("intervals", e.intervals), this.set("precision", e.precision), this.set("margin", e.margin || 0), this.set("borderWidth", e.borderWidth || 5), this.set("stroke", e.stroke !== !1), this.formatCoord = e.formatCoord || function(e) {
			return e;
		}, e.style instanceof Iu ? this.setStyle(e.style) : this.setStyle(new Iu({
			stroke: new Fu({
				color: "#000",
				width: 1
			}),
			fill: new Pu({ color: "#fff" }),
			text: new Hu({
				stroke: new Fu({
					color: "#fff",
					width: 2
				}),
				fill: new Pu({ color: "#000" })
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
			], l = -Infinity, u = Infinity, d = -Infinity, f = Infinity, p = 0, m; m = c[p]; p++) c[p] = gr(m, s.getView().getProjection(), o), l = Math.max(l, c[p][0]), u = Math.min(u, c[p][0]), d = Math.max(d, c[p][1]), f = Math.min(f, c[p][1]);
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
			var T = or(o).getExtent();
			T && (u < T[0] && (u = T[0]), f < T[1] && (f = T[1]), l > T[2] && (l = T[2] + g), d > T[3] && (d = T[3] + g));
			var E = this.getStyle().getStroke() && this.get("stroke"), D = this.getStyle().getText(), O = this.getStyle().getFill();
			t.save(), t.scale(r, r), t.beginPath(), t.rect(y, y, i - 2 * y, a - 2 * y), t.clip(), t.beginPath();
			for (var k = {
				top: [],
				left: [],
				bottom: [],
				right: []
			}, A = u, j, M, N, P; A < l; A += g) for (N = gr([A, f], o, s.getView().getProjection()), N = s.getPixelFromCoordinate(N), E && t.moveTo(N[0], N[1]), M = N, j = f + w; j <= d; j += w) P = gr([A, j], o, s.getView().getProjection()), P = s.getPixelFromCoordinate(P), E && t.lineTo(P[0], P[1]), M[1] > 0 && P[1] < 0 && k.top.push([A, M]), M[1] > a && P[1] < a && k.bottom.push([A, M]), M = P;
			for (j = f; j < d; j += g) for (N = gr([u, j], o, s.getView().getProjection()), N = s.getPixelFromCoordinate(N), E && t.moveTo(N[0], N[1]), M = N, A = u + w; A <= l; A += w) P = gr([A, j], o, s.getView().getProjection()), P = s.getPixelFromCoordinate(P), E && t.lineTo(P[0], P[1]), M[0] < 0 && P[0] > 0 && k.left.push([j, M]), M[0] < i && P[0] > i && k.right.push([j, M]), M = P;
			if (E && (t.strokeStyle = this.getStyle().getStroke().getColor(), t.lineWidth = this.getStyle().getStroke().getWidth(), t.stroke()), D) {
				t.fillStyle = this.getStyle().getText().getFill().getColor(), t.strokeStyle = this.getStyle().getText().getStroke().getColor(), t.lineWidth = this.getStyle().getText().getStroke().getWidth(), t.font = this.getStyle().getText().getFont(), t.textAlign = "center", t.textBaseline = "hanging";
				var F, ee, te = (O ? v : 0) + y + 2;
				for (p = 0; F = k.top[p]; p++) Math.round(F[0] / this.get("step")) % _ || (ee = this.formatCoord(F[0], "top"), t.strokeText(ee, F[1][0], te), t.fillText(ee, F[1][0], te));
				for (t.textBaseline = "alphabetic", p = 0; F = k.bottom[p]; p++) Math.round(F[0] / this.get("step")) % _ || (ee = this.formatCoord(F[0], "bottom"), t.strokeText(ee, F[1][0], a - te), t.fillText(ee, F[1][0], a - te));
				for (t.textBaseline = "middle", t.textAlign = "left", p = 0; F = k.left[p]; p++) Math.round(F[0] / this.get("step")) % _ || (ee = this.formatCoord(F[0], "left"), t.strokeText(ee, te, F[1][1]), t.fillText(ee, te, F[1][1]));
				for (t.textAlign = "right", p = 0; F = k.right[p]; p++) Math.round(F[0] / this.get("step")) % _ || (ee = this.formatCoord(F[0], "right"), t.strokeText(ee, i - te, F[1][1]), t.fillText(ee, i - te, F[1][1]));
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
}, Ag = class extends ke {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = (e.className || "") + " ol-button ol-unselectable ol-control", super({
			element: t,
			target: e.target
		});
		var n = this, r = this.button_ = document.createElement(/ol-text-button/.test(e.className) ? "div" : "button");
		for (var i in e.id ? r.setAttribute("id", e.id) : r.setAttribute("id", "ol-button-" + y(this)), this.button_.className = e.classButton || "", r.type = "button", e.title && (r.title = e.title), e.name && (r.name = e.name), e.html instanceof Element ? r.appendChild(e.html) : r.innerHTML = e.html || "", r.addEventListener("click", function(t) {
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
}, jg = class extends Ag {
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
// src/utils/colours.js
function Mg(e) {
	let t = e.replace("#", "");
	return /^[0-9a-fA-F]{6}$/.test(t) ? {
		r: Number.parseInt(t.slice(0, 2), 16),
		g: Number.parseInt(t.slice(2, 4), 16),
		b: Number.parseInt(t.slice(4, 6), 16)
	} : null;
}
function Ng({ r: e, g: t, b: n }) {
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
function Pg(e) {
	let t = Ng(e);
	return 1.05 / (t + .05) >= (t + .05) / .05 ? "#ffffff" : "#000000";
}
function Fg(e, t, n) {
	return {
		r: Math.round(e.r + (t - e.r) * n),
		g: Math.round(e.g + (t - e.g) * n),
		b: Math.round(e.b + (t - e.b) * n)
	};
}
function Ig(e, t) {
	return `rgba(${e.r}, ${e.g}, ${e.b}, ${t})`;
}
// src/controls/grid.js
var Lg = {
	fine: 32,
	default: 64,
	coarse: 128
}, Rg = 64, zg = {
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
function Bg({ map: e, projection: t, themeSelect: n, gridThemeSelect: r, gridOpacityInput: i, gridOpacityValue: a, gridSpacingSelect: o, gridLabelsVisibleInput: s, graticuleVisibleInput: c, screenSpaceGraticuleVisibleInput: l, onGraticulesChange: u }) {
	function d() {
		return Lg[o.value] ?? Lg.default;
	}
	function f() {
		return r.value === "default" ? n.value === "light" ? "light" : n.value === "high-contrast" ? "dark-contrast" : "dark" : r.value;
	}
	let p = new Hu({
		font: "12px Calibri,sans-serif",
		fill: new Pu({ color: "rgba(0, 0, 0, 1)" }),
		stroke: new Fu({
			color: "rgba(255, 255, 255, 1)",
			width: 3
		})
	}), m = new Iu({
		stroke: new Fu({
			color: "rgba(0, 0, 0, 0.5)",
			width: 1
		}),
		text: p
	});
	function h() {
		let t = zg[f()] ?? zg.dark, n = Number(i.value) / 100, r = m.getStroke(), o = p;
		r.setColor(Ig(t.line, n)), o.getFill().setColor(t.label), o.getStroke().setColor(t.outline), a.textContent = `${i.value}%`, e.renderSync();
	}
	function g() {
		m.setText(s.checked ? p : null), e.renderSync();
	}
	function _(e) {
		return new kg({
			projection: e,
			margin: Rg,
			style: m,
			spacing: d(),
			/* v8 ignore start */
			formatCoord: (e, t) => (e = t === "left" || t === "right" ? -Math.floor(e) : Math.floor(e), e >= 1e6 && (e = e.toExponential(3), e = e.replace("+", "")), e)
		});
	}
	let v = Rg;
	function y(t) {
		let n = d();
		return new kg({
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
	let C = new jg({
		html: "<i class=\"fas fa-ruler-combined\"></i>",
		className: "ol-graticule",
		title: "Toggle Graticule",
		onToggle(t) {
			C.element.classList.toggle("active", t), t ? (w.setActive(!1), w.element.classList.remove("active"), x.setMap(null), b.setMap(e)) : b.setMap(null);
		}
	});
	e.addControl(C);
	let w = new jg({
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
// src/controls/loading-indicator.js
function Vg({ map: e, element: t, onChange: n = () => {} }) {
	let r = 0, i = !0;
	function a() {
		return r > 0 || i;
	}
	function o() {
		let e = a();
		t.hidden = !e, n(e);
	}
	function s() {
		r += 1, o();
	}
	function c() {
		r = Math.max(0, r - 1), r === 0 && (i = !0, e.render()), o();
	}
	return e.on("loadstart", () => {
		i = !0, o();
	}), e.on("rendercomplete", () => {
		i = !1, o();
	}), o(), e.render(), {
		begin: s,
		end: c,
		isLoading: a
	};
}
// src/controls/snapshot.js
function Hg(e, t) {
	if (t === null || t.classList.contains("viewer-control-hidden")) return null;
	let n = e.getTargetElement(), r = t.querySelector(".ol-scale-line-inner");
	if (n === null || r === null) return null;
	let i = n.getBoundingClientRect(), a = t.getBoundingClientRect(), o = r.getBoundingClientRect(), s = window.getComputedStyle(t), c = window.getComputedStyle(r);
	return {
		x: a.left - i.left,
		y: a.top - i.top,
		width: a.width,
		height: a.height,
		backgroundColour: s.backgroundColor,
		inner: {
			x: o.left - i.left,
			y: o.top - i.top,
			width: o.width,
			height: o.height,
			text: r.textContent.trim(),
			colour: c.color,
			font: `${c.fontWeight} ${c.fontSize} ${c.fontFamily}`,
			borderTopWidth: Number.parseFloat(c.borderTopWidth) || 0,
			borderRightWidth: Number.parseFloat(c.borderRightWidth) || 0,
			borderBottomWidth: Number.parseFloat(c.borderBottomWidth) || 0,
			borderLeftWidth: Number.parseFloat(c.borderLeftWidth) || 0,
			borderTopColour: c.borderTopColor,
			borderRightColour: c.borderRightColor,
			borderBottomColour: c.borderBottomColor,
			borderLeftColour: c.borderLeftColor
		}
	};
}
function Ug(e, t) {
	if (t === null) return;
	let n = e.getContext("2d");
	if (n === null) return;
	n.fillStyle = t.backgroundColour, n.fillRect(t.x, t.y, t.width, t.height);
	let r = t.inner;
	n.fillStyle = r.borderTopColour, n.fillRect(r.x, r.y, r.width, r.borderTopWidth), n.fillStyle = r.borderRightColour, n.fillRect(r.x + r.width - r.borderRightWidth, r.y, r.borderRightWidth, r.height), n.fillStyle = r.borderBottomColour, n.fillRect(r.x, r.y + r.height - r.borderBottomWidth, r.width, r.borderBottomWidth), n.fillStyle = r.borderLeftColour, n.fillRect(r.x, r.y, r.borderLeftWidth, r.height), n.fillStyle = r.colour, n.font = r.font, n.textAlign = "center", n.textBaseline = "middle", n.fillText(r.text, r.x + r.width / 2, r.y + r.height / 2);
}
async function Wg(e, t = null) {
	let n = e.getSize();
	if (n === void 0 || n[0] <= 0 || n[1] <= 0) throw Error("The map size is not available for snapshot capture.");
	let r = Hg(e, t), i = e.getTarget(), a = document.createElement("canvas");
	a.width = n[0], a.height = n[1];
	let o = new Promise((t) => {
		e.once("rendercomplete", t);
	});
	e.setTarget(a);
	try {
		return await o, Ug(a, r), await new Promise((e, t) => {
			a.toBlob((n) => {
				n === null ? t(/* @__PURE__ */ Error("The snapshot could not be encoded as PNG.")) : e(n);
			}, "image/png");
		});
	} finally {
		e.setTarget(i), e.updateSize(), e.renderSync();
	}
}
function Gg(e, t) {
	let n = URL.createObjectURL(e), r = document.createElement("a");
	r.href = n, r.download = t, document.body.appendChild(r), r.click(), r.remove(), URL.revokeObjectURL(n);
}
function Kg(e, t) {
	let n = `${JSON.stringify(e, null, 4)}\n`;
	Gg(new Blob([n], { type: "application/json" }), t);
}
function qg({ button: e, hasSlide: t, onSnapshot: n }) {
	let r = !1, i = !1;
	function a() {
		let n = t(), a = r || i;
		e.disabled = a || !n, e.innerHTML = a ? "<i class=\"fas fa-spinner fa-spin\"></i>" : "<i class=\"fas fa-camera\"></i>", e.title = n ? a ? "Waiting for viewer to finish" : "Save snapshot" : "Load a slide to save a snapshot";
	}
	function o(e) {
		r = e, a();
	}
	async function s() {
		if (!(r || i || !t())) {
			i = !0, a();
			try {
				await n();
			} finally {
				i = !1, a();
			}
		}
	}
	return e.addEventListener("click", () => {
		s().catch((e) => {
			console.error("Failed to save snapshot.", e);
		});
	}), a(), {
		saveSnapshot: s,
		setLoading: o,
		updateButton: a
	};
}
// node_modules/ol/control/ScaleLine.js
var Jg = "units", Yg = [
	1,
	2,
	5
], Xg = 25.4 / .28, Zg = class extends ke {
	constructor(e) {
		e ||= {};
		let t = document.createElement("div");
		t.style.pointerEvents = "none", super({
			element: t,
			render: e.render,
			target: e.target
		}), this.on, this.once, this.un;
		let n = e.className === void 0 ? e.bar ? "ol-scale-bar" : "ol-scale-line" : e.className;
		this.innerElement_ = document.createElement("div"), this.innerElement_.className = n + "-inner", this.element.className = n + " " + te, this.element.appendChild(this.innerElement_), this.viewState_ = null, this.minWidth_ = e.minWidth === void 0 ? 64 : e.minWidth, this.maxWidth_ = e.maxWidth, this.renderedVisible_ = !1, this.renderedWidth_ = void 0, this.renderedHTML_ = "", this.addChangeListener(Jg, this.handleUnitsChanged_), this.setUnits(e.units || "metric"), this.scaleBar_ = e.bar || !1, this.scaleBarSteps_ = e.steps || 4, this.scaleBarText_ = e.text || !1, this.dpi_ = e.dpi || void 0;
	}
	getUnits() {
		return this.get(Jg);
	}
	handleUnitsChanged_() {
		this.updateElement_();
	}
	setUnits(e) {
		this.set(Jg, e);
	}
	setDpi(e) {
		this.dpi_ = e;
	}
	updateElement_() {
		let e = this.viewState_;
		if (!e) {
			let e = this.element;
			this.renderedVisible_ && e && (e.style.display = "none", this.renderedVisible_ = !1);
			return;
		}
		let t = e.center, n = e.projection, r = this.getUnits(), i = r == "degrees" ? "degrees" : "m", a = sr(n, e.resolution, t, i), o = this.minWidth_ * (this.dpi_ || Xg) / Xg, s = this.maxWidth_ === void 0 ? void 0 : this.maxWidth_ * (this.dpi_ || Xg) / Xg, c = o * a, l = "";
		if (r == "degrees") {
			let e = Xt.degrees;
			c *= e, c < e / 60 ? (l = "″", a *= 3600) : c < e ? (l = "′", a *= 60) : l = "°";
		} else if (r == "imperial") c < .9144 ? (l = "in", a /= .0254) : c < 1609.344 ? (l = "ft", a /= .3048) : (l = "mi", a /= 1609.344);
		else if (r == "nautical") a /= 1852, l = "NM";
		else if (r == "metric") c < 1e-6 ? (l = "nm", a *= 1e9) : c < .001 ? (l = "μm", a *= 1e6) : c < 1 ? (l = "mm", a *= 1e3) : c < 1e3 ? l = "m" : (l = "km", a /= 1e3);
		else if (r == "us") c < .9144 ? (l = "in", a *= 39.37) : c < 1609.344 ? (l = "ft", a /= .30480061) : (l = "mi", a /= 1609.3472);
		else throw Error("Invalid units");
		let u = 3 * Math.floor(Math.log(o * a) / Math.log(10)), d, f, p = 0, m = 0, h, g;
		for (;;) {
			p = Math.floor(u / 3);
			let e = 10 ** p;
			if (d = Yg[(u % 3 + 3) % 3] * e, f = Math.round(d / a), isNaN(f)) {
				let e = this.element;
				e && (e.style.display = "none"), this.renderedVisible_ = !1;
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
		if (this.renderedHTML_ != _ && (this.innerElement_.innerHTML = _, this.renderedHTML_ = _), this.renderedWidth_ != f && (this.innerElement_.style.width = f + "px", this.renderedWidth_ = f), !this.renderedVisible_) {
			let e = this.element;
			e && (e.style.display = ""), this.renderedVisible_ = !0;
		}
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
		let e = this.viewState_;
		if (!e) return 0;
		let t = sr(e.projection, e.resolution, e.center, "m"), n = this.dpi_ || Xg;
		return 1e3 / 25.4 * t * n;
	}
	render(e) {
		let t = e.frameState;
		this.viewState_ = t ? t.viewState : null, this.updateElement_();
	}
}, Qg = {
	small: 70,
	default: 100,
	large: 140
};
function $g({ map: e, hasSlide: t, enabledInput: n, colourInput: r, opacityInput: i, opacityValue: a, sizeSelect: o, unitsSelect: s, onControlChange: c }) {
	function l() {
		let e = Qg[o.value] ?? Qg.default;
		return new Zg({
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
		let e = Number(i.value) / 100, t = Mg(r.value);
		if (t === null) return;
		let n = Pg(t) === "#ffffff" ? {
			r: 255,
			g: 255,
			b: 255
		} : {
			r: 17,
			g: 17,
			b: 17
		};
		u.element.style.backgroundColor = Ig(n, e), a.textContent = `${i.value}%`;
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
// node_modules/ol/Overlay.js
var e_ = {
	ELEMENT: "element",
	MAP: "map",
	OFFSET: "offset",
	POSITION: "position",
	POSITIONING: "positioning"
}, t_ = class extends j {
	constructor(e) {
		super(), this.on, this.once, this.un, this.options = e, this.id = e.id, this.insertFirst = e.insertFirst === void 0 || e.insertFirst, this.stopEvent = e.stopEvent === void 0 || e.stopEvent, this.element = document.createElement("div"), this.element.className = e.className === void 0 ? "ol-overlay-container " + ee : e.className, this.element.style.position = "absolute", this.element.style.pointerEvents = "auto", this.autoPan = e.autoPan === !0 ? {} : e.autoPan || void 0, this.rendered = {
			transform_: "",
			visible: !0
		}, this.mapPostrenderListenerKey = null, this.addChangeListener(e_.ELEMENT, this.handleElementChanged), this.addChangeListener(e_.MAP, this.handleMapChanged), this.addChangeListener(e_.OFFSET, this.handleOffsetChanged), this.addChangeListener(e_.POSITION, this.handlePositionChanged), this.addChangeListener(e_.POSITIONING, this.handlePositioningChanged), e.element !== void 0 && this.setElement(e.element), this.setOffset(e.offset === void 0 ? [0, 0] : e.offset), this.setPositioning(e.positioning || "top-left"), e.position !== void 0 && this.setPosition(e.position);
	}
	getElement() {
		return this.get(e_.ELEMENT);
	}
	getId() {
		return this.id;
	}
	getMap() {
		return this.get(e_.MAP) || null;
	}
	getOffset() {
		return this.get(e_.OFFSET);
	}
	getPosition() {
		return this.get(e_.POSITION);
	}
	getPositioning() {
		return this.get(e_.POSITIONING);
	}
	handleElementChanged() {
		we(this.element);
		let e = this.getElement();
		e && this.element.appendChild(e);
	}
	handleMapChanged() {
		this.mapPostrenderListenerKey &&= (this.element?.remove(), o(this.mapPostrenderListenerKey), null);
		let e = this.getMap();
		if (e) {
			this.mapPostrenderListenerKey = i(e, Oe.POSTRENDER, this.render, this), this.updatePixelPosition();
			let t = this.stopEvent ? e.getOverlayContainerStopEvent() : e.getOverlayContainer();
			t && (this.insertFirst ? t.insertBefore(this.element, t.childNodes[0] || null) : t.appendChild(this.element)), this.performAutoPan();
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
		this.set(e_.ELEMENT, e);
	}
	setMap(e) {
		this.set(e_.MAP, e);
	}
	setOffset(e) {
		this.set(e_.OFFSET, e);
	}
	setPosition(e) {
		this.set(e_.POSITION, e);
	}
	performAutoPan() {
		this.autoPan && this.panIntoView(this.autoPan);
	}
	panIntoView(e) {
		let t = this.getMap();
		if (!t || !t.getTargetElement() || !this.get(e_.POSITION)) return;
		let n = t.getSize(), r = this.getElement();
		if (!n || !r) return;
		let i = t.getTargetElement();
		if (!i) return;
		let a = this.getRect(i, n), o = this.getRect(r, [xe(r), Se(r)]);
		e ||= {};
		let s = e.margin === void 0 ? 20 : e.margin;
		if (!Ge(a, o)) {
			let n = o[0] - a[0], r = a[2] - o[2], i = o[1] - a[1], c = a[3] - o[3], l = [0, 0];
			if (n < 0 ? l[0] = n - s : r < 0 && (l[0] = Math.abs(r) + s), i < 0 ? l[1] = i - s : c < 0 && (l[1] = Math.abs(c) + s), l[0] !== 0 || l[1] !== 0) {
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
		this.set(e_.POSITIONING, e);
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
		let n = e.getPixelFromCoordinate(t, { wrapX: !0 }), r = e.getSize();
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
}, n_ = .75, r_ = .1, i_ = class extends ke {
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
		let c = new Mf({
			view: e.view,
			controls: new P(),
			interactions: new P()
		});
		this.ovmap_ = c, e.layers && e.layers.forEach(function(e) {
			c.addLayer(e);
		});
		let l = document.createElement("div");
		l.className = "ol-overviewmap-box", l.style.boxSizing = "border-box", this.boxOverlay_ = new t_({
			position: [0, 0],
			positioning: "center-center",
			element: l
		}), this.ovmap_.addOverlay(this.boxOverlay_);
		let u = t + " " + te + " " + re + (this.collapsed_ && this.collapsible_ ? " " + ie : "") + (this.collapsible_ ? "" : " ol-uncollapsible"), d = this.element;
		d.className = u, d.appendChild(this.ovmapDiv_), d.appendChild(o);
		let f = this.boxOverlay_, p = this.boxOverlay_.getElement(), m = (e) => ({
			clientX: e.clientX,
			clientY: e.clientY
		}), h = (e) => {
			let t = m(e), n = c.getEventCoordinate(t);
			f.setPosition(n);
		}, g = (e) => {
			let t = c.getEventCoordinateInternal(e), n = this.getMap();
			if (!n) return;
			n.getView().setCenterInternal(t);
			let r = n.getOwnerDocument();
			r.removeEventListener("pointermove", h), r.removeEventListener("pointerup", g);
		};
		this.ovmapDiv_.addEventListener("pointerdown", (e) => {
			let t = this.getMap();
			if (!t) return;
			let n = t.getOwnerDocument();
			e.target === p && n.addEventListener("pointermove", h), n.addEventListener("pointerup", g);
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
		if (e.key === Pd.VIEW) {
			let t = e.oldValue;
			t && this.unbindView_(t);
			let n = this.getMap();
			if (!n) return;
			let r = n.getView();
			this.bindView_(r);
		} else this.ovmap_.isRendered() || e.key !== Pd.TARGET && e.key !== Pd.SIZE || this.ovmap_.updateSize();
	}
	bindView_(e) {
		if (!this.view_) {
			let t = new us({ projection: e.getProjection() });
			this.ovmap_.setView(t);
		}
		e.addChangeListener(Zo.ROTATION, this.boundHandleRotationChanged_), this.handleRotationChanged_(), e.isDef() && (this.ovmap_.updateSize(), this.resetExtent_());
	}
	unbindView_(e) {
		e.removeChangeListener(Zo.ROTATION, this.boundHandleRotationChanged_);
	}
	handleRotationChanged_() {
		if (this.rotateWithView_) {
			let e = this.getMap();
			if (!e) return;
			this.ovmap_.getView().setRotation(e.getView().getRotation());
		}
	}
	validateExtent_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.ovmap_;
		if (!e.isRendered() || !t.isRendered()) return;
		let n = e.getSize(), r = e.getView().calculateExtentInternal(n);
		if (this.viewExtent_ && $e(r, this.viewExtent_)) return;
		this.viewExtent_ = r;
		let i = t.getSize(), a = t.getView().calculateExtentInternal(i), o = t.getPixelFromCoordinateInternal(ht(r)), s = t.getPixelFromCoordinateInternal(st(r)), c = Math.abs(o[0] - s[0]), l = Math.abs(o[1] - s[1]), u = i[0], d = i[1];
		c < u * r_ || l < d * r_ || c > u * n_ || l > d * n_ ? this.resetExtent_() : Ge(a, r) || this.recenter_();
	}
	resetExtent_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.ovmap_, n = e.getSize(), r = e.getView().calculateExtentInternal(n), i = t.getView();
		bt(r, 1 / (2 ** (Math.log(n_ / r_) / Math.LN2 / 2) * r_)), i.fitInternal($i(r));
	}
	recenter_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.ovmap_, n = e.getView();
		t.getView().setCenterInternal(n.getCenterInternal());
	}
	updateBox_() {
		let e = this.getMap();
		if (!e) return;
		let t = this.ovmap_;
		if (!e.isRendered() || !t.isRendered()) return;
		let n = e.getSize(), r = e.getView(), i = t.getView(), a = this.rotateWithView_ ? 0 : -r.getRotation(), o = this.boxOverlay_, s = this.boxOverlay_.getElement(), c = r.getCenter(), l = r.getResolution(), u = i.getResolution();
		if (l === void 0 || u === void 0) return;
		let d = n[0] * l / u, f = n[1] * l / u;
		if (o.setPosition(c), s) {
			s.style.width = d + "px", s.style.height = f + "px";
			let e = "rotate(" + a + "rad)";
			s.style.transform = e;
		}
	}
	updateBoxAfterOvmapIsRendered_() {
		this.ovmapPostrenderKey_ ||= a(this.ovmap_, Oe.POSTRENDER, (e) => {
			delete this.ovmapPostrenderKey_, this.updateBox_();
		});
	}
	handleClick_(e) {
		e.preventDefault(), this.handleToggle_();
	}
	handleToggle_() {
		let e = this.element;
		if (!e) return;
		e.classList.toggle(ie), this.collapsed_ ? Ce(this.collapseLabel_, this.label_) : Ce(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_;
		let t = this.ovmap_;
		if (!this.collapsed_) {
			if (t.isRendered()) {
				this.viewExtent_ = void 0, t.render();
				return;
			}
			t.updateSize(), this.resetExtent_(), this.updateBoxAfterOvmapIsRendered_();
		}
	}
	getCollapsible() {
		return this.collapsible_;
	}
	setCollapsible(e) {
		if (this.collapsible_ === e) return;
		this.collapsible_ = e;
		let t = this.element;
		t && (t.classList.toggle("ol-uncollapsible"), !e && this.collapsed_ && this.handleToggle_());
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
		if (this.rotateWithView_ === e) return;
		this.rotateWithView_ = e;
		let t = this.getMap();
		t && t.getView().getRotation() !== 0 && (this.rotateWithView_ ? this.handleRotationChanged_() : this.ovmap_.getView().setRotation(0), this.viewExtent_ = void 0, this.validateExtent_(), this.updateBox_());
	}
	getOverviewMap() {
		return this.ovmap_;
	}
	render(e) {
		this.validateExtent_(), this.updateBox_();
	}
}, a_ = {
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
function o_({ map: e, source: t, projection: n, extent: r, sizeSelect: i, visibleInput: a, hasSlide: o }) {
	let s = new Ss();
	t !== null && s.setSource(t);
	function c() {
		return a_[i.value] ?? a_.default;
	}
	function l(e, t) {
		let n = c(), r = [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2], i = t[2] - t[0], a = t[3] - t[1], o = Math.max(i / n.width, a / n.height), s = new us({
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
	let f = new i_({
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
// src/components/file-select.js
var s_ = 0;
function c_(e) {
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
	o.className = "viewer-file-select-options", o.id = `viewer-file-select-${s_}`, o.setAttribute("role", "listbox"), s_ += 1, n.setAttribute("aria-controls", o.id), a.setAttribute("aria-controls", o.id), i.append(a, o), t.append(n, i);
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
		} else u.forEach((e, t) => {
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
		t.classList.contains("open") ? _() : g();
	}), n.addEventListener("keydown", (e) => {
		e.key === "ArrowDown" && (e.preventDefault(), g());
	}), a.addEventListener("input", () => {
		d = -1, h();
	}), a.addEventListener("keydown", (e) => {
		if (e.key === "Escape") e.preventDefault(), _(), n.focus();
		else if (u.length !== 0) {
			if (e.key === "ArrowDown") e.preventDefault(), d = Math.min(d + 1, u.length - 1), h();
			else if (e.key === "ArrowUp") e.preventDefault(), d = d <= 0 ? u.length - 1 : d - 1, h();
			else if (e.key === "Enter") {
				e.preventDefault();
				let t = u[d >= 0 ? d : 0];
				t !== void 0 && m(t);
			}
		}
	}), document.addEventListener("click", (e) => {
		t.contains(e.target) || _();
	}), t.disabled = !0, t;
}
// src/utils/paths.js
function l_(e) {
	let t = e.split(/[\\/]/).pop() ?? e, n = t.lastIndexOf(".");
	return n <= 0 ? t : t.slice(0, n);
}
// src/panels/files.js
function u_({ panel: e, toggle: t, container: n, configuredSlides: r, configuredOverlays: i, getCurrentSlidePath: a, hasSlide: o, hasOverlays: s, onSlideSelected: c, onOverlaySelected: l, onClearSlide: u, onClearOverlays: d, onOpen: f }) {
	let p = document.createElement("div");
	p.className = "viewer-file-selectors";
	let m = c_("Select slide"), h = c_("Load overlay");
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
		let t = l_(e);
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
		e ? (m.disabled = !0, h.disabled = !0, v.disabled = !0, y.disabled = !0) : T();
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
// src/panels/layers.js
function d_({ panel: e, toggle: t, list: n, getSlideLayer: r, getCurrentSlidePath: i, getOverlayLayers: a, onRemoveLayer: o, onOpen: s }) {
	function c(n) {
		n && s(), e.classList.toggle("hidden", !n), t.classList.toggle("active", n);
	}
	function l() {
		let e = [], t = r();
		t.getSource() !== null && e.push({
			id: "slide",
			name: l_(i() ?? "slide"),
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
		} else e.forEach(({ id: e, name: r, layer: i }) => {
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
// src/panels/annotations.js
function f_(e) {
	return typeof e == "string" ? e : `#${e.slice(0, 3).map((e) => Math.round(e * 255).toString(16).padStart(2, "0")).join("")}`;
}
function p_({ panel: e, toggle: t, list: n, selectAllButton: r, deselectAllButton: i, importButton: a, importInput: o, exportButton: s, colourBySelect: c, paletteField: l, paletteSelect: u, colourMapField: d, colourMapSelect: f, secondaryTypeField: p, secondaryTypeSelect: m, propertyField: h, propertySelect: g, propertyLegend: _, propertyLegendCaption: v, propertyMin: y, propertyMax: b, linkOpacityInput: x, getAnnotationGroups: S, getAnnotationTypes: C, getAnnotationColour: w, getDisplayMode: T, getPalette: E, getColourMap: D, getAnnotationProperties: O, getAnnotationProperty: k, getSecondaryType: A, getPropertyRange: j, isAnnotationTypeVisible: M, getAnnotationOpacity: N, getOpacityLinked: P, onColourChange: F, onVisibilityChange: ee, onOpacityChange: te, onOpacityLinkChange: ne, onPaletteChange: re, onColourMapChange: ie, onDisplayModeChange: ae, onPropertyChange: oe, onSecondaryTypeChange: se, onSetAllVisibility: ce, onImport: le, onExport: ue, onOpen: de }) {
	function fe(n) {
		n && (de(), ge()), e.classList.toggle("hidden", !n), t.classList.toggle("active", n);
	}
	function pe(e) {
		e().then(() => {
			ge();
		}).catch((e) => {
			console.error(e), ge();
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
	function ge() {
		n.replaceChildren();
		let e = S(), t = T(), o = E(), ne = D(), re = O(), ie = C(), ae = A(), oe = e.length > 0;
		x.checked = P(), x.disabled = !oe, x.closest(".annotations-panel-link-opacity")?.classList.toggle("disabled", !oe), l.hidden = t !== "type", d.hidden = t === "type", u.value = o, f.value = ne, u.disabled = !oe, f.disabled = !oe || re.length === 0, l.classList.toggle("disabled", !oe), d.classList.toggle("disabled", !oe || re.length === 0), c.value = t, c.disabled = !oe, c.closest(".annotations-panel-display-field")?.classList.toggle("disabled", !oe);
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
		} else for (let { layerName: r, annotationTypes: i } of e) {
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
					pe(() => ee(r, a, l.checked));
				});
				let u = document.createElement("input");
				u.type = "color", u.className = "annotations-panel-colour", u.value = f_(w(r, a)), u.title = `Change ${i} colour`, u.disabled = t === "property" || t === "secondary", u.addEventListener("change", () => {
					pe(() => F(r, a, u.value));
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
		render: ge,
		setOpen: fe
	};
}
// src/panels/settings.js
var m_ = {
	dark: "#111111",
	light: "#f2f2f2",
	"high-contrast": "#000000"
}, h_ = "tiatoolbox-openlayers-settings";
function g_({ viewerApp: e, panel: t, toggle: n, closeButton: r, tabs: i, tabPanels: a, resetDefaultsButton: o, themeSelect: s, controlOpacityInput: c, controlOpacityValue: l, annotationInspectionEnabledInput: u, multipleAnnotationSelectionInput: d, zoomVisibleInput: f, zoomLevelVisibleInput: p, rotationVisibleInput: m, graticuleVisibleInput: h, screenSpaceGraticuleVisibleInput: g, resetViewVisibleInput: _, fullscreenVisibleInput: v, mousePositionVisibleInput: y, overviewMapVisibleInput: b, overviewMapSizeSelect: x, mouseWheelZoomSensitivitySelect: S, zoomButtonStepSelect: C, gridThemeSelect: w, gridOpacityInput: T, gridSpacingSelect: E, gridLabelsVisibleInput: D, scaleBarEnabledInput: O, scaleBarColourInput: k, scaleBarOpacityInput: A, scaleBarSizeSelect: j, scaleBarUnitsSelect: M, snapshotVisibleInput: N, snapshotScaleBarInput: P, snapshotMetadataInput: F, onAnnotationInspectionChange: ee, onMultipleAnnotationSelectionChange: te, onThemeChange: ne, onControlVisibilityChange: re, onReset: ie }) {
	let ae = !1;
	function oe(e) {
		t.classList.toggle("hidden", !e), n.classList.toggle("active", e);
	}
	function se() {
		let t = Mg(m_[s.value] ?? m_.dark);
		/* v8 ignore if */
		if (t === null) return;
		let n = Number(c.value) / 100, r = Pg(t), i = r === "#ffffff" ? {
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
		}, o = r === "#ffffff" ? 255 : 0, u = Fg(t, o, .08), d = Fg(t, o, .16), f = Fg(t, o, .28), p = Fg(t, o, .4), m = Fg(t, o, .58);
		e.style.setProperty("--viewer-control-background", Ig(t, n)), e.style.setProperty("--viewer-control-surface-background", Ig(u, n)), e.style.setProperty("--viewer-control-hover-background", Ig(d, n)), e.style.setProperty("--viewer-control-pressed-background", Ig(f, n)), e.style.setProperty("--viewer-control-foreground", r), e.style.setProperty("--viewer-control-hover-foreground", Pg(d)), e.style.setProperty("--viewer-control-pressed-foreground", Pg(f)), e.style.setProperty("--viewer-control-muted-foreground", Ig(i, .7)), e.style.setProperty("--viewer-control-subtle-foreground", Ig(i, .55)), e.style.setProperty("--viewer-control-border", Ig(p, Math.max(n, .7))), e.style.setProperty("--viewer-control-focus-border", Ig(m, Math.max(n, .9))), e.style.setProperty("--viewer-control-foreground-shadow", Ig(a, .75)), l.textContent = `${c.value}%`;
	}
	function ce() {
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
			},
			snapshot: {
				visible: N.checked,
				scaleBar: P.checked,
				metadata: F.checked
			}
		};
		try {
			window.localStorage.setItem(h_, JSON.stringify(e));
		} catch {}
	}
	function le() {
		let e;
		try {
			let t = window.localStorage.getItem(h_);
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
		let o = e.snapshot;
		typeof o == "object" && o && (typeof o.visible == "boolean" && (N.checked = o.visible), typeof o.scaleBar == "boolean" && (P.checked = o.scaleBar), typeof o.metadata == "boolean" && (F.checked = o.metadata));
	}
	function ue() {
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
		O.checked = !0, k.value = "#ffffff", A.value = "100", j.value = "default", M.value = "metric", N.checked = !0, P.checked = !0, F.checked = !0;
	}
	function de() {
		try {
			window.localStorage.removeItem(h_);
		} catch {}
	}
	function fe() {
		if (!ae) {
			ae = !0, n.addEventListener("click", () => {
				oe(t.classList.contains("hidden"));
			}), r.addEventListener("click", () => {
				oe(!1);
			});
			for (let e of i) e.addEventListener("click", () => {
				let t = e.dataset.settingsTab;
				for (let t of i) t.classList.toggle("active", t === e);
				for (let e of a) e.classList.toggle("hidden", e.dataset.settingsPanel !== t);
			});
			s.addEventListener("change", () => {
				se(), ne(), ce();
			}), c.addEventListener("input", () => {
				se(), ce();
			}), u.addEventListener("change", () => {
				ee(), ce();
			}), d.addEventListener("change", () => {
				te(), ce();
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
				b,
				N
			]) e.addEventListener("change", () => {
				re(), ce();
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
				M,
				N,
				P,
				F
			]) e.addEventListener("change", () => {
				ce();
			});
			for (let e of [
				T,
				k,
				A
			]) e.addEventListener("input", () => {
				ce();
			});
			o.addEventListener("click", () => {
				ie();
			});
		}
	}
	return {
		bindEvents: fe,
		clearSavedSettings: de,
		load: le,
		resetValues: ue,
		setOpen: oe,
		updateAppearance: se
	};
}
// src/utils/annotation-colours.js
async function __(e, t, n, r = {}) {
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
function v_(e) {
	return typeof e == "object" && !!e && !Array.isArray(e);
}
function y_(e, t) {
	if (!v_(e)) throw Error(`${t} must be an object.`);
	for (let [n, r] of Object.entries(e)) if (!Array.isArray(r) || r.length !== 3 && r.length !== 4 || !r.every((e) => typeof e == "number" && Number.isFinite(e) && e >= 0 && e <= 255)) throw Error(`Invalid annotation colour for ${t}.${n}.`);
}
function b_(e) {
	if (!v_(e)) throw Error("Annotation colour config must be an object.");
	let t = Object.hasOwn(e, "color_dict"), n = Object.hasOwn(e, "layer_color_dicts");
	if (!t && !n) throw Error("Annotation colour config must contain color_dict or layer_color_dicts.");
	let r = t ? e.color_dict : {};
	y_(r, "color_dict");
	let i = {};
	if (n) {
		if (!v_(e.layer_color_dicts)) throw Error("layer_color_dicts must be an object.");
		for (let [t, n] of Object.entries(e.layer_color_dicts)) y_(n, `layer_color_dicts.${t}`), i[t] = n;
	}
	return {
		colorDict: r,
		layerColorDicts: i
	};
}
function x_(e, t, n, r = {}) {
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
function S_(e, t) {
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
// src/utils/annotation-filters.js
function C_(e, t) {
	let n = e.filter((e) => t.get(e) ?? !0);
	if (n.length === e.length) return null;
	if (n.length === 0) {
		let t = JSON.stringify(e[0]);
		return `(props["type"]==${t}) & (props["type"]!=${t})`;
	}
	return n.map((e) => `(props["type"]==${JSON.stringify(e)})`).join(" | ");
}
// src/utils/numbers.js
function w_(e) {
	if (e.length === 0) return null;
	let t = Infinity, n = -Infinity;
	for (let r of e) {
		if (typeof r != "number" || !Number.isFinite(r)) return null;
		t = Math.min(t, r), n = Math.max(n, r);
	}
	return [t, n];
}
// src/main.js
function T_(e, t, n) {
	return new hp({
		url: `/tileserver/layer/slide/${e}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${n}`,
		size: t.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
}
var E_ = document.getElementById("map"), D_ = document.querySelector(".viewer-app"), O_ = document.getElementById("viewer-loading-indicator"), k_ = document.getElementById("snapshot-button"), A_ = document.querySelector(".snapshot-control"), j_ = document.getElementById("settings-snapshot-visible"), M_ = document.getElementById("settings-snapshot-scale-bar"), N_ = document.getElementById("settings-snapshot-metadata"), P_ = document.getElementById("viewer-panel"), F_ = document.getElementById("viewer-panel-toggle"), I_ = document.getElementById("viewer-files"), L_ = document.getElementById("layer-editor"), R_ = document.getElementById("layer-editor-toggle"), z_ = document.getElementById("layer-editor-list"), B_ = document.getElementById("annotations-panel"), V_ = document.getElementById("annotations-toggle"), H_ = document.getElementById("annotations-panel-list"), U_ = document.getElementById("annotations-colour-by"), W_ = document.getElementById("annotations-palette-field"), G_ = document.getElementById("annotations-palette"), K_ = document.getElementById("annotations-colour-map-field"), q_ = document.getElementById("annotations-colour-map"), J_ = document.getElementById("annotations-secondary-type-field"), Y_ = document.getElementById("annotations-secondary-type"), X_ = document.getElementById("annotations-property-field"), Z_ = document.getElementById("annotations-property"), Q_ = document.getElementById("annotations-property-legend"), $_ = document.getElementById("annotations-property-legend-caption"), ev = document.getElementById("annotations-property-min"), tv = document.getElementById("annotations-property-max"), nv = document.getElementById("annotations-link-opacity"), rv = document.getElementById("annotation-inspectors"), iv = document.getElementById("annotations-select-all"), av = document.getElementById("annotations-deselect-all"), ov = document.getElementById("annotations-import-colours"), sv = document.getElementById("annotations-import-colours-file"), cv = document.getElementById("annotations-export-colours"), lv = document.getElementById("settings-panel"), uv = document.getElementById("settings-toggle"), dv = document.getElementById("settings-close"), fv = document.querySelectorAll(".settings-tab"), pv = document.querySelectorAll(".settings-tab-panel"), mv = document.getElementById("settings-annotation-inspection"), hv = document.getElementById("settings-multiple-annotation-selection"), gv = document.getElementById("settings-zoom-visible"), _v = document.getElementById("settings-zoom-level-visible"), vv = document.getElementById("settings-rotation-visible"), yv = document.getElementById("settings-graticule-visible"), bv = document.getElementById("settings-screen-space-graticule-visible"), xv = document.getElementById("reset-view-button"), Sv = document.querySelector(".reset-view-control"), Cv = document.getElementById("settings-reset-view-visible"), wv = document.getElementById("settings-fullscreen-visible"), Tv = document.getElementById("settings-mouse-position-visible"), Ev = document.getElementById("settings-overview-map-visible"), Dv = document.getElementById("settings-overview-map-size"), Ov = document.getElementById("settings-mouse-wheel-zoom-sensitivity"), kv = document.getElementById("settings-zoom-button-step"), Av = document.getElementById("settings-scale-bar-enabled"), jv = document.getElementById("settings-theme"), Mv = document.getElementById("settings-grid-theme"), Nv = document.getElementById("settings-grid-opacity"), Pv = document.getElementById("settings-grid-opacity-value"), Fv = document.getElementById("settings-grid-spacing"), Iv = document.getElementById("settings-grid-labels-visible"), Lv = document.getElementById("settings-control-opacity"), Rv = document.getElementById("settings-control-opacity-value"), zv = document.getElementById("settings-reset-defaults"), Bv = document.getElementById("settings-scale-bar-colour"), Vv = document.getElementById("settings-scale-bar-opacity"), Hv = document.getElementById("settings-scale-bar-opacity-value"), Uv = document.getElementById("settings-scale-bar-size"), Wv = document.getElementById("settings-scale-bar-units");
if (E_ === null || D_ === null) throw Error("The OpenLayers viewer could not be found.");
if (O_ === null || k_ === null || A_ === null || j_ === null || M_ === null || N_ === null || P_ === null || F_ === null || L_ === null || R_ === null || z_ === null || B_ === null || V_ === null || H_ === null || iv === null || av === null || ov === null || sv === null || cv === null || U_ === null || W_ === null || G_ === null || K_ === null || q_ === null || X_ === null || Z_ === null || Q_ === null || $_ === null || ev === null || tv === null || nv === null || rv === null || J_ === null || Y_ === null || lv === null || uv === null || dv === null || mv === null || hv === null || gv === null || _v === null || vv === null || yv === null || bv === null || xv === null || Sv === null || Cv === null || wv === null || Tv === null || Ev === null || Dv === null || Ov === null || kv === null || jv === null || Mv === null || Nv === null || Pv === null || Fv === null || Iv === null || Lv === null || Rv === null || zv === null || Av === null || Bv === null || Vv === null || Hv === null || Uv === null || Wv === null || I_ === null) throw Error("The OpenLayers viewer controls could not be found.");
var Gv = 0, Kv = 0, qv = /* @__PURE__ */ new Map(), Jv = new Uf();
function Yv() {
	Gv += 1, Kv += 1;
}
function Xv(e) {
	let t = qv.get(e);
	t !== void 0 && (Jv.removeFeature(t.feature), t.card.remove(), qv.delete(e));
}
function Zv() {
	Jv.clear(), rv.replaceChildren(), qv.clear();
}
function Qv(e) {
	for (let [t, n] of qv) n.layerName === e && Xv(t);
}
function $v() {
	if (qv.size <= 1) return;
	let e = null, t = -1;
	for (let [n, r] of qv) r.order > t && (e = n, t = r.order);
	for (let t of [...qv.keys()]) t !== e && Xv(t);
}
function ey() {
	Yv(), Zv();
}
var ty = {
	dark: "#ffffff",
	light: "#000000",
	"high-contrast": "#ffffff"
}, ny = g_({
	viewerApp: D_,
	panel: lv,
	toggle: uv,
	closeButton: dv,
	tabs: fv,
	tabPanels: pv,
	resetDefaultsButton: zv,
	themeSelect: jv,
	controlOpacityInput: Lv,
	controlOpacityValue: Rv,
	annotationInspectionEnabledInput: mv,
	multipleAnnotationSelectionInput: hv,
	zoomVisibleInput: gv,
	zoomLevelVisibleInput: _v,
	rotationVisibleInput: vv,
	graticuleVisibleInput: yv,
	screenSpaceGraticuleVisibleInput: bv,
	resetViewVisibleInput: Cv,
	fullscreenVisibleInput: wv,
	mousePositionVisibleInput: Tv,
	overviewMapVisibleInput: Ev,
	overviewMapSizeSelect: Dv,
	mouseWheelZoomSensitivitySelect: Ov,
	zoomButtonStepSelect: kv,
	gridThemeSelect: Mv,
	gridOpacityInput: Nv,
	gridSpacingSelect: Fv,
	gridLabelsVisibleInput: Iv,
	scaleBarEnabledInput: Av,
	scaleBarColourInput: Bv,
	scaleBarOpacityInput: Vv,
	scaleBarSizeSelect: Uv,
	scaleBarUnitsSelect: Wv,
	snapshotVisibleInput: j_,
	snapshotScaleBarInput: M_,
	snapshotMetadataInput: N_,
	onAnnotationInspectionChange() {
		mv.checked || ey();
	},
	onMultipleAnnotationSelectionChange() {
		Yv(), hv.checked || $v();
	},
	onThemeChange() {
		Bv.value = ty[jv.value] ?? ty.dark, Db.updateColour(), Db.updateOpacity(), Nb.updateAppearance();
	},
	onControlVisibilityChange() {
		Lb();
	},
	onReset() {
		Rb();
	}
});
ny.load(), ny.updateAppearance();
var ry = d_({
	panel: L_,
	toggle: R_,
	list: z_,
	getSlideLayer: () => lb,
	getCurrentSlidePath: () => ly,
	getOverlayLayers: () => uy,
	onRemoveLayer: (e) => Yb(e),
	onOpen() {
		ab.setOpen(!1), ob.setOpen(!1);
	}
}), iy = JSON.parse(E_.dataset.layers ?? "[]"), ay = null, oy = Date.now(), sy = Date.now(), cy = null, ly = null, uy = {}, dy = /* @__PURE__ */ new Map(), fy = /* @__PURE__ */ new Set(), py = /* @__PURE__ */ new Map(), my = /* @__PURE__ */ new Map(), hy = /* @__PURE__ */ new Map(), gy = /* @__PURE__ */ new Map(), _y = /* @__PURE__ */ new Map(), vy = "type", yy = null, by = null, xy = "automatic", Sy = "viridis", Cy = [], wy = /* @__PURE__ */ new Map(), Ty = !1;
function Ey() {
	let e = /* @__PURE__ */ new Set();
	for (let t of hy.values()) for (let n of t) e.add(n);
	return [...e];
}
function Dy(e = xy) {
	return e === "automatic" || e === "custom" ? null : e;
}
function Oy() {
	return [...hy.entries()].map(([e, t]) => ({
		layerName: e,
		annotationTypes: t
	}));
}
function ky() {
	for (let [e, t] of hy) if (t.length > 0) return {
		layerName: e,
		annotationType: t[0]
	};
	return null;
}
async function Ay() {
	let e = [...fy];
	if (e.length === 0) return [];
	let t = await Promise.all(e.map((e) => sg(e))), n = t[0].filter((e) => !["type", "class"].includes(e.toLowerCase()) && t.slice(1).every((t) => t.includes(e))), r = [];
	for (let t of n) {
		let n = w_((await Promise.all(e.map((e) => cg(e, t)))).flat());
		n !== null && (wy.set(t, n), r.push(t));
	}
	return r.sort();
}
function jy(e) {
	let t = wy.get(e);
	if (t === void 0) throw Error(`Annotation property range is not available: ${e}`);
	let [n, r] = t;
	return n === r ? [n, n + 1] : t;
}
async function My(e) {
	let t = my.get(e);
	if (t === void 0) throw Error(`Annotation colours are not available for layer: ${e}`);
	await mg(e), await ug("type", e), await fg(null, e), await ig(t, e);
}
async function Ny({ refresh: e = !0 } = {}) {
	await Promise.all([...fy].map((e) => My(e))), vy = "type", yy = null, by = null, await Ky({ refresh: e });
}
async function Py() {
	await mg(), await ug("type"), await fg(null);
}
async function Fy(e, { refresh: t = !0 } = {}) {
	let n = jy(e);
	await Promise.all([...fy].map(async (t) => {
		await mg(t), await ug(e, t), await dg(Sy, t), await fg(n, t);
	})), vy = "property", yy = e, by = null, await Ky({ refresh: t });
}
async function Iy(e, t, n, { refresh: r = !0 } = {}) {
	let i = jy(n);
	await Promise.all([...fy].map((e) => My(e))), await pg(t, n, Sy, i, e), vy = "secondary", yy = n, by = {
		layerName: e,
		annotationType: t
	}, await Ky({ refresh: r });
}
async function Ly({ refresh: e = !0 } = {}) {
	if (fy.size === 0) await Py(), Cy = [], wy.clear(), vy = "type", yy = null, by = null, ob.render();
	else {
		if (Cy = [], wy.clear(), ob.render(), Cy = await Ay(), vy === "property" && (yy === null || !Cy.includes(yy) ? await Ny({ refresh: e }) : await Fy(yy, { refresh: e })), vy === "secondary") {
			let t = by, n = t === null ? void 0 : hy.get(t.layerName);
			t === null || n === void 0 || !n.some((e) => Object.is(e, t.annotationType)) || yy === null || !Cy.includes(yy) ? await Ny({ refresh: e }) : await Iy(t.layerName, t.annotationType, yy, { refresh: e });
		}
		ob.render();
	}
}
async function Ry(e, { refresh: t = !0 } = {}) {
	let n = await og(Ey(), Dy(e)), r = [];
	for (let [e, t] of hy) {
		let i = my.get(e);
		if (i === void 0) continue;
		let a = _y.get(e), o = new Map(i);
		for (let e of t) {
			let t = n.get(e);
			if (t === void 0) continue;
			let r = i.get(e), s = a?.get(e) ?? r?.[3] ?? 1;
			o.set(e, [
				t[0],
				t[1],
				t[2],
				s
			]);
		}
		r.push({
			layerName: e,
			previousColours: new Map(i),
			updatedColours: o
		});
	}
	try {
		await Promise.all(r.map(({ layerName: e, updatedColours: t }) => ig(t, e)));
	} catch (e) {
		throw await Promise.allSettled(r.map(({ layerName: e, previousColours: t }) => ig(t, e))), e;
	}
	for (let { layerName: e, updatedColours: t } of r) my.set(e, t);
	for (let [e, t] of n) {
		let n = py.get(e);
		py.set(e, [
			t[0],
			t[1],
			t[2],
			n?.[3] ?? t[3] ?? 1
		]);
	}
	xy = e, t && Uy();
}
async function zy(e) {
	let t;
	try {
		t = JSON.parse(await e.text());
	} catch {
		throw Error("Annotation colour file is not valid JSON.");
	}
	let { colorDict: n, layerColorDicts: r } = b_(t), i = [];
	for (let [e, t] of hy) {
		let a = my.get(e);
		if (a === void 0) continue;
		let o = x_(a, t, n, r[e] ?? {});
		i.push({
			layerName: e,
			previousColours: new Map(a),
			updatedColours: o
		});
	}
	try {
		await Promise.all(i.map(({ layerName: e, updatedColours: t }) => ig(t, e)));
	} catch (e) {
		throw await Promise.allSettled(i.map(({ layerName: e, previousColours: t }) => ig(t, e))), e;
	}
	for (let { layerName: e, updatedColours: t } of i) my.set(e, t);
	xy = "custom", Uy();
}
function By() {
	let e = /* @__PURE__ */ new Map(), t = {};
	for (let [n, r] of hy) {
		let i = my.get(n);
		if (i !== void 0) {
			t[n] = S_(i, r).color_dict;
			for (let t of r) {
				if (e.has(t)) continue;
				let n = i.get(t);
				n !== void 0 && e.set(t, n);
			}
		}
	}
	let n = S_(e, Ey());
	n.layer_color_dicts = t;
	let r = `${JSON.stringify(n, null, 4)}\n`, i = new Blob([r], { type: "application/json" }), a = URL.createObjectURL(i), o = document.createElement("a");
	o.href = a, o.download = "annotation_config.json", document.body.appendChild(o), o.click(), o.remove(), URL.revokeObjectURL(a);
}
function Vy(e, t, n) {
	let r = new Map(t.map((e) => [e, !0])), i = new Map(t.map((e) => [e, n.get(e)?.[3] ?? 1]));
	my.set(e, n), gy.set(e, r), _y.set(e, i);
}
function Hy(e) {
	my.delete(e), gy.delete(e), _y.delete(e);
}
function Uy(e = null) {
	sy += 1;
	for (let t of fy) {
		if (t === e) continue;
		let n = uy[t];
		if (n === void 0) continue;
		let r = new hp({
			url: `/tileserver/layer/${encodeURIComponent(t)}/${ay}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${sy}`,
			size: cy.slide_dimensions,
			crossOrigin: "anonymous",
			zDirection: -1
		});
		n.setSource(r);
	}
	_b.render();
}
function Wy(e, t) {
	return vy === "secondary" && by !== null ? e === by.layerName && Object.is(t, by.annotationType) : gy.get(e)?.get(t) ?? !0;
}
async function Gy(e, { refresh: t = !0 } = {}) {
	if (!fy.has(e)) return;
	let n = hy.get(e) ?? [], r = gy.get(e);
	if (r === void 0) throw Error(`Annotation visibility is not available for layer: ${e}`);
	await rg(C_(n, vy === "secondary" ? new Map(n.map((t) => [t, Wy(e, t)])) : r), e), t && Uy();
}
async function Ky({ refresh: e = !0 } = {}) {
	await Promise.all([...fy].map((e) => Gy(e, { refresh: !1 }))), e && Uy();
}
async function qy(e) {
	let [t, n] = e, r = [...fy].filter((e) => uy[e]?.getVisible() !== !1).sort((e, t) => (uy[t]?.getZIndex() ?? 0) - (uy[e]?.getZIndex() ?? 0));
	for (let e of r) {
		let r = await lg(e, t, -n, { details: !0 });
		if (Object.keys(r).length !== 0) {
			if (r.id === void 0 || r.geometry === void 0 || r.properties === void 0) throw Error("Annotation inspection response is incomplete.");
			if (r.properties.type === void 0 || Wy(e, r.properties.type)) return {
				layerName: e,
				annotationId: r.id,
				geometry: r.geometry,
				properties: r.properties
			};
		}
	}
	return null;
}
function Jy(e) {
	return {
		box: "Bounding box (px)",
		centroid: "Centroid (px)",
		prob: "Probability",
		type: "Type"
	}[e] ?? e;
}
function Yy(e, t) {
	if (e === "prob" && typeof t == "number") return `${t.toFixed(4)} (${(t * 100).toFixed(2)}%)`;
	if (e === "centroid" && Array.isArray(t) && t.length >= 2) {
		let [e, n] = t;
		return `x: ${typeof e == "number" ? e.toFixed(2) : e}, y: ${typeof n == "number" ? n.toFixed(2) : n}`;
	}
	return e === "box" && Array.isArray(t) && t.length === 4 ? `x: ${t[0]}–${t[2]}, y: ${t[1]}–${t[3]}` : Array.isArray(t) ? t.join(", ") : typeof t == "number" && !Number.isInteger(t) ? t.toFixed(4) : typeof t == "object" && t ? JSON.stringify(t) : String(t);
}
function Xy(e, t, n) {
	let r = Math.max(8, D_.clientWidth - e.offsetWidth - 8), i = Math.max(8, D_.clientHeight - e.offsetHeight - 8);
	e.style.left = `${Math.min(Math.max(t, 8), r)}px`, e.style.top = `${Math.min(Math.max(n, 8), i)}px`;
}
function Zy(e, t) {
	let n = !1, r = 0, i = 0;
	t.addEventListener("pointerdown", (a) => {
		if (a.target.closest("button") !== null) return;
		a.preventDefault();
		let o = e.getBoundingClientRect();
		n = !0, r = a.clientX - o.left, i = a.clientY - o.top, t.setPointerCapture(a.pointerId), e.classList.add("dragging");
	}), t.addEventListener("pointermove", (t) => {
		if (!n) return;
		let a = D_.getBoundingClientRect();
		Xy(e, t.clientX - a.left - r, t.clientY - a.top - i);
	});
	let a = () => {
		n = !1, e.classList.remove("dragging");
	};
	t.addEventListener("pointerup", a), t.addEventListener("pointercancel", a);
}
function Qy(e) {
	let t = new pa().readFeature({
		type: "Feature",
		geometry: e.geometry,
		properties: {}
	}), n = t.getGeometry();
	if (n === void 0) throw Error("Annotation geometry is not available.");
	return n.scale(1, -1, [0, 0]), t;
}
function $y(e, t) {
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
		r.className = "annotation-inspector-property-name", r.textContent = Jy(e);
		let i = document.createElement("div");
		i.className = "annotation-inspector-property-value", i.textContent = Yy(e, n), t.append(r, i), s.appendChild(t);
	}
	return n.append(r, s), a.addEventListener("click", () => {
		Xv(e);
	}), Zy(n, r), n;
}
function eb(e, t, n) {
	let r = JSON.stringify([e.layerName, String(e.annotationId)]), i = hv.checked;
	i || Zv();
	let a = qv.get(r), o = E_.getBoundingClientRect(), s = D_.getBoundingClientRect();
	if (a !== void 0) {
		a.order = n, qv.delete(r), qv.set(r, a), rv.appendChild(a.card), Xy(a.card, o.left - s.left + t[0] + 12, o.top - s.top + t[1] + 12);
		return;
	}
	let c = Qy(e), l = $y(r, e), u = i ? qv.size * 18 : 0;
	qv.set(r, {
		layerName: e.layerName,
		feature: c,
		card: l,
		order: n
	}), Jv.addFeature(c), rv.appendChild(l), Xy(l, o.left - s.left + t[0] + 12 + u, o.top - s.top + t[1] + 12 + u);
}
var tb = await Zh("slide"), nb = await Zh("overlay"), rb = nb.config?.color_dict ?? {}, ib = nb.config?.layer_color_dicts ?? {}, ab = u_({
	panel: P_,
	toggle: F_,
	container: I_,
	configuredSlides: tb,
	configuredOverlays: nb,
	getCurrentSlidePath: () => ly,
	hasSlide: () => cy !== null,
	hasOverlays: () => Object.keys(uy).length > 0,
	onSlideSelected: (e) => Kb(e),
	onOverlaySelected: (e) => Jb(e),
	onClearSlide: () => Wb(),
	onClearOverlays: () => Vb(),
	onOpen() {
		ry.setOpen(!1), ob.setOpen(!1);
	}
}), ob = p_({
	panel: B_,
	toggle: V_,
	list: H_,
	colourBySelect: U_,
	paletteField: W_,
	paletteSelect: G_,
	colourMapField: K_,
	colourMapSelect: q_,
	secondaryTypeField: J_,
	secondaryTypeSelect: Y_,
	propertyField: X_,
	propertySelect: Z_,
	propertyLegend: Q_,
	propertyLegendCaption: $_,
	propertyMin: ev,
	propertyMax: tv,
	linkOpacityInput: nv,
	selectAllButton: iv,
	deselectAllButton: av,
	importButton: ov,
	importInput: sv,
	exportButton: cv,
	getAnnotationGroups: Oy,
	getAnnotationTypes: Ey,
	getDisplayMode: () => vy,
	getPalette: () => xy,
	getColourMap: () => Sy,
	getAnnotationProperties: () => Cy,
	getAnnotationProperty: () => yy,
	getSecondaryType: () => by,
	getPropertyRange: () => yy === null ? null : wy.get(yy) ?? null,
	getAnnotationColour: (e, t) => my.get(e)?.get(t) ?? [
		0,
		0,
		0,
		1
	],
	isAnnotationTypeVisible: (e, t) => gy.get(e)?.get(t) ?? !0,
	getAnnotationOpacity: (e, t) => _y.get(e)?.get(t) ?? 1,
	getOpacityLinked: () => Ty,
	async onPaletteChange(e) {
		await Ry(e);
	},
	async onColourMapChange(e) {
		let t = Sy;
		Sy = e;
		try {
			if (vy === "property") {
				if (yy === null) throw Error("No annotation property is selected.");
				await Fy(yy);
				return;
			}
			if (vy === "secondary") {
				if (yy === null || by === null) throw Error("Class + Property selection is not available.");
				await Iy(by.layerName, by.annotationType, yy);
			}
		} catch (e) {
			throw Sy = t, e;
		}
	},
	async onDisplayModeChange(e) {
		if (e === "type") {
			await Ny();
			return;
		}
		let t = yy ?? Cy[0];
		if (t === void 0) throw Error("No annotation properties are available.");
		if (e === "property") await Fy(t);
		else if (e === "secondary") {
			let e = by ?? ky();
			if (e === null) throw Error("No annotation classes are available.");
			await Iy(e.layerName, e.annotationType, t);
		} else throw Error(`Unknown annotation display mode: ${e}`);
	},
	async onPropertyChange(e) {
		if (vy === "secondary") {
			if (by === null) throw Error("No secondary annotation class is selected.");
			await Iy(by.layerName, by.annotationType, e);
		} else await Fy(e);
	},
	async onSecondaryTypeChange(e, t) {
		let n = yy ?? Cy[0];
		if (n === void 0) throw Error("No annotation properties are available.");
		await Iy(e, t, n);
	},
	async onColourChange(e, t, n) {
		let r = Mg(n), i = my.get(e);
		if (i === void 0) throw Error(`Annotation colours are not available for layer: ${e}`);
		let a = i.get(t)?.[3] ?? 1, o = new Map(i);
		o.set(t, [
			r.r / 255,
			r.g / 255,
			r.b / 255,
			a
		]), await ig(o, e), my.set(e, o), xy = "custom", Uy();
	},
	async onVisibilityChange(e, t, n) {
		let r = gy.get(e);
		if (r === void 0) throw Error(`Annotation visibility is not available for layer: ${e}`);
		let i = r.get(t) ?? !0;
		r.set(t, n);
		try {
			await Gy(e);
		} catch (e) {
			throw r.set(t, i), e;
		}
	},
	async onOpacityChange(e, t, n) {
		let r = [], i = (e, t) => {
			let i = _y.get(e);
			if (i === void 0) throw Error(`Annotation opacity is not available for layer: ${e}`);
			let a = my.get(e);
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
		if (Ty) for (let e of fy) i(e, hy.get(e) ?? []);
		else i(e, [t]);
		try {
			await Promise.all(r.map(({ layerName: e, updatedOpacities: t }) => ag(t, e)));
		} catch (e) {
			throw await Promise.allSettled(r.map(({ layerName: e, previousOpacities: t }) => ag(t, e))), e;
		}
		for (let { layerName: e, updatedOpacities: t, updatedColours: n } of r) _y.set(e, t), my.set(e, n);
		Uy();
	},
	async onOpacityLinkChange(e) {
		Ty = e;
	},
	async onSetAllVisibility(e) {
		let t = /* @__PURE__ */ new Map();
		for (let n of fy) {
			let r = gy.get(n);
			if (r === void 0) throw Error(`Annotation visibility is not available for layer: ${n}`);
			t.set(n, new Map(r));
			let i = hy.get(n) ?? [];
			for (let t of i) r.set(t, e);
		}
		try {
			await Promise.all([...fy].map((e) => Gy(e, { refresh: !1 })));
		} catch (e) {
			for (let [e, n] of t) gy.set(e, n);
			throw await Promise.allSettled([...fy].map((e) => Gy(e, { refresh: !1 }))), Uy(), e;
		}
		Uy();
	},
	async onImport(e) {
		await zy(e);
	},
	onExport() {
		By();
	},
	onOpen() {
		ab.setOpen(!1), ry.setOpen(!1);
	}
});
ab.setOpen(!0);
var sb = new URLSearchParams(window.location.search).get("slide") ?? (iy.length === 0 ? tb.files[0]?.path ?? null : null);
if (sb !== null) {
	ly = sb, ay = await Yh();
	let e = await Xh(sb);
	cy = e, ab.setSlide(sb), ab.updateOverlaySelect(), iy = [{
		name: "slide",
		url: `/tileserver/layer/slide/${ay}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${oy}`,
		size: e.slide_dimensions,
		mpp: e.mpp[0]
	}];
} else iy.length === 0 && (ay = await Yh());
ab.updateActionState();
var cb = iy.map((e) => {
	let t = new hp({
		url: e.url,
		size: e.size,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	return new Ss({
		title: e.name,
		source: t
	});
}), lb = cb[0];
lb === void 0 && (lb = new Ss({ title: "slide" }), cb.push(lb)), lb.setZIndex(0);
var ub = lb.getSource(), db, fb, pb;
if (ub !== null) {
	let e = ub.getTileGrid();
	db = e.getResolutions(), fb = e.getExtent(), pb = new Zt({
		code: "ZoomifyProjection",
		units: "pixels",
		extent: fb,
		metersPerUnit: iy[0].mpp * 1e-6,
		getPointResolution(e) {
			return e;
		}
	});
} else db = [1], fb = [
	0,
	-1,
	1,
	0
], pb = new Zt({
	code: "ZoomifyProjectionEmpty",
	units: "pixels",
	extent: fb,
	metersPerUnit: 1,
	getPointResolution(e) {
		return e;
	}
});
var mb = .1;
function hb(e) {
	let t = e[2] - e[0], n = e[3] - e[1], r = t * mb, i = n * mb;
	return [
		e[0] - r,
		e[1] - i,
		e[2] + r,
		e[3] + i
	];
}
ir(pb);
var gb = new us({
	projection: pb,
	resolutions: db,
	extent: hb(fb),
	constrainOnlyCenter: !0,
	smoothExtentConstraint: !0,
	smoothResolutionConstraint: !1,
	center: [.5, -.5],
	resolution: db[0]
}), _b = new Mf({
	target: E_,
	layers: cb,
	view: gb,
	controls: Le({
		zoom: !1,
		rotate: !1
	}),
	interactions: Sf({ mouseWheelZoom: !1 })
}), vb = null;
function yb() {
	let e = _b.getView(), t = e.getCenter(), n = _b.getLayers().getArray(), r = Object.entries(uy).map(([e, t]) => ({
		name: e,
		path: dy.get(e) ?? null,
		visible: t.getVisible(),
		opacity: t.getOpacity(),
		zIndex: t.getZIndex() ?? null,
		order: n.indexOf(t)
	})).sort((e, t) => e.order - t.order);
	return {
		url: window.location.href,
		slide: { path: ly },
		viewport: {
			centre: t === void 0 ? null : [...t],
			zoom: e.getZoom() ?? null,
			rotation: e.getRotation()
		},
		overlays: r,
		scaleBar: {
			enabled: Av.checked,
			included: Av.checked && M_.checked,
			colour: Bv.value,
			opacity: Number(Vv.value),
			size: Uv.value,
			units: Wv.value
		}
	};
}
var bb = qg({
	button: k_,
	hasSlide: () => cy !== null,
	async onSnapshot() {
		if (ly === null) return;
		let e = await Wg(_b, M_.checked ? vb?.element ?? null : null), t = `${l_(ly)}_snapshot`;
		Gg(e, `${t}.png`), N_.checked && Kg(yb(), `${t}.json`);
	}
}), xb = Vg({
	map: _b,
	element: O_,
	onChange(e) {
		bb.setLoading(e);
	}
}), Sb = new Od({
	source: Jv,
	style: new Iu({
		fill: new Pu({ color: "rgba(255, 215, 0, 0.22)" }),
		stroke: new Fu({
			color: "#ffd700",
			width: 3
		}),
		image: new Nu({
			radius: 7,
			fill: new Pu({ color: "rgba(255, 215, 0, 0.45)" }),
			stroke: new Fu({
				color: "#ffd700",
				width: 3
			})
		})
	})
});
Sb.setZIndex(1e4), _b.addLayer(Sb);
var Cb = Eg({
	map: _b,
	viewerApp: D_,
	getSlideSource: () => lb.getSource(),
	zoomVisibleInput: gv,
	zoomLevelVisibleInput: _v,
	rotationVisibleInput: vv,
	resetViewButton: xv,
	resetViewControl: Sv,
	resetViewVisibleInput: Cv,
	fullscreenVisibleInput: wv,
	mousePositionVisibleInput: Tv,
	mouseWheelZoomSensitivitySelect: Ov,
	zoomButtonStepSelect: kv
}), { fullscreen: wb, mousePositionControl: Tb, rotate: Eb } = Cb, Db = $g({
	map: _b,
	hasSlide: () => lb.getSource() !== null,
	enabledInput: Av,
	colourInput: Bv,
	opacityInput: Vv,
	opacityValue: Hv,
	sizeSelect: Uv,
	unitsSelect: Wv,
	onControlChange(e) {
		vb = e, window.scaleLineControl = e;
	}
}), Ob = o_({
	map: _b,
	source: ub,
	projection: pb,
	extent: fb,
	sizeSelect: Dv,
	visibleInput: Ev,
	hasSlide: () => lb.getSource() !== null
}), kb = Ob.control, Ab = new Jh();
_b.addControl(Ab);
var jb = null, Mb = null, Nb = Bg({
	map: _b,
	projection: pb,
	themeSelect: jv,
	gridThemeSelect: Mv,
	gridOpacityInput: Nv,
	gridOpacityValue: Pv,
	gridSpacingSelect: Fv,
	gridLabelsVisibleInput: Iv,
	graticuleVisibleInput: yv,
	screenSpaceGraticuleVisibleInput: bv,
	onGraticulesChange(e, t) {
		jb = e, Mb = t, window.graticule = jb, window.screenSpaceGraticule = Mb;
	}
}), { graticuleToggle: Pb, screenSpaceGraticuleToggle: Fb } = Nb, Ib = document.createElement("div");
Ib.className = "viewer-tools-group ol-unselectable", _b.getOverlayContainerStopEvent().append(Ib), Ib.append(Eb.element, Pb.element, Fb.element);
function Lb() {
	Cb.updateVisibility(), Nb.updateVisibility(), Ob.updateVisibility(), A_.classList.toggle("viewer-control-hidden", !j_.checked);
}
function Rb() {
	ny.resetValues(), Yv(), $v(), ny.updateAppearance(), Nb.updateAppearance(), Nb.updateSpacing(), Nb.updateLabels(), Lb(), Ob.updateSize(), Cb.updateMouseWheelZoomSensitivity(), Cb.updateZoomButtonStep(), Db.updateSize(), Db.updateUnits(), Db.updateVisibility(), Db.updateColour(), Db.updateOpacity(), ny.clearSavedSettings();
}
ny.bindEvents();
function zb(e) {
	Cb.setViewerEnabled(e), Nb.setViewerEnabled(e), Db.setViewerEnabled(e), Ob.setViewerEnabled(e);
}
if (zb(ub !== null), Lb(), ub !== null) {
	_b.getView().fit(fb);
	let e = Hb();
	e !== null && (_b.getView().setCenter(e.center), _b.getView().setZoom(e.zoom));
}
_b.on("moveend", () => {
	Ub(), Cb.updateZoomLevel();
}), _b.on("singleclick", async (e) => {
	if (!mv.checked || fy.size === 0) return;
	Kv += 1;
	let t = Kv, n = Gv, r = hv.checked;
	try {
		let i = await qy(e.coordinate);
		if (n !== Gv || !mv.checked || fy.size === 0 || !r && t !== Kv || i === null) return;
		eb(i, e.pixel, t);
	} catch (e) {
		if (n !== Gv) return;
		console.error("Failed to inspect annotation.", e);
	}
});
function Bb() {
	for (let e of Object.values(uy)) {
		e.setSource(null), _b.removeLayer(e);
		let t = cb.indexOf(e);
		t !== -1 && cb.splice(t, 1);
	}
	for (let e of Object.keys(uy)) delete uy[e];
	dy.clear(), fy.clear(), hy.clear(), my.clear(), gy.clear(), _y.clear(), ey(), vy = "type", yy = null, by = null, Cy = [], wy.clear(), ob.render(), ry.render(), ab.updateActionState();
}
async function Vb() {
	await Qh(), Bb(), await Py();
}
function Hb() {
	let e = new URLSearchParams(window.location.search), t = Number(e.get("x")), n = Number(e.get("y")), r = Number(e.get("zoom"));
	return e.get("x") === null || e.get("y") === null || e.get("zoom") === null || !Number.isFinite(t) || !Number.isFinite(n) || !Number.isFinite(r) ? null : {
		center: [t, n],
		zoom: r
	};
}
function Ub() {
	if (ly === null) return;
	let e = _b.getView(), t = e.getCenter(), n = e.getZoom();
	if (t === void 0 || n === void 0) return;
	let r = new URL(window.location.href);
	r.searchParams.set("slide", ly), r.searchParams.set("x", t[0].toFixed(2)), r.searchParams.set("y", t[1].toFixed(2)), r.searchParams.set("zoom", n.toString());
	let i = r.searchParams.toString().replace(/%2F/gi, "/");
	window.history.replaceState({}, "", `${r.pathname}?${i}${r.hash}`);
}
async function Wb() {
	if (ay === null) throw Error("No TileServer session is available.");
	await $h(), Bb(), await Py(), ly = null, cy = null, iy.length = 0, oy += 1, sy += 1, lb.setSource(null), Ob.setSource(null), ry.render();
	let e = [
		0,
		-1,
		1,
		0
	], t = [1], n = new Zt({
		code: "ZoomifyProjectionEmpty",
		units: "pixels",
		extent: e,
		metersPerUnit: 1,
		getPointResolution(e) {
			return e;
		}
	});
	ir(n);
	let r = new us({
		projection: n,
		resolutions: t,
		constrainOnlyCenter: !0,
		center: [.5, -.5],
		resolution: t[0]
	});
	_b.setView(r), Ob.setView(n, e), Nb.setProjection(n, { preserveActive: !1 });
	let i = new URL(window.location.href);
	i.search = "", i.hash = "", window.history.replaceState({}, "", i), zb(!1), Cb.updateZoomLevel(), ab.updateActionState(), bb.updateButton();
}
async function Gb(e) {
	if (ay === null) throw Error("Dynamic slide switching requires a TileServer session.");
	Bb(), await Py();
	let t = await Xh(e);
	cy = t, ly = e, ab.setSlide(e), ab.updateOverlaySelect(), ab.updateActionState(), oy += 1;
	let n = T_(ay, t, oy), r = n.getTileGrid(), i = r.getExtent(), a = r.getResolutions(), o = new Zt({
		code: "ZoomifyProjection",
		units: "pixels",
		extent: i,
		metersPerUnit: t.mpp[0] * 1e-6
	});
	ir(o);
	let s = [(i[0] + i[2]) / 2, (i[1] + i[3]) / 2], c = new us({
		projection: o,
		resolutions: a,
		extent: hb(i),
		constrainOnlyCenter: !0,
		smoothExtentConstraint: !0,
		smoothResolutionConstraint: !1,
		center: s,
		resolution: a[0]
	});
	c.fit(i, { size: _b.getSize() }), _b.setView(c), Ob.setView(o, i), Nb.setProjection(o), lb.setSource(n), Ob.setSource(n), ry.render(), zb(!0), Ub(), Cb.updateZoomLevel();
}
async function Kb(e) {
	xb.begin();
	try {
		return await Gb(e);
	} finally {
		xb.end();
	}
}
ry.render();
async function qb(e) {
	if (ay === null || cy === null) throw Error("Dynamic overlay loading requires a loaded slide.");
	let t = e.split(".").pop().toLowerCase();
	if (t === "npy" || t === "mha") throw Error("Registration overlays are not supported yet.");
	let n = [
		"db",
		"dat",
		"geojson"
	].includes(t), r = l_(e), i = fy.has(r);
	if (r === "slide") throw Error("The overlay name \"slide\" is reserved.");
	let a = await eg(e, r);
	if (i && (Yv(), Qv(r)), n) {
		fy.add(r);
		let e = [...new Set(a)];
		hy.set(r, e);
		let t = /* @__PURE__ */ new Map(), n = {
			...rb,
			...ib[r] ?? {}
		}, i = e.some((e) => Object.hasOwn(n, String(e)));
		await __(t, e, (e) => og(e, Dy()), n), i && (xy = "custom");
		for (let [e, n] of t) py.has(e) || py.set(e, n);
		Vy(r, e, t), await ag(_y.get(r), r), xy === "custom" ? vy === "type" && await ig(my.get(r), r) : await Ry(xy, { refresh: !1 });
	} else i && (fy.delete(r), hy.delete(r), Hy(r));
	n || i ? (await Ly({ refresh: !1 }), n && await Gy(r, { refresh: !1 }), Uy(r)) : sy += 1, dy.set(r, e);
	let o = new hp({
		url: `/tileserver/layer/${encodeURIComponent(r)}/${ay}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${sy}`,
		size: cy.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	if (uy[r] !== void 0) uy[r].setSource(o), uy[r].setVisible(!0);
	else {
		let e = [lb, ...Object.values(uy)], t = Math.max(...e.map((e) => e.getZIndex() ?? 0)), n = new Ss({
			title: r,
			source: o,
			opacity: .75
		});
		n.setZIndex(t + 1), uy[r] = n, _b.addLayer(n), cb.push(n);
	}
	return ry.render(), ab.updateActionState(), a;
}
async function Jb(e) {
	xb.begin();
	try {
		return await qb(e);
	} finally {
		xb.end();
	}
}
async function Yb(e) {
	let t = uy[e], n = fy.has(e);
	if (t === void 0) throw Error(`Overlay is not loaded: ${e}`);
	let r = t.getSource();
	t.setVisible(!1), t.setSource(null), _b.removeLayer(t);
	try {
		await tg(e);
	} catch (e) {
		throw t.setSource(r), t.setVisible(!0), _b.addLayer(t), e;
	}
	let i = cb.indexOf(t);
	i !== -1 && cb.splice(i, 1), fy.delete(e), hy.delete(e), Hy(e), delete uy[e], dy.delete(e), n && (Yv(), Qv(e), await Ly({ refresh: !1 }), Uy()), ry.render(), ab.updateActionState();
}
async function Xb(e) {
	if (fy.size === 0) throw Error("No annotation overlay is loaded.");
	let t = e instanceof Map ? [...e.entries()] : Object.entries(e), n = new Map(t);
	await ig(n), py.clear();
	for (let [e, t] of n) py.set(e, t);
	Uy();
}
Object.assign(window, {
	clearOverlays: Vb,
	extent: fb,
	fullscreen: wb,
	graticule: jb,
	graticuleToggle: Pb,
	layerSwitcher: Ab,
	layers: cb,
	layersData: iy,
	loadOverlay: Jb,
	map: _b,
	mousePositionControl: Tb,
	overlayLayers: uy,
	overviewMapControl: kb,
	projection: pb,
	removeOverlay: Yb,
	removeSlide: Wb,
	resolutions: db,
	rotate: Eb,
	scaleLineControl: vb,
	screenSpaceGraticule: Mb,
	screenSpaceGraticuleToggle: Fb,
	setAnnotationColors: Xb,
	switchSlide: Kb,
	view: gb
});
