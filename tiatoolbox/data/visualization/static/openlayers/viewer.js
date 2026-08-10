//#region \0rolldown/runtime.js
var e = Object.create, t = Object.defineProperty, n = Object.getOwnPropertyDescriptor, r = Object.getOwnPropertyNames, i = Object.getPrototypeOf, a = Object.prototype.hasOwnProperty, o = (e, t) => () => (t || (e((t = { exports: {} }).exports, t), e = null), t.exports), s = (e, i, o, s) => {
	if (i && typeof i == "object" || typeof i == "function") for (var c = r(i), l = 0, u = c.length, d; l < u; l++) d = c[l], !a.call(e, d) && d !== o && t(e, d, {
		get: ((e) => i[e]).bind(null, d),
		enumerable: !(s = n(i, d)) || s.enumerable
	});
	return e;
}, c = (n, r, a) => (a = n == null ? {} : e(i(n)), s(r || !n || !n.__esModule ? t(a, "default", {
	value: n,
	enumerable: !0
}) : a, n)), l = function() {
	function e(e) {
		this.propagationStopped, this.defaultPrevented, this.type = e, this.target = null;
	}
	return e.prototype.preventDefault = function() {
		this.defaultPrevented = !0;
	}, e.prototype.stopPropagation = function() {
		this.propagationStopped = !0;
	}, e;
}(), u = { PROPERTYCHANGE: "propertychange" }, d = function() {
	function e() {
		this.disposed = !1;
	}
	return e.prototype.dispose = function() {
		this.disposed || (this.disposed = !0, this.disposeInternal());
	}, e.prototype.disposeInternal = function() {}, e;
}();
//#endregion
//#region node_modules/ol/array.js
function f(e, t) {
	return e > t ? 1 : e < t ? -1 : 0;
}
function p(e, t, n) {
	var r = e.length;
	if (e[0] <= t) return 0;
	if (t <= e[r - 1]) return r - 1;
	var i = void 0;
	if (n > 0) {
		for (i = 1; i < r; ++i) if (e[i] < t) return i - 1;
	} else if (n < 0) {
		for (i = 1; i < r; ++i) if (e[i] <= t) return i;
	} else for (i = 1; i < r; ++i) if (e[i] == t) return i;
	else if (e[i] < t) return typeof n == "function" ? n(t, e[i - 1], e[i]) > 0 ? i - 1 : i : e[i - 1] - t < t - e[i] ? i - 1 : i;
	return r - 1;
}
function m(e, t, n) {
	for (; t < n;) {
		var r = e[t];
		e[t] = e[n], e[n] = r, ++t, --n;
	}
}
function h(e, t) {
	for (var n = Array.isArray(t) ? t : [t], r = n.length, i = 0; i < r; i++) e[e.length] = n[i];
}
function g(e, t) {
	var n = e.length;
	if (n !== t.length) return !1;
	for (var r = 0; r < n; r++) if (e[r] !== t[r]) return !1;
	return !0;
}
function _(e, t, n) {
	var r = t || f;
	return e.every(function(t, i) {
		if (i === 0) return !0;
		var a = r(e[i - 1], t);
		return !(a > 0 || n && a === 0);
	});
}
//#endregion
//#region node_modules/ol/functions.js
function v() {
	return !0;
}
function y() {
	return !1;
}
function b() {}
function x(e) {
	var t = !1, n, r, i;
	return function() {
		var a = Array.prototype.slice.call(arguments);
		return (!t || this !== i || !g(a, r)) && (t = !0, i = this, r = a, n = e.apply(this, arguments)), n;
	};
}
//#endregion
//#region node_modules/ol/obj.js
var S = typeof Object.assign == "function" ? Object.assign : function(e, t) {
	if (e == null) throw TypeError("Cannot convert undefined or null to object");
	for (var n = Object(e), r = 1, i = arguments.length; r < i; ++r) {
		var a = arguments[r];
		if (a != null) for (var o in a) a.hasOwnProperty(o) && (n[o] = a[o]);
	}
	return n;
};
function C(e) {
	for (var t in e) delete e[t];
}
var w = typeof Object.values == "function" ? Object.values : function(e) {
	var t = [];
	for (var n in e) t.push(e[n]);
	return t;
};
function T(e) {
	for (var t in e) return !1;
	return !t;
}
//#endregion
//#region node_modules/ol/events/Target.js
var E = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), D = function(e) {
	E(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.eventTarget_ = t, n.pendingRemovals_ = null, n.dispatching_ = null, n.listeners_ = null, n;
	}
	return t.prototype.addEventListener = function(e, t) {
		if (!(!e || !t)) {
			var n = this.listeners_ ||= {}, r = n[e] || (n[e] = []);
			r.indexOf(t) === -1 && r.push(t);
		}
	}, t.prototype.dispatchEvent = function(e) {
		var t = typeof e == "string" ? new l(e) : e, n = t.type;
		t.target ||= this.eventTarget_ || this;
		var r = this.listeners_ && this.listeners_[n], i;
		if (r) {
			var a = this.dispatching_ ||= {}, o = this.pendingRemovals_ ||= {};
			n in a || (a[n] = 0, o[n] = 0), ++a[n];
			for (var s = 0, c = r.length; s < c; ++s) if (i = "handleEvent" in r[s] ? r[s].handleEvent(t) : r[s].call(this, t), i === !1 || t.propagationStopped) {
				i = !1;
				break;
			}
			if (--a[n], a[n] === 0) {
				var u = o[n];
				for (delete o[n]; u--;) this.removeEventListener(n, b);
				delete a[n];
			}
			return i;
		}
	}, t.prototype.disposeInternal = function() {
		this.listeners_ && C(this.listeners_);
	}, t.prototype.getListeners = function(e) {
		return this.listeners_ && this.listeners_[e] || void 0;
	}, t.prototype.hasListener = function(e) {
		return this.listeners_ ? e ? e in this.listeners_ : Object.keys(this.listeners_).length > 0 : !1;
	}, t.prototype.removeEventListener = function(e, t) {
		var n = this.listeners_ && this.listeners_[e];
		if (n) {
			var r = n.indexOf(t);
			r !== -1 && (this.pendingRemovals_ && e in this.pendingRemovals_ ? (n[r] = b, ++this.pendingRemovals_[e]) : (n.splice(r, 1), n.length === 0 && delete this.listeners_[e]));
		}
	}, t;
}(d), O = {
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
};
//#endregion
//#region node_modules/ol/events.js
function k(e, t, n, r, i) {
	if (r && r !== e && (n = n.bind(r)), i) {
		var a = n;
		n = function() {
			e.removeEventListener(t, n), a.apply(this, arguments);
		};
	}
	var o = {
		target: e,
		type: t,
		listener: n
	};
	return e.addEventListener(t, n), o;
}
function A(e, t, n, r) {
	return k(e, t, n, r, !0);
}
function j(e) {
	e && e.target && (e.target.removeEventListener(e.type, e.listener), C(e));
}
//#endregion
//#region node_modules/ol/Observable.js
var M = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), N = function(e) {
	M(t, e);
	function t() {
		var t = e.call(this) || this;
		return t.on = t.onInternal, t.once = t.onceInternal, t.un = t.unInternal, t.revision_ = 0, t;
	}
	return t.prototype.changed = function() {
		++this.revision_, this.dispatchEvent(O.CHANGE);
	}, t.prototype.getRevision = function() {
		return this.revision_;
	}, t.prototype.onInternal = function(e, t) {
		if (Array.isArray(e)) {
			for (var n = e.length, r = Array(n), i = 0; i < n; ++i) r[i] = k(this, e[i], t);
			return r;
		} else return k(this, e, t);
	}, t.prototype.onceInternal = function(e, t) {
		var n;
		if (Array.isArray(e)) {
			var r = e.length;
			n = Array(r);
			for (var i = 0; i < r; ++i) n[i] = A(this, e[i], t);
		} else n = A(this, e, t);
		return t.ol_key = n, n;
	}, t.prototype.unInternal = function(e, t) {
		var n = t.ol_key;
		if (n) P(n);
		else if (Array.isArray(e)) for (var r = 0, i = e.length; r < i; ++r) this.removeEventListener(e[r], t);
		else this.removeEventListener(e, t);
	}, t;
}(D);
N.prototype.on, N.prototype.once, N.prototype.un;
function P(e) {
	if (Array.isArray(e)) for (var t = 0, n = e.length; t < n; ++t) j(e[t]);
	else j(e);
}
//#endregion
//#region node_modules/ol/util.js
function F() {
	return (function() {
		throw Error("Unimplemented abstract method.");
	})();
}
var I = 0;
function L(e) {
	return e.ol_uid ||= String(++I);
}
var R = "6.9.0", ee = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), te = function(e) {
	ee(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.key = n, i.oldValue = r, i;
	}
	return t;
}(l), z = function(e) {
	ee(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.on, n.once, n.un, L(n), n.values_ = null, t !== void 0 && n.setProperties(t), n;
	}
	return t.prototype.get = function(e) {
		var t;
		return this.values_ && this.values_.hasOwnProperty(e) && (t = this.values_[e]), t;
	}, t.prototype.getKeys = function() {
		return this.values_ && Object.keys(this.values_) || [];
	}, t.prototype.getProperties = function() {
		return this.values_ && S({}, this.values_) || {};
	}, t.prototype.hasProperties = function() {
		return !!this.values_;
	}, t.prototype.notify = function(e, t) {
		var n = "change:" + e;
		this.dispatchEvent(new te(n, e, t)), n = u.PROPERTYCHANGE, this.dispatchEvent(new te(n, e, t));
	}, t.prototype.addChangeListener = function(e, t) {
		this.addEventListener("change:" + e, t);
	}, t.prototype.removeChangeListener = function(e, t) {
		this.removeEventListener("change:" + e, t);
	}, t.prototype.set = function(e, t, n) {
		var r = this.values_ ||= {};
		if (n) r[e] = t;
		else {
			var i = r[e];
			r[e] = t, i !== t && this.notify(e, i);
		}
	}, t.prototype.setProperties = function(e, t) {
		for (var n in e) this.set(n, e[n], t);
	}, t.prototype.applyProperties = function(e) {
		e.values_ && S(this.values_ ||= {}, e.values_);
	}, t.prototype.unset = function(e, t) {
		if (this.values_ && e in this.values_) {
			var n = this.values_[e];
			delete this.values_[e], T(this.values_) && (this.values_ = null), t || this.notify(e, n);
		}
	}, t;
}(N), ne = {
	POSTRENDER: "postrender",
	MOVESTART: "movestart",
	MOVEEND: "moveend"
}, re = typeof navigator < "u" && navigator.userAgent !== void 0 ? navigator.userAgent.toLowerCase() : "", ie = re.indexOf("firefox") !== -1;
re.indexOf("safari") !== -1 && re.indexOf("chrom");
var ae = re.indexOf("webkit") !== -1 && re.indexOf("edge") == -1, oe = re.indexOf("macintosh") !== -1, se = typeof devicePixelRatio < "u" ? devicePixelRatio : 1, ce = typeof WorkerGlobalScope < "u" && typeof OffscreenCanvas < "u" && self instanceof WorkerGlobalScope, le = typeof Image < "u" && Image.prototype.decode, ue = (function() {
	var e = !1;
	try {
		var t = Object.defineProperty({}, "passive", { get: function() {
			e = !0;
		} });
		window.addEventListener("_", null, t), window.removeEventListener("_", null, t);
	} catch {}
	return e;
})();
//#endregion
//#region node_modules/ol/dom.js
function de(e, t, n, r) {
	var i;
	return n && n.length ? i = n.shift() : ce ? i = new OffscreenCanvas(e || 300, t || 300) : (i = document.createElement("canvas"), i.style.all = "unset"), e && (i.width = e), t && (i.height = t), i.getContext("2d", r);
}
function fe(e) {
	var t = e.offsetWidth, n = getComputedStyle(e);
	return t += parseInt(n.marginLeft, 10) + parseInt(n.marginRight, 10), t;
}
function pe(e) {
	var t = e.offsetHeight, n = getComputedStyle(e);
	return t += parseInt(n.marginTop, 10) + parseInt(n.marginBottom, 10), t;
}
function me(e, t) {
	var n = t.parentNode;
	n && n.replaceChild(e, t);
}
function he(e) {
	return e && e.parentNode ? e.parentNode.removeChild(e) : null;
}
function ge(e) {
	for (; e.lastChild;) e.removeChild(e.lastChild);
}
function _e(e, t) {
	for (var n = e.childNodes, r = 0;; ++r) {
		var i = n[r], a = t[r];
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
//#endregion
//#region node_modules/ol/control/Control.js
var ve = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ye = function(e) {
	ve(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t.element;
		return r && !t.target && !r.style.pointerEvents && (r.style.pointerEvents = "auto"), n.element = r || null, n.target_ = null, n.map_ = null, n.listenerKeys = [], t.render && (n.render = t.render), t.target && n.setTarget(t.target), n;
	}
	return t.prototype.disposeInternal = function() {
		he(this.element), e.prototype.disposeInternal.call(this);
	}, t.prototype.getMap = function() {
		return this.map_;
	}, t.prototype.setMap = function(e) {
		this.map_ && he(this.element);
		for (var t = 0, n = this.listenerKeys.length; t < n; ++t) j(this.listenerKeys[t]);
		this.listenerKeys.length = 0, this.map_ = e, this.map_ && ((this.target_ ? this.target_ : e.getOverlayContainerStopEvent()).appendChild(this.element), this.render !== b && this.listenerKeys.push(k(e, ne.POSTRENDER, this.render, this)), e.render());
	}, t.prototype.render = function(e) {}, t.prototype.setTarget = function(e) {
		this.target_ = typeof e == "string" ? document.getElementById(e) : e;
	}, t;
}(z), be = "ol-hidden", xe = "ol-selectable", Se = "ol-unselectable", Ce = "ol-unsupported", we = "ol-control", Te = "ol-collapsed", Ee = new RegExp([
	"^\\s*(?=(?:(?:[-a-z]+\\s*){0,2}(italic|oblique))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(small-caps))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)",
	"(?:(?:normal|\\1|\\2|\\3)\\s*){0,3}((?:xx?-)?",
	"(?:small|large)|medium|smaller|larger|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx]))",
	"(?:\\s*\\/\\s*(normal|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx])?))",
	"?\\s*([-,\\\"\\'\\sa-z]+?)\\s*$"
].join(""), "i"), De = [
	"style",
	"variant",
	"weight",
	"size",
	"lineHeight",
	"family"
], Oe = function(e) {
	var t = e.match(Ee);
	if (!t) return null;
	for (var n = {
		lineHeight: "normal",
		size: "1.2em",
		style: "normal",
		weight: "normal",
		variant: "normal"
	}, r = 0, i = De.length; r < i; ++r) {
		var a = t[r + 1];
		a !== void 0 && (n[De[r]] = a);
	}
	return n.families = n.family.split(/,\s?/), n;
};
function ke(e) {
	return e === 1 ? "" : String(Math.round(e * 100) / 100);
}
//#endregion
//#region node_modules/ol/control/FullScreen.js
var Ae = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), je = [
	"fullscreenchange",
	"webkitfullscreenchange",
	"MSFullscreenChange"
], Me = {
	ENTERFULLSCREEN: "enterfullscreen",
	LEAVEFULLSCREEN: "leavefullscreen"
}, Ne = function(e) {
	Ae(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, {
			element: document.createElement("div"),
			target: r.target
		}) || this, n.on, n.once, n.un, n.cssClassName_ = r.className === void 0 ? "ol-full-screen" : r.className, n.activeClassName_ = r.activeClassName === void 0 ? [n.cssClassName_ + "-true"] : r.activeClassName.split(" "), n.inactiveClassName_ = r.inactiveClassName === void 0 ? [n.cssClassName_ + "-false"] : r.inactiveClassName.split(" ");
		var i = r.label === void 0 ? "⤢" : r.label;
		n.labelNode_ = typeof i == "string" ? document.createTextNode(i) : i;
		var a = r.labelActive === void 0 ? "×" : r.labelActive;
		n.labelActiveNode_ = typeof a == "string" ? document.createTextNode(a) : a, n.button_ = document.createElement("button");
		var o = r.tipLabel ? r.tipLabel : "Toggle full-screen";
		n.setClassName_(n.button_, Fe()), n.button_.setAttribute("type", "button"), n.button_.title = o, n.button_.appendChild(n.labelNode_), n.button_.addEventListener(O.CLICK, n.handleClick_.bind(n), !1);
		var s = n.cssClassName_ + " " + Se + " " + we + " " + (Pe() ? "" : Ce), c = n.element;
		return c.className = s, c.appendChild(n.button_), n.keys_ = r.keys !== void 0 && r.keys, n.source_ = r.source, n;
	}
	return t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.handleFullScreen_();
	}, t.prototype.handleFullScreen_ = function() {
		if (Pe()) {
			var e = this.getMap();
			if (e) if (Fe()) Re();
			else {
				var t = void 0;
				t = this.source_ ? typeof this.source_ == "string" ? document.getElementById(this.source_) : this.source_ : e.getTargetElement(), this.keys_ ? Le(t) : Ie(t);
			}
		}
	}, t.prototype.handleFullScreenChange_ = function() {
		var e = this.getMap();
		Fe() ? (this.setClassName_(this.button_, !0), me(this.labelActiveNode_, this.labelNode_), this.dispatchEvent(Me.ENTERFULLSCREEN)) : (this.setClassName_(this.button_, !1), me(this.labelNode_, this.labelActiveNode_), this.dispatchEvent(Me.LEAVEFULLSCREEN)), e && e.updateSize();
	}, t.prototype.setClassName_ = function(e, t) {
		var n, r, i, a = this.activeClassName_, o = this.inactiveClassName_, s = t ? a : o;
		(n = e.classList).remove.apply(n, a), (r = e.classList).remove.apply(r, o), (i = e.classList).add.apply(i, s);
	}, t.prototype.setMap = function(t) {
		if (e.prototype.setMap.call(this, t), t) for (var n = 0, r = je.length; n < r; ++n) this.listenerKeys.push(k(document, je[n], this.handleFullScreenChange_, this));
	}, t;
}(ye);
function Pe() {
	var e = document.body;
	return !!(e.webkitRequestFullscreen || e.msRequestFullscreen && document.msFullscreenEnabled || e.requestFullscreen && document.fullscreenEnabled);
}
function Fe() {
	return !!(document.webkitIsFullScreen || document.msFullscreenElement || document.fullscreenElement);
}
function Ie(e) {
	e.requestFullscreen ? e.requestFullscreen() : e.msRequestFullscreen ? e.msRequestFullscreen() : e.webkitRequestFullscreen && e.webkitRequestFullscreen();
}
function Le(e) {
	e.webkitRequestFullscreen ? e.webkitRequestFullscreen() : Ie(e);
}
function Re() {
	document.exitFullscreen ? document.exitFullscreen() : document.msExitFullscreen ? document.msExitFullscreen() : document.webkitExitFullscreen && document.webkitExitFullscreen();
}
//#endregion
//#region node_modules/ol/pointer/EventType.js
var ze = {
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, B = {
	RADIANS: "radians",
	DEGREES: "degrees",
	FEET: "ft",
	METERS: "m",
	PIXELS: "pixels",
	TILE_PIXELS: "tile-pixels",
	USFEET: "us-ft"
};
B.METERS, B.FEET, B.USFEET, B.RADIANS, B.DEGREES;
var Be = {};
Be[B.RADIANS] = 6370997 / (2 * Math.PI), Be[B.DEGREES] = 2 * Math.PI * 6370997 / 360, Be[B.FEET] = .3048, Be[B.METERS] = 1, Be[B.USFEET] = 1200 / 3937;
//#endregion
//#region node_modules/ol/proj/Projection.js
var Ve = function() {
	function e(e) {
		this.code_ = e.code, this.units_ = e.units, this.extent_ = e.extent === void 0 ? null : e.extent, this.worldExtent_ = e.worldExtent === void 0 ? null : e.worldExtent, this.axisOrientation_ = e.axisOrientation === void 0 ? "enu" : e.axisOrientation, this.global_ = e.global !== void 0 && e.global, this.canWrapX_ = !!(this.global_ && this.extent_), this.getPointResolutionFunc_ = e.getPointResolution, this.defaultTileGrid_ = null, this.metersPerUnit_ = e.metersPerUnit;
	}
	return e.prototype.canWrapX = function() {
		return this.canWrapX_;
	}, e.prototype.getCode = function() {
		return this.code_;
	}, e.prototype.getExtent = function() {
		return this.extent_;
	}, e.prototype.getUnits = function() {
		return this.units_;
	}, e.prototype.getMetersPerUnit = function() {
		return this.metersPerUnit_ || Be[this.units_];
	}, e.prototype.getWorldExtent = function() {
		return this.worldExtent_;
	}, e.prototype.getAxisOrientation = function() {
		return this.axisOrientation_;
	}, e.prototype.isGlobal = function() {
		return this.global_;
	}, e.prototype.setGlobal = function(e) {
		this.global_ = e, this.canWrapX_ = !!(e && this.extent_);
	}, e.prototype.getDefaultTileGrid = function() {
		return this.defaultTileGrid_;
	}, e.prototype.setDefaultTileGrid = function(e) {
		this.defaultTileGrid_ = e;
	}, e.prototype.setExtent = function(e) {
		this.extent_ = e, this.canWrapX_ = !!(this.global_ && e);
	}, e.prototype.setWorldExtent = function(e) {
		this.worldExtent_ = e;
	}, e.prototype.setGetPointResolution = function(e) {
		this.getPointResolutionFunc_ = e;
	}, e.prototype.getPointResolutionFunc = function() {
		return this.getPointResolutionFunc_;
	}, e;
}();
//#endregion
//#region node_modules/ol/math.js
function V(e, t, n) {
	return Math.min(Math.max(e, t), n);
}
var He = (function() {
	return "cosh" in Math ? Math.cosh : function(e) {
		var t = Math.exp(e);
		return (t + 1 / t) / 2;
	};
})(), Ue = (function() {
	return "log2" in Math ? Math.log2 : function(e) {
		return Math.log(e) * Math.LOG2E;
	};
})();
function We(e, t, n, r, i, a) {
	var o = i - n, s = a - r;
	if (o !== 0 || s !== 0) {
		var c = ((e - n) * o + (t - r) * s) / (o * o + s * s);
		c > 1 ? (n = i, r = a) : c > 0 && (n += o * c, r += s * c);
	}
	return Ge(e, t, n, r);
}
function Ge(e, t, n, r) {
	var i = n - e, a = r - t;
	return i * i + a * a;
}
function Ke(e) {
	for (var t = e.length, n = 0; n < t; n++) {
		for (var r = n, i = Math.abs(e[n][n]), a = n + 1; a < t; a++) {
			var o = Math.abs(e[a][n]);
			o > i && (i = o, r = a);
		}
		if (i === 0) return null;
		var s = e[r];
		e[r] = e[n], e[n] = s;
		for (var c = n + 1; c < t; c++) for (var l = -e[c][n] / e[n][n], u = n; u < t + 1; u++) n == u ? e[c][u] = 0 : e[c][u] += l * e[n][u];
	}
	for (var d = Array(t), f = t - 1; f >= 0; f--) {
		d[f] = e[f][t] / e[f][f];
		for (var p = f - 1; p >= 0; p--) e[p][t] -= e[p][f] * d[f];
	}
	return d;
}
function qe(e) {
	return e * Math.PI / 180;
}
function Je(e, t) {
	var n = e % t;
	return n * t < 0 ? n + t : n;
}
function Ye(e, t, n) {
	return e + n * (t - e);
}
//#endregion
//#region node_modules/ol/proj/epsg3857.js
var Xe = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ze = 6378137, Qe = Math.PI * Ze, $e = [
	-Qe,
	-Qe,
	Qe,
	Qe
], et = [
	-180,
	-85,
	180,
	85
], tt = Ze * Math.log(Math.tan(Math.PI / 2)), nt = function(e) {
	Xe(t, e);
	function t(t) {
		return e.call(this, {
			code: t,
			units: B.METERS,
			extent: $e,
			global: !0,
			worldExtent: et,
			getPointResolution: function(e, t) {
				return e / He(t[1] / 6378137);
			}
		}) || this;
	}
	return t;
}(Ve), rt = [
	new nt("EPSG:3857"),
	new nt("EPSG:102100"),
	new nt("EPSG:102113"),
	new nt("EPSG:900913"),
	new nt("http://www.opengis.net/def/crs/EPSG/0/3857"),
	new nt("http://www.opengis.net/gml/srs/epsg.xml#3857")
];
function it(e, t, n) {
	var r = e.length, i = n > 1 ? n : 2, a = t;
	a === void 0 && (a = i > 2 ? e.slice() : Array(r));
	for (var o = 0; o < r; o += i) {
		a[o] = Qe * e[o] / 180;
		var s = Ze * Math.log(Math.tan(Math.PI * (+e[o + 1] + 90) / 360));
		s > tt ? s = tt : s < -tt && (s = -tt), a[o + 1] = s;
	}
	return a;
}
function at(e, t, n) {
	var r = e.length, i = n > 1 ? n : 2, a = t;
	a === void 0 && (a = i > 2 ? e.slice() : Array(r));
	for (var o = 0; o < r; o += i) a[o] = 180 * e[o] / Qe, a[o + 1] = 360 * Math.atan(Math.exp(e[o + 1] / Ze)) / Math.PI - 90;
	return a;
}
//#endregion
//#region node_modules/ol/proj/epsg4326.js
var ot = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), st = 6378137, ct = [
	-180,
	-90,
	180,
	90
], lt = Math.PI * st / 180, ut = function(e) {
	ot(t, e);
	function t(t, n) {
		return e.call(this, {
			code: t,
			units: B.DEGREES,
			extent: ct,
			axisOrientation: n,
			global: !0,
			metersPerUnit: lt,
			worldExtent: ct
		}) || this;
	}
	return t;
}(Ve), dt = [
	new ut("CRS:84"),
	new ut("EPSG:4326", "neu"),
	new ut("urn:ogc:def:crs:OGC:1.3:CRS84"),
	new ut("urn:ogc:def:crs:OGC:2:84"),
	new ut("http://www.opengis.net/def/crs/OGC/1.3/CRS84", "neu"),
	new ut("http://www.opengis.net/gml/srs/epsg.xml#4326", "neu"),
	new ut("http://www.opengis.net/def/crs/EPSG/0/4326", "neu")
], ft = {};
function pt(e) {
	return ft[e] || ft[e.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\w+)$/, "EPSG:$3")] || null;
}
function mt(e, t) {
	ft[e] = t;
}
//#endregion
//#region node_modules/ol/proj/transforms.js
var ht = {};
function gt(e, t, n) {
	var r = e.getCode(), i = t.getCode();
	r in ht || (ht[r] = {}), ht[r][i] = n;
}
function _t(e, t) {
	var n;
	return e in ht && t in ht[e] && (n = ht[e][t]), n;
}
//#endregion
//#region node_modules/ol/extent/Corner.js
var vt = {
	BOTTOM_LEFT: "bottom-left",
	BOTTOM_RIGHT: "bottom-right",
	TOP_LEFT: "top-left",
	TOP_RIGHT: "top-right"
}, yt = {
	UNKNOWN: 0,
	INTERSECTING: 1,
	ABOVE: 2,
	RIGHT: 4,
	BELOW: 8,
	LEFT: 16
}, bt = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), xt = function(e) {
	bt(t, e);
	function t(t) {
		var n = this, r = "Assertion failed. See https://openlayers.org/en/" + ("v" + R.split("-")[0]) + "/doc/errors/#" + t + " for details.";
		return n = e.call(this, r) || this, n.code = t, n.name = "AssertionError", n.message = r, n;
	}
	return t;
}(Error);
//#endregion
//#region node_modules/ol/asserts.js
function H(e, t) {
	if (!e) throw new xt(t);
}
//#endregion
//#region node_modules/ol/extent.js
function St(e) {
	for (var t = At(), n = 0, r = e.length; n < r; ++n) Lt(t, e[n]);
	return t;
}
function Ct(e, t, n) {
	return n ? (n[0] = e[0] - t, n[1] = e[1] - t, n[2] = e[2] + t, n[3] = e[3] + t, n) : [
		e[0] - t,
		e[1] - t,
		e[2] + t,
		e[3] + t
	];
}
function wt(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e.slice();
}
function Tt(e, t, n) {
	var r = t < e[0] ? e[0] - t : e[2] < t ? t - e[2] : 0, i = n < e[1] ? e[1] - n : e[3] < n ? n - e[3] : 0;
	return r * r + i * i;
}
function Et(e, t) {
	return Ot(e, t[0], t[1]);
}
function Dt(e, t) {
	return e[0] <= t[0] && t[2] <= e[2] && e[1] <= t[1] && t[3] <= e[3];
}
function Ot(e, t, n) {
	return e[0] <= t && t <= e[2] && e[1] <= n && n <= e[3];
}
function kt(e, t) {
	var n = e[0], r = e[1], i = e[2], a = e[3], o = t[0], s = t[1], c = yt.UNKNOWN;
	return o < n ? c |= yt.LEFT : o > i && (c |= yt.RIGHT), s < r ? c |= yt.BELOW : s > a && (c |= yt.ABOVE), c === yt.UNKNOWN && (c = yt.INTERSECTING), c;
}
function At() {
	return [
		Infinity,
		Infinity,
		-Infinity,
		-Infinity
	];
}
function jt(e, t, n, r, i) {
	return i ? (i[0] = e, i[1] = t, i[2] = n, i[3] = r, i) : [
		e,
		t,
		n,
		r
	];
}
function Mt(e) {
	return jt(Infinity, Infinity, -Infinity, -Infinity, e);
}
function Nt(e, t) {
	var n = e[0], r = e[1];
	return jt(n, r, n, r, t);
}
function Pt(e, t, n, r, i) {
	return Rt(Mt(i), e, t, n, r);
}
function Ft(e, t) {
	return e[0] == t[0] && e[2] == t[2] && e[1] == t[1] && e[3] == t[3];
}
function It(e, t) {
	return t[0] < e[0] && (e[0] = t[0]), t[2] > e[2] && (e[2] = t[2]), t[1] < e[1] && (e[1] = t[1]), t[3] > e[3] && (e[3] = t[3]), e;
}
function Lt(e, t) {
	t[0] < e[0] && (e[0] = t[0]), t[0] > e[2] && (e[2] = t[0]), t[1] < e[1] && (e[1] = t[1]), t[1] > e[3] && (e[3] = t[1]);
}
function Rt(e, t, n, r, i) {
	for (; n < r; n += i) zt(e, t[n], t[n + 1]);
	return e;
}
function zt(e, t, n) {
	e[0] = Math.min(e[0], t), e[1] = Math.min(e[1], n), e[2] = Math.max(e[2], t), e[3] = Math.max(e[3], n);
}
function Bt(e, t) {
	var n = t(Ht(e));
	return n || (n = t(Ut(e)), n) || (n = t(Xt(e)), n) || (n = t(Yt(e)), n) ? n : !1;
}
function Vt(e) {
	var t = 0;
	return Qt(e) || (t = U(e) * qt(e)), t;
}
function Ht(e) {
	return [e[0], e[1]];
}
function Ut(e) {
	return [e[2], e[1]];
}
function Wt(e) {
	return [(e[0] + e[2]) / 2, (e[1] + e[3]) / 2];
}
function Gt(e, t) {
	var n;
	return t === vt.BOTTOM_LEFT ? n = Ht(e) : t === vt.BOTTOM_RIGHT ? n = Ut(e) : t === vt.TOP_LEFT ? n = Yt(e) : t === vt.TOP_RIGHT ? n = Xt(e) : H(!1, 13), n;
}
function Kt(e, t, n, r, i) {
	var a = t * r[0] / 2, o = t * r[1] / 2, s = Math.cos(n), c = Math.sin(n), l = a * s, u = a * c, d = o * s, f = o * c, p = e[0], m = e[1], h = p - l + f, g = p - l - f, _ = p + l - f, v = p + l + f, y = m - u - d, b = m - u + d, x = m + u + d, S = m + u - d;
	return jt(Math.min(h, g, _, v), Math.min(y, b, x, S), Math.max(h, g, _, v), Math.max(y, b, x, S), i);
}
function qt(e) {
	return e[3] - e[1];
}
function Jt(e, t, n) {
	var r = n || At();
	return Zt(e, t) ? (e[0] > t[0] ? r[0] = e[0] : r[0] = t[0], e[1] > t[1] ? r[1] = e[1] : r[1] = t[1], e[2] < t[2] ? r[2] = e[2] : r[2] = t[2], e[3] < t[3] ? r[3] = e[3] : r[3] = t[3]) : Mt(r), r;
}
function Yt(e) {
	return [e[0], e[3]];
}
function Xt(e) {
	return [e[2], e[3]];
}
function U(e) {
	return e[2] - e[0];
}
function Zt(e, t) {
	return e[0] <= t[2] && e[2] >= t[0] && e[1] <= t[3] && e[3] >= t[1];
}
function Qt(e) {
	return e[2] < e[0] || e[3] < e[1];
}
function $t(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e;
}
function en(e, t) {
	var n = (e[2] - e[0]) / 2 * (t - 1), r = (e[3] - e[1]) / 2 * (t - 1);
	e[0] -= n, e[2] += n, e[1] -= r, e[3] += r;
}
function tn(e, t, n) {
	var r = !1, i = kt(e, t), a = kt(e, n);
	if (i === yt.INTERSECTING || a === yt.INTERSECTING) r = !0;
	else {
		var o = e[0], s = e[1], c = e[2], l = e[3], u = t[0], d = t[1], f = n[0], p = n[1], m = (p - d) / (f - u), h = void 0, g = void 0;
		a & yt.ABOVE && !(i & yt.ABOVE) && (h = f - (p - l) / m, r = h >= o && h <= c), !r && a & yt.RIGHT && !(i & yt.RIGHT) && (g = p - (f - c) * m, r = g >= s && g <= l), !r && a & yt.BELOW && !(i & yt.BELOW) && (h = f - (p - s) / m, r = h >= o && h <= c), !r && a & yt.LEFT && !(i & yt.LEFT) && (g = p - (f - o) * m, r = g >= s && g <= l);
	}
	return r;
}
function nn(e, t) {
	var n = t.getExtent(), r = Wt(e);
	if (t.canWrapX() && (r[0] < n[0] || r[0] >= n[2])) {
		var i = U(n), a = Math.floor((r[0] - n[0]) / i) * i;
		e[0] -= a, e[2] -= a;
	}
	return e;
}
//#endregion
//#region node_modules/ol/geom/GeometryType.js
var W = {
	POINT: "Point",
	LINE_STRING: "LineString",
	LINEAR_RING: "LinearRing",
	POLYGON: "Polygon",
	MULTI_POINT: "MultiPoint",
	MULTI_LINE_STRING: "MultiLineString",
	MULTI_POLYGON: "MultiPolygon",
	GEOMETRY_COLLECTION: "GeometryCollection",
	CIRCLE: "Circle"
};
function rn(e, t, n) {
	var r = n || 6371008.8, i = qe(e[1]), a = qe(t[1]), o = (a - i) / 2, s = qe(t[0] - e[0]) / 2, c = Math.sin(o) * Math.sin(o) + Math.sin(s) * Math.sin(s) * Math.cos(i) * Math.cos(a);
	return 2 * r * Math.atan2(Math.sqrt(c), Math.sqrt(1 - c));
}
//#endregion
//#region node_modules/ol/coordinate.js
function an(e, t) {
	return e[0] += +t[0], e[1] += +t[1], e;
}
function on(e, t, n) {
	return e ? t.replace("{x}", e[0].toFixed(n)).replace("{y}", e[1].toFixed(n)) : "";
}
function sn(e, t) {
	for (var n = !0, r = e.length - 1; r >= 0; --r) if (e[r] != t[r]) {
		n = !1;
		break;
	}
	return n;
}
function cn(e, t) {
	var n = Math.cos(t), r = Math.sin(t), i = e[0] * n - e[1] * r, a = e[1] * n + e[0] * r;
	return e[0] = i, e[1] = a, e;
}
function ln(e, t) {
	return e[0] *= t, e[1] *= t, e;
}
function un(e, t) {
	if (t.canWrapX()) {
		var n = U(t.getExtent()), r = dn(e, t, n);
		r && (e[0] -= r * n);
	}
	return e;
}
function dn(e, t, n) {
	var r = t.getExtent(), i = 0;
	if (t.canWrapX() && (e[0] < r[0] || e[0] > r[2])) {
		var a = n || U(r);
		i = Math.floor((e[0] - r[0]) / a);
	}
	return i;
}
//#endregion
//#region node_modules/ol/proj.js
function fn(e, t, n) {
	var r;
	if (t !== void 0) {
		for (var i = 0, a = e.length; i < a; ++i) t[i] = e[i];
		r = t;
	} else r = e.slice();
	return r;
}
function pn(e, t, n) {
	if (t !== void 0 && e !== t) {
		for (var r = 0, i = e.length; r < i; ++r) t[r] = e[r];
		e = t;
	}
	return e;
}
function mn(e) {
	mt(e.getCode(), e), gt(e, e, fn);
}
function hn(e) {
	e.forEach(mn);
}
function gn(e) {
	return typeof e == "string" ? pt(e) : e || null;
}
function _n(e, t, n, r) {
	e = gn(e);
	var i, a = e.getPointResolutionFunc();
	if (a) {
		if (i = a(t, n), r && r !== e.getUnits()) {
			var o = e.getMetersPerUnit();
			o && (i = i * o / Be[r]);
		}
	} else {
		var s = e.getUnits();
		if (s == B.DEGREES && !r || r == B.DEGREES) i = t;
		else {
			var c = Sn(e, gn("EPSG:4326"));
			if (c === pn && s !== B.DEGREES) i = t * e.getMetersPerUnit();
			else {
				var l = [
					n[0] - t / 2,
					n[1],
					n[0] + t / 2,
					n[1],
					n[0],
					n[1] - t / 2,
					n[0],
					n[1] + t / 2
				];
				l = c(l, l, 2), i = (rn(l.slice(0, 2), l.slice(2, 4)) + rn(l.slice(4, 6), l.slice(6, 8))) / 2;
			}
			var o = r ? Be[r] : e.getMetersPerUnit();
			o !== void 0 && (i /= o);
		}
	}
	return i;
}
function vn(e) {
	hn(e), e.forEach(function(t) {
		e.forEach(function(e) {
			t !== e && gt(t, e, fn);
		});
	});
}
function yn(e, t, n, r) {
	e.forEach(function(e) {
		t.forEach(function(t) {
			gt(e, t, n), gt(t, e, r);
		});
	});
}
function bn(e, t) {
	return e ? typeof e == "string" ? gn(e) : e : gn(t);
}
function xn(e, t) {
	if (e === t) return !0;
	var n = e.getUnits() === t.getUnits();
	return (e.getCode() === t.getCode() || Sn(e, t) === fn) && n;
}
function Sn(e, t) {
	var n = _t(e.getCode(), t.getCode());
	return n ||= pn, n;
}
function Cn(e, t) {
	return Sn(gn(e), gn(t));
}
function wn(e, t, n) {
	return Cn(t, n)(e, void 0, e.length);
}
var Tn = null;
function En() {
	return Tn;
}
function Dn(e, t) {
	return e;
}
function On(e, t) {
	return e;
}
function kn(e, t) {
	return e;
}
function An(e, t) {
	return e;
}
function jn(e, t) {
	return e;
}
function Mn() {
	vn(rt), vn(dt), yn(dt, rt, it, at);
}
Mn();
//#endregion
//#region node_modules/ol/control/MousePosition.js
var Nn = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Pn = "projection", Fn = "coordinateFormat", In = function(e) {
	Nn(t, e);
	function t(t) {
		var n = this, r = t || {}, i = document.createElement("div");
		i.className = r.className === void 0 ? "ol-mouse-position" : r.className, n = e.call(this, {
			element: i,
			render: r.render,
			target: r.target
		}) || this, n.on, n.once, n.un, n.addChangeListener(Pn, n.handleProjectionChanged_), r.coordinateFormat && n.setCoordinateFormat(r.coordinateFormat), r.projection && n.setProjection(r.projection);
		var a = !0, o = "&#160;";
		return "undefinedHTML" in r ? (r.undefinedHTML !== void 0 && (o = r.undefinedHTML), a = !!o) : "placeholder" in r && (r.placeholder === !1 ? a = !1 : o = String(r.placeholder)), n.placeholder_ = o, n.renderOnMouseOut_ = a, n.renderedHTML_ = i.innerHTML, n.mapProjection_ = null, n.transform_ = null, n;
	}
	return t.prototype.handleProjectionChanged_ = function() {
		this.transform_ = null;
	}, t.prototype.getCoordinateFormat = function() {
		return this.get(Fn);
	}, t.prototype.getProjection = function() {
		return this.get(Pn);
	}, t.prototype.handleMouseMove = function(e) {
		var t = this.getMap();
		this.updateHTML_(t.getEventPixel(e));
	}, t.prototype.handleMouseOut = function(e) {
		this.updateHTML_(null);
	}, t.prototype.setMap = function(t) {
		if (e.prototype.setMap.call(this, t), t) {
			var n = t.getViewport();
			this.listenerKeys.push(k(n, ze.POINTERMOVE, this.handleMouseMove, this)), this.renderOnMouseOut_ && this.listenerKeys.push(k(n, ze.POINTEROUT, this.handleMouseOut, this)), this.updateHTML_(null);
		}
	}, t.prototype.setCoordinateFormat = function(e) {
		this.set(Fn, e);
	}, t.prototype.setProjection = function(e) {
		this.set(Pn, gn(e));
	}, t.prototype.updateHTML_ = function(e) {
		var t = this.placeholder_;
		if (e && this.mapProjection_) {
			if (!this.transform_) {
				var n = this.getProjection();
				n ? this.transform_ = Sn(this.mapProjection_, n) : this.transform_ = pn;
			}
			var r = this.getMap().getCoordinateFromPixelInternal(e);
			if (r) {
				var i = En();
				i && (this.transform_ = Sn(this.mapProjection_, i)), this.transform_(r, r);
				var a = this.getCoordinateFormat();
				t = a ? a(r) : r.toString();
			}
		}
		(!this.renderedHTML_ || t !== this.renderedHTML_) && (this.element.innerHTML = t, this.renderedHTML_ = t);
	}, t.prototype.render = function(e) {
		var t = e.frameState;
		t ? this.mapProjection_ != t.viewState.projection && (this.mapProjection_ = t.viewState.projection, this.transform_ = null) : this.mapProjection_ = null;
	}, t;
}(ye), Ln = [
	,
	,
	,
	,
	,
	,
];
function Rn() {
	return [
		1,
		0,
		0,
		1,
		0,
		0
	];
}
function zn(e) {
	return Vn(e, 1, 0, 0, 1, 0, 0);
}
function Bn(e, t) {
	var n = e[0], r = e[1], i = e[2], a = e[3], o = e[4], s = e[5], c = t[0], l = t[1], u = t[2], d = t[3], f = t[4], p = t[5];
	return e[0] = n * c + i * l, e[1] = r * c + a * l, e[2] = n * u + i * d, e[3] = r * u + a * d, e[4] = n * f + i * p + o, e[5] = r * f + a * p + s, e;
}
function Vn(e, t, n, r, i, a, o) {
	return e[0] = t, e[1] = n, e[2] = r, e[3] = i, e[4] = a, e[5] = o, e;
}
function Hn(e, t) {
	return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e;
}
function G(e, t) {
	var n = t[0], r = t[1];
	return t[0] = e[0] * n + e[2] * r + e[4], t[1] = e[1] * n + e[3] * r + e[5], t;
}
function Un(e, t) {
	var n = Math.cos(t), r = Math.sin(t);
	return Bn(e, Vn(Ln, n, r, -r, n, 0, 0));
}
function Wn(e, t, n) {
	return Bn(e, Vn(Ln, t, 0, 0, n, 0, 0));
}
function Gn(e, t, n) {
	return Vn(e, t, 0, 0, n, 0, 0);
}
function Kn(e, t, n) {
	return Bn(e, Vn(Ln, 1, 0, 0, 1, t, n));
}
function qn(e, t, n, r, i, a, o, s) {
	var c = Math.sin(a), l = Math.cos(a);
	return e[0] = r * l, e[1] = i * c, e[2] = -r * c, e[3] = i * l, e[4] = o * r * l - s * r * c + t, e[5] = o * i * c + s * i * l + n, e;
}
function Jn(e, t) {
	var n = Yn(t);
	H(n !== 0, 32);
	var r = t[0], i = t[1], a = t[2], o = t[3], s = t[4], c = t[5];
	return e[0] = o / n, e[1] = -i / n, e[2] = -a / n, e[3] = r / n, e[4] = (a * c - o * s) / n, e[5] = -(r * c - i * s) / n, e;
}
function Yn(e) {
	return e[0] * e[3] - e[1] * e[2];
}
var Xn;
function Zn(e) {
	var t = "matrix(" + e.join(", ") + ")";
	if (ce) return t;
	var n = Xn ||= document.createElement("div");
	return n.style.transform = t, n.style.transform;
}
//#endregion
//#region node_modules/ol/color.js
var Qn = /^#([a-f0-9]{3}|[a-f0-9]{4}(?:[a-f0-9]{2}){0,2})$/i, $n = /^([a-z]*)$|^hsla?\(.*\)$/i;
function er(e) {
	return typeof e == "string" ? e : or(e);
}
function tr(e) {
	var t = document.createElement("div");
	if (t.style.color = e, t.style.color !== "") {
		document.body.appendChild(t);
		var n = getComputedStyle(t).color;
		return document.body.removeChild(t), n;
	} else return "";
}
var nr = (function() {
	var e = {}, t = 0;
	return (function(n) {
		var r;
		if (e.hasOwnProperty(n)) r = e[n];
		else {
			if (t >= 1024) {
				var i = 0;
				for (var a in e) i++ & 3 || (delete e[a], --t);
			}
			r = ir(n), e[n] = r, ++t;
		}
		return r;
	});
})();
function rr(e) {
	return Array.isArray(e) ? e : nr(e);
}
function ir(e) {
	var t, n, r, i, a;
	if ($n.exec(e) && (e = tr(e)), Qn.exec(e)) {
		var o = e.length - 1, s = void 0;
		s = o <= 4 ? 1 : 2;
		var c = o === 4 || o === 8;
		t = parseInt(e.substr(1 + 0 * s, s), 16), n = parseInt(e.substr(1 + 1 * s, s), 16), r = parseInt(e.substr(1 + 2 * s, s), 16), i = c ? parseInt(e.substr(1 + 3 * s, s), 16) : 255, s == 1 && (t = (t << 4) + t, n = (n << 4) + n, r = (r << 4) + r, c && (i = (i << 4) + i)), a = [
			t,
			n,
			r,
			i / 255
		];
	} else e.indexOf("rgba(") == 0 ? (a = e.slice(5, -1).split(",").map(Number), ar(a)) : e.indexOf("rgb(") == 0 ? (a = e.slice(4, -1).split(",").map(Number), a.push(1), ar(a)) : H(!1, 14);
	return a;
}
function ar(e) {
	return e[0] = V(e[0] + .5 | 0, 0, 255), e[1] = V(e[1] + .5 | 0, 0, 255), e[2] = V(e[2] + .5 | 0, 0, 255), e[3] = V(e[3], 0, 1), e;
}
function or(e) {
	var t = e[0];
	t != (t | 0) && (t = t + .5 | 0);
	var n = e[1];
	n != (n | 0) && (n = n + .5 | 0);
	var r = e[2];
	r != (r | 0) && (r = r + .5 | 0);
	var i = e[3] === void 0 ? 1 : e[3];
	return "rgba(" + t + "," + n + "," + r + "," + i + ")";
}
//#endregion
//#region node_modules/ol/style/IconImageCache.js
var sr = function() {
	function e() {
		this.cache_ = {}, this.cacheSize_ = 0, this.maxCacheSize_ = 32;
	}
	return e.prototype.clear = function() {
		this.cache_ = {}, this.cacheSize_ = 0;
	}, e.prototype.canExpireCache = function() {
		return this.cacheSize_ > this.maxCacheSize_;
	}, e.prototype.expire = function() {
		if (this.canExpireCache()) {
			var e = 0;
			for (var t in this.cache_) {
				var n = this.cache_[t];
				!(e++ & 3) && !n.hasListener() && (delete this.cache_[t], --this.cacheSize_);
			}
		}
	}, e.prototype.get = function(e, t, n) {
		var r = cr(e, t, n);
		return r in this.cache_ ? this.cache_[r] : null;
	}, e.prototype.set = function(e, t, n, r) {
		var i = cr(e, t, n);
		this.cache_[i] = r, ++this.cacheSize_;
	}, e.prototype.setSize = function(e) {
		this.maxCacheSize_ = e, this.expire();
	}, e;
}();
function cr(e, t, n) {
	var r = n ? er(n) : "null";
	return t + ":" + e + ":" + r;
}
var lr = new sr(), K = {
	OPACITY: "opacity",
	VISIBLE: "visible",
	EXTENT: "extent",
	Z_INDEX: "zIndex",
	MAX_RESOLUTION: "maxResolution",
	MIN_RESOLUTION: "minResolution",
	MAX_ZOOM: "maxZoom",
	MIN_ZOOM: "minZoom",
	SOURCE: "source"
}, ur = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), dr = function(e) {
	ur(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = S({}, t);
		return typeof t.properties == "object" && (delete r.properties, S(r, t.properties)), r[K.OPACITY] = t.opacity === void 0 ? 1 : t.opacity, H(typeof r[K.OPACITY] == "number", 64), r[K.VISIBLE] = t.visible === void 0 || t.visible, r[K.Z_INDEX] = t.zIndex, r[K.MAX_RESOLUTION] = t.maxResolution === void 0 ? Infinity : t.maxResolution, r[K.MIN_RESOLUTION] = t.minResolution === void 0 ? 0 : t.minResolution, r[K.MIN_ZOOM] = t.minZoom === void 0 ? -Infinity : t.minZoom, r[K.MAX_ZOOM] = t.maxZoom === void 0 ? Infinity : t.maxZoom, n.className_ = r.className === void 0 ? "ol-layer" : t.className, delete r.className, n.setProperties(r), n.state_ = null, n;
	}
	return t.prototype.getClassName = function() {
		return this.className_;
	}, t.prototype.getLayerState = function(e) {
		var t = this.state_ || {
			layer: this,
			managed: e === void 0 || e
		}, n = this.getZIndex();
		return t.opacity = V(Math.round(this.getOpacity() * 100) / 100, 0, 1), t.sourceState = this.getSourceState(), t.visible = this.getVisible(), t.extent = this.getExtent(), t.zIndex = n === void 0 && !t.managed ? Infinity : n, t.maxResolution = this.getMaxResolution(), t.minResolution = Math.max(this.getMinResolution(), 0), t.minZoom = this.getMinZoom(), t.maxZoom = this.getMaxZoom(), this.state_ = t, t;
	}, t.prototype.getLayersArray = function(e) {
		return F();
	}, t.prototype.getLayerStatesArray = function(e) {
		return F();
	}, t.prototype.getExtent = function() {
		return this.get(K.EXTENT);
	}, t.prototype.getMaxResolution = function() {
		return this.get(K.MAX_RESOLUTION);
	}, t.prototype.getMinResolution = function() {
		return this.get(K.MIN_RESOLUTION);
	}, t.prototype.getMinZoom = function() {
		return this.get(K.MIN_ZOOM);
	}, t.prototype.getMaxZoom = function() {
		return this.get(K.MAX_ZOOM);
	}, t.prototype.getOpacity = function() {
		return this.get(K.OPACITY);
	}, t.prototype.getSourceState = function() {
		return F();
	}, t.prototype.getVisible = function() {
		return this.get(K.VISIBLE);
	}, t.prototype.getZIndex = function() {
		return this.get(K.Z_INDEX);
	}, t.prototype.setExtent = function(e) {
		this.set(K.EXTENT, e);
	}, t.prototype.setMaxResolution = function(e) {
		this.set(K.MAX_RESOLUTION, e);
	}, t.prototype.setMinResolution = function(e) {
		this.set(K.MIN_RESOLUTION, e);
	}, t.prototype.setMaxZoom = function(e) {
		this.set(K.MAX_ZOOM, e);
	}, t.prototype.setMinZoom = function(e) {
		this.set(K.MIN_ZOOM, e);
	}, t.prototype.setOpacity = function(e) {
		H(typeof e == "number", 64), this.set(K.OPACITY, e);
	}, t.prototype.setVisible = function(e) {
		this.set(K.VISIBLE, e);
	}, t.prototype.setZIndex = function(e) {
		this.set(K.Z_INDEX, e);
	}, t.prototype.disposeInternal = function() {
		this.state_ &&= (this.state_.layer = null, null), e.prototype.disposeInternal.call(this);
	}, t;
}(z), fr = {
	PRERENDER: "prerender",
	POSTRENDER: "postrender",
	PRECOMPOSE: "precompose",
	POSTCOMPOSE: "postcompose",
	RENDERCOMPLETE: "rendercomplete"
}, pr = {
	UNDEFINED: "undefined",
	LOADING: "loading",
	READY: "ready",
	ERROR: "error"
}, mr = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), hr = function(e) {
	mr(t, e);
	function t(t) {
		var n = this, r = S({}, t);
		delete r.source, n = e.call(this, r) || this, n.on, n.once, n.un, n.mapPrecomposeKey_ = null, n.mapRenderKey_ = null, n.sourceChangeKey_ = null, n.renderer_ = null, t.render && (n.render = t.render), t.map && n.setMap(t.map), n.addChangeListener(K.SOURCE, n.handleSourcePropertyChange_);
		var i = t.source ? t.source : null;
		return n.setSource(i), n;
	}
	return t.prototype.getLayersArray = function(e) {
		var t = e || [];
		return t.push(this), t;
	}, t.prototype.getLayerStatesArray = function(e) {
		var t = e || [];
		return t.push(this.getLayerState()), t;
	}, t.prototype.getSource = function() {
		return this.get(K.SOURCE) || null;
	}, t.prototype.getSourceState = function() {
		var e = this.getSource();
		return e ? e.getState() : pr.UNDEFINED;
	}, t.prototype.handleSourceChange_ = function() {
		this.changed();
	}, t.prototype.handleSourcePropertyChange_ = function() {
		this.sourceChangeKey_ &&= (j(this.sourceChangeKey_), null);
		var e = this.getSource();
		e && (this.sourceChangeKey_ = k(e, O.CHANGE, this.handleSourceChange_, this)), this.changed();
	}, t.prototype.getFeatures = function(e) {
		return this.renderer_ ? this.renderer_.getFeatures(e) : new Promise(function(e) {
			return e([]);
		});
	}, t.prototype.render = function(e, t) {
		var n = this.getRenderer();
		if (n.prepareFrame(e)) return n.renderFrame(e, t);
	}, t.prototype.setMap = function(e) {
		this.mapPrecomposeKey_ &&= (j(this.mapPrecomposeKey_), null), e || this.changed(), this.mapRenderKey_ &&= (j(this.mapRenderKey_), null), e && (this.mapPrecomposeKey_ = k(e, fr.PRECOMPOSE, function(e) {
			var t = e.frameState.layerStatesArray, n = this.getLayerState(!1);
			H(!t.some(function(e) {
				return e.layer === n.layer;
			}), 67), t.push(n);
		}, this), this.mapRenderKey_ = k(this, O.CHANGE, e.render, e), this.changed());
	}, t.prototype.setSource = function(e) {
		this.set(K.SOURCE, e);
	}, t.prototype.getRenderer = function() {
		return this.renderer_ ||= this.createRenderer(), this.renderer_;
	}, t.prototype.hasRenderer = function() {
		return !!this.renderer_;
	}, t.prototype.createRenderer = function() {
		return null;
	}, t.prototype.disposeInternal = function() {
		this.renderer_ && (this.renderer_.dispose(), delete this.renderer_), this.setSource(null), e.prototype.disposeInternal.call(this);
	}, t;
}(dr);
function gr(e, t) {
	if (!e.visible) return !1;
	var n = t.resolution;
	if (n < e.minResolution || n >= e.maxResolution) return !1;
	var r = t.zoom;
	return r > e.minZoom && r <= e.maxZoom;
}
//#endregion
//#region node_modules/ol/renderer/Map.js
var _r = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), vr = function(e) {
	_r(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.map_ = t, n;
	}
	return t.prototype.dispatchRenderEvent = function(e, t) {
		F();
	}, t.prototype.calculateMatrices2D = function(e) {
		var t = e.viewState, n = e.coordinateToPixelTransform, r = e.pixelToCoordinateTransform;
		qn(n, e.size[0] / 2, e.size[1] / 2, 1 / t.resolution, -1 / t.resolution, -t.rotation, -t.center[0], -t.center[1]), Jn(r, n);
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i, a, o, s) {
		var c, l = t.viewState;
		function u(e, t, n, r) {
			return i.call(a, t, e ? n : null, r);
		}
		var d = l.projection, f = un(e.slice(), d), p = [[0, 0]];
		if (d.canWrapX() && r) {
			var m = U(d.getExtent());
			p.push([-m, 0], [m, 0]);
		}
		for (var h = t.layerStatesArray, g = h.length, _ = [], v = [], y = 0; y < p.length; y++) for (var b = g - 1; b >= 0; --b) {
			var x = h[b], S = x.layer;
			if (S.hasRenderer() && gr(x, l) && o.call(s, S)) {
				var C = S.getRenderer(), w = S.getSource();
				if (C && w) {
					var T = w.getWrapX() ? f : e, E = u.bind(null, x.managed);
					v[0] = T[0] + p[y][0], v[1] = T[1] + p[y][1], c = C.forEachFeatureAtCoordinate(v, t, n, E, _);
				}
				if (c) return c;
			}
		}
		if (_.length !== 0) {
			var D = 1 / _.length;
			return _.forEach(function(e, t) {
				return e.distanceSq += t * D;
			}), _.sort(function(e, t) {
				return e.distanceSq - t.distanceSq;
			}), _.some(function(e) {
				return c = e.callback(e.feature, e.layer, e.geometry);
			}), c;
		}
	}, t.prototype.forEachLayerAtPixel = function(e, t, n, r, i) {
		return F();
	}, t.prototype.hasFeatureAtCoordinate = function(e, t, n, r, i, a) {
		return this.forEachFeatureAtCoordinate(e, t, n, r, v, this, i, a) !== void 0;
	}, t.prototype.getMap = function() {
		return this.map_;
	}, t.prototype.renderFrame = function(e) {
		F();
	}, t.prototype.scheduleExpireIconCache = function(e) {
		lr.canExpireCache() && e.postRenderFunctions.push(yr);
	}, t;
}(d);
function yr(e, t) {
	lr.expire();
}
//#endregion
//#region node_modules/ol/render/Event.js
var br = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), xr = function(e) {
	br(t, e);
	function t(t, n, r, i) {
		var a = e.call(this, t) || this;
		return a.inversePixelTransform = n, a.frameState = r, a.context = i, a;
	}
	return t;
}(l), Sr = "10px sans-serif", Cr = "#000", wr = "round", Tr = [], Er = "round", Dr = "#000", Or = "center", kr = "middle", Ar = [
	0,
	0,
	0,
	0
], jr = new z(), Mr = new D();
Mr.setSize = function() {
	console.warn("labelCache is deprecated.");
};
var Nr = null, Pr, Fr = {}, Ir = (function() {
	var e = ["monospace", "serif"], t = e.length, n = "wmytzilWMYTZIL@#/&?$%10", r, i;
	function a(r, a, o) {
		for (var s = !0, c = 0; c < t; ++c) {
			var l = e[c];
			if (i = zr(r + " " + a + " 32px " + l, n), o != l) {
				var u = zr(r + " " + a + " 32px " + o + "," + l, n);
				s &&= u != i;
			}
		}
		return !!s;
	}
	function o() {
		for (var e = !0, t = jr.getKeys(), n = 0, i = t.length; n < i; ++n) {
			var o = t[n];
			jr.get(o) < 100 && (a.apply(this, o.split("\n")) ? (C(Fr), Nr = null, Pr = void 0, jr.set(o, 100)) : (jr.set(o, jr.get(o) + 1, !0), e = !1));
		}
		e && (clearInterval(r), r = void 0);
	}
	return function(e) {
		var t = Oe(e);
		if (t) for (var n = t.families, i = 0, s = n.length; i < s; ++i) {
			var c = n[i], l = t.style + "\n" + t.weight + "\n" + c;
			jr.get(l) === void 0 && (jr.set(l, 100, !0), a(t.style, t.weight, c) || (jr.set(l, 0, !0), r === void 0 && (r = setInterval(o, 32))));
		}
	};
})(), Lr = (function() {
	var e;
	return function(t) {
		var n = Fr[t];
		if (n == null) {
			if (ce) {
				var r = Oe(t), i = Rr(t, "Žg");
				n = (isNaN(Number(r.lineHeight)) ? 1.2 : Number(r.lineHeight)) * (i.actualBoundingBoxAscent + i.actualBoundingBoxDescent);
			} else e || (e = document.createElement("div"), e.innerHTML = "M", e.style.minHeight = "0", e.style.maxHeight = "none", e.style.height = "auto", e.style.padding = "0", e.style.border = "none", e.style.position = "absolute", e.style.display = "block", e.style.left = "-99999px"), e.style.font = t, document.body.appendChild(e), n = e.offsetHeight, document.body.removeChild(e);
			Fr[t] = n;
		}
		return n;
	};
})();
function Rr(e, t) {
	return Nr ||= de(1, 1), e != Pr && (Nr.font = e, Pr = Nr.font), Nr.measureText(t);
}
function zr(e, t) {
	return Rr(e, t).width;
}
function Br(e, t, n) {
	if (t in n) return n[t];
	var r = zr(e, t);
	return n[t] = r, r;
}
function Vr(e, t, n) {
	for (var r = t.length, i = 0, a = 0; a < r; ++a) {
		var o = zr(e, t[a]);
		i = Math.max(i, o), n.push(o);
	}
	return i;
}
function Hr(e, t, n, r, i, a, o, s, c, l, u) {
	e.save(), n !== 1 && (e.globalAlpha *= n), t && e.setTransform.apply(e, t), r.contextInstructions ? (e.translate(c, l), e.scale(u[0], u[1]), Ur(r, e)) : u[0] < 0 || u[1] < 0 ? (e.translate(c, l), e.scale(u[0], u[1]), e.drawImage(r, i, a, o, s, 0, 0, o, s)) : e.drawImage(r, i, a, o, s, c, l, o * u[0], s * u[1]), e.restore();
}
function Ur(e, t) {
	for (var n = e.contextInstructions, r = 0, i = n.length; r < i; r += 2) Array.isArray(n[r + 1]) ? t[n[r]].apply(t, n[r + 1]) : t[n[r]] = n[r + 1];
}
//#endregion
//#region node_modules/ol/renderer/Composite.js
var Wr = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Gr = function(e) {
	Wr(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		n.fontChangeListenerKey_ = k(jr, u.PROPERTYCHANGE, t.redrawText.bind(t)), n.element_ = document.createElement("div");
		var r = n.element_.style;
		r.position = "absolute", r.width = "100%", r.height = "100%", r.zIndex = "0", n.element_.className = Se + " ol-layers";
		var i = t.getViewport();
		return i.insertBefore(n.element_, i.firstChild || null), n.children_ = [], n.renderedVisible_ = !0, n;
	}
	return t.prototype.dispatchRenderEvent = function(e, t) {
		var n = this.getMap();
		if (n.hasListener(e)) {
			var r = new xr(e, void 0, t);
			n.dispatchEvent(r);
		}
	}, t.prototype.disposeInternal = function() {
		j(this.fontChangeListenerKey_), this.element_.parentNode.removeChild(this.element_), e.prototype.disposeInternal.call(this);
	}, t.prototype.renderFrame = function(e) {
		if (!e) {
			this.renderedVisible_ &&= (this.element_.style.display = "none", !1);
			return;
		}
		this.calculateMatrices2D(e), this.dispatchRenderEvent(fr.PRECOMPOSE, e);
		var t = e.layerStatesArray.sort(function(e, t) {
			return e.zIndex - t.zIndex;
		}), n = e.viewState;
		this.children_.length = 0;
		for (var r = [], i = null, a = 0, o = t.length; a < o; ++a) {
			var s = t[a];
			if (e.layerIndex = a, !(!gr(s, n) || s.sourceState != pr.READY && s.sourceState != pr.UNDEFINED)) {
				var c = s.layer, l = c.render(e, i);
				l && (l !== i && (this.children_.push(l), i = l), "getDeclutter" in c && r.push(c));
			}
		}
		for (var a = r.length - 1; a >= 0; --a) r[a].renderDeclutter(e);
		_e(this.element_, this.children_), this.dispatchRenderEvent(fr.POSTCOMPOSE, e), this.renderedVisible_ ||= (this.element_.style.display = "", !0), this.scheduleExpireIconCache(e);
	}, t.prototype.forEachLayerAtPixel = function(e, t, n, r, i) {
		for (var a = t.viewState, o = t.layerStatesArray, s = o.length - 1; s >= 0; --s) {
			var c = o[s], l = c.layer;
			if (l.hasRenderer() && gr(c, a) && i(l)) {
				var u = l.getRenderer().getDataAtPixel(e, t, n);
				if (u) {
					var d = r(l, u);
					if (d) return d;
				}
			}
		}
	}, t;
}(vr), Kr = {
	LAYERGROUP: "layergroup",
	SIZE: "size",
	TARGET: "target",
	VIEW: "view"
}, qr = {
	BOTTOM_LEFT: "bottom-left",
	BOTTOM_CENTER: "bottom-center",
	BOTTOM_RIGHT: "bottom-right",
	CENTER_LEFT: "center-left",
	CENTER_CENTER: "center-center",
	CENTER_RIGHT: "center-right",
	TOP_LEFT: "top-left",
	TOP_CENTER: "top-center",
	TOP_RIGHT: "top-right"
}, Jr = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Yr = {
	ELEMENT: "element",
	MAP: "map",
	OFFSET: "offset",
	POSITION: "position",
	POSITIONING: "positioning"
}, Xr = function(e) {
	Jr(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un, n.options = t, n.id = t.id, n.insertFirst = t.insertFirst === void 0 || t.insertFirst, n.stopEvent = t.stopEvent === void 0 || t.stopEvent, n.element = document.createElement("div"), n.element.className = t.className === void 0 ? "ol-overlay-container " + xe : t.className, n.element.style.position = "absolute", n.element.style.pointerEvents = "auto";
		var r = t.autoPan;
		return r && typeof r != "object" && (r = {
			animation: t.autoPanAnimation,
			margin: t.autoPanMargin
		}), n.autoPan = r || !1, n.rendered = {
			transform_: "",
			visible: !0
		}, n.mapPostrenderListenerKey = null, n.addChangeListener(Yr.ELEMENT, n.handleElementChanged), n.addChangeListener(Yr.MAP, n.handleMapChanged), n.addChangeListener(Yr.OFFSET, n.handleOffsetChanged), n.addChangeListener(Yr.POSITION, n.handlePositionChanged), n.addChangeListener(Yr.POSITIONING, n.handlePositioningChanged), t.element !== void 0 && n.setElement(t.element), n.setOffset(t.offset === void 0 ? [0, 0] : t.offset), n.setPositioning(t.positioning === void 0 ? qr.TOP_LEFT : t.positioning), t.position !== void 0 && n.setPosition(t.position), n;
	}
	return t.prototype.getElement = function() {
		return this.get(Yr.ELEMENT);
	}, t.prototype.getId = function() {
		return this.id;
	}, t.prototype.getMap = function() {
		return this.get(Yr.MAP);
	}, t.prototype.getOffset = function() {
		return this.get(Yr.OFFSET);
	}, t.prototype.getPosition = function() {
		return this.get(Yr.POSITION);
	}, t.prototype.getPositioning = function() {
		return this.get(Yr.POSITIONING);
	}, t.prototype.handleElementChanged = function() {
		ge(this.element);
		var e = this.getElement();
		e && this.element.appendChild(e);
	}, t.prototype.handleMapChanged = function() {
		this.mapPostrenderListenerKey &&= (he(this.element), j(this.mapPostrenderListenerKey), null);
		var e = this.getMap();
		if (e) {
			this.mapPostrenderListenerKey = k(e, ne.POSTRENDER, this.render, this), this.updatePixelPosition();
			var t = this.stopEvent ? e.getOverlayContainerStopEvent() : e.getOverlayContainer();
			this.insertFirst ? t.insertBefore(this.element, t.childNodes[0] || null) : t.appendChild(this.element), this.performAutoPan();
		}
	}, t.prototype.render = function() {
		this.updatePixelPosition();
	}, t.prototype.handleOffsetChanged = function() {
		this.updatePixelPosition();
	}, t.prototype.handlePositionChanged = function() {
		this.updatePixelPosition(), this.performAutoPan();
	}, t.prototype.handlePositioningChanged = function() {
		this.updatePixelPosition();
	}, t.prototype.setElement = function(e) {
		this.set(Yr.ELEMENT, e);
	}, t.prototype.setMap = function(e) {
		this.set(Yr.MAP, e);
	}, t.prototype.setOffset = function(e) {
		this.set(Yr.OFFSET, e);
	}, t.prototype.setPosition = function(e) {
		this.set(Yr.POSITION, e);
	}, t.prototype.performAutoPan = function() {
		this.autoPan && this.panIntoView(this.autoPan);
	}, t.prototype.panIntoView = function(e) {
		var t = this.getMap();
		if (!(!t || !t.getTargetElement() || !this.get(Yr.POSITION))) {
			var n = this.getRect(t.getTargetElement(), t.getSize()), r = this.getElement(), i = this.getRect(r, [fe(r), pe(r)]), a = e || {}, o = a.margin === void 0 ? 20 : a.margin;
			if (!Dt(n, i)) {
				var s = i[0] - n[0], c = n[2] - i[2], l = i[1] - n[1], u = n[3] - i[3], d = [0, 0];
				if (s < 0 ? d[0] = s - o : c < 0 && (d[0] = Math.abs(c) + o), l < 0 ? d[1] = l - o : u < 0 && (d[1] = Math.abs(u) + o), d[0] !== 0 || d[1] !== 0) {
					var f = t.getView().getCenterInternal(), p = t.getPixelFromCoordinateInternal(f);
					if (!p) return;
					var m = [p[0] + d[0], p[1] + d[1]], h = a.animation || {};
					t.getView().animateInternal({
						center: t.getCoordinateFromPixelInternal(m),
						duration: h.duration,
						easing: h.easing
					});
				}
			}
		}
	}, t.prototype.getRect = function(e, t) {
		var n = e.getBoundingClientRect(), r = n.left + window.pageXOffset, i = n.top + window.pageYOffset;
		return [
			r,
			i,
			r + t[0],
			i + t[1]
		];
	}, t.prototype.setPositioning = function(e) {
		this.set(Yr.POSITIONING, e);
	}, t.prototype.setVisible = function(e) {
		this.rendered.visible !== e && (this.element.style.display = e ? "" : "none", this.rendered.visible = e);
	}, t.prototype.updatePixelPosition = function() {
		var e = this.getMap(), t = this.getPosition();
		if (!e || !e.isRendered() || !t) {
			this.setVisible(!1);
			return;
		}
		var n = e.getPixelFromCoordinate(t), r = e.getSize();
		this.updateRenderedPosition(n, r);
	}, t.prototype.updateRenderedPosition = function(e, t) {
		var n = this.element.style, r = this.getOffset(), i = this.getPositioning();
		this.setVisible(!0);
		var a = Math.round(e[0] + r[0]) + "px", o = Math.round(e[1] + r[1]) + "px", s = "0%", c = "0%";
		i == qr.BOTTOM_RIGHT || i == qr.CENTER_RIGHT || i == qr.TOP_RIGHT ? s = "-100%" : (i == qr.BOTTOM_CENTER || i == qr.CENTER_CENTER || i == qr.TOP_CENTER) && (s = "-50%"), i == qr.BOTTOM_LEFT || i == qr.BOTTOM_CENTER || i == qr.BOTTOM_RIGHT ? c = "-100%" : (i == qr.CENTER_LEFT || i == qr.CENTER_CENTER || i == qr.CENTER_RIGHT) && (c = "-50%");
		var l = "translate(" + s + ", " + c + ") translate(" + a + ", " + o + ")";
		this.rendered.transform_ != l && (this.rendered.transform_ = l, n.transform = l, n.msTransform = l);
	}, t.prototype.getOptions = function() {
		return this.options;
	}, t;
}(z), Zr = {
	ADD: "add",
	REMOVE: "remove"
}, Qr = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), $r = { LENGTH: "length" }, ei = function(e) {
	Qr(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.element = n, i.index = r, i;
	}
	return t;
}(l), ti = function(e) {
	Qr(t, e);
	function t(t, n) {
		var r = e.call(this) || this;
		if (r.on, r.once, r.un, r.unique_ = !!(n || {}).unique, r.array_ = t || [], r.unique_) for (var i = 0, a = r.array_.length; i < a; ++i) r.assertUnique_(r.array_[i], i);
		return r.updateLength_(), r;
	}
	return t.prototype.clear = function() {
		for (; this.getLength() > 0;) this.pop();
	}, t.prototype.extend = function(e) {
		for (var t = 0, n = e.length; t < n; ++t) this.push(e[t]);
		return this;
	}, t.prototype.forEach = function(e) {
		for (var t = this.array_, n = 0, r = t.length; n < r; ++n) e(t[n], n, t);
	}, t.prototype.getArray = function() {
		return this.array_;
	}, t.prototype.item = function(e) {
		return this.array_[e];
	}, t.prototype.getLength = function() {
		return this.get($r.LENGTH);
	}, t.prototype.insertAt = function(e, t) {
		this.unique_ && this.assertUnique_(t), this.array_.splice(e, 0, t), this.updateLength_(), this.dispatchEvent(new ei(Zr.ADD, t, e));
	}, t.prototype.pop = function() {
		return this.removeAt(this.getLength() - 1);
	}, t.prototype.push = function(e) {
		this.unique_ && this.assertUnique_(e);
		var t = this.getLength();
		return this.insertAt(t, e), this.getLength();
	}, t.prototype.remove = function(e) {
		for (var t = this.array_, n = 0, r = t.length; n < r; ++n) if (t[n] === e) return this.removeAt(n);
	}, t.prototype.removeAt = function(e) {
		var t = this.array_[e];
		return this.array_.splice(e, 1), this.updateLength_(), this.dispatchEvent(new ei(Zr.REMOVE, t, e)), t;
	}, t.prototype.setAt = function(e, t) {
		var n = this.getLength();
		if (e < n) {
			this.unique_ && this.assertUnique_(t, e);
			var r = this.array_[e];
			this.array_[e] = t, this.dispatchEvent(new ei(Zr.REMOVE, r, e)), this.dispatchEvent(new ei(Zr.ADD, t, e));
		} else {
			for (var i = n; i < e; ++i) this.insertAt(i, void 0);
			this.insertAt(e, t);
		}
	}, t.prototype.updateLength_ = function() {
		this.set($r.LENGTH, this.array_.length);
	}, t.prototype.assertUnique_ = function(e, t) {
		for (var n = 0, r = this.array_.length; n < r; ++n) if (this.array_[n] === e && n !== t) throw new xt(58);
	}, t;
}(z), ni = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ri = { LAYERS: "layers" }, ii = function(e) {
	ni(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		delete i.layers;
		var a = r.layers;
		return n = e.call(this, i) || this, n.on, n.once, n.un, n.layersListenerKeys_ = [], n.listenerKeys_ = {}, n.addChangeListener(ri.LAYERS, n.handleLayersChanged_), a ? Array.isArray(a) ? a = new ti(a.slice(), { unique: !0 }) : H(typeof a.getArray == "function", 43) : a = new ti(void 0, { unique: !0 }), n.setLayers(a), n;
	}
	return t.prototype.handleLayerChange_ = function() {
		this.changed();
	}, t.prototype.handleLayersChanged_ = function() {
		this.layersListenerKeys_.forEach(j), this.layersListenerKeys_.length = 0;
		var e = this.getLayers();
		for (var t in this.layersListenerKeys_.push(k(e, Zr.ADD, this.handleLayersAdd_, this), k(e, Zr.REMOVE, this.handleLayersRemove_, this)), this.listenerKeys_) this.listenerKeys_[t].forEach(j);
		C(this.listenerKeys_);
		for (var n = e.getArray(), r = 0, i = n.length; r < i; r++) {
			var a = n[r];
			this.listenerKeys_[L(a)] = [k(a, u.PROPERTYCHANGE, this.handleLayerChange_, this), k(a, O.CHANGE, this.handleLayerChange_, this)];
		}
		this.changed();
	}, t.prototype.handleLayersAdd_ = function(e) {
		var t = e.element;
		this.listenerKeys_[L(t)] = [k(t, u.PROPERTYCHANGE, this.handleLayerChange_, this), k(t, O.CHANGE, this.handleLayerChange_, this)], this.changed();
	}, t.prototype.handleLayersRemove_ = function(e) {
		var t = e.element, n = L(t);
		this.listenerKeys_[n].forEach(j), delete this.listenerKeys_[n], this.changed();
	}, t.prototype.getLayers = function() {
		return this.get(ri.LAYERS);
	}, t.prototype.setLayers = function(e) {
		this.set(ri.LAYERS, e);
	}, t.prototype.getLayersArray = function(e) {
		var t = e === void 0 ? [] : e;
		return this.getLayers().forEach(function(e) {
			e.getLayersArray(t);
		}), t;
	}, t.prototype.getLayerStatesArray = function(e) {
		var t = e === void 0 ? [] : e, n = t.length;
		this.getLayers().forEach(function(e) {
			e.getLayerStatesArray(t);
		});
		var r = this.getLayerState(), i = r.zIndex;
		!e && r.zIndex === void 0 && (i = 0);
		for (var a = n, o = t.length; a < o; a++) {
			var s = t[a];
			s.opacity *= r.opacity, s.visible = s.visible && r.visible, s.maxResolution = Math.min(s.maxResolution, r.maxResolution), s.minResolution = Math.max(s.minResolution, r.minResolution), s.minZoom = Math.max(s.minZoom, r.minZoom), s.maxZoom = Math.min(s.maxZoom, r.maxZoom), r.extent !== void 0 && (s.extent === void 0 ? s.extent = r.extent : s.extent = Jt(s.extent, r.extent)), s.zIndex === void 0 && (s.zIndex = i);
		}
		return t;
	}, t.prototype.getSourceState = function() {
		return pr.READY;
	}, t;
}(dr), ai = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), oi = function(e) {
	ai(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.map = n, i.frameState = r === void 0 ? null : r, i;
	}
	return t;
}(l), si = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ci = function(e) {
	si(t, e);
	function t(t, n, r, i, a) {
		var o = e.call(this, t, n, a) || this;
		return o.originalEvent = r, o.pixel_ = null, o.coordinate_ = null, o.dragging = i !== void 0 && i, o;
	}
	return Object.defineProperty(t.prototype, "pixel", {
		get: function() {
			return this.pixel_ ||= this.map.getEventPixel(this.originalEvent), this.pixel_;
		},
		set: function(e) {
			this.pixel_ = e;
		},
		enumerable: !1,
		configurable: !0
	}), Object.defineProperty(t.prototype, "coordinate", {
		get: function() {
			return this.coordinate_ ||= this.map.getCoordinateFromPixel(this.pixel), this.coordinate_;
		},
		set: function(e) {
			this.coordinate_ = e;
		},
		enumerable: !1,
		configurable: !0
	}), t.prototype.preventDefault = function() {
		e.prototype.preventDefault.call(this), "preventDefault" in this.originalEvent && this.originalEvent.preventDefault();
	}, t.prototype.stopPropagation = function() {
		e.prototype.stopPropagation.call(this), "stopPropagation" in this.originalEvent && this.originalEvent.stopPropagation();
	}, t;
}(oi), q = {
	SINGLECLICK: "singleclick",
	CLICK: O.CLICK,
	DBLCLICK: O.DBLCLICK,
	POINTERDRAG: "pointerdrag",
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, li = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ui = function(e) {
	li(t, e);
	function t(t, n) {
		var r = e.call(this, t) || this;
		r.map_ = t, r.clickTimeoutId_, r.emulateClicks_ = !1, r.dragging_ = !1, r.dragListenerKeys_ = [], r.moveTolerance_ = n === void 0 ? 1 : n, r.down_ = null;
		var i = r.map_.getViewport();
		return r.activePointers_ = 0, r.trackedTouches_ = {}, r.element_ = i, r.pointerdownListenerKey_ = k(i, ze.POINTERDOWN, r.handlePointerDown_, r), r.originalPointerMoveEvent_, r.relayedListenerKey_ = k(i, ze.POINTERMOVE, r.relayEvent_, r), r.boundHandleTouchMove_ = r.handleTouchMove_.bind(r), r.element_.addEventListener(O.TOUCHMOVE, r.boundHandleTouchMove_, ue ? { passive: !1 } : !1), r;
	}
	return t.prototype.emulateClick_ = function(e) {
		var t = new ci(q.CLICK, this.map_, e);
		this.dispatchEvent(t), this.clickTimeoutId_ === void 0 ? this.clickTimeoutId_ = setTimeout(function() {
			this.clickTimeoutId_ = void 0;
			var t = new ci(q.SINGLECLICK, this.map_, e);
			this.dispatchEvent(t);
		}.bind(this), 250) : (clearTimeout(this.clickTimeoutId_), this.clickTimeoutId_ = void 0, t = new ci(q.DBLCLICK, this.map_, e), this.dispatchEvent(t));
	}, t.prototype.updateActivePointers_ = function(e) {
		var t = e;
		t.type == q.POINTERUP || t.type == q.POINTERCANCEL ? delete this.trackedTouches_[t.pointerId] : t.type == q.POINTERDOWN && (this.trackedTouches_[t.pointerId] = !0), this.activePointers_ = Object.keys(this.trackedTouches_).length;
	}, t.prototype.handlePointerUp_ = function(e) {
		this.updateActivePointers_(e);
		var t = new ci(q.POINTERUP, this.map_, e);
		this.dispatchEvent(t), this.emulateClicks_ && !t.defaultPrevented && !this.dragging_ && this.isMouseActionButton_(e) && this.emulateClick_(this.down_), this.activePointers_ === 0 && (this.dragListenerKeys_.forEach(j), this.dragListenerKeys_.length = 0, this.dragging_ = !1, this.down_ = null);
	}, t.prototype.isMouseActionButton_ = function(e) {
		return e.button === 0;
	}, t.prototype.handlePointerDown_ = function(e) {
		this.emulateClicks_ = this.activePointers_ === 0, this.updateActivePointers_(e);
		var t = new ci(q.POINTERDOWN, this.map_, e);
		for (var n in this.dispatchEvent(t), this.down_ = {}, e) {
			var r = e[n];
			this.down_[n] = typeof r == "function" ? b : r;
		}
		if (this.dragListenerKeys_.length === 0) {
			var i = this.map_.getOwnerDocument();
			this.dragListenerKeys_.push(k(i, q.POINTERMOVE, this.handlePointerMove_, this), k(i, q.POINTERUP, this.handlePointerUp_, this), k(this.element_, q.POINTERCANCEL, this.handlePointerUp_, this)), this.element_.getRootNode && this.element_.getRootNode() !== i && this.dragListenerKeys_.push(k(this.element_.getRootNode(), q.POINTERUP, this.handlePointerUp_, this));
		}
	}, t.prototype.handlePointerMove_ = function(e) {
		if (this.isMoving_(e)) {
			this.dragging_ = !0;
			var t = new ci(q.POINTERDRAG, this.map_, e, this.dragging_);
			this.dispatchEvent(t);
		}
	}, t.prototype.relayEvent_ = function(e) {
		this.originalPointerMoveEvent_ = e;
		var t = !!(this.down_ && this.isMoving_(e));
		this.dispatchEvent(new ci(e.type, this.map_, e, t));
	}, t.prototype.handleTouchMove_ = function(e) {
		var t = this.originalPointerMoveEvent_;
		(!t || t.defaultPrevented) && (typeof e.cancelable != "boolean" || e.cancelable === !0) && e.preventDefault();
	}, t.prototype.isMoving_ = function(e) {
		return this.dragging_ || Math.abs(e.clientX - this.down_.clientX) > this.moveTolerance_ || Math.abs(e.clientY - this.down_.clientY) > this.moveTolerance_;
	}, t.prototype.disposeInternal = function() {
		this.relayedListenerKey_ &&= (j(this.relayedListenerKey_), null), this.element_.removeEventListener(O.TOUCHMOVE, this.boundHandleTouchMove_), this.pointerdownListenerKey_ &&= (j(this.pointerdownListenerKey_), null), this.dragListenerKeys_.forEach(j), this.dragListenerKeys_.length = 0, this.element_ = null, e.prototype.disposeInternal.call(this);
	}, t;
}(D), di = Infinity, fi = function() {
	function e(e, t) {
		this.priorityFunction_ = e, this.keyFunction_ = t, this.elements_ = [], this.priorities_ = [], this.queuedElements_ = {};
	}
	return e.prototype.clear = function() {
		this.elements_.length = 0, this.priorities_.length = 0, C(this.queuedElements_);
	}, e.prototype.dequeue = function() {
		var e = this.elements_, t = this.priorities_, n = e[0];
		e.length == 1 ? (e.length = 0, t.length = 0) : (e[0] = e.pop(), t[0] = t.pop(), this.siftUp_(0));
		var r = this.keyFunction_(n);
		return delete this.queuedElements_[r], n;
	}, e.prototype.enqueue = function(e) {
		H(!(this.keyFunction_(e) in this.queuedElements_), 31);
		var t = this.priorityFunction_(e);
		return t == Infinity ? !1 : (this.elements_.push(e), this.priorities_.push(t), this.queuedElements_[this.keyFunction_(e)] = !0, this.siftDown_(0, this.elements_.length - 1), !0);
	}, e.prototype.getCount = function() {
		return this.elements_.length;
	}, e.prototype.getLeftChildIndex_ = function(e) {
		return e * 2 + 1;
	}, e.prototype.getRightChildIndex_ = function(e) {
		return e * 2 + 2;
	}, e.prototype.getParentIndex_ = function(e) {
		return e - 1 >> 1;
	}, e.prototype.heapify_ = function() {
		var e;
		for (e = (this.elements_.length >> 1) - 1; e >= 0; e--) this.siftUp_(e);
	}, e.prototype.isEmpty = function() {
		return this.elements_.length === 0;
	}, e.prototype.isKeyQueued = function(e) {
		return e in this.queuedElements_;
	}, e.prototype.isQueued = function(e) {
		return this.isKeyQueued(this.keyFunction_(e));
	}, e.prototype.siftUp_ = function(e) {
		for (var t = this.elements_, n = this.priorities_, r = t.length, i = t[e], a = n[e], o = e; e < r >> 1;) {
			var s = this.getLeftChildIndex_(e), c = this.getRightChildIndex_(e), l = c < r && n[c] < n[s] ? c : s;
			t[e] = t[l], n[e] = n[l], e = l;
		}
		t[e] = i, n[e] = a, this.siftDown_(o, e);
	}, e.prototype.siftDown_ = function(e, t) {
		for (var n = this.elements_, r = this.priorities_, i = n[t], a = r[t]; t > e;) {
			var o = this.getParentIndex_(t);
			if (r[o] > a) n[t] = n[o], r[t] = r[o], t = o;
			else break;
		}
		n[t] = i, r[t] = a;
	}, e.prototype.reprioritize = function() {
		var e = this.priorityFunction_, t = this.elements_, n = this.priorities_, r = 0, i = t.length, a, o, s;
		for (o = 0; o < i; ++o) a = t[o], s = e(a), s == Infinity ? delete this.queuedElements_[this.keyFunction_(a)] : (n[r] = s, t[r++] = a);
		t.length = r, n.length = r, this.heapify_();
	}, e;
}(), J = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
}, pi = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), mi = function(e) {
	pi(t, e);
	function t(t, n) {
		var r = e.call(this, function(e) {
			return t.apply(null, e);
		}, function(e) {
			return e[0].getKey();
		}) || this;
		return r.boundHandleTileChange_ = r.handleTileChange.bind(r), r.tileChangeCallback_ = n, r.tilesLoading_ = 0, r.tilesLoadingKeys_ = {}, r;
	}
	return t.prototype.enqueue = function(t) {
		var n = e.prototype.enqueue.call(this, t);
		return n && t[0].addEventListener(O.CHANGE, this.boundHandleTileChange_), n;
	}, t.prototype.getTilesLoading = function() {
		return this.tilesLoading_;
	}, t.prototype.handleTileChange = function(e) {
		var t = e.target, n = t.getState();
		if (n === J.LOADED || n === J.ERROR || n === J.EMPTY) {
			t.removeEventListener(O.CHANGE, this.boundHandleTileChange_);
			var r = t.getKey();
			r in this.tilesLoadingKeys_ && (delete this.tilesLoadingKeys_[r], --this.tilesLoading_), this.tileChangeCallback_();
		}
	}, t.prototype.loadMoreTiles = function(e, t) {
		for (var n = 0, r, i, a; this.tilesLoading_ < e && n < t && this.getCount() > 0;) i = this.dequeue()[0], a = i.getKey(), r = i.getState(), r === J.IDLE && !(a in this.tilesLoadingKeys_) && (this.tilesLoadingKeys_[a] = !0, ++this.tilesLoading_, ++n, i.load());
	}, t;
}(fi);
function hi(e, t, n, r, i) {
	if (!e || !(n in e.wantedTiles) || !e.wantedTiles[n][t.getKey()]) return di;
	var a = e.viewState.center, o = r[0] - a[0], s = r[1] - a[1];
	return 65536 * Math.log(i) + Math.sqrt(o * o + s * s) / i;
}
//#endregion
//#region node_modules/ol/ViewHint.js
var Y = {
	ANIMATING: 0,
	INTERACTING: 1
}, gi = {
	CENTER: "center",
	RESOLUTION: "resolution",
	ROTATION: "rotation"
};
//#endregion
//#region node_modules/ol/centerconstraint.js
function _i(e, t, n) {
	return (function(r, i, a, o, s) {
		if (r) {
			var c = t ? 0 : a[0] * i, l = t ? 0 : a[1] * i, u = s ? s[0] : 0, d = s ? s[1] : 0, f = e[0] + c / 2 + u, p = e[2] - c / 2 + u, m = e[1] + l / 2 + d, h = e[3] - l / 2 + d;
			f > p && (f = (p + f) / 2, p = f), m > h && (m = (h + m) / 2, h = m);
			var g = V(r[0], f, p), _ = V(r[1], m, h), v = 30 * i;
			return o && n && (g += -v * Math.log(1 + Math.max(0, f - r[0]) / v) + v * Math.log(1 + Math.max(0, r[0] - p) / v), _ += -v * Math.log(1 + Math.max(0, m - r[1]) / v) + v * Math.log(1 + Math.max(0, r[1] - h) / v)), [g, _];
		} else return;
	});
}
function vi(e) {
	return e;
}
//#endregion
//#region node_modules/ol/resolutionconstraint.js
function yi(e, t, n, r) {
	var i = U(t) / n[0], a = qt(t) / n[1];
	return r ? Math.min(e, Math.max(i, a)) : Math.min(e, Math.min(i, a));
}
function bi(e, t, n) {
	var r = Math.min(e, t), i = 50;
	return r *= Math.log(1 + i * Math.max(0, e / t - 1)) / i + 1, n && (r = Math.max(r, n), r /= Math.log(1 + i * Math.max(0, n / e - 1)) / i + 1), V(r, n / 2, t * 2);
}
function xi(e, t, n, r) {
	return (function(i, a, o, s) {
		if (i !== void 0) {
			var c = e[0], l = e[e.length - 1], u = n ? yi(c, n, o, r) : c;
			if (s) return t === void 0 || t ? bi(i, u, l) : V(i, l, u);
			var d = Math.floor(p(e, Math.min(u, i), a));
			return e[d] > u && d < e.length - 1 ? e[d + 1] : e[d];
		} else return;
	});
}
function Si(e, t, n, r, i, a) {
	return (function(o, s, c, l) {
		if (o !== void 0) {
			var u = i ? yi(t, i, c, a) : t, d = n === void 0 ? 0 : n;
			if (l) return r === void 0 || r ? bi(o, u, d) : V(o, d, u);
			var f = 1e-9, p = Math.ceil(Math.log(t / u) / Math.log(e) - f), m = -s * (.5 - f) + .5, h = Math.floor(Math.log(t / Math.min(u, o)) / Math.log(e) + m);
			return V(t / e ** +Math.max(p, h), d, u);
		} else return;
	});
}
function Ci(e, t, n, r, i) {
	return (function(a, o, s, c) {
		if (a !== void 0) {
			var l = r ? yi(e, r, s, i) : e;
			return !(n === void 0 || n) || !c ? V(a, t, l) : bi(a, l, t);
		} else return;
	});
}
//#endregion
//#region node_modules/ol/rotationconstraint.js
function wi(e) {
	if (e !== void 0) return 0;
}
function Ti(e) {
	if (e !== void 0) return e;
}
function Ei(e) {
	var t = 2 * Math.PI / e;
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return e = Math.floor(e / t + .5) * t, e;
	});
}
function Di(e) {
	var t = e || qe(5);
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return Math.abs(e) <= t ? 0 : e;
	});
}
//#endregion
//#region node_modules/ol/easing.js
function Oi(e) {
	return e ** 3;
}
function ki(e) {
	return 1 - Oi(1 - e);
}
function Ai(e) {
	return 3 * e * e - 2 * e * e * e;
}
function ji(e) {
	return e;
}
//#endregion
//#region node_modules/ol/geom/GeometryLayout.js
var Mi = {
	XY: "XY",
	XYZ: "XYZ",
	XYM: "XYM",
	XYZM: "XYZM"
};
//#endregion
//#region node_modules/ol/geom/flat/transform.js
function Ni(e, t, n, r, i, a) {
	for (var o = a || [], s = 0, c = t; c < n; c += r) {
		var l = e[c], u = e[c + 1];
		o[s++] = i[0] * l + i[2] * u + i[4], o[s++] = i[1] * l + i[3] * u + i[5];
	}
	return a && o.length != s && (o.length = s), o;
}
function Pi(e, t, n, r, i, a, o) {
	for (var s = o || [], c = Math.cos(i), l = Math.sin(i), u = a[0], d = a[1], f = 0, p = t; p < n; p += r) {
		var m = e[p] - u, h = e[p + 1] - d;
		s[f++] = u + m * c - h * l, s[f++] = d + m * l + h * c;
		for (var g = p + 2; g < p + r; ++g) s[f++] = e[g];
	}
	return o && s.length != f && (s.length = f), s;
}
function Fi(e, t, n, r, i, a, o, s) {
	for (var c = s || [], l = o[0], u = o[1], d = 0, f = t; f < n; f += r) {
		var p = e[f] - l, m = e[f + 1] - u;
		c[d++] = l + i * p, c[d++] = u + a * m;
		for (var h = f + 2; h < f + r; ++h) c[d++] = e[h];
	}
	return s && c.length != d && (c.length = d), c;
}
function Ii(e, t, n, r, i, a, o) {
	for (var s = o || [], c = 0, l = t; l < n; l += r) {
		s[c++] = e[l] + i, s[c++] = e[l + 1] + a;
		for (var u = l + 2; u < l + r; ++u) s[c++] = e[u];
	}
	return o && s.length != c && (s.length = c), s;
}
//#endregion
//#region node_modules/ol/geom/Geometry.js
var Li = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ri = Rn(), zi = function(e) {
	Li(t, e);
	function t() {
		var t = e.call(this) || this;
		return t.extent_ = At(), t.extentRevision_ = -1, t.simplifiedGeometryMaxMinSquaredTolerance = 0, t.simplifiedGeometryRevision = 0, t.simplifyTransformedInternal = x(function(e, t, n) {
			if (!n) return this.getSimplifiedGeometry(t);
			var r = this.clone();
			return r.applyTransform(n), r.getSimplifiedGeometry(t);
		}), t;
	}
	return t.prototype.simplifyTransformed = function(e, t) {
		return this.simplifyTransformedInternal(this.getRevision(), e, t);
	}, t.prototype.clone = function() {
		return F();
	}, t.prototype.closestPointXY = function(e, t, n, r) {
		return F();
	}, t.prototype.containsXY = function(e, t) {
		var n = this.getClosestPoint([e, t]);
		return n[0] === e && n[1] === t;
	}, t.prototype.getClosestPoint = function(e, t) {
		var n = t || [NaN, NaN];
		return this.closestPointXY(e[0], e[1], n, Infinity), n;
	}, t.prototype.intersectsCoordinate = function(e) {
		return this.containsXY(e[0], e[1]);
	}, t.prototype.computeExtent = function(e) {
		return F();
	}, t.prototype.getExtent = function(e) {
		if (this.extentRevision_ != this.getRevision()) {
			var t = this.computeExtent(this.extent_);
			(isNaN(t[0]) || isNaN(t[1])) && Mt(t), this.extentRevision_ = this.getRevision();
		}
		return $t(this.extent_, e);
	}, t.prototype.rotate = function(e, t) {
		F();
	}, t.prototype.scale = function(e, t, n) {
		F();
	}, t.prototype.simplify = function(e) {
		return this.getSimplifiedGeometry(e * e);
	}, t.prototype.getSimplifiedGeometry = function(e) {
		return F();
	}, t.prototype.getType = function() {
		return F();
	}, t.prototype.applyTransform = function(e) {
		F();
	}, t.prototype.intersectsExtent = function(e) {
		return F();
	}, t.prototype.translate = function(e, t) {
		F();
	}, t.prototype.transform = function(e, t) {
		var n = gn(e), r = n.getUnits() == B.TILE_PIXELS ? function(e, r, i) {
			var a = n.getExtent(), o = n.getWorldExtent(), s = qt(o) / qt(a);
			return qn(Ri, o[0], o[3], s, -s, 0, 0, 0), Ni(e, 0, e.length, i, Ri, r), Cn(n, t)(e, r, i);
		} : Cn(n, t);
		return this.applyTransform(r), this;
	}, t;
}(z), Bi = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Vi = function(e) {
	Bi(t, e);
	function t() {
		var t = e.call(this) || this;
		return t.layout = Mi.XY, t.stride = 2, t.flatCoordinates = null, t;
	}
	return t.prototype.computeExtent = function(e) {
		return Pt(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
	}, t.prototype.getCoordinates = function() {
		return F();
	}, t.prototype.getFirstCoordinate = function() {
		return this.flatCoordinates.slice(0, this.stride);
	}, t.prototype.getFlatCoordinates = function() {
		return this.flatCoordinates;
	}, t.prototype.getLastCoordinate = function() {
		return this.flatCoordinates.slice(this.flatCoordinates.length - this.stride);
	}, t.prototype.getLayout = function() {
		return this.layout;
	}, t.prototype.getSimplifiedGeometry = function(e) {
		if (this.simplifiedGeometryRevision !== this.getRevision() && (this.simplifiedGeometryMaxMinSquaredTolerance = 0, this.simplifiedGeometryRevision = this.getRevision()), e < 0 || this.simplifiedGeometryMaxMinSquaredTolerance !== 0 && e <= this.simplifiedGeometryMaxMinSquaredTolerance) return this;
		var t = this.getSimplifiedGeometryInternal(e);
		return t.getFlatCoordinates().length < this.flatCoordinates.length ? t : (this.simplifiedGeometryMaxMinSquaredTolerance = e, this);
	}, t.prototype.getSimplifiedGeometryInternal = function(e) {
		return this;
	}, t.prototype.getStride = function() {
		return this.stride;
	}, t.prototype.setFlatCoordinates = function(e, t) {
		this.stride = Ui(e), this.layout = e, this.flatCoordinates = t;
	}, t.prototype.setCoordinates = function(e, t) {
		F();
	}, t.prototype.setLayout = function(e, t, n) {
		var r;
		if (e) r = Ui(e);
		else {
			for (var i = 0; i < n; ++i) if (t.length === 0) {
				this.layout = Mi.XY, this.stride = 2;
				return;
			} else t = t[0];
			r = t.length, e = Hi(r);
		}
		this.layout = e, this.stride = r;
	}, t.prototype.applyTransform = function(e) {
		this.flatCoordinates && (e(this.flatCoordinates, this.flatCoordinates, this.stride), this.changed());
	}, t.prototype.rotate = function(e, t) {
		var n = this.getFlatCoordinates();
		if (n) {
			var r = this.getStride();
			Pi(n, 0, n.length, r, e, t, n), this.changed();
		}
	}, t.prototype.scale = function(e, t, n) {
		var r = t;
		r === void 0 && (r = e);
		var i = n;
		i ||= Wt(this.getExtent());
		var a = this.getFlatCoordinates();
		if (a) {
			var o = this.getStride();
			Fi(a, 0, a.length, o, e, r, i, a), this.changed();
		}
	}, t.prototype.translate = function(e, t) {
		var n = this.getFlatCoordinates();
		if (n) {
			var r = this.getStride();
			Ii(n, 0, n.length, r, e, t, n), this.changed();
		}
	}, t;
}(zi);
function Hi(e) {
	var t;
	return e == 2 ? t = Mi.XY : e == 3 ? t = Mi.XYZ : e == 4 && (t = Mi.XYZM), t;
}
function Ui(e) {
	var t;
	return e == Mi.XY ? t = 2 : e == Mi.XYZ || e == Mi.XYM ? t = 3 : e == Mi.XYZM && (t = 4), t;
}
function Wi(e, t, n) {
	var r = e.getFlatCoordinates();
	if (r) {
		var i = e.getStride();
		return Ni(r, 0, r.length, i, t, n);
	} else return null;
}
//#endregion
//#region node_modules/ol/geom/flat/closest.js
function Gi(e, t, n, r, i, a, o) {
	var s = e[t], c = e[t + 1], l = e[n] - s, u = e[n + 1] - c, d;
	if (l === 0 && u === 0) d = t;
	else {
		var f = ((i - s) * l + (a - c) * u) / (l * l + u * u);
		if (f > 1) d = n;
		else if (f > 0) {
			for (var p = 0; p < r; ++p) o[p] = Ye(e[t + p], e[n + p], f);
			o.length = r;
			return;
		} else d = t;
	}
	for (var p = 0; p < r; ++p) o[p] = e[d + p];
	o.length = r;
}
function Ki(e, t, n, r, i) {
	var a = e[t], o = e[t + 1];
	for (t += r; t < n; t += r) {
		var s = e[t], c = e[t + 1], l = Ge(a, o, s, c);
		l > i && (i = l), a = s, o = c;
	}
	return i;
}
function qi(e, t, n, r, i) {
	for (var a = 0, o = n.length; a < o; ++a) {
		var s = n[a];
		i = Ki(e, t, s, r, i), t = s;
	}
	return i;
}
function Ji(e, t, n, r, i, a, o, s, c, l, u) {
	if (t == n) return l;
	var d, f;
	if (i === 0) if (f = Ge(o, s, e[t], e[t + 1]), f < l) {
		for (d = 0; d < r; ++d) c[d] = e[t + d];
		return c.length = r, f;
	} else return l;
	for (var p = u || [NaN, NaN], m = t + r; m < n;) if (Gi(e, m - r, m, r, o, s, p), f = Ge(o, s, p[0], p[1]), f < l) {
		for (l = f, d = 0; d < r; ++d) c[d] = p[d];
		c.length = r, m += r;
	} else m += r * Math.max((Math.sqrt(f) - Math.sqrt(l)) / i | 0, 1);
	if (a && (Gi(e, n - r, t, r, o, s, p), f = Ge(o, s, p[0], p[1]), f < l)) {
		for (l = f, d = 0; d < r; ++d) c[d] = p[d];
		c.length = r;
	}
	return l;
}
function Yi(e, t, n, r, i, a, o, s, c, l, u) {
	for (var d = u || [NaN, NaN], f = 0, p = n.length; f < p; ++f) {
		var m = n[f];
		l = Ji(e, t, m, r, i, a, o, s, c, l, d), t = m;
	}
	return l;
}
//#endregion
//#region node_modules/ol/geom/flat/deflate.js
function Xi(e, t, n, r) {
	for (var i = 0, a = n.length; i < a; ++i) e[t++] = n[i];
	return t;
}
function Zi(e, t, n, r) {
	for (var i = 0, a = n.length; i < a; ++i) for (var o = n[i], s = 0; s < r; ++s) e[t++] = o[s];
	return t;
}
function Qi(e, t, n, r, i) {
	for (var a = i || [], o = 0, s = 0, c = n.length; s < c; ++s) {
		var l = Zi(e, t, n[s], r);
		a[o++] = l, t = l;
	}
	return a.length = o, a;
}
//#endregion
//#region node_modules/ol/geom/flat/simplify.js
function $i(e, t, n, r, i, a, o) {
	var s = (n - t) / r;
	if (s < 3) {
		for (; t < n; t += r) a[o++] = e[t], a[o++] = e[t + 1];
		return o;
	}
	var c = Array(s);
	c[0] = 1, c[s - 1] = 1;
	for (var l = [t, n - r], u = 0; l.length > 0;) {
		for (var d = l.pop(), f = l.pop(), p = 0, m = e[f], h = e[f + 1], g = e[d], _ = e[d + 1], v = f + r; v < d; v += r) {
			var y = e[v], b = e[v + 1], x = We(y, b, m, h, g, _);
			x > p && (u = v, p = x);
		}
		p > i && (c[(u - t) / r] = 1, f + r < u && l.push(f, u), u + r < d && l.push(u, d));
	}
	for (var v = 0; v < s; ++v) c[v] && (a[o++] = e[t + v * r], a[o++] = e[t + v * r + 1]);
	return o;
}
function ea(e, t) {
	return t * Math.round(e / t);
}
function ta(e, t, n, r, i, a, o) {
	if (t == n) return o;
	var s = ea(e[t], i), c = ea(e[t + 1], i);
	t += r, a[o++] = s, a[o++] = c;
	var l, u;
	do
		if (l = ea(e[t], i), u = ea(e[t + 1], i), t += r, t == n) return a[o++] = l, a[o++] = u, o;
	while (l == s && u == c);
	for (; t < n;) {
		var d = ea(e[t], i), f = ea(e[t + 1], i);
		if (t += r, !(d == l && f == u)) {
			var p = l - s, m = u - c, h = d - s, g = f - c;
			if (p * g == m * h && (p < 0 && h < p || p == h || p > 0 && h > p) && (m < 0 && g < m || m == g || m > 0 && g > m)) {
				l = d, u = f;
				continue;
			}
			a[o++] = l, a[o++] = u, s = l, c = u, l = d, u = f;
		}
	}
	return a[o++] = l, a[o++] = u, o;
}
function na(e, t, n, r, i, a, o, s) {
	for (var c = 0, l = n.length; c < l; ++c) {
		var u = n[c];
		o = ta(e, t, u, r, i, a, o), s.push(o), t = u;
	}
	return o;
}
//#endregion
//#region node_modules/ol/geom/flat/inflate.js
function ra(e, t, n, r, i) {
	for (var a = i === void 0 ? [] : i, o = 0, s = t; s < n; s += r) a[o++] = e.slice(s, s + r);
	return a.length = o, a;
}
function ia(e, t, n, r, i) {
	for (var a = i === void 0 ? [] : i, o = 0, s = 0, c = n.length; s < c; ++s) {
		var l = n[s];
		a[o++] = ra(e, t, l, r, a[o]), t = l;
	}
	return a.length = o, a;
}
function aa(e, t, n, r, i) {
	for (var a = i === void 0 ? [] : i, o = 0, s = 0, c = n.length; s < c; ++s) {
		var l = n[s];
		a[o++] = ia(e, t, l, r, a[o]), t = l[l.length - 1];
	}
	return a.length = o, a;
}
//#endregion
//#region node_modules/ol/geom/flat/area.js
function oa(e, t, n, r) {
	for (var i = 0, a = e[n - r], o = e[n - r + 1]; t < n; t += r) {
		var s = e[t], c = e[t + 1];
		i += o * s - a * c, a = s, o = c;
	}
	return i / 2;
}
function sa(e, t, n, r) {
	for (var i = 0, a = 0, o = n.length; a < o; ++a) {
		var s = n[a];
		i += oa(e, t, s, r), t = s;
	}
	return i;
}
//#endregion
//#region node_modules/ol/geom/LinearRing.js
var ca = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), la = function(e) {
	ca(t, e);
	function t(t, n) {
		var r = e.call(this) || this;
		return r.maxDelta_ = -1, r.maxDeltaRevision_ = -1, n !== void 0 && !Array.isArray(t[0]) ? r.setFlatCoordinates(n, t) : r.setCoordinates(t, n), r;
	}
	return t.prototype.clone = function() {
		return new t(this.flatCoordinates.slice(), this.layout);
	}, t.prototype.closestPointXY = function(e, t, n, r) {
		return r < Tt(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(Ki(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), Ji(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !0, e, t, n, r));
	}, t.prototype.getArea = function() {
		return oa(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}, t.prototype.getCoordinates = function() {
		return ra(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}, t.prototype.getSimplifiedGeometryInternal = function(e) {
		var n = [];
		return n.length = $i(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, n, 0), new t(n, Mi.XY);
	}, t.prototype.getType = function() {
		return W.LINEAR_RING;
	}, t.prototype.intersectsExtent = function(e) {
		return !1;
	}, t.prototype.setCoordinates = function(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = Zi(this.flatCoordinates, 0, e, this.stride), this.changed();
	}, t;
}(Vi), ua = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), da = function(e) {
	ua(t, e);
	function t(t, n) {
		var r = e.call(this) || this;
		return r.setCoordinates(t, n), r;
	}
	return t.prototype.clone = function() {
		var e = new t(this.flatCoordinates.slice(), this.layout);
		return e.applyProperties(this), e;
	}, t.prototype.closestPointXY = function(e, t, n, r) {
		var i = this.flatCoordinates, a = Ge(e, t, i[0], i[1]);
		if (a < r) {
			for (var o = this.stride, s = 0; s < o; ++s) n[s] = i[s];
			return n.length = o, a;
		} else return r;
	}, t.prototype.getCoordinates = function() {
		return this.flatCoordinates ? this.flatCoordinates.slice() : [];
	}, t.prototype.computeExtent = function(e) {
		return Nt(this.flatCoordinates, e);
	}, t.prototype.getType = function() {
		return W.POINT;
	}, t.prototype.intersectsExtent = function(e) {
		return Ot(e, this.flatCoordinates[0], this.flatCoordinates[1]);
	}, t.prototype.setCoordinates = function(e, t) {
		this.setLayout(t, e, 0), this.flatCoordinates ||= [], this.flatCoordinates.length = Xi(this.flatCoordinates, 0, e, this.stride), this.changed();
	}, t;
}(Vi);
//#endregion
//#region node_modules/ol/geom/flat/contains.js
function fa(e, t, n, r, i) {
	return !Bt(i, function(i) {
		return !pa(e, t, n, r, i[0], i[1]);
	});
}
function pa(e, t, n, r, i, a) {
	for (var o = 0, s = e[n - r], c = e[n - r + 1]; t < n; t += r) {
		var l = e[t], u = e[t + 1];
		c <= a ? u > a && (l - s) * (a - c) - (i - s) * (u - c) > 0 && o++ : u <= a && (l - s) * (a - c) - (i - s) * (u - c) < 0 && o--, s = l, c = u;
	}
	return o !== 0;
}
function ma(e, t, n, r, i, a) {
	if (n.length === 0 || !pa(e, t, n[0], r, i, a)) return !1;
	for (var o = 1, s = n.length; o < s; ++o) if (pa(e, n[o - 1], n[o], r, i, a)) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/interiorpoint.js
function ha(e, t, n, r, i, a, o) {
	for (var s, c, l, u, d, p, m, h = i[a + 1], g = [], _ = 0, v = n.length; _ < v; ++_) {
		var y = n[_];
		for (u = e[y - r], p = e[y - r + 1], s = t; s < y; s += r) d = e[s], m = e[s + 1], (h <= p && m <= h || p <= h && h <= m) && (l = (h - p) / (m - p) * (d - u) + u, g.push(l)), u = d, p = m;
	}
	var b = NaN, x = -Infinity;
	for (g.sort(f), u = g[0], s = 1, c = g.length; s < c; ++s) {
		d = g[s];
		var S = Math.abs(d - u);
		S > x && (l = (u + d) / 2, ma(e, t, n, r, l, h) && (b = l, x = S)), u = d;
	}
	return isNaN(b) && (b = i[a]), o ? (o.push(b, h, x), o) : [
		b,
		h,
		x
	];
}
//#endregion
//#region node_modules/ol/geom/flat/segments.js
function ga(e, t, n, r, i) {
	var a;
	for (t += r; t < n; t += r) if (a = i(e.slice(t - r, t), e.slice(t, t + r)), a) return a;
	return !1;
}
//#endregion
//#region node_modules/ol/geom/flat/intersectsextent.js
function _a(e, t, n, r, i) {
	var a = Rt(At(), e, t, n, r);
	return Zt(i, a) ? Dt(i, a) || a[0] >= i[0] && a[2] <= i[2] || a[1] >= i[1] && a[3] <= i[3] ? !0 : ga(e, t, n, r, function(e, t) {
		return tn(i, e, t);
	}) : !1;
}
function va(e, t, n, r, i) {
	return !!(_a(e, t, n, r, i) || pa(e, t, n, r, i[0], i[1]) || pa(e, t, n, r, i[0], i[3]) || pa(e, t, n, r, i[2], i[1]) || pa(e, t, n, r, i[2], i[3]));
}
function ya(e, t, n, r, i) {
	if (!va(e, t, n[0], r, i)) return !1;
	if (n.length === 1) return !0;
	for (var a = 1, o = n.length; a < o; ++a) if (fa(e, n[a - 1], n[a], r, i) && !_a(e, n[a - 1], n[a], r, i)) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/reverse.js
function ba(e, t, n, r) {
	for (; t < n - r;) {
		for (var i = 0; i < r; ++i) {
			var a = e[t + i];
			e[t + i] = e[n - r + i], e[n - r + i] = a;
		}
		t += r, n -= r;
	}
}
//#endregion
//#region node_modules/ol/geom/flat/orient.js
function xa(e, t, n, r) {
	for (var i = 0, a = e[n - r], o = e[n - r + 1]; t < n; t += r) {
		var s = e[t], c = e[t + 1];
		i += (s - a) * (c + o), a = s, o = c;
	}
	return i === 0 ? void 0 : i > 0;
}
function Sa(e, t, n, r, i) {
	for (var a = i !== void 0 && i, o = 0, s = n.length; o < s; ++o) {
		var c = n[o], l = xa(e, t, c, r);
		if (o === 0) {
			if (a && l || !a && !l) return !1;
		} else if (a && !l || !a && l) return !1;
		t = c;
	}
	return !0;
}
function Ca(e, t, n, r, i) {
	for (var a = i !== void 0 && i, o = 0, s = n.length; o < s; ++o) {
		var c = n[o], l = xa(e, t, c, r);
		(o === 0 ? a && l || !a && !l : a && !l || !a && l) && ba(e, t, c, r), t = c;
	}
	return t;
}
//#endregion
//#region node_modules/ol/geom/Polygon.js
var wa = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ta = function(e) {
	wa(t, e);
	function t(t, n, r) {
		var i = e.call(this) || this;
		return i.ends_ = [], i.flatInteriorPointRevision_ = -1, i.flatInteriorPoint_ = null, i.maxDelta_ = -1, i.maxDeltaRevision_ = -1, i.orientedRevision_ = -1, i.orientedFlatCoordinates_ = null, n !== void 0 && r ? (i.setFlatCoordinates(n, t), i.ends_ = r) : i.setCoordinates(t, n), i;
	}
	return t.prototype.appendLinearRing = function(e) {
		this.flatCoordinates ? h(this.flatCoordinates, e.getFlatCoordinates()) : this.flatCoordinates = e.getFlatCoordinates().slice(), this.ends_.push(this.flatCoordinates.length), this.changed();
	}, t.prototype.clone = function() {
		var e = new t(this.flatCoordinates.slice(), this.layout, this.ends_.slice());
		return e.applyProperties(this), e;
	}, t.prototype.closestPointXY = function(e, t, n, r) {
		return r < Tt(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(qi(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), Yi(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}, t.prototype.containsXY = function(e, t) {
		return ma(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, t);
	}, t.prototype.getArea = function() {
		return sa(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride);
	}, t.prototype.getCoordinates = function(e) {
		var t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), Ca(t, 0, this.ends_, this.stride, e)), ia(t, 0, this.ends_, this.stride);
	}, t.prototype.getEnds = function() {
		return this.ends_;
	}, t.prototype.getFlatInteriorPoint = function() {
		if (this.flatInteriorPointRevision_ != this.getRevision()) {
			var e = Wt(this.getExtent());
			this.flatInteriorPoint_ = ha(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, 0), this.flatInteriorPointRevision_ = this.getRevision();
		}
		return this.flatInteriorPoint_;
	}, t.prototype.getInteriorPoint = function() {
		return new da(this.getFlatInteriorPoint(), Mi.XYM);
	}, t.prototype.getLinearRingCount = function() {
		return this.ends_.length;
	}, t.prototype.getLinearRing = function(e) {
		return e < 0 || this.ends_.length <= e ? null : new la(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}, t.prototype.getLinearRings = function() {
		for (var e = this.layout, t = this.flatCoordinates, n = this.ends_, r = [], i = 0, a = 0, o = n.length; a < o; ++a) {
			var s = n[a], c = new la(t.slice(i, s), e);
			r.push(c), i = s;
		}
		return r;
	}, t.prototype.getOrientedFlatCoordinates = function() {
		if (this.orientedRevision_ != this.getRevision()) {
			var e = this.flatCoordinates;
			Sa(e, 0, this.ends_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = Ca(this.orientedFlatCoordinates_, 0, this.ends_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}, t.prototype.getSimplifiedGeometryInternal = function(e) {
		var n = [], r = [];
		return n.length = na(this.flatCoordinates, 0, this.ends_, this.stride, Math.sqrt(e), n, 0, r), new t(n, Mi.XY, r);
	}, t.prototype.getType = function() {
		return W.POLYGON;
	}, t.prototype.intersectsExtent = function(e) {
		return ya(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e);
	}, t.prototype.setCoordinates = function(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		var n = Qi(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}, t;
}(Vi);
function Ea(e) {
	var t = e[0], n = e[1], r = e[2], i = e[3], a = [
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
	return new Ta(a, Mi.XY, [a.length]);
}
//#endregion
//#region node_modules/ol/View.js
var Da = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Oa = 0, ka = function(e) {
	Da(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = S({}, t);
		return n.hints_ = [0, 0], n.animations_ = [], n.updateAnimationKey_, n.projection_ = bn(r.projection, "EPSG:3857"), n.viewportSize_ = [100, 100], n.targetCenter_ = null, n.targetResolution_, n.targetRotation_, n.nextCenter_ = null, n.nextResolution_, n.nextRotation_, n.cancelAnchor_ = void 0, r.center &&= On(r.center, n.projection_), r.extent &&= An(r.extent, n.projection_), n.applyOptions_(r), n;
	}
	return t.prototype.applyOptions_ = function(e) {
		var t = {}, n = Ma(e);
		this.maxResolution_ = n.maxResolution, this.minResolution_ = n.minResolution, this.zoomFactor_ = n.zoomFactor, this.resolutions_ = e.resolutions, this.padding_ = e.padding, this.minZoom_ = n.minZoom;
		var r = ja(e), i = n.constraint, a = Na(e);
		this.constraints_ = {
			center: r,
			resolution: i,
			rotation: a
		}, this.setRotation(e.rotation === void 0 ? 0 : e.rotation), this.setCenterInternal(e.center === void 0 ? null : e.center), e.resolution === void 0 ? e.zoom !== void 0 && this.setZoom(e.zoom) : this.setResolution(e.resolution), this.setProperties(t), this.options_ = e;
	}, Object.defineProperty(t.prototype, "padding", {
		get: function() {
			return this.padding_;
		},
		set: function(e) {
			var t = this.padding_;
			this.padding_ = e;
			var n = this.getCenter();
			if (n) {
				var r = e || [
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
				var i = this.getResolution(), a = i / 2 * (r[3] - t[3] + t[1] - r[1]), o = i / 2 * (r[0] - t[0] + t[2] - r[2]);
				this.setCenterInternal([n[0] + a, n[1] - o]);
			}
		},
		enumerable: !1,
		configurable: !0
	}), t.prototype.getUpdatedOptions_ = function(e) {
		var t = S({}, this.options_);
		return t.resolution === void 0 ? t.zoom = this.getZoom() : t.resolution = this.getResolution(), t.center = this.getCenterInternal(), t.rotation = this.getRotation(), S({}, t, e);
	}, t.prototype.animate = function(e) {
		this.isDef() && !this.getAnimating() && this.resolveConstraints(0);
		for (var t = Array(arguments.length), n = 0; n < t.length; ++n) {
			var r = arguments[n];
			r.center && (r = S({}, r), r.center = On(r.center, this.getProjection())), r.anchor && (r = S({}, r), r.anchor = On(r.anchor, this.getProjection())), t[n] = r;
		}
		this.animateInternal.apply(this, t);
	}, t.prototype.animateInternal = function(e) {
		var t = arguments.length, n;
		t > 1 && typeof arguments[t - 1] == "function" && (n = arguments[t - 1], --t);
		for (var r = 0; r < t && !this.isDef(); ++r) {
			var i = arguments[r];
			i.center && this.setCenterInternal(i.center), i.zoom === void 0 ? i.resolution && this.setResolution(i.resolution) : this.setZoom(i.zoom), i.rotation !== void 0 && this.setRotation(i.rotation);
		}
		if (r === t) {
			n && Aa(n, !0);
			return;
		}
		for (var a = Date.now(), o = this.targetCenter_.slice(), s = this.targetResolution_, c = this.targetRotation_, l = []; r < t; ++r) {
			var u = arguments[r], d = {
				start: a,
				complete: !1,
				anchor: u.anchor,
				duration: u.duration === void 0 ? 1e3 : u.duration,
				easing: u.easing || Ai,
				callback: n
			};
			if (u.center && (d.sourceCenter = o, d.targetCenter = u.center.slice(), o = d.targetCenter), u.zoom === void 0 ? u.resolution && (d.sourceResolution = s, d.targetResolution = u.resolution, s = d.targetResolution) : (d.sourceResolution = s, d.targetResolution = this.getResolutionForZoom(u.zoom), s = d.targetResolution), u.rotation !== void 0) {
				d.sourceRotation = c;
				var f = Je(u.rotation - c + Math.PI, 2 * Math.PI) - Math.PI;
				d.targetRotation = c + f, c = d.targetRotation;
			}
			Pa(d) ? d.complete = !0 : a += d.duration, l.push(d);
		}
		this.animations_.push(l), this.setHint(Y.ANIMATING, 1), this.updateAnimations_();
	}, t.prototype.getAnimating = function() {
		return this.hints_[Y.ANIMATING] > 0;
	}, t.prototype.getInteracting = function() {
		return this.hints_[Y.INTERACTING] > 0;
	}, t.prototype.cancelAnimations = function() {
		this.setHint(Y.ANIMATING, -this.hints_[Y.ANIMATING]);
		for (var e, t = 0, n = this.animations_.length; t < n; ++t) {
			var r = this.animations_[t];
			if (r[0].callback && Aa(r[0].callback, !1), !e) for (var i = 0, a = r.length; i < a; ++i) {
				var o = r[i];
				if (!o.complete) {
					e = o.anchor;
					break;
				}
			}
		}
		this.animations_.length = 0, this.cancelAnchor_ = e, this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
	}, t.prototype.updateAnimations_ = function() {
		if (this.updateAnimationKey_ !== void 0 && (cancelAnimationFrame(this.updateAnimationKey_), this.updateAnimationKey_ = void 0), this.getAnimating()) {
			for (var e = Date.now(), t = !1, n = this.animations_.length - 1; n >= 0; --n) {
				for (var r = this.animations_[n], i = !0, a = 0, o = r.length; a < o; ++a) {
					var s = r[a];
					if (!s.complete) {
						var c = e - s.start, l = s.duration > 0 ? c / s.duration : 1;
						l >= 1 ? (s.complete = !0, l = 1) : i = !1;
						var u = s.easing(l);
						if (s.sourceCenter) {
							var d = s.sourceCenter[0], f = s.sourceCenter[1], p = s.targetCenter[0], m = s.targetCenter[1];
							this.nextCenter_ = s.targetCenter;
							var h = d + u * (p - d), g = f + u * (m - f);
							this.targetCenter_ = [h, g];
						}
						if (s.sourceResolution && s.targetResolution) {
							var _ = u === 1 ? s.targetResolution : s.sourceResolution + u * (s.targetResolution - s.sourceResolution);
							if (s.anchor) {
								var v = this.getViewportSize_(this.getRotation()), y = this.constraints_.resolution(_, 0, v, !0);
								this.targetCenter_ = this.calculateCenterZoom(y, s.anchor);
							}
							this.nextResolution_ = s.targetResolution, this.targetResolution_ = _, this.applyTargetState_(!0);
						}
						if (s.sourceRotation !== void 0 && s.targetRotation !== void 0) {
							var b = u === 1 ? Je(s.targetRotation + Math.PI, 2 * Math.PI) - Math.PI : s.sourceRotation + u * (s.targetRotation - s.sourceRotation);
							if (s.anchor) {
								var x = this.constraints_.rotation(b, !0);
								this.targetCenter_ = this.calculateCenterRotate(x, s.anchor);
							}
							this.nextRotation_ = s.targetRotation, this.targetRotation_ = b;
						}
						if (this.applyTargetState_(!0), t = !0, !s.complete) break;
					}
				}
				if (i) {
					this.animations_[n] = null, this.setHint(Y.ANIMATING, -1), this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
					var S = r[0].callback;
					S && Aa(S, !0);
				}
			}
			this.animations_ = this.animations_.filter(Boolean), t && this.updateAnimationKey_ === void 0 && (this.updateAnimationKey_ = requestAnimationFrame(this.updateAnimations_.bind(this)));
		}
	}, t.prototype.calculateCenterRotate = function(e, t) {
		var n, r = this.getCenterInternal();
		return r !== void 0 && (n = [r[0] - t[0], r[1] - t[1]], cn(n, e - this.getRotation()), an(n, t)), n;
	}, t.prototype.calculateCenterZoom = function(e, t) {
		var n, r = this.getCenterInternal(), i = this.getResolution();
		return r !== void 0 && i !== void 0 && (n = [t[0] - e * (t[0] - r[0]) / i, t[1] - e * (t[1] - r[1]) / i]), n;
	}, t.prototype.getViewportSize_ = function(e) {
		var t = this.viewportSize_;
		if (e) {
			var n = t[0], r = t[1];
			return [Math.abs(n * Math.cos(e)) + Math.abs(r * Math.sin(e)), Math.abs(n * Math.sin(e)) + Math.abs(r * Math.cos(e))];
		} else return t;
	}, t.prototype.setViewportSize = function(e) {
		this.viewportSize_ = Array.isArray(e) ? e.slice() : [100, 100], this.getAnimating() || this.resolveConstraints(0);
	}, t.prototype.getCenter = function() {
		var e = this.getCenterInternal();
		return e && Dn(e, this.getProjection());
	}, t.prototype.getCenterInternal = function() {
		return this.get(gi.CENTER);
	}, t.prototype.getConstraints = function() {
		return this.constraints_;
	}, t.prototype.getConstrainResolution = function() {
		return this.options_.constrainResolution;
	}, t.prototype.getHints = function(e) {
		return e === void 0 ? this.hints_.slice() : (e[0] = this.hints_[0], e[1] = this.hints_[1], e);
	}, t.prototype.calculateExtent = function(e) {
		return kn(this.calculateExtentInternal(e), this.getProjection());
	}, t.prototype.calculateExtentInternal = function(e) {
		var t = e || this.getViewportSizeMinusPadding_(), n = this.getCenterInternal();
		H(n, 1);
		var r = this.getResolution();
		H(r !== void 0, 2);
		var i = this.getRotation();
		return H(i !== void 0, 3), Kt(n, r, i, t);
	}, t.prototype.getMaxResolution = function() {
		return this.maxResolution_;
	}, t.prototype.getMinResolution = function() {
		return this.minResolution_;
	}, t.prototype.getMaxZoom = function() {
		return this.getZoomForResolution(this.minResolution_);
	}, t.prototype.setMaxZoom = function(e) {
		this.applyOptions_(this.getUpdatedOptions_({ maxZoom: e }));
	}, t.prototype.getMinZoom = function() {
		return this.getZoomForResolution(this.maxResolution_);
	}, t.prototype.setMinZoom = function(e) {
		this.applyOptions_(this.getUpdatedOptions_({ minZoom: e }));
	}, t.prototype.setConstrainResolution = function(e) {
		this.applyOptions_(this.getUpdatedOptions_({ constrainResolution: e }));
	}, t.prototype.getProjection = function() {
		return this.projection_;
	}, t.prototype.getResolution = function() {
		return this.get(gi.RESOLUTION);
	}, t.prototype.getResolutions = function() {
		return this.resolutions_;
	}, t.prototype.getResolutionForExtent = function(e, t) {
		return this.getResolutionForExtentInternal(An(e, this.getProjection()), t);
	}, t.prototype.getResolutionForExtentInternal = function(e, t) {
		var n = t || this.getViewportSizeMinusPadding_(), r = U(e) / n[0], i = qt(e) / n[1];
		return Math.max(r, i);
	}, t.prototype.getResolutionForValueFunction = function(e) {
		var t = e || 2, n = this.getConstrainedResolution(this.maxResolution_), r = this.minResolution_, i = Math.log(n / r) / Math.log(t);
		return (function(e) {
			return n / t ** +(e * i);
		});
	}, t.prototype.getRotation = function() {
		return this.get(gi.ROTATION);
	}, t.prototype.getValueForResolutionFunction = function(e) {
		var t = Math.log(e || 2), n = this.getConstrainedResolution(this.maxResolution_), r = this.minResolution_, i = Math.log(n / r) / t;
		return (function(e) {
			return Math.log(n / e) / t / i;
		});
	}, t.prototype.getViewportSizeMinusPadding_ = function(e) {
		var t = this.getViewportSize_(e), n = this.padding_;
		return n && (t = [t[0] - n[1] - n[3], t[1] - n[0] - n[2]]), t;
	}, t.prototype.getState = function() {
		var e = this.getProjection(), t = this.getResolution(), n = this.getRotation(), r = this.getCenterInternal(), i = this.padding_;
		if (i) {
			var a = this.getViewportSizeMinusPadding_();
			r = Fa(r, this.getViewportSize_(), [a[0] / 2 + i[3], a[1] / 2 + i[0]], t, n);
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
	}, t.prototype.getZoom = function() {
		var e, t = this.getResolution();
		return t !== void 0 && (e = this.getZoomForResolution(t)), e;
	}, t.prototype.getZoomForResolution = function(e) {
		var t = this.minZoom_ || 0, n, r;
		if (this.resolutions_) {
			var i = p(this.resolutions_, e, 1);
			t = i, n = this.resolutions_[i], r = i == this.resolutions_.length - 1 ? 2 : n / this.resolutions_[i + 1];
		} else n = this.maxResolution_, r = this.zoomFactor_;
		return t + Math.log(n / e) / Math.log(r);
	}, t.prototype.getResolutionForZoom = function(e) {
		if (this.resolutions_) {
			if (this.resolutions_.length <= 1) return 0;
			var t = V(Math.floor(e), 0, this.resolutions_.length - 2), n = this.resolutions_[t] / this.resolutions_[t + 1];
			return this.resolutions_[t] / n ** +V(e - t, 0, 1);
		} else return this.maxResolution_ / this.zoomFactor_ ** +(e - this.minZoom_);
	}, t.prototype.fit = function(e, t) {
		var n;
		if (H(Array.isArray(e) || typeof e.getSimplifiedGeometry == "function", 24), Array.isArray(e)) {
			H(!Qt(e), 25);
			var r = An(e, this.getProjection());
			n = Ea(r);
		} else if (e.getType() === W.CIRCLE) {
			var r = An(e.getExtent(), this.getProjection());
			n = Ea(r), n.rotate(this.getRotation(), Wt(r));
		} else {
			var i = En();
			n = i ? e.clone().transform(i, this.getProjection()) : e;
		}
		this.fitInternal(n, t);
	}, t.prototype.rotatedExtentForGeometry = function(e) {
		for (var t = this.getRotation(), n = Math.cos(t), r = Math.sin(-t), i = e.getFlatCoordinates(), a = e.getStride(), o = Infinity, s = Infinity, c = -Infinity, l = -Infinity, u = 0, d = i.length; u < d; u += a) {
			var f = i[u] * n - i[u + 1] * r, p = i[u] * r + i[u + 1] * n;
			o = Math.min(o, f), s = Math.min(s, p), c = Math.max(c, f), l = Math.max(l, p);
		}
		return [
			o,
			s,
			c,
			l
		];
	}, t.prototype.fitInternal = function(e, t) {
		var n = t || {}, r = n.size;
		r ||= this.getViewportSizeMinusPadding_();
		var i = n.padding === void 0 ? [
			0,
			0,
			0,
			0
		] : n.padding, a = n.nearest !== void 0 && n.nearest, o = n.minResolution === void 0 ? n.maxZoom === void 0 ? 0 : this.getResolutionForZoom(n.maxZoom) : n.minResolution, s = this.rotatedExtentForGeometry(e), c = this.getResolutionForExtentInternal(s, [r[0] - i[1] - i[3], r[1] - i[0] - i[2]]);
		c = isNaN(c) ? o : Math.max(c, o), c = this.getConstrainedResolution(c, +!a);
		var l = this.getRotation(), u = Math.sin(l), d = Math.cos(l), f = Wt(s);
		f[0] += (i[1] - i[3]) / 2 * c, f[1] += (i[0] - i[2]) / 2 * c;
		var p = f[0] * d - f[1] * u, m = f[1] * d + f[0] * u, h = this.getConstrainedCenter([p, m], c), g = n.callback ? n.callback : b;
		n.duration === void 0 ? (this.targetResolution_ = c, this.targetCenter_ = h, this.applyTargetState_(!1, !0), Aa(g, !0)) : this.animateInternal({
			resolution: c,
			center: h,
			duration: n.duration,
			easing: n.easing
		}, g);
	}, t.prototype.centerOn = function(e, t, n) {
		this.centerOnInternal(On(e, this.getProjection()), t, n);
	}, t.prototype.centerOnInternal = function(e, t, n) {
		this.setCenterInternal(Fa(e, t, n, this.getResolution(), this.getRotation()));
	}, t.prototype.calculateCenterShift = function(e, t, n, r) {
		var i, a = this.padding_;
		if (a && e) {
			var o = this.getViewportSizeMinusPadding_(-n), s = Fa(e, r, [o[0] / 2 + a[3], o[1] / 2 + a[0]], t, n);
			i = [e[0] - s[0], e[1] - s[1]];
		}
		return i;
	}, t.prototype.isDef = function() {
		return !!this.getCenterInternal() && this.getResolution() !== void 0;
	}, t.prototype.adjustCenter = function(e) {
		var t = Dn(this.targetCenter_, this.getProjection());
		this.setCenter([t[0] + e[0], t[1] + e[1]]);
	}, t.prototype.adjustCenterInternal = function(e) {
		var t = this.targetCenter_;
		this.setCenterInternal([t[0] + e[0], t[1] + e[1]]);
	}, t.prototype.adjustResolution = function(e, t) {
		var n = t && On(t, this.getProjection());
		this.adjustResolutionInternal(e, n);
	}, t.prototype.adjustResolutionInternal = function(e, t) {
		var n = this.getAnimating() || this.getInteracting(), r = this.getViewportSize_(this.getRotation()), i = this.constraints_.resolution(this.targetResolution_ * e, 0, r, n);
		t && (this.targetCenter_ = this.calculateCenterZoom(i, t)), this.targetResolution_ *= e, this.applyTargetState_();
	}, t.prototype.adjustZoom = function(e, t) {
		this.adjustResolution(this.zoomFactor_ ** +-e, t);
	}, t.prototype.adjustRotation = function(e, t) {
		t &&= On(t, this.getProjection()), this.adjustRotationInternal(e, t);
	}, t.prototype.adjustRotationInternal = function(e, t) {
		var n = this.getAnimating() || this.getInteracting(), r = this.constraints_.rotation(this.targetRotation_ + e, n);
		t && (this.targetCenter_ = this.calculateCenterRotate(r, t)), this.targetRotation_ += e, this.applyTargetState_();
	}, t.prototype.setCenter = function(e) {
		this.setCenterInternal(On(e, this.getProjection()));
	}, t.prototype.setCenterInternal = function(e) {
		this.targetCenter_ = e, this.applyTargetState_();
	}, t.prototype.setHint = function(e, t) {
		return this.hints_[e] += t, this.changed(), this.hints_[e];
	}, t.prototype.setResolution = function(e) {
		this.targetResolution_ = e, this.applyTargetState_();
	}, t.prototype.setRotation = function(e) {
		this.targetRotation_ = e, this.applyTargetState_();
	}, t.prototype.setZoom = function(e) {
		this.setResolution(this.getResolutionForZoom(e));
	}, t.prototype.applyTargetState_ = function(e, t) {
		var n = this.getAnimating() || this.getInteracting() || t, r = this.constraints_.rotation(this.targetRotation_, n), i = this.getViewportSize_(r), a = this.constraints_.resolution(this.targetResolution_, 0, i, n), o = this.constraints_.center(this.targetCenter_, a, i, n, this.calculateCenterShift(this.targetCenter_, a, r, i));
		this.get(gi.ROTATION) !== r && this.set(gi.ROTATION, r), this.get(gi.RESOLUTION) !== a && this.set(gi.RESOLUTION, a), (!this.get(gi.CENTER) || !sn(this.get(gi.CENTER), o)) && this.set(gi.CENTER, o), this.getAnimating() && !e && this.cancelAnimations(), this.cancelAnchor_ = void 0;
	}, t.prototype.resolveConstraints = function(e, t, n) {
		var r = e === void 0 ? 200 : e, i = t || 0, a = this.constraints_.rotation(this.targetRotation_), o = this.getViewportSize_(a), s = this.constraints_.resolution(this.targetResolution_, i, o), c = this.constraints_.center(this.targetCenter_, s, o, !1, this.calculateCenterShift(this.targetCenter_, s, a, o));
		if (r === 0 && !this.cancelAnchor_) {
			this.targetResolution_ = s, this.targetRotation_ = a, this.targetCenter_ = c, this.applyTargetState_();
			return;
		}
		var l = n || (r === 0 ? this.cancelAnchor_ : void 0);
		this.cancelAnchor_ = void 0, (this.getResolution() !== s || this.getRotation() !== a || !this.getCenterInternal() || !sn(this.getCenterInternal(), c)) && (this.getAnimating() && this.cancelAnimations(), this.animateInternal({
			rotation: a,
			center: c,
			resolution: s,
			duration: r,
			easing: ki,
			anchor: l
		}));
	}, t.prototype.beginInteraction = function() {
		this.resolveConstraints(0), this.setHint(Y.INTERACTING, 1);
	}, t.prototype.endInteraction = function(e, t, n) {
		var r = n && On(n, this.getProjection());
		this.endInteractionInternal(e, t, r);
	}, t.prototype.endInteractionInternal = function(e, t, n) {
		this.setHint(Y.INTERACTING, -1), this.resolveConstraints(e, t, n);
	}, t.prototype.getConstrainedCenter = function(e, t) {
		var n = this.getViewportSize_(this.getRotation());
		return this.constraints_.center(e, t || this.getResolution(), n);
	}, t.prototype.getConstrainedZoom = function(e, t) {
		var n = this.getResolutionForZoom(e);
		return this.getZoomForResolution(this.getConstrainedResolution(n, t));
	}, t.prototype.getConstrainedResolution = function(e, t) {
		var n = t || 0, r = this.getViewportSize_(this.getRotation());
		return this.constraints_.resolution(e, n, r);
	}, t;
}(z);
function Aa(e, t) {
	setTimeout(function() {
		e(t);
	}, 0);
}
function ja(e) {
	if (e.extent !== void 0) {
		var t = e.smoothExtentConstraint === void 0 || e.smoothExtentConstraint;
		return _i(e.extent, e.constrainOnlyCenter, t);
	}
	var n = bn(e.projection, "EPSG:3857");
	if (e.multiWorld !== !0 && n.isGlobal()) {
		var r = n.getExtent().slice();
		return r[0] = -Infinity, r[2] = Infinity, _i(r, !1, !1);
	}
	return vi;
}
function Ma(e) {
	var t, n, r, i = 28, a = 2, o = e.minZoom === void 0 ? Oa : e.minZoom, s = e.maxZoom === void 0 ? i : e.maxZoom, c = e.zoomFactor === void 0 ? a : e.zoomFactor, l = e.multiWorld !== void 0 && e.multiWorld, u = e.smoothResolutionConstraint === void 0 || e.smoothResolutionConstraint, d = e.showFullExtent !== void 0 && e.showFullExtent, f = bn(e.projection, "EPSG:3857"), p = f.getExtent(), m = e.constrainOnlyCenter, h = e.extent;
	if (!l && !h && f.isGlobal() && (m = !1, h = p), e.resolutions !== void 0) {
		var g = e.resolutions;
		n = g[o], r = g[s] === void 0 ? g[g.length - 1] : g[s], t = e.constrainResolution ? xi(g, u, !m && h, d) : Ci(n, r, u, !m && h, d);
	} else {
		var _ = (p ? Math.max(U(p), qt(p)) : 360 * Be[B.DEGREES] / f.getMetersPerUnit()) / 256 / a ** +Oa, v = _ / a ** +(i - Oa);
		n = e.maxResolution, n === void 0 ? n = _ / c ** +o : o = 0, r = e.minResolution, r === void 0 && (r = e.maxZoom === void 0 ? v : e.maxResolution === void 0 ? _ / c ** +s : n / c ** +s), s = o + Math.floor(Math.log(n / r) / Math.log(c)), r = n / c ** +(s - o), t = e.constrainResolution ? Si(c, n, r, u, !m && h, d) : Ci(n, r, u, !m && h, d);
	}
	return {
		constraint: t,
		maxResolution: n,
		minResolution: r,
		minZoom: o,
		zoomFactor: c
	};
}
function Na(e) {
	if (e.enableRotation === void 0 || e.enableRotation) {
		var t = e.constrainRotation;
		return t === void 0 || t === !0 ? Di() : t === !1 ? Ti : typeof t == "number" ? Ei(t) : Ti;
	} else return wi;
}
function Pa(e) {
	return !(e.sourceCenter && e.targetCenter && !sn(e.sourceCenter, e.targetCenter) || e.sourceResolution !== e.targetResolution || e.sourceRotation !== e.targetRotation);
}
function Fa(e, t, n, r, i) {
	var a = Math.cos(-i), o = Math.sin(-i), s = e[0] * a - e[1] * o, c = e[1] * a + e[0] * o;
	return s += (t[0] / 2 - n[0]) * r, c += (n[1] - t[1] / 2) * r, o = -o, [s * a - c * o, c * a + s * o];
}
//#endregion
//#region node_modules/ol/size.js
function Ia(e) {
	return e[0] > 0 && e[1] > 0;
}
function La(e, t, n) {
	return n === void 0 && (n = [0, 0]), n[0] = e[0] * t + .5 | 0, n[1] = e[1] * t + .5 | 0, n;
}
function Ra(e, t) {
	return Array.isArray(e) ? e : (t === void 0 ? t = [e, e] : (t[0] = e, t[1] = e), t);
}
//#endregion
//#region node_modules/ol/PluggableMap.js
var za = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ba = function(e) {
	za(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = Va(t);
		n.boundHandleBrowserEvent_ = n.handleBrowserEvent.bind(n), n.maxTilesLoading_ = t.maxTilesLoading === void 0 ? 16 : t.maxTilesLoading, n.pixelRatio_ = t.pixelRatio === void 0 ? se : t.pixelRatio, n.postRenderTimeoutHandle_, n.animationDelayKey_, n.animationDelay_ = function() {
			this.animationDelayKey_ = void 0, this.renderFrame_(Date.now());
		}.bind(n), n.coordinateToPixelTransform_ = Rn(), n.pixelToCoordinateTransform_ = Rn(), n.frameIndex_ = 0, n.frameState_ = null, n.previousExtent_ = null, n.viewPropertyListenerKey_ = null, n.viewChangeListenerKey_ = null, n.layerGroupPropertyListenerKeys_ = null, n.viewport_ = document.createElement("div"), n.viewport_.className = "ol-viewport" + ("ontouchstart" in window ? " ol-touch" : ""), n.viewport_.style.position = "relative", n.viewport_.style.overflow = "hidden", n.viewport_.style.width = "100%", n.viewport_.style.height = "100%", n.overlayContainer_ = document.createElement("div"), n.overlayContainer_.style.position = "absolute", n.overlayContainer_.style.zIndex = "0", n.overlayContainer_.style.width = "100%", n.overlayContainer_.style.height = "100%", n.overlayContainer_.style.pointerEvents = "none", n.overlayContainer_.className = "ol-overlaycontainer", n.viewport_.appendChild(n.overlayContainer_), n.overlayContainerStopEvent_ = document.createElement("div"), n.overlayContainerStopEvent_.style.position = "absolute", n.overlayContainerStopEvent_.style.zIndex = "0", n.overlayContainerStopEvent_.style.width = "100%", n.overlayContainerStopEvent_.style.height = "100%", n.overlayContainerStopEvent_.style.pointerEvents = "none", n.overlayContainerStopEvent_.className = "ol-overlaycontainer-stopevent", n.viewport_.appendChild(n.overlayContainerStopEvent_), n.mapBrowserEventHandler_ = null, n.moveTolerance_ = t.moveTolerance, n.keyboardEventTarget_ = r.keyboardEventTarget, n.keyHandlerKeys_ = null, n.controls = r.controls || new ti(), n.interactions = r.interactions || new ti(), n.overlays_ = r.overlays, n.overlayIdIndex_ = {}, n.renderer_ = null, n.handleResize_, n.postRenderFunctions_ = [], n.tileQueue_ = new mi(n.getTilePriority.bind(n), n.handleTileChange_.bind(n)), n.addChangeListener(Kr.LAYERGROUP, n.handleLayerGroupChanged_), n.addChangeListener(Kr.VIEW, n.handleViewChanged_), n.addChangeListener(Kr.SIZE, n.handleSizeChanged_), n.addChangeListener(Kr.TARGET, n.handleTargetChanged_), n.setProperties(r.values);
		var i = n;
		return t.view && !(t.view instanceof ka) && t.view.then(function(e) {
			i.setView(new ka(e));
		}), n.controls.addEventListener(Zr.ADD, function(e) {
			e.element.setMap(this);
		}.bind(n)), n.controls.addEventListener(Zr.REMOVE, function(e) {
			e.element.setMap(null);
		}.bind(n)), n.interactions.addEventListener(Zr.ADD, function(e) {
			e.element.setMap(this);
		}.bind(n)), n.interactions.addEventListener(Zr.REMOVE, function(e) {
			e.element.setMap(null);
		}.bind(n)), n.overlays_.addEventListener(Zr.ADD, function(e) {
			this.addOverlayInternal_(e.element);
		}.bind(n)), n.overlays_.addEventListener(Zr.REMOVE, function(e) {
			var t = e.element.getId();
			t !== void 0 && delete this.overlayIdIndex_[t.toString()], e.element.setMap(null);
		}.bind(n)), n.controls.forEach(function(e) {
			e.setMap(this);
		}.bind(n)), n.interactions.forEach(function(e) {
			e.setMap(this);
		}.bind(n)), n.overlays_.forEach(n.addOverlayInternal_.bind(n)), n;
	}
	return t.prototype.createRenderer = function() {
		throw Error("Use a map type that has a createRenderer method");
	}, t.prototype.addControl = function(e) {
		this.getControls().push(e);
	}, t.prototype.addInteraction = function(e) {
		this.getInteractions().push(e);
	}, t.prototype.addLayer = function(e) {
		this.getLayerGroup().getLayers().push(e);
	}, t.prototype.addOverlay = function(e) {
		this.getOverlays().push(e);
	}, t.prototype.addOverlayInternal_ = function(e) {
		var t = e.getId();
		t !== void 0 && (this.overlayIdIndex_[t.toString()] = e), e.setMap(this);
	}, t.prototype.disposeInternal = function() {
		this.setTarget(null), e.prototype.disposeInternal.call(this);
	}, t.prototype.forEachFeatureAtPixel = function(e, t, n) {
		if (this.frameState_) {
			var r = this.getCoordinateFromPixelInternal(e);
			n = n === void 0 ? {} : n;
			var i = n.hitTolerance === void 0 ? 0 : n.hitTolerance, a = n.layerFilter === void 0 ? v : n.layerFilter, o = n.checkWrapped !== !1;
			return this.renderer_.forEachFeatureAtCoordinate(r, this.frameState_, i, o, t, null, a, null);
		}
	}, t.prototype.getFeaturesAtPixel = function(e, t) {
		var n = [];
		return this.forEachFeatureAtPixel(e, function(e) {
			n.push(e);
		}, t), n;
	}, t.prototype.forEachLayerAtPixel = function(e, t, n) {
		if (this.frameState_) {
			var r = n || {}, i = r.hitTolerance === void 0 ? 0 : r.hitTolerance, a = r.layerFilter || v;
			return this.renderer_.forEachLayerAtPixel(e, this.frameState_, i, t, a);
		}
	}, t.prototype.hasFeatureAtPixel = function(e, t) {
		if (!this.frameState_) return !1;
		var n = this.getCoordinateFromPixelInternal(e);
		t = t === void 0 ? {} : t;
		var r = t.layerFilter === void 0 ? v : t.layerFilter, i = t.hitTolerance === void 0 ? 0 : t.hitTolerance, a = t.checkWrapped !== !1;
		return this.renderer_.hasFeatureAtCoordinate(n, this.frameState_, i, a, r, null);
	}, t.prototype.getEventCoordinate = function(e) {
		return this.getCoordinateFromPixel(this.getEventPixel(e));
	}, t.prototype.getEventCoordinateInternal = function(e) {
		return this.getCoordinateFromPixelInternal(this.getEventPixel(e));
	}, t.prototype.getEventPixel = function(e) {
		var t = this.viewport_.getBoundingClientRect(), n = "changedTouches" in e ? e.changedTouches[0] : e;
		return [n.clientX - t.left, n.clientY - t.top];
	}, t.prototype.getTarget = function() {
		return this.get(Kr.TARGET);
	}, t.prototype.getTargetElement = function() {
		var e = this.getTarget();
		return e === void 0 ? null : typeof e == "string" ? document.getElementById(e) : e;
	}, t.prototype.getCoordinateFromPixel = function(e) {
		return Dn(this.getCoordinateFromPixelInternal(e), this.getView().getProjection());
	}, t.prototype.getCoordinateFromPixelInternal = function(e) {
		var t = this.frameState_;
		return t ? G(t.pixelToCoordinateTransform, e.slice()) : null;
	}, t.prototype.getControls = function() {
		return this.controls;
	}, t.prototype.getOverlays = function() {
		return this.overlays_;
	}, t.prototype.getOverlayById = function(e) {
		var t = this.overlayIdIndex_[e.toString()];
		return t === void 0 ? null : t;
	}, t.prototype.getInteractions = function() {
		return this.interactions;
	}, t.prototype.getLayerGroup = function() {
		return this.get(Kr.LAYERGROUP);
	}, t.prototype.setLayers = function(e) {
		var t = this.getLayerGroup();
		if (e instanceof ti) {
			t.setLayers(e);
			return;
		}
		var n = t.getLayers();
		n.clear(), n.extend(e);
	}, t.prototype.getLayers = function() {
		return this.getLayerGroup().getLayers();
	}, t.prototype.getLoading = function() {
		for (var e = this.getLayerGroup().getLayerStatesArray(), t = 0, n = e.length; t < n; ++t) {
			var r = e[t].layer.getSource();
			if (r && r.loading) return !0;
		}
		return !1;
	}, t.prototype.getPixelFromCoordinate = function(e) {
		var t = On(e, this.getView().getProjection());
		return this.getPixelFromCoordinateInternal(t);
	}, t.prototype.getPixelFromCoordinateInternal = function(e) {
		var t = this.frameState_;
		return t ? G(t.coordinateToPixelTransform, e.slice(0, 2)) : null;
	}, t.prototype.getRenderer = function() {
		return this.renderer_;
	}, t.prototype.getSize = function() {
		return this.get(Kr.SIZE);
	}, t.prototype.getView = function() {
		return this.get(Kr.VIEW);
	}, t.prototype.getViewport = function() {
		return this.viewport_;
	}, t.prototype.getOverlayContainer = function() {
		return this.overlayContainer_;
	}, t.prototype.getOverlayContainerStopEvent = function() {
		return this.overlayContainerStopEvent_;
	}, t.prototype.getOwnerDocument = function() {
		var e = this.getTargetElement();
		return e ? e.ownerDocument : document;
	}, t.prototype.getTilePriority = function(e, t, n, r) {
		return hi(this.frameState_, e, t, n, r);
	}, t.prototype.handleBrowserEvent = function(e, t) {
		var n = new ci(t || e.type, this, e);
		this.handleMapBrowserEvent(n);
	}, t.prototype.handleMapBrowserEvent = function(e) {
		if (this.frameState_) {
			var t = e.originalEvent, n = t.type;
			if (n === ze.POINTERDOWN || n === O.WHEEL || n === O.KEYDOWN) {
				var r = this.getOwnerDocument(), i = this.viewport_.getRootNode ? this.viewport_.getRootNode() : r, a = t.target;
				if (this.overlayContainerStopEvent_.contains(a) || !(i === r ? r.documentElement : i).contains(a)) return;
			}
			if (e.frameState = this.frameState_, this.dispatchEvent(e) !== !1) for (var o = this.getInteractions().getArray().slice(), s = o.length - 1; s >= 0; s--) {
				var c = o[s];
				if (!(c.getMap() !== this || !c.getActive() || !this.getTargetElement()) && (!c.handleEvent(e) || e.propagationStopped)) break;
			}
		}
	}, t.prototype.handlePostRender = function() {
		var e = this.frameState_, t = this.tileQueue_;
		if (!t.isEmpty()) {
			var n = this.maxTilesLoading_, r = n;
			if (e) {
				var i = e.viewHints;
				if (i[Y.ANIMATING] || i[Y.INTERACTING]) {
					var a = Date.now() - e.time > 8;
					n = a ? 0 : 8, r = a ? 0 : 2;
				}
			}
			t.getTilesLoading() < n && (t.reprioritize(), t.loadMoreTiles(n, r));
		}
		e && this.hasListener(fr.RENDERCOMPLETE) && !e.animate && !this.tileQueue_.getTilesLoading() && !this.getLoading() && this.renderer_.dispatchRenderEvent(fr.RENDERCOMPLETE, e);
		for (var o = this.postRenderFunctions_, s = 0, c = o.length; s < c; ++s) o[s](this, e);
		o.length = 0;
	}, t.prototype.handleSizeChanged_ = function() {
		this.getView() && !this.getView().getAnimating() && this.getView().resolveConstraints(0), this.render();
	}, t.prototype.handleTargetChanged_ = function() {
		var e;
		if (this.getTarget() && (e = this.getTargetElement()), this.mapBrowserEventHandler_) {
			for (var t = 0, n = this.keyHandlerKeys_.length; t < n; ++t) j(this.keyHandlerKeys_[t]);
			this.keyHandlerKeys_ = null, this.viewport_.removeEventListener(O.CONTEXTMENU, this.boundHandleBrowserEvent_), this.viewport_.removeEventListener(O.WHEEL, this.boundHandleBrowserEvent_), this.handleResize_ !== void 0 && (removeEventListener(O.RESIZE, this.handleResize_, !1), this.handleResize_ = void 0), this.mapBrowserEventHandler_.dispose(), this.mapBrowserEventHandler_ = null, he(this.viewport_);
		}
		if (!e) this.renderer_ &&= (clearTimeout(this.postRenderTimeoutHandle_), this.postRenderTimeoutHandle_ = void 0, this.postRenderFunctions_.length = 0, this.renderer_.dispose(), null), this.animationDelayKey_ &&= (cancelAnimationFrame(this.animationDelayKey_), void 0);
		else {
			for (var r in e.appendChild(this.viewport_), this.renderer_ ||= this.createRenderer(), this.mapBrowserEventHandler_ = new ui(this, this.moveTolerance_), q) this.mapBrowserEventHandler_.addEventListener(q[r], this.handleMapBrowserEvent.bind(this));
			this.viewport_.addEventListener(O.CONTEXTMENU, this.boundHandleBrowserEvent_, !1), this.viewport_.addEventListener(O.WHEEL, this.boundHandleBrowserEvent_, ue ? { passive: !1 } : !1);
			var i = this.keyboardEventTarget_ ? this.keyboardEventTarget_ : e;
			this.keyHandlerKeys_ = [k(i, O.KEYDOWN, this.handleBrowserEvent, this), k(i, O.KEYPRESS, this.handleBrowserEvent, this)], this.handleResize_ || (this.handleResize_ = this.updateSize.bind(this), window.addEventListener(O.RESIZE, this.handleResize_, !1));
		}
		this.updateSize();
	}, t.prototype.handleTileChange_ = function() {
		this.render();
	}, t.prototype.handleViewPropertyChanged_ = function() {
		this.render();
	}, t.prototype.handleViewChanged_ = function() {
		this.viewPropertyListenerKey_ &&= (j(this.viewPropertyListenerKey_), null), this.viewChangeListenerKey_ &&= (j(this.viewChangeListenerKey_), null);
		var e = this.getView();
		e && (this.updateViewportSize_(), this.viewPropertyListenerKey_ = k(e, u.PROPERTYCHANGE, this.handleViewPropertyChanged_, this), this.viewChangeListenerKey_ = k(e, O.CHANGE, this.handleViewPropertyChanged_, this), e.resolveConstraints(0)), this.render();
	}, t.prototype.handleLayerGroupChanged_ = function() {
		this.layerGroupPropertyListenerKeys_ &&= (this.layerGroupPropertyListenerKeys_.forEach(j), null);
		var e = this.getLayerGroup();
		e && (this.layerGroupPropertyListenerKeys_ = [k(e, u.PROPERTYCHANGE, this.render, this), k(e, O.CHANGE, this.render, this)]), this.render();
	}, t.prototype.isRendered = function() {
		return !!this.frameState_;
	}, t.prototype.renderSync = function() {
		this.animationDelayKey_ && cancelAnimationFrame(this.animationDelayKey_), this.animationDelay_();
	}, t.prototype.redrawText = function() {
		for (var e = this.getLayerGroup().getLayerStatesArray(), t = 0, n = e.length; t < n; ++t) {
			var r = e[t].layer;
			r.hasRenderer() && r.getRenderer().handleFontsChanged();
		}
	}, t.prototype.render = function() {
		this.renderer_ && this.animationDelayKey_ === void 0 && (this.animationDelayKey_ = requestAnimationFrame(this.animationDelay_));
	}, t.prototype.removeControl = function(e) {
		return this.getControls().remove(e);
	}, t.prototype.removeInteraction = function(e) {
		return this.getInteractions().remove(e);
	}, t.prototype.removeLayer = function(e) {
		return this.getLayerGroup().getLayers().remove(e);
	}, t.prototype.removeOverlay = function(e) {
		return this.getOverlays().remove(e);
	}, t.prototype.renderFrame_ = function(e) {
		var t = this, n = this.getSize(), r = this.getView(), i = this.frameState_, a = null;
		if (n !== void 0 && Ia(n) && r && r.isDef()) {
			var o = r.getHints(this.frameState_ ? this.frameState_.viewHints : void 0), s = r.getState();
			if (a = {
				animate: !1,
				coordinateToPixelTransform: this.coordinateToPixelTransform_,
				declutterTree: null,
				extent: Kt(s.center, s.resolution, s.rotation, n),
				index: this.frameIndex_++,
				layerIndex: 0,
				layerStatesArray: this.getLayerGroup().getLayerStatesArray(),
				pixelRatio: this.pixelRatio_,
				pixelToCoordinateTransform: this.pixelToCoordinateTransform_,
				postRenderFunctions: [],
				size: n,
				tileQueue: this.tileQueue_,
				time: e,
				usedTiles: {},
				viewState: s,
				viewHints: o,
				wantedTiles: {}
			}, s.nextCenter && s.nextResolution) {
				var c = isNaN(s.nextRotation) ? s.rotation : s.nextRotation;
				a.nextExtent = Kt(s.nextCenter, s.nextResolution, c, n);
			}
		}
		this.frameState_ = a, this.renderer_.renderFrame(a), a && (a.animate && this.render(), Array.prototype.push.apply(this.postRenderFunctions_, a.postRenderFunctions), i && (!this.previousExtent_ || !Qt(this.previousExtent_) && !Ft(a.extent, this.previousExtent_)) && (this.dispatchEvent(new oi(ne.MOVESTART, this, i)), this.previousExtent_ = Mt(this.previousExtent_)), this.previousExtent_ && !a.viewHints[Y.ANIMATING] && !a.viewHints[Y.INTERACTING] && !Ft(a.extent, this.previousExtent_) && (this.dispatchEvent(new oi(ne.MOVEEND, this, a)), wt(a.extent, this.previousExtent_))), this.dispatchEvent(new oi(ne.POSTRENDER, this, a)), this.postRenderTimeoutHandle_ ||= setTimeout(function() {
			t.postRenderTimeoutHandle_ = void 0, t.handlePostRender();
		}, 0);
	}, t.prototype.setLayerGroup = function(e) {
		this.set(Kr.LAYERGROUP, e);
	}, t.prototype.setSize = function(e) {
		this.set(Kr.SIZE, e);
	}, t.prototype.setTarget = function(e) {
		this.set(Kr.TARGET, e);
	}, t.prototype.setView = function(e) {
		if (!e || e instanceof ka) {
			this.set(Kr.VIEW, e);
			return;
		}
		this.set(Kr.VIEW, new ka());
		var t = this;
		e.then(function(e) {
			t.setView(new ka(e));
		});
	}, t.prototype.updateSize = function() {
		var e = this.getTargetElement(), t = void 0;
		if (e) {
			var n = getComputedStyle(e), r = e.offsetWidth - parseFloat(n.borderLeftWidth) - parseFloat(n.paddingLeft) - parseFloat(n.paddingRight) - parseFloat(n.borderRightWidth), i = e.offsetHeight - parseFloat(n.borderTopWidth) - parseFloat(n.paddingTop) - parseFloat(n.paddingBottom) - parseFloat(n.borderBottomWidth);
			!isNaN(r) && !isNaN(i) && (t = [r, i], !Ia(t) && (e.offsetWidth || e.offsetHeight || e.getClientRects().length) && console.warn("No map visible because the map container's width or height are 0."));
		}
		this.setSize(t), this.updateViewportSize_();
	}, t.prototype.updateViewportSize_ = function() {
		var e = this.getView();
		if (e) {
			var t = void 0, n = getComputedStyle(this.viewport_);
			n.width && n.height && (t = [parseInt(n.width, 10), parseInt(n.height, 10)]), e.setViewportSize(t);
		}
	}, t;
}(z);
function Va(e) {
	var t = null;
	e.keyboardEventTarget !== void 0 && (t = typeof e.keyboardEventTarget == "string" ? document.getElementById(e.keyboardEventTarget) : e.keyboardEventTarget);
	var n = {}, r = e.layers && typeof e.layers.getLayers == "function" ? e.layers : new ii({ layers: e.layers });
	n[Kr.LAYERGROUP] = r, n[Kr.TARGET] = e.target, n[Kr.VIEW] = e.view instanceof ka ? e.view : new ka();
	var i;
	e.controls !== void 0 && (Array.isArray(e.controls) ? i = new ti(e.controls.slice()) : (H(typeof e.controls.getArray == "function", 47), i = e.controls));
	var a;
	e.interactions !== void 0 && (Array.isArray(e.interactions) ? a = new ti(e.interactions.slice()) : (H(typeof e.interactions.getArray == "function", 48), a = e.interactions));
	var o;
	return e.overlays === void 0 ? o = new ti() : Array.isArray(e.overlays) ? o = new ti(e.overlays.slice()) : (H(typeof e.overlays.getArray == "function", 49), o = e.overlays), {
		controls: i,
		interactions: a,
		keyboardEventTarget: t,
		overlays: o,
		values: n
	};
}
//#endregion
//#region node_modules/ol/control/OverviewMap.js
var Ha = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ua = .75, Wa = .1, Ga = function(e) {
	Ha(t, e);
	function t() {
		return e !== null && e.apply(this, arguments) || this;
	}
	return t.prototype.createRenderer = function() {
		return new Gr(this);
	}, t;
}(Ba), Ka = function(e) {
	Ha(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, {
			element: document.createElement("div"),
			render: r.render,
			target: r.target
		}) || this, n.boundHandleRotationChanged_ = n.handleRotationChanged_.bind(n), n.collapsed_ = r.collapsed === void 0 || r.collapsed, n.collapsible_ = r.collapsible === void 0 || r.collapsible, n.collapsible_ || (n.collapsed_ = !1), n.rotateWithView_ = r.rotateWithView !== void 0 && r.rotateWithView, n.viewExtent_ = void 0;
		var i = r.className === void 0 ? "ol-overviewmap" : r.className, a = r.tipLabel === void 0 ? "Overview map" : r.tipLabel, o = r.collapseLabel === void 0 ? "‹" : r.collapseLabel;
		typeof o == "string" ? (n.collapseLabel_ = document.createElement("span"), n.collapseLabel_.textContent = o) : n.collapseLabel_ = o;
		var s = r.label === void 0 ? "›" : r.label;
		typeof s == "string" ? (n.label_ = document.createElement("span"), n.label_.textContent = s) : n.label_ = s;
		var c = n.collapsible_ && !n.collapsed_ ? n.collapseLabel_ : n.label_, l = document.createElement("button");
		l.setAttribute("type", "button"), l.title = a, l.appendChild(c), l.addEventListener(O.CLICK, n.handleClick_.bind(n), !1), n.ovmapDiv_ = document.createElement("div"), n.ovmapDiv_.className = "ol-overviewmap-map", n.view_ = r.view, n.ovmap_ = new Ga({ view: r.view });
		var u = n.ovmap_;
		r.layers && r.layers.forEach(function(e) {
			u.addLayer(e);
		});
		var d = document.createElement("div");
		d.className = "ol-overviewmap-box", d.style.boxSizing = "border-box", n.boxOverlay_ = new Xr({
			position: [0, 0],
			positioning: qr.CENTER_CENTER,
			element: d
		}), n.ovmap_.addOverlay(n.boxOverlay_);
		var f = i + " " + Se + " " + we + (n.collapsed_ && n.collapsible_ ? " " + Te : "") + (n.collapsible_ ? "" : " ol-uncollapsible"), p = n.element;
		p.className = f, p.appendChild(n.ovmapDiv_), p.appendChild(l);
		var m = n, h = n.boxOverlay_, g = n.boxOverlay_.getElement(), _ = function(e) {
			return {
				clientX: e.clientX,
				clientY: e.clientY
			};
		}, v = function(e) {
			var t = _(e), n = u.getEventCoordinateInternal(t);
			h.setPosition(n);
		}, y = function(e) {
			var t = u.getEventCoordinateInternal(e);
			m.getMap().getView().setCenterInternal(t), window.removeEventListener("mousemove", v), window.removeEventListener("mouseup", y);
		};
		return g.addEventListener("mousedown", function() {
			window.addEventListener("mousemove", v), window.addEventListener("mouseup", y);
		}), n;
	}
	return t.prototype.setMap = function(t) {
		var n = this.getMap();
		if (t !== n) {
			if (n) {
				var r = n.getView();
				r && this.unbindView_(r), this.ovmap_.setTarget(null);
			}
			if (e.prototype.setMap.call(this, t), t) {
				this.ovmap_.setTarget(this.ovmapDiv_), this.listenerKeys.push(k(t, u.PROPERTYCHANGE, this.handleMapPropertyChange_, this));
				var i = t.getView();
				i && (this.bindView_(i), i.isDef() && (this.ovmap_.updateSize(), this.resetExtent_())), this.ovmap_.isRendered() || this.updateBoxAfterOvmapIsRendered_();
			}
		}
	}, t.prototype.handleMapPropertyChange_ = function(e) {
		if (e.key === Kr.VIEW) {
			var t = e.oldValue;
			t && this.unbindView_(t);
			var n = this.getMap().getView();
			this.bindView_(n);
		} else !this.ovmap_.isRendered() && (e.key === Kr.TARGET || e.key === Kr.SIZE) && this.ovmap_.updateSize();
	}, t.prototype.bindView_ = function(e) {
		if (!this.view_) {
			var t = new ka({ projection: e.getProjection() });
			this.ovmap_.setView(t);
		}
		e.addChangeListener(gi.ROTATION, this.boundHandleRotationChanged_), this.handleRotationChanged_();
	}, t.prototype.unbindView_ = function(e) {
		e.removeChangeListener(gi.ROTATION, this.boundHandleRotationChanged_);
	}, t.prototype.handleRotationChanged_ = function() {
		this.rotateWithView_ && this.ovmap_.getView().setRotation(this.getMap().getView().getRotation());
	}, t.prototype.validateExtent_ = function() {
		var e = this.getMap(), t = this.ovmap_;
		if (!(!e.isRendered() || !t.isRendered())) {
			var n = e.getSize(), r = e.getView().calculateExtentInternal(n);
			if (!(this.viewExtent_ && Ft(r, this.viewExtent_))) {
				this.viewExtent_ = r;
				var i = t.getSize(), a = t.getView().calculateExtentInternal(i), o = t.getPixelFromCoordinateInternal(Yt(r)), s = t.getPixelFromCoordinateInternal(Ut(r)), c = Math.abs(o[0] - s[0]), l = Math.abs(o[1] - s[1]), u = i[0], d = i[1];
				c < u * Wa || l < d * Wa || c > u * Ua || l > d * Ua ? this.resetExtent_() : Dt(a, r) || this.recenter_();
			}
		}
	}, t.prototype.resetExtent_ = function() {
		if (!(Ua === 0 || Wa === 0)) {
			var e = this.getMap(), t = this.ovmap_, n = e.getSize(), r = e.getView().calculateExtentInternal(n), i = t.getView();
			en(r, 1 / (2 ** (Math.log(Ua / Wa) / Math.LN2 / 2) * Wa)), i.fitInternal(Ea(r));
		}
	}, t.prototype.recenter_ = function() {
		var e = this.getMap(), t = this.ovmap_, n = e.getView();
		t.getView().setCenterInternal(n.getCenterInternal());
	}, t.prototype.updateBox_ = function() {
		var e = this.getMap(), t = this.ovmap_;
		if (!(!e.isRendered() || !t.isRendered())) {
			var n = e.getSize(), r = e.getView(), i = t.getView(), a = this.rotateWithView_ ? 0 : -r.getRotation(), o = this.boxOverlay_, s = this.boxOverlay_.getElement(), c = r.getCenterInternal(), l = r.getResolution(), u = i.getResolution(), d = n[0] * l / u, f = n[1] * l / u;
			if (o.setPosition(c), s) {
				s.style.width = d + "px", s.style.height = f + "px";
				var p = "rotate(" + a + "rad)";
				s.style.transform = p;
			}
		}
	}, t.prototype.updateBoxAfterOvmapIsRendered_ = function() {
		this.ovmapPostrenderKey_ ||= A(this.ovmap_, ne.POSTRENDER, function(e) {
			delete this.ovmapPostrenderKey_, this.updateBox_();
		}, this);
	}, t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.handleToggle_();
	}, t.prototype.handleToggle_ = function() {
		this.element.classList.toggle(Te), this.collapsed_ ? me(this.collapseLabel_, this.label_) : me(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_;
		var e = this.ovmap_;
		if (!this.collapsed_) {
			if (e.isRendered()) {
				this.viewExtent_ = void 0, e.render();
				return;
			}
			e.updateSize(), this.resetExtent_(), this.updateBoxAfterOvmapIsRendered_();
		}
	}, t.prototype.getCollapsible = function() {
		return this.collapsible_;
	}, t.prototype.setCollapsible = function(e) {
		this.collapsible_ !== e && (this.collapsible_ = e, this.element.classList.toggle("ol-uncollapsible"), !e && this.collapsed_ && this.handleToggle_());
	}, t.prototype.setCollapsed = function(e) {
		!this.collapsible_ || this.collapsed_ === e || this.handleToggle_();
	}, t.prototype.getCollapsed = function() {
		return this.collapsed_;
	}, t.prototype.getRotateWithView = function() {
		return this.rotateWithView_;
	}, t.prototype.setRotateWithView = function(e) {
		this.rotateWithView_ !== e && (this.rotateWithView_ = e, this.getMap().getView().getRotation() !== 0 && (this.rotateWithView_ ? this.handleRotationChanged_() : this.ovmap_.getView().setRotation(0), this.viewExtent_ = void 0, this.validateExtent_(), this.updateBox_()));
	}, t.prototype.getOverviewMap = function() {
		return this.ovmap_;
	}, t.prototype.render = function(e) {
		this.validateExtent_(), this.updateBox_();
	}, t;
}(ye), qa = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ja = function(e) {
	qa(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, {
			element: document.createElement("div"),
			render: r.render,
			target: r.target
		}) || this;
		var i = r.className === void 0 ? "ol-rotate" : r.className, a = r.label === void 0 ? "⇧" : r.label, o = r.compassClassName === void 0 ? "ol-compass" : r.compassClassName;
		n.label_ = null, typeof a == "string" ? (n.label_ = document.createElement("span"), n.label_.className = o, n.label_.textContent = a) : (n.label_ = a, n.label_.classList.add(o));
		var s = r.tipLabel ? r.tipLabel : "Reset rotation", c = document.createElement("button");
		c.className = i + "-reset", c.setAttribute("type", "button"), c.title = s, c.appendChild(n.label_), c.addEventListener(O.CLICK, n.handleClick_.bind(n), !1);
		var l = i + " " + Se + " " + we, u = n.element;
		return u.className = l, u.appendChild(c), n.callResetNorth_ = r.resetNorth ? r.resetNorth : void 0, n.duration_ = r.duration === void 0 ? 250 : r.duration, n.autoHide_ = r.autoHide === void 0 || r.autoHide, n.rotation_ = void 0, n.autoHide_ && n.element.classList.add(be), n;
	}
	return t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.callResetNorth_ === void 0 ? this.resetNorth_() : this.callResetNorth_();
	}, t.prototype.resetNorth_ = function() {
		var e = this.getMap().getView();
		if (e) {
			var t = e.getRotation();
			t !== void 0 && (this.duration_ > 0 && t % (2 * Math.PI) != 0 ? e.animate({
				rotation: 0,
				duration: this.duration_,
				easing: ki
			}) : e.setRotation(0));
		}
	}, t.prototype.render = function(e) {
		var t = e.frameState;
		if (t) {
			var n = t.viewState.rotation;
			if (n != this.rotation_) {
				var r = "rotate(" + n + "rad)";
				if (this.autoHide_) {
					var i = this.element.classList.contains(be);
					!i && n === 0 ? this.element.classList.add(be) : i && n !== 0 && this.element.classList.remove(be);
				}
				this.label_.style.transform = r;
			}
			this.rotation_ = n;
		}
	}, t;
}(ye), Ya = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Xa = "units", Za = {
	DEGREES: "degrees",
	IMPERIAL: "imperial",
	NAUTICAL: "nautical",
	METRIC: "metric",
	US: "us"
}, Qa = [
	1,
	2,
	5
], $a = 25.4 / .28, eo = function(e) {
	Ya(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r.className === void 0 ? r.bar ? "ol-scale-bar" : "ol-scale-line" : r.className;
		return n = e.call(this, {
			element: document.createElement("div"),
			render: r.render,
			target: r.target
		}) || this, n.on, n.once, n.un, n.innerElement_ = document.createElement("div"), n.innerElement_.className = i + "-inner", n.element.className = i + " " + Se, n.element.appendChild(n.innerElement_), n.viewState_ = null, n.minWidth_ = r.minWidth === void 0 ? 64 : r.minWidth, n.renderedVisible_ = !1, n.renderedWidth_ = void 0, n.renderedHTML_ = "", n.addChangeListener(Xa, n.handleUnitsChanged_), n.setUnits(r.units || Za.METRIC), n.scaleBar_ = r.bar || !1, n.scaleBarSteps_ = r.steps || 4, n.scaleBarText_ = r.text || !1, n.dpi_ = r.dpi || void 0, n;
	}
	return t.prototype.getUnits = function() {
		return this.get(Xa);
	}, t.prototype.handleUnitsChanged_ = function() {
		this.updateElement_();
	}, t.prototype.setUnits = function(e) {
		this.set(Xa, e);
	}, t.prototype.setDpi = function(e) {
		this.dpi_ = e;
	}, t.prototype.updateElement_ = function() {
		var e = this.viewState_;
		if (!e) {
			this.renderedVisible_ &&= (this.element.style.display = "none", !1);
			return;
		}
		var t = e.center, n = e.projection, r = this.getUnits(), i = r == Za.DEGREES ? B.DEGREES : B.METERS, a = _n(n, e.resolution, t, i), o = this.minWidth_ * (this.dpi_ || $a) / $a, s = o * a, c = "";
		if (r == Za.DEGREES) {
			var l = Be[B.DEGREES];
			s *= l, s < l / 60 ? (c = "″", a *= 3600) : s < l ? (c = "′", a *= 60) : c = "°";
		} else r == Za.IMPERIAL ? s < .9144 ? (c = "in", a /= .0254) : s < 1609.344 ? (c = "ft", a /= .3048) : (c = "mi", a /= 1609.344) : r == Za.NAUTICAL ? (a /= 1852, c = "nm") : r == Za.METRIC ? s < .001 ? (c = "μm", a *= 1e6) : s < 1 ? (c = "mm", a *= 1e3) : s < 1e3 ? c = "m" : (c = "km", a /= 1e3) : r == Za.US ? s < .9144 ? (c = "in", a *= 39.37) : s < 1609.344 ? (c = "ft", a /= .30480061) : (c = "mi", a /= 1609.3472) : H(!1, 33);
		for (var u = 3 * Math.floor(Math.log(o * a) / Math.log(10)), d, f, p;;) {
			p = Math.floor(u / 3);
			var m = 10 ** p;
			if (d = Qa[(u % 3 + 3) % 3] * m, f = Math.round(d / a), isNaN(f)) {
				this.element.style.display = "none", this.renderedVisible_ = !1;
				return;
			} else if (f >= o) break;
			++u;
		}
		var h = this.scaleBar_ ? this.createScaleBar(f, d, c) : d.toFixed(p < 0 ? -p : 0) + " " + c;
		this.renderedHTML_ != h && (this.innerElement_.innerHTML = h, this.renderedHTML_ = h), this.renderedWidth_ != f && (this.innerElement_.style.width = f + "px", this.renderedWidth_ = f), this.renderedVisible_ ||= (this.element.style.display = "", !0);
	}, t.prototype.createScaleBar = function(e, t, n) {
		for (var r = "1 : " + Math.round(this.getScaleForResolution()).toLocaleString(), i = [], a = e / this.scaleBarSteps_, o = "#ffffff", s = 0; s < this.scaleBarSteps_; s++) s === 0 && i.push(this.createMarker("absolute", s)), i.push("<div><div class=\"ol-scale-singlebar\" style=\"width: " + a + "px;background-color: " + o + ";\"></div>" + this.createMarker("relative", s) + (s % 2 == 0 || this.scaleBarSteps_ === 2 ? this.createStepText(s, e, !1, t, n) : "") + "</div>"), s === this.scaleBarSteps_ - 1 && i.push(this.createStepText(s + 1, e, !0, t, n)), o = o === "#ffffff" ? "#000000" : "#ffffff";
		return "<div style=\"display: flex;\">" + (this.scaleBarText_ ? "<div class=\"ol-scale-text\" style=\"width: " + e + "px;\">" + r + "</div>" : "") + i.join("") + "</div>";
	}, t.prototype.createMarker = function(e, t) {
		var n = e === "absolute" ? 3 : -10;
		return "<div class=\"ol-scale-step-marker\" style=\"position: " + e + ";top: " + n + "px;\"></div>";
	}, t.prototype.createStepText = function(e, t, n, r, i) {
		var a = (e === 0 ? 0 : Math.round(r / this.scaleBarSteps_ * e * 100) / 100) + (e === 0 ? "" : " " + i), o = e === 0 ? -3 : t / this.scaleBarSteps_ * -1, s = e === 0 ? 0 : t / this.scaleBarSteps_ * 2;
		return "<div class=\"ol-scale-step-text\" style=\"margin-left: " + o + "px;text-align: " + (e === 0 ? "left" : "center") + "; min-width: " + s + "px;left: " + (n ? t + "px" : "unset") + ";\">" + a + "</div>";
	}, t.prototype.getScaleForResolution = function() {
		var e = _n(this.viewState_.projection, this.viewState_.resolution, this.viewState_.center), t = this.dpi_ || $a, n = this.viewState_.projection.getMetersPerUnit();
		return parseFloat(e.toString()) * n * (1e3 / 25.4) * t;
	}, t.prototype.render = function(e) {
		var t = e.frameState;
		t ? this.viewState_ = t.viewState : this.viewState_ = null, this.updateElement_();
	}, t;
}(ye), to = {
	PRELOAD: "preload",
	USE_INTERIM_TILES_ON_ERROR: "useInterimTilesOnError"
}, no = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ro = function(e) {
	no(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		return delete i.preload, delete i.useInterimTilesOnError, n = e.call(this, i) || this, n.on, n.once, n.un, n.setPreload(r.preload === void 0 ? 0 : r.preload), n.setUseInterimTilesOnError(r.useInterimTilesOnError === void 0 || r.useInterimTilesOnError), n;
	}
	return t.prototype.getPreload = function() {
		return this.get(to.PRELOAD);
	}, t.prototype.setPreload = function(e) {
		this.set(to.PRELOAD, e);
	}, t.prototype.getUseInterimTilesOnError = function() {
		return this.get(to.USE_INTERIM_TILES_ON_ERROR);
	}, t.prototype.setUseInterimTilesOnError = function(e) {
		this.set(to.USE_INTERIM_TILES_ON_ERROR, e);
	}, t;
}(hr), X = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
}, io = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ao = function(e) {
	io(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.boundHandleImageChange_ = n.handleImageChange_.bind(n), n.layer_ = t, n.declutterExecutorGroup = null, n;
	}
	return t.prototype.getFeatures = function(e) {
		return F();
	}, t.prototype.prepareFrame = function(e) {
		return F();
	}, t.prototype.renderFrame = function(e, t) {
		return F();
	}, t.prototype.loadedTileCallback = function(e, t, n) {
		e[t] || (e[t] = {}), e[t][n.tileCoord.toString()] = n;
	}, t.prototype.createLoadedTileFinder = function(e, t, n) {
		return function(r, i) {
			var a = this.loadedTileCallback.bind(this, n, r);
			return e.forEachLoadedTile(t, r, i, a);
		}.bind(this);
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {}, t.prototype.getDataAtPixel = function(e, t, n) {
		return null;
	}, t.prototype.getLayer = function() {
		return this.layer_;
	}, t.prototype.handleFontsChanged = function() {}, t.prototype.handleImageChange_ = function(e) {
		e.target.getState() === X.LOADED && this.renderIfReadyAndVisible();
	}, t.prototype.loadImage = function(e) {
		var t = e.getState();
		return t != X.LOADED && t != X.ERROR && e.addEventListener(O.CHANGE, this.boundHandleImageChange_), t == X.IDLE && (e.load(), t = e.getState()), t == X.LOADED;
	}, t.prototype.renderIfReadyAndVisible = function() {
		var e = this.getLayer();
		e.getVisible() && e.getSourceState() == pr.READY && e.changed();
	}, t;
}(N), oo = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), so = function(e) {
	oo(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.container = null, n.renderedResolution, n.tempTransform = Rn(), n.pixelTransform = Rn(), n.inversePixelTransform = Rn(), n.context = null, n.containerReused = !1, n;
	}
	return t.prototype.useContainer = function(e, t, n) {
		var r = this.getLayer().getClassName(), i, a;
		if (e && e.style.opacity === ke(n) && e.className === r) {
			var o = e.firstElementChild;
			o instanceof HTMLCanvasElement && (a = o.getContext("2d"));
		}
		if (a && a.canvas.style.transform === t ? (this.container = e, this.context = a, this.containerReused = !0) : this.containerReused &&= (this.container = null, this.context = null, !1), !this.container) {
			i = document.createElement("div"), i.className = r;
			var s = i.style;
			s.position = "absolute", s.width = "100%", s.height = "100%", a = de();
			var o = a.canvas;
			i.appendChild(o), s = o.style, s.position = "absolute", s.left = "0", s.transformOrigin = "top left", this.container = i, this.context = a;
		}
	}, t.prototype.clipUnrotated = function(e, t, n) {
		var r = Yt(n), i = Xt(n), a = Ut(n), o = Ht(n);
		G(t.coordinateToPixelTransform, r), G(t.coordinateToPixelTransform, i), G(t.coordinateToPixelTransform, a), G(t.coordinateToPixelTransform, o);
		var s = this.inversePixelTransform;
		G(s, r), G(s, i), G(s, a), G(s, o), e.save(), e.beginPath(), e.moveTo(Math.round(r[0]), Math.round(r[1])), e.lineTo(Math.round(i[0]), Math.round(i[1])), e.lineTo(Math.round(a[0]), Math.round(a[1])), e.lineTo(Math.round(o[0]), Math.round(o[1])), e.clip();
	}, t.prototype.dispatchRenderEvent_ = function(e, t, n) {
		var r = this.getLayer();
		if (r.hasListener(e)) {
			var i = new xr(e, this.inversePixelTransform, n, t);
			r.dispatchEvent(i);
		}
	}, t.prototype.preRender = function(e, t) {
		this.dispatchRenderEvent_(fr.PRERENDER, e, t);
	}, t.prototype.postRender = function(e, t) {
		this.dispatchRenderEvent_(fr.POSTRENDER, e, t);
	}, t.prototype.getRenderTransform = function(e, t, n, r, i, a, o) {
		var s = i / 2, c = a / 2, l = r / t, u = -l, d = -e[0] + o, f = -e[1];
		return qn(this.tempTransform, s, c, l, u, -n, d, f);
	}, t.prototype.getDataAtPixel = function(e, t, n) {
		var r = G(this.inversePixelTransform, e.slice()), i = this.context, a = this.getLayer().getExtent();
		if (a && !Et(a, G(t.pixelToCoordinateTransform, e.slice()))) return null;
		var o;
		try {
			var s = Math.round(r[0]), c = Math.round(r[1]), l = document.createElement("canvas"), u = l.getContext("2d");
			l.width = 1, l.height = 1, u.clearRect(0, 0, 1, 1), u.drawImage(i.canvas, s, c, 1, 1, 0, 0, 1, 1), o = u.getImageData(0, 0, 1, 1).data;
		} catch (e) {
			return e.name === "SecurityError" ? /* @__PURE__ */ new Uint8Array() : o;
		}
		return o[3] === 0 ? null : o;
	}, t;
}(ao), co = function() {
	function e(e, t, n, r) {
		this.minX = e, this.maxX = t, this.minY = n, this.maxY = r;
	}
	return e.prototype.contains = function(e) {
		return this.containsXY(e[1], e[2]);
	}, e.prototype.containsTileRange = function(e) {
		return this.minX <= e.minX && e.maxX <= this.maxX && this.minY <= e.minY && e.maxY <= this.maxY;
	}, e.prototype.containsXY = function(e, t) {
		return this.minX <= e && e <= this.maxX && this.minY <= t && t <= this.maxY;
	}, e.prototype.equals = function(e) {
		return this.minX == e.minX && this.minY == e.minY && this.maxX == e.maxX && this.maxY == e.maxY;
	}, e.prototype.extend = function(e) {
		e.minX < this.minX && (this.minX = e.minX), e.maxX > this.maxX && (this.maxX = e.maxX), e.minY < this.minY && (this.minY = e.minY), e.maxY > this.maxY && (this.maxY = e.maxY);
	}, e.prototype.getHeight = function() {
		return this.maxY - this.minY + 1;
	}, e.prototype.getSize = function() {
		return [this.getWidth(), this.getHeight()];
	}, e.prototype.getWidth = function() {
		return this.maxX - this.minX + 1;
	}, e.prototype.intersects = function(e) {
		return this.minX <= e.maxX && this.maxX >= e.minX && this.minY <= e.maxY && this.maxY >= e.minY;
	}, e;
}();
function lo(e, t, n, r, i) {
	return i === void 0 ? new co(e, t, n, r) : (i.minX = e, i.maxX = t, i.minY = n, i.maxY = r, i);
}
//#endregion
//#region node_modules/ol/renderer/canvas/TileLayer.js
var uo = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), fo = function(e) {
	uo(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.extentChanged = !0, n.renderedExtent_ = null, n.renderedPixelRatio, n.renderedProjection = null, n.renderedRevision, n.renderedTiles = [], n.newTiles_ = !1, n.tmpExtent = At(), n.tmpTileRange_ = new co(0, 0, 0, 0), n;
	}
	return t.prototype.isDrawableTile = function(e) {
		var t = this.getLayer(), n = e.getState(), r = t.getUseInterimTilesOnError();
		return n == J.LOADED || n == J.EMPTY || n == J.ERROR && !r;
	}, t.prototype.getTile = function(e, t, n, r) {
		var i = r.pixelRatio, a = r.viewState.projection, o = this.getLayer(), s = o.getSource().getTile(e, t, n, i, a);
		return s.getState() == J.ERROR && (o.getUseInterimTilesOnError() ? o.getPreload() > 0 && (this.newTiles_ = !0) : s.setState(J.LOADED)), this.isDrawableTile(s) || (s = s.getInterimTile()), s;
	}, t.prototype.loadedTileCallback = function(t, n, r) {
		return this.isDrawableTile(r) ? e.prototype.loadedTileCallback.call(this, t, n, r) : !1;
	}, t.prototype.prepareFrame = function(e) {
		return !!this.getLayer().getSource();
	}, t.prototype.renderFrame = function(e, t) {
		var n = e.layerStatesArray[e.layerIndex], r = e.viewState, i = r.projection, a = r.resolution, o = r.center, s = r.rotation, c = e.pixelRatio, l = this.getLayer(), u = l.getSource(), d = u.getRevision(), p = u.getTileGridForProjection(i), m = p.getZForResolution(a, u.zDirection), h = p.getResolution(m), g = e.extent, _ = n.extent && An(n.extent, i);
		_ && (g = Jt(g, An(n.extent, i)));
		var v = u.getTilePixelRatio(c), y = Math.round(e.size[0] * v), b = Math.round(e.size[1] * v);
		if (s) {
			var x = Math.round(Math.sqrt(y * y + b * b));
			y = x, b = x;
		}
		var C = h * y / 2 / v, w = h * b / 2 / v, T = [
			o[0] - C,
			o[1] - w,
			o[0] + C,
			o[1] + w
		], E = p.getTileRangeForExtentAndZ(g, m), D = {};
		D[m] = {};
		var O = this.createLoadedTileFinder(u, i, D), k = this.tmpExtent, A = this.tmpTileRange_;
		this.newTiles_ = !1;
		for (var j = E.minX; j <= E.maxX; ++j) for (var M = E.minY; M <= E.maxY; ++M) {
			var N = this.getTile(m, j, M, e);
			if (this.isDrawableTile(N)) {
				var P = L(this);
				if (N.getState() == J.LOADED) {
					D[m][N.tileCoord.toString()] = N;
					var F = N.inTransition(P);
					!this.newTiles_ && (F || this.renderedTiles.indexOf(N) === -1) && (this.newTiles_ = !0);
				}
				if (N.getAlpha(P, e.time) === 1) continue;
			}
			var I = p.getTileCoordChildTileRange(N.tileCoord, A, k), R = !1;
			I && (R = O(m + 1, I)), R || p.forEachTileCoordParentTileRange(N.tileCoord, O, A, k);
		}
		var ee = h / a;
		qn(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / v, 1 / v, s, -y / 2, -b / 2);
		var te = Zn(this.pixelTransform);
		this.useContainer(t, te, n.opacity);
		var z = this.context, ne = z.canvas;
		Jn(this.inversePixelTransform, this.pixelTransform), qn(this.tempTransform, y / 2, b / 2, ee, ee, 0, -y / 2, -b / 2), ne.width != y || ne.height != b ? (ne.width = y, ne.height = b) : this.containerReused || z.clearRect(0, 0, y, b), _ && this.clipUnrotated(z, e, _), S(z, u.getContextOptions()), this.preRender(z, e), this.renderedTiles.length = 0;
		var re = Object.keys(D).map(Number);
		re.sort(f);
		var ie, ae, oe;
		n.opacity === 1 && (!this.containerReused || u.getOpaque(e.viewState.projection)) ? re = re.reverse() : (ie = [], ae = []);
		for (var se = re.length - 1; se >= 0; --se) {
			var ce = re[se], le = u.getTilePixelSize(ce, c, i), ue = p.getResolution(ce) / h, de = le[0] * ue * ee, fe = le[1] * ue * ee, pe = p.getTileCoordForCoordAndZ(Yt(T), ce), me = p.getTileCoordExtent(pe), he = G(this.tempTransform, [v * (me[0] - T[0]) / h, v * (T[3] - me[3]) / h]), ge = v * u.getGutterForProjection(i), _e = D[ce];
			for (var ve in _e) {
				var N = _e[ve], ye = N.tileCoord, be = pe[1] - ye[1], xe = Math.round(he[0] - (be - 1) * de), Se = pe[2] - ye[2], Ce = Math.round(he[1] - (Se - 1) * fe), j = Math.round(he[0] - be * de), M = Math.round(he[1] - Se * fe), we = xe - j, Te = Ce - M, Ee = m === ce, F = Ee && N.getAlpha(L(this), e.time) !== 1;
				if (!F) if (ie) {
					z.save(), oe = [
						j,
						M,
						j + we,
						M,
						j + we,
						M + Te,
						j,
						M + Te
					];
					for (var De = 0, Oe = ie.length; De < Oe; ++De) if (m !== ce && ce < ae[De]) {
						var Ae = ie[De];
						z.beginPath(), z.moveTo(oe[0], oe[1]), z.lineTo(oe[2], oe[3]), z.lineTo(oe[4], oe[5]), z.lineTo(oe[6], oe[7]), z.moveTo(Ae[6], Ae[7]), z.lineTo(Ae[4], Ae[5]), z.lineTo(Ae[2], Ae[3]), z.lineTo(Ae[0], Ae[1]), z.clip();
					}
					ie.push(oe), ae.push(ce);
				} else z.clearRect(j, M, we, Te);
				this.drawTileImage(N, e, j, M, we, Te, ge, Ee), ie && !F ? (z.restore(), this.renderedTiles.unshift(N)) : this.renderedTiles.push(N), this.updateUsedTiles(e.usedTiles, u, N);
			}
		}
		this.renderedRevision = d, this.renderedResolution = h, this.extentChanged = !this.renderedExtent_ || !Ft(this.renderedExtent_, T), this.renderedExtent_ = T, this.renderedPixelRatio = c, this.renderedProjection = i, this.manageTilePyramid(e, u, p, c, i, g, m, l.getPreload()), this.scheduleExpireCache(e, u), this.postRender(z, e), n.extent && z.restore(), te !== ne.style.transform && (ne.style.transform = te);
		var je = ke(n.opacity), Me = this.container;
		return je !== Me.style.opacity && (Me.style.opacity = je), this.container;
	}, t.prototype.drawTileImage = function(e, t, n, r, i, a, o, s) {
		var c = this.getTileImage(e);
		if (c) {
			var l = L(this), u = s ? e.getAlpha(l, t.time) : 1, d = u !== this.context.globalAlpha;
			d && (this.context.save(), this.context.globalAlpha = u), this.context.drawImage(c, o, o, c.width - 2 * o, c.height - 2 * o, n, r, i, a), d && this.context.restore(), u === 1 ? s && e.endTransition(l) : t.animate = !0;
		}
	}, t.prototype.getImage = function() {
		var e = this.context;
		return e ? e.canvas : null;
	}, t.prototype.getTileImage = function(e) {
		return e.getImage();
	}, t.prototype.scheduleExpireCache = function(e, t) {
		if (t.canExpireCache()) {
			var n = function(e, t, n) {
				var r = L(e);
				r in n.usedTiles && e.expireCache(n.viewState.projection, n.usedTiles[r]);
			}.bind(null, t);
			e.postRenderFunctions.push(n);
		}
	}, t.prototype.updateUsedTiles = function(e, t, n) {
		var r = L(t);
		r in e || (e[r] = {}), e[r][n.getKey()] = !0;
	}, t.prototype.manageTilePyramid = function(e, t, n, r, i, a, o, s, c) {
		var l = L(t);
		l in e.wantedTiles || (e.wantedTiles[l] = {});
		var u = e.wantedTiles[l], d = e.tileQueue, f = n.getMinZoom(), p = 0, m, h, g, _, v, y;
		for (y = f; y <= o; ++y) for (h = n.getTileRangeForExtentAndZ(a, y, h), g = n.getResolution(y), _ = h.minX; _ <= h.maxX; ++_) for (v = h.minY; v <= h.maxY; ++v) o - y <= s ? (++p, m = t.getTile(y, _, v, r, i), m.getState() == J.IDLE && (u[m.getKey()] = !0, d.isKeyQueued(m.getKey()) || d.enqueue([
			m,
			l,
			n.getTileCoordCenter(m.tileCoord),
			g
		])), c !== void 0 && c(m)) : t.useTile(y, _, v, i);
		t.updateCacheSize(p, i);
	}, t;
}(so);
fo.prototype.getLayer;
//#endregion
//#region node_modules/ol/layer/Tile.js
var po = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), mo = function(e) {
	po(t, e);
	function t(t) {
		return e.call(this, t) || this;
	}
	return t.prototype.createRenderer = function() {
		return new fo(this);
	}, t;
}(ro), ho = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), go = function(e) {
	ho(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, {
			element: document.createElement("div"),
			render: r.render,
			target: r.target
		}) || this, n.ulElement_ = document.createElement("ul"), n.collapsed_ = r.collapsed === void 0 || r.collapsed, n.userCollapsed_ = n.collapsed_, n.overrideCollapsible_ = r.collapsible !== void 0, n.collapsible_ = r.collapsible === void 0 || r.collapsible, n.collapsible_ || (n.collapsed_ = !1);
		var i = r.className === void 0 ? "ol-attribution" : r.className, a = r.tipLabel === void 0 ? "Attributions" : r.tipLabel, o = r.expandClassName === void 0 ? i + "-expand" : r.expandClassName, s = r.collapseLabel === void 0 ? "›" : r.collapseLabel, c = r.collapseClassName === void 0 ? i + "-collpase" : r.collapseClassName;
		typeof s == "string" ? (n.collapseLabel_ = document.createElement("span"), n.collapseLabel_.textContent = s, n.collapseLabel_.className = c) : n.collapseLabel_ = s;
		var l = r.label === void 0 ? "i" : r.label;
		typeof l == "string" ? (n.label_ = document.createElement("span"), n.label_.textContent = l, n.label_.className = o) : n.label_ = l;
		var u = n.collapsible_ && !n.collapsed_ ? n.collapseLabel_ : n.label_;
		n.toggleButton_ = document.createElement("button"), n.toggleButton_.setAttribute("type", "button"), n.toggleButton_.setAttribute("aria-expanded", String(!n.collapsed_)), n.toggleButton_.title = a, n.toggleButton_.appendChild(u), n.toggleButton_.addEventListener(O.CLICK, n.handleClick_.bind(n), !1);
		var d = i + " " + Se + " " + we + (n.collapsed_ && n.collapsible_ ? " " + Te : "") + (n.collapsible_ ? "" : " ol-uncollapsible"), f = n.element;
		return f.className = d, f.appendChild(n.toggleButton_), f.appendChild(n.ulElement_), n.renderedAttributions_ = [], n.renderedVisible_ = !0, n;
	}
	return t.prototype.collectSourceAttributions_ = function(e) {
		for (var t = {}, n = [], r = !0, i = e.layerStatesArray, a = 0, o = i.length; a < o; ++a) {
			var s = i[a];
			if (gr(s, e.viewState)) {
				var c = s.layer.getSource();
				if (c) {
					var l = c.getAttributions();
					if (l) {
						var u = l(e);
						if (u) if (r &&= c.getAttributionsCollapsible() !== !1, Array.isArray(u)) for (var d = 0, f = u.length; d < f; ++d) u[d] in t || (n.push(u[d]), t[u[d]] = !0);
						else u in t || (n.push(u), t[u] = !0);
					}
				}
			}
		}
		return this.overrideCollapsible_ || this.setCollapsible(r), n;
	}, t.prototype.updateElement_ = function(e) {
		if (!e) {
			this.renderedVisible_ &&= (this.element.style.display = "none", !1);
			return;
		}
		var t = this.collectSourceAttributions_(e), n = t.length > 0;
		if (this.renderedVisible_ != n && (this.element.style.display = n ? "" : "none", this.renderedVisible_ = n), !g(t, this.renderedAttributions_)) {
			ge(this.ulElement_);
			for (var r = 0, i = t.length; r < i; ++r) {
				var a = document.createElement("li");
				a.innerHTML = t[r], this.ulElement_.appendChild(a);
			}
			this.renderedAttributions_ = t;
		}
	}, t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.handleToggle_(), this.userCollapsed_ = this.collapsed_;
	}, t.prototype.handleToggle_ = function() {
		this.element.classList.toggle(Te), this.collapsed_ ? me(this.collapseLabel_, this.label_) : me(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_, this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_));
	}, t.prototype.getCollapsible = function() {
		return this.collapsible_;
	}, t.prototype.setCollapsible = function(e) {
		this.collapsible_ !== e && (this.collapsible_ = e, this.element.classList.toggle("ol-uncollapsible"), this.userCollapsed_ && this.handleToggle_());
	}, t.prototype.setCollapsed = function(e) {
		this.userCollapsed_ = e, !(!this.collapsible_ || this.collapsed_ === e) && this.handleToggle_();
	}, t.prototype.getCollapsed = function() {
		return this.collapsed_;
	}, t.prototype.render = function(e) {
		this.updateElement_(e.frameState);
	}, t;
}(ye), _o = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), vo = function(e) {
	_o(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, {
			element: document.createElement("div"),
			target: r.target
		}) || this;
		var i = r.className === void 0 ? "ol-zoom" : r.className, a = r.delta === void 0 ? 1 : r.delta, o = r.zoomInClassName === void 0 ? i + "-in" : r.zoomInClassName, s = r.zoomOutClassName === void 0 ? i + "-out" : r.zoomOutClassName, c = r.zoomInLabel === void 0 ? "+" : r.zoomInLabel, l = r.zoomOutLabel === void 0 ? "–" : r.zoomOutLabel, u = r.zoomInTipLabel === void 0 ? "Zoom in" : r.zoomInTipLabel, d = r.zoomOutTipLabel === void 0 ? "Zoom out" : r.zoomOutTipLabel, f = document.createElement("button");
		f.className = o, f.setAttribute("type", "button"), f.title = u, f.appendChild(typeof c == "string" ? document.createTextNode(c) : c), f.addEventListener(O.CLICK, n.handleClick_.bind(n, a), !1);
		var p = document.createElement("button");
		p.className = s, p.setAttribute("type", "button"), p.title = d, p.appendChild(typeof l == "string" ? document.createTextNode(l) : l), p.addEventListener(O.CLICK, n.handleClick_.bind(n, -a), !1);
		var m = i + " " + Se + " " + we, h = n.element;
		return h.className = m, h.appendChild(f), h.appendChild(p), n.duration_ = r.duration === void 0 ? 250 : r.duration, n;
	}
	return t.prototype.handleClick_ = function(e, t) {
		t.preventDefault(), this.zoomByDelta_(e);
	}, t.prototype.zoomByDelta_ = function(e) {
		var t = this.getMap().getView();
		if (t) {
			var n = t.getZoom();
			if (n !== void 0) {
				var r = t.getConstrainedZoom(n + e);
				this.duration_ > 0 ? (t.getAnimating() && t.cancelAnimations(), t.animate({
					zoom: r,
					duration: this.duration_,
					easing: ki
				})) : t.setZoom(r);
			}
		}
	}, t;
}(ye);
//#endregion
//#region node_modules/ol/control.js
function yo(e) {
	var t = e || {}, n = new ti();
	return (t.zoom === void 0 || t.zoom) && n.push(new vo(t.zoomOptions)), (t.rotate === void 0 || t.rotate) && n.push(new Ja(t.rotateOptions)), (t.attribution === void 0 || t.attribution) && n.push(new go(t.attributionOptions)), n;
}
//#endregion
//#region node_modules/ol/interaction/Property.js
var bo = { ACTIVE: "active" }, xo = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), So = function(e) {
	xo(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.on, n.once, n.un, t && t.handleEvent && (n.handleEvent = t.handleEvent), n.map_ = null, n.setActive(!0), n;
	}
	return t.prototype.getActive = function() {
		return this.get(bo.ACTIVE);
	}, t.prototype.getMap = function() {
		return this.map_;
	}, t.prototype.handleEvent = function(e) {
		return !0;
	}, t.prototype.setActive = function(e) {
		this.set(bo.ACTIVE, e);
	}, t.prototype.setMap = function(e) {
		this.map_ = e;
	}, t;
}(z);
function Co(e, t, n) {
	var r = e.getCenterInternal();
	if (r) {
		var i = [r[0] + t[0], r[1] + t[1]];
		e.animateInternal({
			duration: n === void 0 ? 250 : n,
			easing: ji,
			center: e.getConstrainedCenter(i)
		});
	}
}
function wo(e, t, n, r) {
	var i = e.getZoom();
	if (i !== void 0) {
		var a = e.getConstrainedZoom(i + t), o = e.getResolutionForZoom(a);
		e.getAnimating() && e.cancelAnimations(), e.animate({
			resolution: o,
			anchor: n,
			duration: r === void 0 ? 250 : r,
			easing: ki
		});
	}
}
//#endregion
//#region node_modules/ol/interaction/DoubleClickZoom.js
var To = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Eo = function(e) {
	To(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		return n.delta_ = r.delta ? r.delta : 1, n.duration_ = r.duration === void 0 ? 250 : r.duration, n;
	}
	return t.prototype.handleEvent = function(e) {
		var t = !1;
		if (e.type == q.DBLCLICK) {
			var n = e.originalEvent, r = e.map, i = e.coordinate, a = n.shiftKey ? -this.delta_ : this.delta_;
			wo(r.getView(), a, i, this.duration_), n.preventDefault(), t = !0;
		}
		return !t;
	}, t;
}(So), Do = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Oo = function(e) {
	Do(t, e);
	function t(t) {
		var n = this, r = t || {};
		return n = e.call(this, r) || this, r.handleDownEvent && (n.handleDownEvent = r.handleDownEvent), r.handleDragEvent && (n.handleDragEvent = r.handleDragEvent), r.handleMoveEvent && (n.handleMoveEvent = r.handleMoveEvent), r.handleUpEvent && (n.handleUpEvent = r.handleUpEvent), r.stopDown && (n.stopDown = r.stopDown), n.handlingDownUpSequence = !1, n.trackedPointers_ = {}, n.targetPointers = [], n;
	}
	return t.prototype.getPointerCount = function() {
		return this.targetPointers.length;
	}, t.prototype.handleDownEvent = function(e) {
		return !1;
	}, t.prototype.handleDragEvent = function(e) {}, t.prototype.handleEvent = function(e) {
		if (!e.originalEvent) return !0;
		var t = !1;
		if (this.updateTrackedPointers_(e), this.handlingDownUpSequence) {
			if (e.type == q.POINTERDRAG) this.handleDragEvent(e), e.originalEvent.preventDefault();
			else if (e.type == q.POINTERUP) {
				var n = this.handleUpEvent(e);
				this.handlingDownUpSequence = n && this.targetPointers.length > 0;
			}
		} else if (e.type == q.POINTERDOWN) {
			var r = this.handleDownEvent(e);
			this.handlingDownUpSequence = r, t = this.stopDown(r);
		} else e.type == q.POINTERMOVE && this.handleMoveEvent(e);
		return !t;
	}, t.prototype.handleMoveEvent = function(e) {}, t.prototype.handleUpEvent = function(e) {
		return !1;
	}, t.prototype.stopDown = function(e) {
		return e;
	}, t.prototype.updateTrackedPointers_ = function(e) {
		if (Ao(e)) {
			var t = e.originalEvent, n = t.pointerId.toString();
			e.type == q.POINTERUP ? delete this.trackedPointers_[n] : (e.type == q.POINTERDOWN || n in this.trackedPointers_) && (this.trackedPointers_[n] = t), this.targetPointers = w(this.trackedPointers_);
		}
	}, t;
}(So);
function ko(e) {
	for (var t = e.length, n = 0, r = 0, i = 0; i < t; i++) n += e[i].clientX, r += e[i].clientY;
	return [n / t, r / t];
}
function Ao(e) {
	var t = e.type;
	return t === q.POINTERDOWN || t === q.POINTERDRAG || t === q.POINTERUP;
}
//#endregion
//#region node_modules/ol/events/condition.js
function jo(e) {
	var t = arguments;
	return function(e) {
		for (var n = !0, r = 0, i = t.length; r < i && (n &&= t[r](e), n); ++r);
		return n;
	};
}
var Mo = function(e) {
	var t = e.originalEvent;
	return t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, No = function(e) {
	return e.target.getTargetElement().contains(document.activeElement);
}, Po = function(e) {
	return !e.map.getTargetElement().hasAttribute("tabindex") || No(e);
}, Fo = v, Io = function(e) {
	var t = e.originalEvent;
	return t.button == 0 && !(ae && oe && t.ctrlKey);
}, Lo = function(e) {
	var t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && !t.shiftKey;
}, Ro = function(e) {
	var t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, zo = function(e) {
	var t = e.originalEvent.target.tagName;
	return t !== "INPUT" && t !== "SELECT" && t !== "TEXTAREA";
}, Bo = function(e) {
	var t = e.originalEvent;
	return H(t !== void 0, 56), t.pointerType == "mouse";
}, Vo = function(e) {
	var t = e.originalEvent;
	return H(t !== void 0, 56), t.isPrimary && t.button === 0;
}, Ho = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Uo = function(e) {
	Ho(t, e);
	function t(t) {
		var n = e.call(this, { stopDown: y }) || this, r = t || {};
		n.kinetic_ = r.kinetic, n.lastCentroid = null, n.lastPointersCount_, n.panning_ = !1;
		var i = r.condition ? r.condition : jo(Lo, Vo);
		return n.condition_ = r.onFocusOnly ? jo(Po, i) : i, n.noKinetic_ = !1, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		this.panning_ || (this.panning_ = !0, this.getMap().getView().beginInteraction());
		var t = this.targetPointers, n = ko(t);
		if (t.length == this.lastPointersCount_) {
			if (this.kinetic_ && this.kinetic_.update(n[0], n[1]), this.lastCentroid) {
				var r = [this.lastCentroid[0] - n[0], n[1] - this.lastCentroid[1]], i = e.map.getView();
				ln(r, i.getResolution()), cn(r, i.getRotation()), i.adjustCenterInternal(r);
			}
		} else this.kinetic_ && this.kinetic_.begin();
		this.lastCentroid = n, this.lastPointersCount_ = t.length, e.originalEvent.preventDefault();
	}, t.prototype.handleUpEvent = function(e) {
		var t = e.map, n = t.getView();
		if (this.targetPointers.length === 0) {
			if (!this.noKinetic_ && this.kinetic_ && this.kinetic_.end()) {
				var r = this.kinetic_.getDistance(), i = this.kinetic_.getAngle(), a = n.getCenterInternal(), o = t.getPixelFromCoordinateInternal(a), s = t.getCoordinateFromPixelInternal([o[0] - r * Math.cos(i), o[1] - r * Math.sin(i)]);
				n.animateInternal({
					center: n.getConstrainedCenter(s),
					duration: 500,
					easing: ki
				});
			}
			return this.panning_ && (this.panning_ = !1, n.endInteraction()), !1;
		} else return this.kinetic_ && this.kinetic_.begin(), this.lastCentroid = null, !0;
	}, t.prototype.handleDownEvent = function(e) {
		if (this.targetPointers.length > 0 && this.condition_(e)) {
			var t = e.map.getView();
			return this.lastCentroid = null, t.getAnimating() && t.cancelAnimations(), this.kinetic_ && this.kinetic_.begin(), this.noKinetic_ = this.targetPointers.length > 1, !0;
		} else return !1;
	}, t;
}(Oo), Wo = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Go = function(e) {
	Wo(t, e);
	function t(t) {
		var n = this, r = t || {};
		return n = e.call(this, { stopDown: y }) || this, n.condition_ = r.condition ? r.condition : Mo, n.lastAngle_ = void 0, n.duration_ = r.duration === void 0 ? 250 : r.duration, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		if (Bo(e)) {
			var t = e.map, n = t.getView();
			if (n.getConstraints().rotation !== wi) {
				var r = t.getSize(), i = e.pixel, a = Math.atan2(r[1] / 2 - i[1], i[0] - r[0] / 2);
				if (this.lastAngle_ !== void 0) {
					var o = a - this.lastAngle_;
					n.adjustRotationInternal(-o);
				}
				this.lastAngle_ = a;
			}
		}
	}, t.prototype.handleUpEvent = function(e) {
		return Bo(e) ? (e.map.getView().endInteraction(this.duration_), !1) : !0;
	}, t.prototype.handleDownEvent = function(e) {
		return Bo(e) && Io(e) && this.condition_(e) ? (e.map.getView().beginInteraction(), this.lastAngle_ = void 0, !0) : !1;
	}, t;
}(Oo), Ko = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), qo = function(e) {
	Ko(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.geometry_ = null, n.element_ = document.createElement("div"), n.element_.style.position = "absolute", n.element_.style.pointerEvents = "auto", n.element_.className = "ol-box " + t, n.map_ = null, n.startPixel_ = null, n.endPixel_ = null, n;
	}
	return t.prototype.disposeInternal = function() {
		this.setMap(null);
	}, t.prototype.render_ = function() {
		var e = this.startPixel_, t = this.endPixel_, n = "px", r = this.element_.style;
		r.left = Math.min(e[0], t[0]) + n, r.top = Math.min(e[1], t[1]) + n, r.width = Math.abs(t[0] - e[0]) + n, r.height = Math.abs(t[1] - e[1]) + n;
	}, t.prototype.setMap = function(e) {
		if (this.map_) {
			this.map_.getOverlayContainer().removeChild(this.element_);
			var t = this.element_.style;
			t.left = "inherit", t.top = "inherit", t.width = "inherit", t.height = "inherit";
		}
		this.map_ = e, this.map_ && this.map_.getOverlayContainer().appendChild(this.element_);
	}, t.prototype.setPixels = function(e, t) {
		this.startPixel_ = e, this.endPixel_ = t, this.createOrUpdateGeometry(), this.render_();
	}, t.prototype.createOrUpdateGeometry = function() {
		var e = this.startPixel_, t = this.endPixel_, n = [
			e,
			[e[0], t[1]],
			t,
			[t[0], e[1]]
		].map(this.map_.getCoordinateFromPixelInternal, this.map_);
		n[4] = n[0].slice(), this.geometry_ ? this.geometry_.setCoordinates([n]) : this.geometry_ = new Ta([n]);
	}, t.prototype.getGeometry = function() {
		return this.geometry_;
	}, t;
}(d), Jo = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Yo = {
	BOXSTART: "boxstart",
	BOXDRAG: "boxdrag",
	BOXEND: "boxend",
	BOXCANCEL: "boxcancel"
}, Xo = function(e) {
	Jo(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.coordinate = n, i.mapBrowserEvent = r, i;
	}
	return t;
}(l), Zo = function(e) {
	Jo(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = t || {};
		return n.box_ = new qo(r.className || "ol-dragbox"), n.minArea_ = r.minArea === void 0 ? 64 : r.minArea, r.onBoxEnd && (n.onBoxEnd = r.onBoxEnd), n.startPixel_ = null, n.condition_ = r.condition ? r.condition : Io, n.boxEndCondition_ = r.boxEndCondition ? r.boxEndCondition : n.defaultBoxEndCondition, n;
	}
	return t.prototype.defaultBoxEndCondition = function(e, t, n) {
		var r = n[0] - t[0], i = n[1] - t[1];
		return r * r + i * i >= this.minArea_;
	}, t.prototype.getGeometry = function() {
		return this.box_.getGeometry();
	}, t.prototype.handleDragEvent = function(e) {
		this.box_.setPixels(this.startPixel_, e.pixel), this.dispatchEvent(new Xo(Yo.BOXDRAG, e.coordinate, e));
	}, t.prototype.handleUpEvent = function(e) {
		this.box_.setMap(null);
		var t = this.boxEndCondition_(e, this.startPixel_, e.pixel);
		return t && this.onBoxEnd(e), this.dispatchEvent(new Xo(t ? Yo.BOXEND : Yo.BOXCANCEL, e.coordinate, e)), !1;
	}, t.prototype.handleDownEvent = function(e) {
		return this.condition_(e) ? (this.startPixel_ = e.pixel, this.box_.setMap(e.map), this.box_.setPixels(this.startPixel_, this.startPixel_), this.dispatchEvent(new Xo(Yo.BOXSTART, e.coordinate, e)), !0) : !1;
	}, t.prototype.onBoxEnd = function(e) {}, t;
}(Oo), Qo = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), $o = function(e) {
	Qo(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r.condition ? r.condition : Ro;
		return n = e.call(this, {
			condition: i,
			className: r.className || "ol-dragzoom",
			minArea: r.minArea
		}) || this, n.duration_ = r.duration === void 0 ? 200 : r.duration, n.out_ = r.out !== void 0 && r.out, n;
	}
	return t.prototype.onBoxEnd = function(e) {
		var t = this.getMap().getView(), n = this.getGeometry();
		if (this.out_) {
			var r = t.rotatedExtentForGeometry(n), i = t.getResolutionForExtentInternal(r), a = t.getResolution() / i;
			n = n.clone(), n.scale(a * a);
		}
		t.fitInternal(n, {
			duration: this.duration_,
			easing: ki
		});
	}, t;
}(Zo), es = {
	LEFT: 37,
	UP: 38,
	RIGHT: 39,
	DOWN: 40
}, ts = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ns = function(e) {
	ts(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		return n.defaultCondition_ = function(e) {
			return Lo(e) && zo(e);
		}, n.condition_ = r.condition === void 0 ? n.defaultCondition_ : r.condition, n.duration_ = r.duration === void 0 ? 100 : r.duration, n.pixelDelta_ = r.pixelDelta === void 0 ? 128 : r.pixelDelta, n;
	}
	return t.prototype.handleEvent = function(e) {
		var t = !1;
		if (e.type == O.KEYDOWN) {
			var n = e.originalEvent, r = n.keyCode;
			if (this.condition_(e) && (r == es.DOWN || r == es.LEFT || r == es.RIGHT || r == es.UP)) {
				var i = e.map.getView(), a = i.getResolution() * this.pixelDelta_, o = 0, s = 0;
				r == es.DOWN ? s = -a : r == es.LEFT ? o = -a : r == es.RIGHT ? o = a : s = a;
				var c = [o, s];
				cn(c, i.getRotation()), Co(i, c, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}, t;
}(So), rs = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), is = function(e) {
	rs(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		return n.condition_ = r.condition ? r.condition : zo, n.delta_ = r.delta ? r.delta : 1, n.duration_ = r.duration === void 0 ? 100 : r.duration, n;
	}
	return t.prototype.handleEvent = function(e) {
		var t = !1;
		if (e.type == O.KEYDOWN || e.type == O.KEYPRESS) {
			var n = e.originalEvent, r = n.charCode;
			if (this.condition_(e) && (r == 43 || r == 45)) {
				var i = e.map, a = r == 43 ? this.delta_ : -this.delta_;
				wo(i.getView(), a, void 0, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}, t;
}(So), as = function() {
	function e(e, t, n) {
		this.decay_ = e, this.minVelocity_ = t, this.delay_ = n, this.points_ = [], this.angle_ = 0, this.initialVelocity_ = 0;
	}
	return e.prototype.begin = function() {
		this.points_.length = 0, this.angle_ = 0, this.initialVelocity_ = 0;
	}, e.prototype.update = function(e, t) {
		this.points_.push(e, t, Date.now());
	}, e.prototype.end = function() {
		if (this.points_.length < 6) return !1;
		var e = Date.now() - this.delay_, t = this.points_.length - 3;
		if (this.points_[t + 2] < e) return !1;
		for (var n = t - 3; n > 0 && this.points_[n + 2] > e;) n -= 3;
		var r = this.points_[t + 2] - this.points_[n + 2];
		if (r < 1e3 / 60) return !1;
		var i = this.points_[t] - this.points_[n], a = this.points_[t + 1] - this.points_[n + 1];
		return this.angle_ = Math.atan2(a, i), this.initialVelocity_ = Math.sqrt(i * i + a * a) / r, this.initialVelocity_ > this.minVelocity_;
	}, e.prototype.getDistance = function() {
		return (this.minVelocity_ - this.initialVelocity_) / this.decay_;
	}, e.prototype.getAngle = function() {
		return this.angle_;
	}, e;
}(), os = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ss = {
	TRACKPAD: "trackpad",
	WHEEL: "wheel"
}, cs = function(e) {
	os(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, r) || this, n.totalDelta_ = 0, n.lastDelta_ = 0, n.maxDelta_ = r.maxDelta === void 0 ? 1 : r.maxDelta, n.duration_ = r.duration === void 0 ? 250 : r.duration, n.timeout_ = r.timeout === void 0 ? 80 : r.timeout, n.useAnchor_ = r.useAnchor === void 0 || r.useAnchor, n.constrainResolution_ = r.constrainResolution !== void 0 && r.constrainResolution;
		var i = r.condition ? r.condition : Fo;
		return n.condition_ = r.onFocusOnly ? jo(Po, i) : i, n.lastAnchor_ = null, n.startTime_ = void 0, n.timeoutId_, n.mode_ = void 0, n.trackpadEventGap_ = 400, n.trackpadTimeoutId_, n.deltaPerZoom_ = 300, n;
	}
	return t.prototype.endInteraction_ = function() {
		this.trackpadTimeoutId_ = void 0, this.getMap().getView().endInteraction(void 0, this.lastDelta_ ? this.lastDelta_ > 0 ? 1 : -1 : 0, this.lastAnchor_);
	}, t.prototype.handleEvent = function(e) {
		if (!this.condition_(e) || e.type !== O.WHEEL) return !0;
		var t = e.map, n = e.originalEvent;
		n.preventDefault(), this.useAnchor_ && (this.lastAnchor_ = e.coordinate);
		var r;
		if (e.type == O.WHEEL && (r = n.deltaY, ie && n.deltaMode === WheelEvent.DOM_DELTA_PIXEL && (r /= se), n.deltaMode === WheelEvent.DOM_DELTA_LINE && (r *= 40)), r === 0) return !1;
		this.lastDelta_ = r;
		var i = Date.now();
		this.startTime_ === void 0 && (this.startTime_ = i), (!this.mode_ || i - this.startTime_ > this.trackpadEventGap_) && (this.mode_ = Math.abs(r) < 4 ? ss.TRACKPAD : ss.WHEEL);
		var a = t.getView();
		if (this.mode_ === ss.TRACKPAD && !(a.getConstrainResolution() || this.constrainResolution_)) return this.trackpadTimeoutId_ ? clearTimeout(this.trackpadTimeoutId_) : (a.getAnimating() && a.cancelAnimations(), a.beginInteraction()), this.trackpadTimeoutId_ = setTimeout(this.endInteraction_.bind(this), this.timeout_), a.adjustZoom(-r / this.deltaPerZoom_, this.lastAnchor_), this.startTime_ = i, !1;
		this.totalDelta_ += r;
		var o = Math.max(this.timeout_ - (i - this.startTime_), 0);
		return clearTimeout(this.timeoutId_), this.timeoutId_ = setTimeout(this.handleWheelZoom_.bind(this, t), o), !1;
	}, t.prototype.handleWheelZoom_ = function(e) {
		var t = e.getView();
		t.getAnimating() && t.cancelAnimations();
		var n = -V(this.totalDelta_, -this.maxDelta_ * this.deltaPerZoom_, this.maxDelta_ * this.deltaPerZoom_) / this.deltaPerZoom_;
		(t.getConstrainResolution() || this.constrainResolution_) && (n = n ? n > 0 ? 1 : -1 : 0), wo(t, n, this.lastAnchor_, this.duration_), this.mode_ = void 0, this.totalDelta_ = 0, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_ = void 0;
	}, t.prototype.setMouseAnchor = function(e) {
		this.useAnchor_ = e, e || (this.lastAnchor_ = null);
	}, t;
}(So), ls = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), us = function(e) {
	ls(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r;
		return i.stopDown ||= y, n = e.call(this, i) || this, n.anchor_ = null, n.lastAngle_ = void 0, n.rotating_ = !1, n.rotationDelta_ = 0, n.threshold_ = r.threshold === void 0 ? .3 : r.threshold, n.duration_ = r.duration === void 0 ? 250 : r.duration, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		var t = 0, n = this.targetPointers[0], r = this.targetPointers[1], i = Math.atan2(r.clientY - n.clientY, r.clientX - n.clientX);
		if (this.lastAngle_ !== void 0) {
			var a = i - this.lastAngle_;
			this.rotationDelta_ += a, !this.rotating_ && Math.abs(this.rotationDelta_) > this.threshold_ && (this.rotating_ = !0), t = a;
		}
		this.lastAngle_ = i;
		var o = e.map, s = o.getView();
		if (s.getConstraints().rotation !== wi) {
			var c = o.getViewport().getBoundingClientRect(), l = ko(this.targetPointers);
			l[0] -= c.left, l[1] -= c.top, this.anchor_ = o.getCoordinateFromPixelInternal(l), this.rotating_ && (o.render(), s.adjustRotationInternal(t, this.anchor_));
		}
	}, t.prototype.handleUpEvent = function(e) {
		return this.targetPointers.length < 2 ? (e.map.getView().endInteraction(this.duration_), !1) : !0;
	}, t.prototype.handleDownEvent = function(e) {
		if (this.targetPointers.length >= 2) {
			var t = e.map;
			return this.anchor_ = null, this.lastAngle_ = void 0, this.rotating_ = !1, this.rotationDelta_ = 0, this.handlingDownUpSequence || t.getView().beginInteraction(), !0;
		} else return !1;
	}, t;
}(Oo), ds = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), fs = function(e) {
	ds(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r;
		return i.stopDown ||= y, n = e.call(this, i) || this, n.anchor_ = null, n.duration_ = r.duration === void 0 ? 400 : r.duration, n.lastDistance_ = void 0, n.lastScaleDelta_ = 1, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		var t = 1, n = this.targetPointers[0], r = this.targetPointers[1], i = n.clientX - r.clientX, a = n.clientY - r.clientY, o = Math.sqrt(i * i + a * a);
		this.lastDistance_ !== void 0 && (t = this.lastDistance_ / o), this.lastDistance_ = o;
		var s = e.map, c = s.getView();
		t != 1 && (this.lastScaleDelta_ = t);
		var l = s.getViewport().getBoundingClientRect(), u = ko(this.targetPointers);
		u[0] -= l.left, u[1] -= l.top, this.anchor_ = s.getCoordinateFromPixelInternal(u), s.render(), c.adjustResolutionInternal(t, this.anchor_);
	}, t.prototype.handleUpEvent = function(e) {
		if (this.targetPointers.length < 2) {
			var t = e.map.getView(), n = this.lastScaleDelta_ > 1 ? 1 : -1;
			return t.endInteraction(this.duration_, n), !1;
		} else return !0;
	}, t.prototype.handleDownEvent = function(e) {
		if (this.targetPointers.length >= 2) {
			var t = e.map;
			return this.anchor_ = null, this.lastDistance_ = void 0, this.lastScaleDelta_ = 1, this.handlingDownUpSequence || t.getView().beginInteraction(), !0;
		} else return !1;
	}, t;
}(Oo);
//#endregion
//#region node_modules/ol/geom/flat/length.js
function ps(e, t, n, r) {
	for (var i = e[t], a = e[t + 1], o = 0, s = t + r; s < n; s += r) {
		var c = e[s], l = e[s + 1];
		o += Math.sqrt((c - i) * (c - i) + (l - a) * (l - a)), i = c, a = l;
	}
	return o;
}
//#endregion
//#region node_modules/rbush/rbush.min.js
var ms = /* @__PURE__ */ o(((e, t) => {
	(function(n, r) {
		typeof e == "object" && t !== void 0 ? t.exports = r() : typeof define == "function" && define.amd ? define(r) : (n ||= self).RBush = r();
	})(e, function() {
		function e(e, r, i, a, o) {
			(function e(n, r, i, a, o) {
				for (; a > i;) {
					if (a - i > 600) {
						var s = a - i + 1, c = r - i + 1, l = Math.log(s), u = .5 * Math.exp(2 * l / 3), d = .5 * Math.sqrt(l * u * (s - u) / s) * (c - s / 2 < 0 ? -1 : 1);
						e(n, r, Math.max(i, Math.floor(r - c * u / s + d)), Math.min(a, Math.floor(r + (s - c) * u / s + d)), o);
					}
					var f = n[r], p = i, m = a;
					for (t(n, i, r), o(n[a], f) > 0 && t(n, i, a); p < m;) {
						for (t(n, p, m), p++, m--; o(n[p], f) < 0;) p++;
						for (; o(n[m], f) > 0;) m--;
					}
					o(n[i], f) === 0 ? t(n, i, m) : t(n, ++m, a), m <= r && (i = m + 1), r <= m && (a = m - 1);
				}
			})(e, r, i || 0, a || e.length - 1, o || n);
		}
		function t(e, t, n) {
			var r = e[t];
			e[t] = e[n], e[n] = r;
		}
		function n(e, t) {
			return e < t ? -1 : +(e > t);
		}
		var r = function(e) {
			e === void 0 && (e = 9), this._maxEntries = Math.max(4, e), this._minEntries = Math.max(2, Math.ceil(.4 * this._maxEntries)), this.clear();
		};
		function i(e, t, n) {
			if (!n) return t.indexOf(e);
			for (var r = 0; r < t.length; r++) if (n(e, t[r])) return r;
			return -1;
		}
		function a(e, t) {
			o(e, 0, e.children.length, t, e);
		}
		function o(e, t, n, r, i) {
			i ||= m(null), i.minX = Infinity, i.minY = Infinity, i.maxX = -Infinity, i.maxY = -Infinity;
			for (var a = t; a < n; a++) {
				var o = e.children[a];
				s(i, e.leaf ? r(o) : o);
			}
			return i;
		}
		function s(e, t) {
			return e.minX = Math.min(e.minX, t.minX), e.minY = Math.min(e.minY, t.minY), e.maxX = Math.max(e.maxX, t.maxX), e.maxY = Math.max(e.maxY, t.maxY), e;
		}
		function c(e, t) {
			return e.minX - t.minX;
		}
		function l(e, t) {
			return e.minY - t.minY;
		}
		function u(e) {
			return (e.maxX - e.minX) * (e.maxY - e.minY);
		}
		function d(e) {
			return e.maxX - e.minX + (e.maxY - e.minY);
		}
		function f(e, t) {
			return e.minX <= t.minX && e.minY <= t.minY && t.maxX <= e.maxX && t.maxY <= e.maxY;
		}
		function p(e, t) {
			return t.minX <= e.maxX && t.minY <= e.maxY && t.maxX >= e.minX && t.maxY >= e.minY;
		}
		function m(e) {
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
		function h(t, n, r, i, a) {
			for (var o = [n, r]; o.length;) if (!((r = o.pop()) - (n = o.pop()) <= i)) {
				var s = n + Math.ceil((r - n) / i / 2) * i;
				e(t, s, n, r, a), o.push(n, s, s, r);
			}
		}
		return r.prototype.all = function() {
			return this._all(this.data, []);
		}, r.prototype.search = function(e) {
			var t = this.data, n = [];
			if (!p(e, t)) return n;
			for (var r = this.toBBox, i = []; t;) {
				for (var a = 0; a < t.children.length; a++) {
					var o = t.children[a], s = t.leaf ? r(o) : o;
					p(e, s) && (t.leaf ? n.push(o) : f(e, s) ? this._all(o, n) : i.push(o));
				}
				t = i.pop();
			}
			return n;
		}, r.prototype.collides = function(e) {
			var t = this.data;
			if (!p(e, t)) return !1;
			for (var n = []; t;) {
				for (var r = 0; r < t.children.length; r++) {
					var i = t.children[r], a = t.leaf ? this.toBBox(i) : i;
					if (p(e, a)) {
						if (t.leaf || f(e, a)) return !0;
						n.push(i);
					}
				}
				t = n.pop();
			}
			return !1;
		}, r.prototype.load = function(e) {
			if (!e || !e.length) return this;
			if (e.length < this._minEntries) {
				for (var t = 0; t < e.length; t++) this.insert(e[t]);
				return this;
			}
			var n = this._build(e.slice(), 0, e.length - 1, 0);
			if (this.data.children.length) if (this.data.height === n.height) this._splitRoot(this.data, n);
			else {
				if (this.data.height < n.height) {
					var r = this.data;
					this.data = n, n = r;
				}
				this._insert(n, this.data.height - n.height - 1, !0);
			}
			else this.data = n;
			return this;
		}, r.prototype.insert = function(e) {
			return e && this._insert(e, this.data.height - 1), this;
		}, r.prototype.clear = function() {
			return this.data = m([]), this;
		}, r.prototype.remove = function(e, t) {
			if (!e) return this;
			for (var n, r, a, o = this.data, s = this.toBBox(e), c = [], l = []; o || c.length;) {
				if (o || (o = c.pop(), r = c[c.length - 1], n = l.pop(), a = !0), o.leaf) {
					var u = i(e, o.children, t);
					if (u !== -1) return o.children.splice(u, 1), c.push(o), this._condense(c), this;
				}
				a || o.leaf || !f(o, s) ? r ? (n++, o = r.children[n], a = !1) : o = null : (c.push(o), l.push(n), n = 0, r = o, o = o.children[0]);
			}
			return this;
		}, r.prototype.toBBox = function(e) {
			return e;
		}, r.prototype.compareMinX = function(e, t) {
			return e.minX - t.minX;
		}, r.prototype.compareMinY = function(e, t) {
			return e.minY - t.minY;
		}, r.prototype.toJSON = function() {
			return this.data;
		}, r.prototype.fromJSON = function(e) {
			return this.data = e, this;
		}, r.prototype._all = function(e, t) {
			for (var n = []; e;) e.leaf ? t.push.apply(t, e.children) : n.push.apply(n, e.children), e = n.pop();
			return t;
		}, r.prototype._build = function(e, t, n, r) {
			var i, o = n - t + 1, s = this._maxEntries;
			if (o <= s) return a(i = m(e.slice(t, n + 1)), this.toBBox), i;
			r || (r = Math.ceil(Math.log(o) / Math.log(s)), s = Math.ceil(o / s ** (r - 1))), (i = m([])).leaf = !1, i.height = r;
			var c = Math.ceil(o / s), l = c * Math.ceil(Math.sqrt(s));
			h(e, t, n, l, this.compareMinX);
			for (var u = t; u <= n; u += l) {
				var d = Math.min(u + l - 1, n);
				h(e, u, d, c, this.compareMinY);
				for (var f = u; f <= d; f += c) {
					var p = Math.min(f + c - 1, d);
					i.children.push(this._build(e, f, p, r - 1));
				}
			}
			return a(i, this.toBBox), i;
		}, r.prototype._chooseSubtree = function(e, t, n, r) {
			for (; r.push(t), !t.leaf && r.length - 1 !== n;) {
				for (var i = Infinity, a = Infinity, o = void 0, s = 0; s < t.children.length; s++) {
					var c = t.children[s], l = u(c), d = (f = e, p = c, (Math.max(p.maxX, f.maxX) - Math.min(p.minX, f.minX)) * (Math.max(p.maxY, f.maxY) - Math.min(p.minY, f.minY)) - l);
					d < a ? (a = d, i = l < i ? l : i, o = c) : d === a && l < i && (i = l, o = c);
				}
				t = o || t.children[0];
			}
			var f, p;
			return t;
		}, r.prototype._insert = function(e, t, n) {
			var r = n ? e : this.toBBox(e), i = [], a = this._chooseSubtree(r, this.data, t, i);
			for (a.children.push(e), s(a, r); t >= 0 && i[t].children.length > this._maxEntries;) this._split(i, t), t--;
			this._adjustParentBBoxes(r, i, t);
		}, r.prototype._split = function(e, t) {
			var n = e[t], r = n.children.length, i = this._minEntries;
			this._chooseSplitAxis(n, i, r);
			var o = this._chooseSplitIndex(n, i, r), s = m(n.children.splice(o, n.children.length - o));
			s.height = n.height, s.leaf = n.leaf, a(n, this.toBBox), a(s, this.toBBox), t ? e[t - 1].children.push(s) : this._splitRoot(n, s);
		}, r.prototype._splitRoot = function(e, t) {
			this.data = m([e, t]), this.data.height = e.height + 1, this.data.leaf = !1, a(this.data, this.toBBox);
		}, r.prototype._chooseSplitIndex = function(e, t, n) {
			for (var r, i, a, s, c, l, d, f = Infinity, p = Infinity, m = t; m <= n - t; m++) {
				var h = o(e, 0, m, this.toBBox), g = o(e, m, n, this.toBBox), _ = (i = h, a = g, s = void 0, c = void 0, l = void 0, d = void 0, s = Math.max(i.minX, a.minX), c = Math.max(i.minY, a.minY), l = Math.min(i.maxX, a.maxX), d = Math.min(i.maxY, a.maxY), Math.max(0, l - s) * Math.max(0, d - c)), v = u(h) + u(g);
				_ < f ? (f = _, r = m, p = v < p ? v : p) : _ === f && v < p && (p = v, r = m);
			}
			return r || n - t;
		}, r.prototype._chooseSplitAxis = function(e, t, n) {
			var r = e.leaf ? this.compareMinX : c, i = e.leaf ? this.compareMinY : l;
			this._allDistMargin(e, t, n, r) < this._allDistMargin(e, t, n, i) && e.children.sort(r);
		}, r.prototype._allDistMargin = function(e, t, n, r) {
			e.children.sort(r);
			for (var i = this.toBBox, a = o(e, 0, t, i), c = o(e, n - t, n, i), l = d(a) + d(c), u = t; u < n - t; u++) {
				var f = e.children[u];
				s(a, e.leaf ? i(f) : f), l += d(a);
			}
			for (var p = n - t - 1; p >= t; p--) {
				var m = e.children[p];
				s(c, e.leaf ? i(m) : m), l += d(c);
			}
			return l;
		}, r.prototype._adjustParentBBoxes = function(e, t, n) {
			for (var r = n; r >= 0; r--) s(t[r], e);
		}, r.prototype._condense = function(e) {
			for (var t = e.length - 1, n = void 0; t >= 0; t--) e[t].children.length === 0 ? t > 0 ? (n = e[t - 1].children).splice(n.indexOf(e[t]), 1) : this.clear() : a(e[t], this.toBBox);
		}, r;
	});
})), hs = function() {
	function e(e) {
		this.opacity_ = e.opacity, this.rotateWithView_ = e.rotateWithView, this.rotation_ = e.rotation, this.scale_ = e.scale, this.scaleArray_ = Ra(e.scale), this.displacement_ = e.displacement;
	}
	return e.prototype.clone = function() {
		var t = this.getScale();
		return new e({
			opacity: this.getOpacity(),
			scale: Array.isArray(t) ? t.slice() : t,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			displacement: this.getDisplacement().slice()
		});
	}, e.prototype.getOpacity = function() {
		return this.opacity_;
	}, e.prototype.getRotateWithView = function() {
		return this.rotateWithView_;
	}, e.prototype.getRotation = function() {
		return this.rotation_;
	}, e.prototype.getScale = function() {
		return this.scale_;
	}, e.prototype.getScaleArray = function() {
		return this.scaleArray_;
	}, e.prototype.getDisplacement = function() {
		return this.displacement_;
	}, e.prototype.getAnchor = function() {
		return F();
	}, e.prototype.getImage = function(e) {
		return F();
	}, e.prototype.getHitDetectionImage = function() {
		return F();
	}, e.prototype.getPixelRatio = function(e) {
		return 1;
	}, e.prototype.getImageState = function() {
		return F();
	}, e.prototype.getImageSize = function() {
		return F();
	}, e.prototype.getOrigin = function() {
		return F();
	}, e.prototype.getSize = function() {
		return F();
	}, e.prototype.setOpacity = function(e) {
		this.opacity_ = e;
	}, e.prototype.setRotateWithView = function(e) {
		this.rotateWithView_ = e;
	}, e.prototype.setRotation = function(e) {
		this.rotation_ = e;
	}, e.prototype.setScale = function(e) {
		this.scale_ = e, this.scaleArray_ = Ra(e);
	}, e.prototype.listenImageChange = function(e) {
		F();
	}, e.prototype.load = function() {
		F();
	}, e.prototype.unlistenImageChange = function(e) {
		F();
	}, e;
}();
//#endregion
//#region node_modules/ol/colorlike.js
function gs(e) {
	return Array.isArray(e) ? or(e) : e;
}
//#endregion
//#region node_modules/ol/style/RegularShape.js
var _s = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), vs = function(e) {
	_s(t, e);
	function t(t) {
		var n = this, r = t.rotateWithView !== void 0 && t.rotateWithView;
		return n = e.call(this, {
			opacity: 1,
			rotateWithView: r,
			rotation: t.rotation === void 0 ? 0 : t.rotation,
			scale: t.scale === void 0 ? 1 : t.scale,
			displacement: t.displacement === void 0 ? [0, 0] : t.displacement
		}) || this, n.canvas_ = void 0, n.hitDetectionCanvas_ = null, n.fill_ = t.fill === void 0 ? null : t.fill, n.origin_ = [0, 0], n.points_ = t.points, n.radius_ = t.radius === void 0 ? t.radius1 : t.radius, n.radius2_ = t.radius2, n.angle_ = t.angle === void 0 ? 0 : t.angle, n.stroke_ = t.stroke === void 0 ? null : t.stroke, n.anchor_ = null, n.size_ = null, n.renderOptions_ = null, n.render(), n;
	}
	return t.prototype.clone = function() {
		var e = this.getScale(), n = new t({
			fill: this.getFill() ? this.getFill().clone() : void 0,
			points: this.getPoints(),
			radius: this.getRadius(),
			radius2: this.getRadius2(),
			angle: this.getAngle(),
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			scale: Array.isArray(e) ? e.slice() : e,
			displacement: this.getDisplacement().slice()
		});
		return n.setOpacity(this.getOpacity()), n;
	}, t.prototype.getAnchor = function() {
		return this.anchor_;
	}, t.prototype.getAngle = function() {
		return this.angle_;
	}, t.prototype.getFill = function() {
		return this.fill_;
	}, t.prototype.getHitDetectionImage = function() {
		return this.hitDetectionCanvas_ || this.createHitDetectionCanvas_(this.renderOptions_), this.hitDetectionCanvas_;
	}, t.prototype.getImage = function(e) {
		var t = this.canvas_[e];
		if (!t) {
			var n = this.renderOptions_, r = de(n.size * e, n.size * e);
			this.draw_(n, r, e), t = r.canvas, this.canvas_[e] = t;
		}
		return t;
	}, t.prototype.getPixelRatio = function(e) {
		return e;
	}, t.prototype.getImageSize = function() {
		return this.size_;
	}, t.prototype.getImageState = function() {
		return X.LOADED;
	}, t.prototype.getOrigin = function() {
		return this.origin_;
	}, t.prototype.getPoints = function() {
		return this.points_;
	}, t.prototype.getRadius = function() {
		return this.radius_;
	}, t.prototype.getRadius2 = function() {
		return this.radius2_;
	}, t.prototype.getSize = function() {
		return this.size_;
	}, t.prototype.getStroke = function() {
		return this.stroke_;
	}, t.prototype.listenImageChange = function(e) {}, t.prototype.load = function() {}, t.prototype.unlistenImageChange = function(e) {}, t.prototype.calculateLineJoinSize_ = function(e, t, n) {
		if (t === 0 || this.points_ === Infinity || e !== "bevel" && e !== "miter") return t;
		var r = this.radius_, i = this.radius2_ === void 0 ? r : this.radius2_;
		if (r < i) {
			var a = r;
			r = i, i = a;
		}
		var o = this.radius2_ === void 0 ? this.points_ : this.points_ * 2, s = 2 * Math.PI / o, c = i * Math.sin(s), l = Math.sqrt(i * i - c * c), u = r - l, d = Math.sqrt(c * c + u * u), f = d / c;
		if (e === "miter" && f <= n) return f * t;
		var p = t / 2 / f, m = t / 2 * (u / d), h = Math.sqrt((r + p) * (r + p) + m * m) - r;
		if (this.radius2_ === void 0 || e === "bevel") return h * 2;
		var g = r * Math.sin(s), _ = Math.sqrt(r * r - g * g), v = i - _, y = Math.sqrt(g * g + v * v) / g;
		if (y <= n) {
			var b = y * t / 2 - i - r;
			return 2 * Math.max(h, b);
		}
		return h * 2;
	}, t.prototype.createRenderOptions = function() {
		var e = Er, t = 0, n = null, r = 0, i, a = 0;
		this.stroke_ && (i = this.stroke_.getColor(), i === null && (i = Dr), i = gs(i), a = this.stroke_.getWidth(), a === void 0 && (a = 1), n = this.stroke_.getLineDash(), r = this.stroke_.getLineDashOffset(), e = this.stroke_.getLineJoin(), e === void 0 && (e = Er), t = this.stroke_.getMiterLimit(), t === void 0 && (t = 10));
		var o = this.calculateLineJoinSize_(e, a, t), s = Math.max(this.radius_, this.radius2_ || 0), c = Math.ceil(2 * s + o);
		return {
			strokeStyle: i,
			strokeWidth: a,
			size: c,
			lineDash: n,
			lineDashOffset: r,
			lineJoin: e,
			miterLimit: t
		};
	}, t.prototype.render = function() {
		this.renderOptions_ = this.createRenderOptions();
		var e = this.renderOptions_.size, t = this.getDisplacement();
		this.canvas_ = {}, this.anchor_ = [e / 2 - t[0], e / 2 + t[1]], this.size_ = [e, e];
	}, t.prototype.draw_ = function(e, t, n) {
		if (t.scale(n, n), t.translate(e.size / 2, e.size / 2), this.createPath_(t), this.fill_) {
			var r = this.fill_.getColor();
			r === null && (r = Cr), t.fillStyle = gs(r), t.fill();
		}
		this.stroke_ && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, t.setLineDash && e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}, t.prototype.createHitDetectionCanvas_ = function(e) {
		if (this.fill_) {
			var t = this.fill_.getColor(), n = 0;
			if (typeof t == "string" && (t = rr(t)), t === null ? n = 1 : Array.isArray(t) && (n = t.length === 4 ? t[3] : 1), n === 0) {
				var r = de(e.size, e.size);
				this.hitDetectionCanvas_ = r.canvas, this.drawHitDetectionCanvas_(e, r);
			}
		}
		this.hitDetectionCanvas_ ||= this.getImage(1);
	}, t.prototype.createPath_ = function(e) {
		var t = this.points_, n = this.radius_;
		if (t === Infinity) e.arc(0, 0, n, 0, 2 * Math.PI);
		else {
			var r = this.radius2_ === void 0 ? n : this.radius2_;
			this.radius2_ !== void 0 && (t *= 2);
			for (var i = this.angle_ - Math.PI / 2, a = 2 * Math.PI / t, o = 0; o < t; o++) {
				var s = i + o * a, c = o % 2 == 0 ? n : r;
				e.lineTo(c * Math.cos(s), c * Math.sin(s));
			}
			e.closePath();
		}
	}, t.prototype.drawHitDetectionCanvas_ = function(e, t) {
		t.translate(e.size / 2, e.size / 2), this.createPath_(t), t.fillStyle = Cr, t.fill(), this.stroke_ && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}, t;
}(hs), ys = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), bs = function(e) {
	ys(t, e);
	function t(t) {
		var n = this, r = t || {};
		return n = e.call(this, {
			points: Infinity,
			fill: r.fill,
			radius: r.radius,
			stroke: r.stroke,
			scale: r.scale === void 0 ? 1 : r.scale,
			rotation: r.rotation === void 0 ? 0 : r.rotation,
			rotateWithView: r.rotateWithView !== void 0 && r.rotateWithView,
			displacement: r.displacement === void 0 ? [0, 0] : r.displacement
		}) || this, n;
	}
	return t.prototype.clone = function() {
		var e = this.getScale(), n = new t({
			fill: this.getFill() ? this.getFill().clone() : void 0,
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			radius: this.getRadius(),
			scale: Array.isArray(e) ? e.slice() : e,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			displacement: this.getDisplacement().slice()
		});
		return n.setOpacity(this.getOpacity()), n;
	}, t.prototype.setRadius = function(e) {
		this.radius_ = e, this.render();
	}, t;
}(vs), xs = function() {
	function e(e) {
		var t = e || {};
		this.color_ = t.color === void 0 ? null : t.color;
	}
	return e.prototype.clone = function() {
		var t = this.getColor();
		return new e({ color: Array.isArray(t) ? t.slice() : t || void 0 });
	}, e.prototype.getColor = function() {
		return this.color_;
	}, e.prototype.setColor = function(e) {
		this.color_ = e;
	}, e;
}(), Ss = function() {
	function e(e) {
		var t = e || {};
		this.color_ = t.color === void 0 ? null : t.color, this.lineCap_ = t.lineCap, this.lineDash_ = t.lineDash === void 0 ? null : t.lineDash, this.lineDashOffset_ = t.lineDashOffset, this.lineJoin_ = t.lineJoin, this.miterLimit_ = t.miterLimit, this.width_ = t.width;
	}
	return e.prototype.clone = function() {
		var t = this.getColor();
		return new e({
			color: Array.isArray(t) ? t.slice() : t || void 0,
			lineCap: this.getLineCap(),
			lineDash: this.getLineDash() ? this.getLineDash().slice() : void 0,
			lineDashOffset: this.getLineDashOffset(),
			lineJoin: this.getLineJoin(),
			miterLimit: this.getMiterLimit(),
			width: this.getWidth()
		});
	}, e.prototype.getColor = function() {
		return this.color_;
	}, e.prototype.getLineCap = function() {
		return this.lineCap_;
	}, e.prototype.getLineDash = function() {
		return this.lineDash_;
	}, e.prototype.getLineDashOffset = function() {
		return this.lineDashOffset_;
	}, e.prototype.getLineJoin = function() {
		return this.lineJoin_;
	}, e.prototype.getMiterLimit = function() {
		return this.miterLimit_;
	}, e.prototype.getWidth = function() {
		return this.width_;
	}, e.prototype.setColor = function(e) {
		this.color_ = e;
	}, e.prototype.setLineCap = function(e) {
		this.lineCap_ = e;
	}, e.prototype.setLineDash = function(e) {
		this.lineDash_ = e;
	}, e.prototype.setLineDashOffset = function(e) {
		this.lineDashOffset_ = e;
	}, e.prototype.setLineJoin = function(e) {
		this.lineJoin_ = e;
	}, e.prototype.setMiterLimit = function(e) {
		this.miterLimit_ = e;
	}, e.prototype.setWidth = function(e) {
		this.width_ = e;
	}, e;
}(), Cs = function() {
	function e(e) {
		var t = e || {};
		this.geometry_ = null, this.geometryFunction_ = Ds, t.geometry !== void 0 && this.setGeometry(t.geometry), this.fill_ = t.fill === void 0 ? null : t.fill, this.image_ = t.image === void 0 ? null : t.image, this.renderer_ = t.renderer === void 0 ? null : t.renderer, this.hitDetectionRenderer_ = t.hitDetectionRenderer === void 0 ? null : t.hitDetectionRenderer, this.stroke_ = t.stroke === void 0 ? null : t.stroke, this.text_ = t.text === void 0 ? null : t.text, this.zIndex_ = t.zIndex;
	}
	return e.prototype.clone = function() {
		var t = this.getGeometry();
		return t && typeof t == "object" && (t = t.clone()), new e({
			geometry: t,
			fill: this.getFill() ? this.getFill().clone() : void 0,
			image: this.getImage() ? this.getImage().clone() : void 0,
			renderer: this.getRenderer(),
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			text: this.getText() ? this.getText().clone() : void 0,
			zIndex: this.getZIndex()
		});
	}, e.prototype.getRenderer = function() {
		return this.renderer_;
	}, e.prototype.setRenderer = function(e) {
		this.renderer_ = e;
	}, e.prototype.setHitDetectionRenderer = function(e) {
		this.hitDetectionRenderer_ = e;
	}, e.prototype.getHitDetectionRenderer = function() {
		return this.hitDetectionRenderer_;
	}, e.prototype.getGeometry = function() {
		return this.geometry_;
	}, e.prototype.getGeometryFunction = function() {
		return this.geometryFunction_;
	}, e.prototype.getFill = function() {
		return this.fill_;
	}, e.prototype.setFill = function(e) {
		this.fill_ = e;
	}, e.prototype.getImage = function() {
		return this.image_;
	}, e.prototype.setImage = function(e) {
		this.image_ = e;
	}, e.prototype.getStroke = function() {
		return this.stroke_;
	}, e.prototype.setStroke = function(e) {
		this.stroke_ = e;
	}, e.prototype.getText = function() {
		return this.text_;
	}, e.prototype.setText = function(e) {
		this.text_ = e;
	}, e.prototype.getZIndex = function() {
		return this.zIndex_;
	}, e.prototype.setGeometry = function(e) {
		typeof e == "function" ? this.geometryFunction_ = e : typeof e == "string" ? this.geometryFunction_ = function(t) {
			return t.get(e);
		} : e ? e !== void 0 && (this.geometryFunction_ = function() {
			return e;
		}) : this.geometryFunction_ = Ds, this.geometry_ = e;
	}, e.prototype.setZIndex = function(e) {
		this.zIndex_ = e;
	}, e;
}();
function ws(e) {
	var t;
	if (typeof e == "function") t = e;
	else {
		var n;
		Array.isArray(e) ? n = e : (H(typeof e.getZIndex == "function", 41), n = [e]), t = function() {
			return n;
		};
	}
	return t;
}
var Ts = null;
function Es(e, t) {
	if (!Ts) {
		var n = new xs({ color: "rgba(255,255,255,0.4)" }), r = new Ss({
			color: "#3399CC",
			width: 1.25
		});
		Ts = [new Cs({
			image: new bs({
				fill: n,
				stroke: r,
				radius: 5
			}),
			fill: n,
			stroke: r
		})];
	}
	return Ts;
}
function Ds(e) {
	return e.getGeometry();
}
//#endregion
//#region node_modules/ol/layer/BaseVector.js
var Os = /* @__PURE__ */ c(ms(), 1), ks = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), As = { RENDER_ORDER: "renderOrder" }, js = function(e) {
	ks(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		return delete i.style, delete i.renderBuffer, delete i.updateWhileAnimating, delete i.updateWhileInteracting, n = e.call(this, i) || this, n.declutter_ = r.declutter !== void 0 && r.declutter, n.renderBuffer_ = r.renderBuffer === void 0 ? 100 : r.renderBuffer, n.style_ = null, n.styleFunction_ = void 0, n.setStyle(r.style), n.updateWhileAnimating_ = r.updateWhileAnimating !== void 0 && r.updateWhileAnimating, n.updateWhileInteracting_ = r.updateWhileInteracting !== void 0 && r.updateWhileInteracting, n;
	}
	return t.prototype.getDeclutter = function() {
		return this.declutter_;
	}, t.prototype.getFeatures = function(t) {
		return e.prototype.getFeatures.call(this, t);
	}, t.prototype.getRenderBuffer = function() {
		return this.renderBuffer_;
	}, t.prototype.getRenderOrder = function() {
		return this.get(As.RENDER_ORDER);
	}, t.prototype.getStyle = function() {
		return this.style_;
	}, t.prototype.getStyleFunction = function() {
		return this.styleFunction_;
	}, t.prototype.getUpdateWhileAnimating = function() {
		return this.updateWhileAnimating_;
	}, t.prototype.getUpdateWhileInteracting = function() {
		return this.updateWhileInteracting_;
	}, t.prototype.renderDeclutter = function(e) {
		e.declutterTree ||= new Os.default(9), this.getRenderer().renderDeclutter(e);
	}, t.prototype.setRenderOrder = function(e) {
		this.set(As.RENDER_ORDER, e);
	}, t.prototype.setStyle = function(e) {
		this.style_ = e === void 0 ? Es : e, this.styleFunction_ = e === null ? void 0 : ws(this.style_), this.changed();
	}, t;
}(hr), Z = {
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
}, Ms = [Z.FILL], Ns = [Z.STROKE], Ps = [Z.BEGIN_PATH], Fs = [Z.CLOSE_PATH], Is = function() {
	function e() {}
	return e.prototype.drawCustom = function(e, t, n, r) {}, e.prototype.drawGeometry = function(e) {}, e.prototype.setStyle = function(e) {}, e.prototype.drawCircle = function(e, t) {}, e.prototype.drawFeature = function(e, t) {}, e.prototype.drawGeometryCollection = function(e, t) {}, e.prototype.drawLineString = function(e, t) {}, e.prototype.drawMultiLineString = function(e, t) {}, e.prototype.drawMultiPoint = function(e, t) {}, e.prototype.drawMultiPolygon = function(e, t) {}, e.prototype.drawPoint = function(e, t) {}, e.prototype.drawPolygon = function(e, t) {}, e.prototype.drawText = function(e, t) {}, e.prototype.setFillStrokeStyle = function(e, t) {}, e.prototype.setImageStyle = function(e, t) {}, e.prototype.setTextStyle = function(e, t) {}, e;
}(), Ls = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Rs = function(e) {
	Ls(t, e);
	function t(t, n, r, i) {
		var a = e.call(this) || this;
		return a.tolerance = t, a.maxExtent = n, a.pixelRatio = i, a.maxLineWidth = 0, a.resolution = r, a.beginGeometryInstruction1_ = null, a.beginGeometryInstruction2_ = null, a.bufferedMaxExtent_ = null, a.instructions = [], a.coordinates = [], a.tmpCoordinate_ = [], a.hitDetectionInstructions = [], a.state = {}, a;
	}
	return t.prototype.applyPixelRatio = function(e) {
		var t = this.pixelRatio;
		return t == 1 ? e : e.map(function(e) {
			return e * t;
		});
	}, t.prototype.appendFlatPointCoordinates = function(e, t) {
		for (var n = this.getBufferedMaxExtent(), r = this.tmpCoordinate_, i = this.coordinates, a = i.length, o = 0, s = e.length; o < s; o += t) r[0] = e[o], r[1] = e[o + 1], Et(n, r) && (i[a++] = r[0], i[a++] = r[1]);
		return a;
	}, t.prototype.appendFlatLineCoordinates = function(e, t, n, r, i, a) {
		var o = this.coordinates, s = o.length, c = this.getBufferedMaxExtent();
		a && (t += r);
		var l = e[t], u = e[t + 1], d = this.tmpCoordinate_, f = !0, p, m, h;
		for (p = t + r; p < n; p += r) d[0] = e[p], d[1] = e[p + 1], h = kt(c, d), h === m ? h === yt.INTERSECTING ? (o[s++] = d[0], o[s++] = d[1], f = !1) : f = !0 : (f &&= (o[s++] = l, o[s++] = u, !1), o[s++] = d[0], o[s++] = d[1]), l = d[0], u = d[1], m = h;
		return (i && f || p === t + r) && (o[s++] = l, o[s++] = u), s;
	}, t.prototype.drawCustomCoordinates_ = function(e, t, n, r, i) {
		for (var a = 0, o = n.length; a < o; ++a) {
			var s = n[a], c = this.appendFlatLineCoordinates(e, t, s, r, !1, !1);
			i.push(c), t = s;
		}
		return t;
	}, t.prototype.drawCustom = function(e, t, n, r) {
		this.beginGeometry(e, t);
		var i = e.getType(), a = e.getStride(), o = this.coordinates.length, s, c, l, u, d;
		switch (i) {
			case W.MULTI_POLYGON:
				s = e.getOrientedFlatCoordinates(), u = [];
				var f = e.getEndss();
				d = 0;
				for (var p = 0, m = f.length; p < m; ++p) {
					var h = [];
					d = this.drawCustomCoordinates_(s, d, f[p], a, h), u.push(h);
				}
				this.instructions.push([
					Z.CUSTOM,
					o,
					u,
					e,
					n,
					aa
				]), this.hitDetectionInstructions.push([
					Z.CUSTOM,
					o,
					u,
					e,
					r || n,
					aa
				]);
				break;
			case W.POLYGON:
			case W.MULTI_LINE_STRING:
				l = [], s = i == W.POLYGON ? e.getOrientedFlatCoordinates() : e.getFlatCoordinates(), d = this.drawCustomCoordinates_(s, 0, e.getEnds(), a, l), this.instructions.push([
					Z.CUSTOM,
					o,
					l,
					e,
					n,
					ia
				]), this.hitDetectionInstructions.push([
					Z.CUSTOM,
					o,
					l,
					e,
					r || n,
					ia
				]);
				break;
			case W.LINE_STRING:
			case W.CIRCLE:
				s = e.getFlatCoordinates(), c = this.appendFlatLineCoordinates(s, 0, s.length, a, !1, !1), this.instructions.push([
					Z.CUSTOM,
					o,
					c,
					e,
					n,
					ra
				]), this.hitDetectionInstructions.push([
					Z.CUSTOM,
					o,
					c,
					e,
					r || n,
					ra
				]);
				break;
			case W.MULTI_POINT:
				s = e.getFlatCoordinates(), c = this.appendFlatPointCoordinates(s, a), c > o && (this.instructions.push([
					Z.CUSTOM,
					o,
					c,
					e,
					n,
					ra
				]), this.hitDetectionInstructions.push([
					Z.CUSTOM,
					o,
					c,
					e,
					r || n,
					ra
				]));
				break;
			case W.POINT:
				s = e.getFlatCoordinates(), this.coordinates.push(s[0], s[1]), c = this.coordinates.length, this.instructions.push([
					Z.CUSTOM,
					o,
					c,
					e,
					n
				]), this.hitDetectionInstructions.push([
					Z.CUSTOM,
					o,
					c,
					e,
					r || n
				]);
				break;
			default:
		}
		this.endGeometry(t);
	}, t.prototype.beginGeometry = function(e, t) {
		this.beginGeometryInstruction1_ = [
			Z.BEGIN_GEOMETRY,
			t,
			0,
			e
		], this.instructions.push(this.beginGeometryInstruction1_), this.beginGeometryInstruction2_ = [
			Z.BEGIN_GEOMETRY,
			t,
			0,
			e
		], this.hitDetectionInstructions.push(this.beginGeometryInstruction2_);
	}, t.prototype.finish = function() {
		return {
			instructions: this.instructions,
			hitDetectionInstructions: this.hitDetectionInstructions,
			coordinates: this.coordinates
		};
	}, t.prototype.reverseHitDetectionInstructions = function() {
		var e = this.hitDetectionInstructions;
		e.reverse();
		var t, n = e.length, r, i, a = -1;
		for (t = 0; t < n; ++t) r = e[t], i = r[0], i == Z.END_GEOMETRY ? a = t : i == Z.BEGIN_GEOMETRY && (r[2] = t, m(this.hitDetectionInstructions, a, t), a = -1);
	}, t.prototype.setFillStrokeStyle = function(e, t) {
		var n = this.state;
		if (e ? n.fillStyle = gs(e.getColor() || Cr) : n.fillStyle = void 0, t) {
			n.strokeStyle = gs(t.getColor() || Dr);
			var r = t.getLineCap();
			n.lineCap = r === void 0 ? wr : r;
			var i = t.getLineDash();
			n.lineDash = i ? i.slice() : Tr, n.lineDashOffset = t.getLineDashOffset() || 0;
			var a = t.getLineJoin();
			n.lineJoin = a === void 0 ? Er : a;
			var o = t.getWidth();
			n.lineWidth = o === void 0 ? 1 : o;
			var s = t.getMiterLimit();
			n.miterLimit = s === void 0 ? 10 : s, n.lineWidth > this.maxLineWidth && (this.maxLineWidth = n.lineWidth, this.bufferedMaxExtent_ = null);
		} else n.strokeStyle = void 0, n.lineCap = void 0, n.lineDash = null, n.lineDashOffset = void 0, n.lineJoin = void 0, n.lineWidth = void 0, n.miterLimit = void 0;
	}, t.prototype.createFill = function(e) {
		var t = e.fillStyle, n = [Z.SET_FILL_STYLE, t];
		return typeof t != "string" && n.push(!0), n;
	}, t.prototype.applyStroke = function(e) {
		this.instructions.push(this.createStroke(e));
	}, t.prototype.createStroke = function(e) {
		return [
			Z.SET_STROKE_STYLE,
			e.strokeStyle,
			e.lineWidth * this.pixelRatio,
			e.lineCap,
			e.lineJoin,
			e.miterLimit,
			this.applyPixelRatio(e.lineDash),
			e.lineDashOffset * this.pixelRatio
		];
	}, t.prototype.updateFillStyle = function(e, t) {
		var n = e.fillStyle;
		(typeof n != "string" || e.currentFillStyle != n) && (n !== void 0 && this.instructions.push(t.call(this, e)), e.currentFillStyle = n);
	}, t.prototype.updateStrokeStyle = function(e, t) {
		var n = e.strokeStyle, r = e.lineCap, i = e.lineDash, a = e.lineDashOffset, o = e.lineJoin, s = e.lineWidth, c = e.miterLimit;
		(e.currentStrokeStyle != n || e.currentLineCap != r || i != e.currentLineDash && !g(e.currentLineDash, i) || e.currentLineDashOffset != a || e.currentLineJoin != o || e.currentLineWidth != s || e.currentMiterLimit != c) && (n !== void 0 && t.call(this, e), e.currentStrokeStyle = n, e.currentLineCap = r, e.currentLineDash = i, e.currentLineDashOffset = a, e.currentLineJoin = o, e.currentLineWidth = s, e.currentMiterLimit = c);
	}, t.prototype.endGeometry = function(e) {
		this.beginGeometryInstruction1_[2] = this.instructions.length, this.beginGeometryInstruction1_ = null, this.beginGeometryInstruction2_[2] = this.hitDetectionInstructions.length, this.beginGeometryInstruction2_ = null;
		var t = [Z.END_GEOMETRY, e];
		this.instructions.push(t), this.hitDetectionInstructions.push(t);
	}, t.prototype.getBufferedMaxExtent = function() {
		if (!this.bufferedMaxExtent_ && (this.bufferedMaxExtent_ = wt(this.maxExtent), this.maxLineWidth > 0)) {
			var e = this.resolution * (this.maxLineWidth + 1) / 2;
			Ct(this.bufferedMaxExtent_, e, this.bufferedMaxExtent_);
		}
		return this.bufferedMaxExtent_;
	}, t;
}(Is), zs = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Bs = function(e) {
	zs(t, e);
	function t(t, n, r, i) {
		var a = e.call(this, t, n, r, i) || this;
		return a.hitDetectionImage_ = null, a.image_ = null, a.imagePixelRatio_ = void 0, a.anchorX_ = void 0, a.anchorY_ = void 0, a.height_ = void 0, a.opacity_ = void 0, a.originX_ = void 0, a.originY_ = void 0, a.rotateWithView_ = void 0, a.rotation_ = void 0, a.scale_ = void 0, a.width_ = void 0, a.declutterImageWithText_ = void 0, a;
	}
	return t.prototype.drawPoint = function(e, t) {
		if (this.image_) {
			this.beginGeometry(e, t);
			var n = e.getFlatCoordinates(), r = e.getStride(), i = this.coordinates.length, a = this.appendFlatPointCoordinates(n, r);
			this.instructions.push([
				Z.DRAW_IMAGE,
				i,
				a,
				this.image_,
				this.anchorX_ * this.imagePixelRatio_,
				this.anchorY_ * this.imagePixelRatio_,
				Math.ceil(this.height_ * this.imagePixelRatio_),
				this.opacity_,
				this.originX_,
				this.originY_,
				this.rotateWithView_,
				this.rotation_,
				[this.scale_[0] * this.pixelRatio / this.imagePixelRatio_, this.scale_[1] * this.pixelRatio / this.imagePixelRatio_],
				Math.ceil(this.width_ * this.imagePixelRatio_),
				this.declutterImageWithText_
			]), this.hitDetectionInstructions.push([
				Z.DRAW_IMAGE,
				i,
				a,
				this.hitDetectionImage_,
				this.anchorX_,
				this.anchorY_,
				this.height_,
				this.opacity_,
				this.originX_,
				this.originY_,
				this.rotateWithView_,
				this.rotation_,
				this.scale_,
				this.width_,
				this.declutterImageWithText_
			]), this.endGeometry(t);
		}
	}, t.prototype.drawMultiPoint = function(e, t) {
		if (this.image_) {
			this.beginGeometry(e, t);
			var n = e.getFlatCoordinates(), r = e.getStride(), i = this.coordinates.length, a = this.appendFlatPointCoordinates(n, r);
			this.instructions.push([
				Z.DRAW_IMAGE,
				i,
				a,
				this.image_,
				this.anchorX_ * this.imagePixelRatio_,
				this.anchorY_ * this.imagePixelRatio_,
				Math.ceil(this.height_ * this.imagePixelRatio_),
				this.opacity_,
				this.originX_,
				this.originY_,
				this.rotateWithView_,
				this.rotation_,
				[this.scale_[0] * this.pixelRatio / this.imagePixelRatio_, this.scale_[1] * this.pixelRatio / this.imagePixelRatio_],
				Math.ceil(this.width_ * this.imagePixelRatio_),
				this.declutterImageWithText_
			]), this.hitDetectionInstructions.push([
				Z.DRAW_IMAGE,
				i,
				a,
				this.hitDetectionImage_,
				this.anchorX_,
				this.anchorY_,
				this.height_,
				this.opacity_,
				this.originX_,
				this.originY_,
				this.rotateWithView_,
				this.rotation_,
				this.scale_,
				this.width_,
				this.declutterImageWithText_
			]), this.endGeometry(t);
		}
	}, t.prototype.finish = function() {
		return this.reverseHitDetectionInstructions(), this.anchorX_ = void 0, this.anchorY_ = void 0, this.hitDetectionImage_ = null, this.image_ = null, this.imagePixelRatio_ = void 0, this.height_ = void 0, this.scale_ = void 0, this.opacity_ = void 0, this.originX_ = void 0, this.originY_ = void 0, this.rotateWithView_ = void 0, this.rotation_ = void 0, this.width_ = void 0, e.prototype.finish.call(this);
	}, t.prototype.setImageStyle = function(e, t) {
		var n = e.getAnchor(), r = e.getSize(), i = e.getHitDetectionImage(), a = e.getImage(this.pixelRatio), o = e.getOrigin();
		this.imagePixelRatio_ = e.getPixelRatio(this.pixelRatio), this.anchorX_ = n[0], this.anchorY_ = n[1], this.hitDetectionImage_ = i, this.image_ = a, this.height_ = r[1], this.opacity_ = e.getOpacity(), this.originX_ = o[0] * this.imagePixelRatio_, this.originY_ = o[1] * this.imagePixelRatio_, this.rotateWithView_ = e.getRotateWithView(), this.rotation_ = e.getRotation(), this.scale_ = e.getScaleArray(), this.width_ = r[0], this.declutterImageWithText_ = t;
	}, t;
}(Rs), Vs = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Hs = function(e) {
	Vs(t, e);
	function t(t, n, r, i) {
		return e.call(this, t, n, r, i) || this;
	}
	return t.prototype.drawFlatCoordinates_ = function(e, t, n, r) {
		var i = this.coordinates.length, a = this.appendFlatLineCoordinates(e, t, n, r, !1, !1), o = [
			Z.MOVE_TO_LINE_TO,
			i,
			a
		];
		return this.instructions.push(o), this.hitDetectionInstructions.push(o), n;
	}, t.prototype.drawLineString = function(e, t) {
		var n = this.state, r = n.strokeStyle, i = n.lineWidth;
		if (!(r === void 0 || i === void 0)) {
			this.updateStrokeStyle(n, this.applyStroke), this.beginGeometry(e, t), this.hitDetectionInstructions.push([
				Z.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				Tr,
				0
			], Ps);
			var a = e.getFlatCoordinates(), o = e.getStride();
			this.drawFlatCoordinates_(a, 0, a.length, o), this.hitDetectionInstructions.push(Ns), this.endGeometry(t);
		}
	}, t.prototype.drawMultiLineString = function(e, t) {
		var n = this.state, r = n.strokeStyle, i = n.lineWidth;
		if (!(r === void 0 || i === void 0)) {
			this.updateStrokeStyle(n, this.applyStroke), this.beginGeometry(e, t), this.hitDetectionInstructions.push([
				Z.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				n.lineDash,
				n.lineDashOffset
			], Ps);
			for (var a = e.getEnds(), o = e.getFlatCoordinates(), s = e.getStride(), c = 0, l = 0, u = a.length; l < u; ++l) c = this.drawFlatCoordinates_(o, c, a[l], s);
			this.hitDetectionInstructions.push(Ns), this.endGeometry(t);
		}
	}, t.prototype.finish = function() {
		var t = this.state;
		return t.lastStroke != null && t.lastStroke != this.coordinates.length && this.instructions.push(Ns), this.reverseHitDetectionInstructions(), this.state = null, e.prototype.finish.call(this);
	}, t.prototype.applyStroke = function(t) {
		t.lastStroke != null && t.lastStroke != this.coordinates.length && (this.instructions.push(Ns), t.lastStroke = this.coordinates.length), t.lastStroke = 0, e.prototype.applyStroke.call(this, t), this.instructions.push(Ps);
	}, t;
}(Rs), Us = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ws = function(e) {
	Us(t, e);
	function t(t, n, r, i) {
		return e.call(this, t, n, r, i) || this;
	}
	return t.prototype.drawFlatCoordinatess_ = function(e, t, n, r) {
		var i = this.state, a = i.fillStyle !== void 0, o = i.strokeStyle !== void 0, s = n.length;
		this.instructions.push(Ps), this.hitDetectionInstructions.push(Ps);
		for (var c = 0; c < s; ++c) {
			var l = n[c], u = this.coordinates.length, d = this.appendFlatLineCoordinates(e, t, l, r, !0, !o), f = [
				Z.MOVE_TO_LINE_TO,
				u,
				d
			];
			this.instructions.push(f), this.hitDetectionInstructions.push(f), o && (this.instructions.push(Fs), this.hitDetectionInstructions.push(Fs)), t = l;
		}
		return a && (this.instructions.push(Ms), this.hitDetectionInstructions.push(Ms)), o && (this.instructions.push(Ns), this.hitDetectionInstructions.push(Ns)), t;
	}, t.prototype.drawCircle = function(e, t) {
		var n = this.state, r = n.fillStyle, i = n.strokeStyle;
		if (!(r === void 0 && i === void 0)) {
			this.setFillStrokeStyles_(), this.beginGeometry(e, t), n.fillStyle !== void 0 && this.hitDetectionInstructions.push([Z.SET_FILL_STYLE, Cr]), n.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
				Z.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				n.lineDash,
				n.lineDashOffset
			]);
			var a = e.getFlatCoordinates(), o = e.getStride(), s = this.coordinates.length;
			this.appendFlatLineCoordinates(a, 0, a.length, o, !1, !1);
			var c = [Z.CIRCLE, s];
			this.instructions.push(Ps, c), this.hitDetectionInstructions.push(Ps, c), n.fillStyle !== void 0 && (this.instructions.push(Ms), this.hitDetectionInstructions.push(Ms)), n.strokeStyle !== void 0 && (this.instructions.push(Ns), this.hitDetectionInstructions.push(Ns)), this.endGeometry(t);
		}
	}, t.prototype.drawPolygon = function(e, t) {
		var n = this.state, r = n.fillStyle, i = n.strokeStyle;
		if (!(r === void 0 && i === void 0)) {
			this.setFillStrokeStyles_(), this.beginGeometry(e, t), n.fillStyle !== void 0 && this.hitDetectionInstructions.push([Z.SET_FILL_STYLE, Cr]), n.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
				Z.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				n.lineDash,
				n.lineDashOffset
			]);
			var a = e.getEnds(), o = e.getOrientedFlatCoordinates(), s = e.getStride();
			this.drawFlatCoordinatess_(o, 0, a, s), this.endGeometry(t);
		}
	}, t.prototype.drawMultiPolygon = function(e, t) {
		var n = this.state, r = n.fillStyle, i = n.strokeStyle;
		if (!(r === void 0 && i === void 0)) {
			this.setFillStrokeStyles_(), this.beginGeometry(e, t), n.fillStyle !== void 0 && this.hitDetectionInstructions.push([Z.SET_FILL_STYLE, Cr]), n.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
				Z.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				n.lineDash,
				n.lineDashOffset
			]);
			for (var a = e.getEndss(), o = e.getOrientedFlatCoordinates(), s = e.getStride(), c = 0, l = 0, u = a.length; l < u; ++l) c = this.drawFlatCoordinatess_(o, c, a[l], s);
			this.endGeometry(t);
		}
	}, t.prototype.finish = function() {
		this.reverseHitDetectionInstructions(), this.state = null;
		var t = this.tolerance;
		if (t !== 0) for (var n = this.coordinates, r = 0, i = n.length; r < i; ++r) n[r] = ea(n[r], t);
		return e.prototype.finish.call(this);
	}, t.prototype.setFillStrokeStyles_ = function() {
		var e = this.state;
		e.fillStyle !== void 0 && this.updateFillStyle(e, this.createFill), e.strokeStyle !== void 0 && this.updateStrokeStyle(e, this.applyStroke);
	}, t;
}(Rs), Gs = {
	POINT: "point",
	LINE: "line"
};
//#endregion
//#region node_modules/ol/geom/flat/straightchunk.js
function Ks(e, t, n, r, i) {
	var a = n, o = n, s = 0, c = 0, l = n, u, d, f, p, m, h, g, _, v, y;
	for (d = n; d < r; d += i) {
		var b = t[d], x = t[d + 1];
		m !== void 0 && (v = b - m, y = x - h, p = Math.sqrt(v * v + y * y), g !== void 0 && (c += f, u = Math.acos((g * v + _ * y) / (f * p)), u > e && (c > s && (s = c, a = l, o = d), c = 0, l = d - i)), f = p, g = v, _ = y), m = b, h = x;
	}
	return c += p, c > s ? [l, d] : [a, o];
}
//#endregion
//#region node_modules/ol/render/canvas/TextBuilder.js
var qs = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Js = {
	left: 0,
	end: 0,
	center: .5,
	right: 1,
	start: 1,
	top: 0,
	middle: .5,
	hanging: .2,
	alphabetic: .8,
	ideographic: .8,
	bottom: 1
}, Ys = {
	Circle: Ws,
	Default: Rs,
	Image: Bs,
	LineString: Hs,
	Polygon: Ws,
	Text: function(e) {
		qs(t, e);
		function t(t, n, r, i) {
			var a = e.call(this, t, n, r, i) || this;
			return a.labels_ = null, a.text_ = "", a.textOffsetX_ = 0, a.textOffsetY_ = 0, a.textRotateWithView_ = void 0, a.textRotation_ = 0, a.textFillState_ = null, a.fillStates = {}, a.textStrokeState_ = null, a.strokeStates = {}, a.textState_ = {}, a.textStates = {}, a.textKey_ = "", a.fillKey_ = "", a.strokeKey_ = "", a.declutterImageWithText_ = void 0, a;
		}
		return t.prototype.finish = function() {
			var t = e.prototype.finish.call(this);
			return t.textStates = this.textStates, t.fillStates = this.fillStates, t.strokeStates = this.strokeStates, t;
		}, t.prototype.drawText = function(e, t) {
			var n = this.textFillState_, r = this.textStrokeState_, i = this.textState_;
			if (!(this.text_ === "" || !i || !n && !r)) {
				var a = this.coordinates, o = a.length, s = e.getType(), c = null, l = e.getStride();
				if (i.placement === Gs.LINE && (s == W.LINE_STRING || s == W.MULTI_LINE_STRING || s == W.POLYGON || s == W.MULTI_POLYGON)) {
					if (!Zt(this.getBufferedMaxExtent(), e.getExtent())) return;
					var u = void 0;
					if (c = e.getFlatCoordinates(), s == W.LINE_STRING) u = [c.length];
					else if (s == W.MULTI_LINE_STRING) u = e.getEnds();
					else if (s == W.POLYGON) u = e.getEnds().slice(0, 1);
					else if (s == W.MULTI_POLYGON) {
						var d = e.getEndss();
						u = [];
						for (var f = 0, p = d.length; f < p; ++f) u.push(d[f][0]);
					}
					this.beginGeometry(e, t);
					for (var m = i.textAlign, h = 0, g = void 0, _ = 0, v = u.length; _ < v; ++_) {
						if (m == null) {
							var y = Ks(i.maxAngle, c, h, u[_], l);
							h = y[0], g = y[1];
						} else g = u[_];
						for (var f = h; f < g; f += l) a.push(c[f], c[f + 1]);
						var b = a.length;
						h = u[_], this.drawChars_(o, b), o = b;
					}
					this.endGeometry(t);
				} else {
					var x = i.overflow ? null : [];
					switch (s) {
						case W.POINT:
						case W.MULTI_POINT:
							c = e.getFlatCoordinates();
							break;
						case W.LINE_STRING:
							c = e.getFlatMidpoint();
							break;
						case W.CIRCLE:
							c = e.getCenter();
							break;
						case W.MULTI_LINE_STRING:
							c = e.getFlatMidpoints(), l = 2;
							break;
						case W.POLYGON:
							c = e.getFlatInteriorPoint(), i.overflow || x.push(c[2] / this.resolution), l = 3;
							break;
						case W.MULTI_POLYGON:
							var S = e.getFlatInteriorPoints();
							c = [];
							for (var f = 0, p = S.length; f < p; f += 3) i.overflow || x.push(S[f + 2] / this.resolution), c.push(S[f], S[f + 1]);
							if (c.length === 0) return;
							l = 2;
							break;
						default:
					}
					var b = this.appendFlatPointCoordinates(c, l);
					if (b === o) return;
					if (x && (b - o) / 2 !== c.length / l) {
						var C = o / 2;
						x = x.filter(function(e, t) {
							var n = a[(C + t) * 2] === c[t * l] && a[(C + t) * 2 + 1] === c[t * l + 1];
							return n || --C, n;
						});
					}
					this.saveTextStates_(), (i.backgroundFill || i.backgroundStroke) && (this.setFillStrokeStyle(i.backgroundFill, i.backgroundStroke), i.backgroundFill && (this.updateFillStyle(this.state, this.createFill), this.hitDetectionInstructions.push(this.createFill(this.state))), i.backgroundStroke && (this.updateStrokeStyle(this.state, this.applyStroke), this.hitDetectionInstructions.push(this.createStroke(this.state)))), this.beginGeometry(e, t);
					var w = i.padding;
					if (w != Ar && (i.scale[0] < 0 || i.scale[1] < 0)) {
						var T = i.padding[0], E = i.padding[1], D = i.padding[2], O = i.padding[3];
						i.scale[0] < 0 && (E = -E, O = -O), i.scale[1] < 0 && (T = -T, D = -D), w = [
							T,
							E,
							D,
							O
						];
					}
					var k = this.pixelRatio;
					this.instructions.push([
						Z.DRAW_IMAGE,
						o,
						b,
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
						this.declutterImageWithText_,
						w == Ar ? Ar : w.map(function(e) {
							return e * k;
						}),
						!!i.backgroundFill,
						!!i.backgroundStroke,
						this.text_,
						this.textKey_,
						this.strokeKey_,
						this.fillKey_,
						this.textOffsetX_,
						this.textOffsetY_,
						x
					]);
					var A = 1 / k;
					this.hitDetectionInstructions.push([
						Z.DRAW_IMAGE,
						o,
						b,
						null,
						NaN,
						NaN,
						NaN,
						1,
						0,
						0,
						this.textRotateWithView_,
						this.textRotation_,
						[A, A],
						NaN,
						this.declutterImageWithText_,
						w,
						!!i.backgroundFill,
						!!i.backgroundStroke,
						this.text_,
						this.textKey_,
						this.strokeKey_,
						this.fillKey_,
						this.textOffsetX_,
						this.textOffsetY_,
						x
					]), this.endGeometry(t);
				}
			}
		}, t.prototype.saveTextStates_ = function() {
			var e = this.textStrokeState_, t = this.textState_, n = this.textFillState_, r = this.strokeKey_;
			e && (r in this.strokeStates || (this.strokeStates[r] = {
				strokeStyle: e.strokeStyle,
				lineCap: e.lineCap,
				lineDashOffset: e.lineDashOffset,
				lineWidth: e.lineWidth,
				lineJoin: e.lineJoin,
				miterLimit: e.miterLimit,
				lineDash: e.lineDash
			}));
			var i = this.textKey_;
			i in this.textStates || (this.textStates[i] = {
				font: t.font,
				textAlign: t.textAlign || "center",
				textBaseline: t.textBaseline || "middle",
				scale: t.scale
			});
			var a = this.fillKey_;
			n && (a in this.fillStates || (this.fillStates[a] = { fillStyle: n.fillStyle }));
		}, t.prototype.drawChars_ = function(e, t) {
			var n = this.textStrokeState_, r = this.textState_, i = this.strokeKey_, a = this.textKey_, o = this.fillKey_;
			this.saveTextStates_();
			var s = this.pixelRatio, c = Js[r.textBaseline], l = this.textOffsetY_ * s, u = this.text_, d = n ? n.lineWidth * Math.abs(r.scale[0]) / 2 : 0;
			this.instructions.push([
				Z.DRAW_CHARS,
				e,
				t,
				c,
				r.overflow,
				o,
				r.maxAngle,
				s,
				l,
				i,
				d * s,
				u,
				a,
				1
			]), this.hitDetectionInstructions.push([
				Z.DRAW_CHARS,
				e,
				t,
				c,
				r.overflow,
				o,
				r.maxAngle,
				1,
				l,
				i,
				d,
				u,
				a,
				1 / s
			]);
		}, t.prototype.setTextStyle = function(e, t) {
			var n, r, i;
			if (!e) this.text_ = "";
			else {
				var a = e.getFill();
				a ? (r = this.textFillState_, r || (r = {}, this.textFillState_ = r), r.fillStyle = gs(a.getColor() || "#000")) : (r = null, this.textFillState_ = r);
				var o = e.getStroke();
				if (!o) i = null, this.textStrokeState_ = i;
				else {
					i = this.textStrokeState_, i || (i = {}, this.textStrokeState_ = i);
					var s = o.getLineDash(), c = o.getLineDashOffset(), l = o.getWidth(), u = o.getMiterLimit();
					i.lineCap = o.getLineCap() || "round", i.lineDash = s ? s.slice() : Tr, i.lineDashOffset = c === void 0 ? 0 : c, i.lineJoin = o.getLineJoin() || "round", i.lineWidth = l === void 0 ? 1 : l, i.miterLimit = u === void 0 ? 10 : u, i.strokeStyle = gs(o.getColor() || "#000");
				}
				n = this.textState_;
				var d = e.getFont() || "10px sans-serif";
				Ir(d);
				var f = e.getScaleArray();
				n.overflow = e.getOverflow(), n.font = d, n.maxAngle = e.getMaxAngle(), n.placement = e.getPlacement(), n.textAlign = e.getTextAlign(), n.textBaseline = e.getTextBaseline() || "middle", n.backgroundFill = e.getBackgroundFill(), n.backgroundStroke = e.getBackgroundStroke(), n.padding = e.getPadding() || Ar, n.scale = f === void 0 ? [1, 1] : f;
				var p = e.getOffsetX(), m = e.getOffsetY(), h = e.getRotateWithView(), g = e.getRotation();
				this.text_ = e.getText() || "", this.textOffsetX_ = p === void 0 ? 0 : p, this.textOffsetY_ = m === void 0 ? 0 : m, this.textRotateWithView_ = h !== void 0 && h, this.textRotation_ = g === void 0 ? 0 : g, this.strokeKey_ = i ? (typeof i.strokeStyle == "string" ? i.strokeStyle : L(i.strokeStyle)) + i.lineCap + i.lineDashOffset + "|" + i.lineWidth + i.lineJoin + i.miterLimit + "[" + i.lineDash.join() + "]" : "", this.textKey_ = n.font + n.scale + (n.textAlign || "?") + (n.textBaseline || "?"), this.fillKey_ = r ? typeof r.fillStyle == "string" ? r.fillStyle : "|" + L(r.fillStyle) : "";
			}
			this.declutterImageWithText_ = t;
		}, t;
	}(Rs)
}, Xs = function() {
	function e(e, t, n, r) {
		this.tolerance_ = e, this.maxExtent_ = t, this.pixelRatio_ = r, this.resolution_ = n, this.buildersByZIndex_ = {};
	}
	return e.prototype.finish = function() {
		var e = {};
		for (var t in this.buildersByZIndex_) {
			e[t] = e[t] || {};
			var n = this.buildersByZIndex_[t];
			for (var r in n) {
				var i = n[r].finish();
				e[t][r] = i;
			}
		}
		return e;
	}, e.prototype.getBuilder = function(e, t) {
		var n = e === void 0 ? "0" : e.toString(), r = this.buildersByZIndex_[n];
		r === void 0 && (r = {}, this.buildersByZIndex_[n] = r);
		var i = r[t];
		if (i === void 0) {
			var a = Ys[t];
			i = new a(this.tolerance_, this.maxExtent_, this.resolution_, this.pixelRatio_), r[t] = i;
		}
		return i;
	}, e;
}(), Q = {
	CIRCLE: "Circle",
	DEFAULT: "Default",
	IMAGE: "Image",
	LINE_STRING: "LineString",
	POLYGON: "Polygon",
	TEXT: "Text"
};
//#endregion
//#region node_modules/ol/geom/flat/textpath.js
function Zs(e, t, n, r, i, a, o, s, c, l, u, d) {
	var f = e[t], p = e[t + 1], m = 0, h = 0, g = 0, _ = 0;
	function v() {
		m = f, h = p, t += r, f = e[t], p = e[t + 1], _ += g, g = Math.sqrt((f - m) * (f - m) + (p - h) * (p - h));
	}
	do
		v();
	while (t < n - r && _ + g < a);
	for (var y = g === 0 ? 0 : (a - _) / g, b = Ye(m, f, y), x = Ye(h, p, y), S = t - r, C = _, w = a + s * c(l, i, u); t < n - r && _ + g < w;) v();
	y = g === 0 ? 0 : (w - _) / g;
	var T = Ye(m, f, y), E = Ye(h, p, y), D;
	if (d) {
		var O = [
			b,
			x,
			T,
			E
		];
		Pi(O, 0, 4, 2, d, O, O), D = O[0] > O[2];
	} else D = b > T;
	var k = Math.PI, A = [], j = S + r === t;
	t = S, g = 0, _ = C, f = e[t], p = e[t + 1];
	var M;
	if (j) {
		v(), M = Math.atan2(p - h, f - m), D && (M += M > 0 ? -k : k);
		var N = (T + b) / 2, P = (E + x) / 2;
		return A[0] = [
			N,
			P,
			(w - a) / 2,
			M,
			i
		], A;
	}
	for (var F = 0, I = i.length; F < I;) {
		v();
		var L = Math.atan2(p - h, f - m);
		if (D && (L += L > 0 ? -k : k), M !== void 0) {
			var R = L - M;
			if (R += R > k ? -2 * k : R < -k ? 2 * k : 0, Math.abs(R) > o) return null;
		}
		M = L;
		for (var ee = F, te = 0; F < I; ++F) {
			var z = s * c(l, i[D ? I - F - 1 : F], u);
			if (t + r < n && _ + g < a + te + z / 2) break;
			te += z;
		}
		if (F !== ee) {
			var ne = D ? i.substring(I - ee, I - F) : i.substring(ee, F);
			y = g === 0 ? 0 : (a + te / 2 - _) / g;
			var N = Ye(m, f, y), P = Ye(h, p, y);
			A.push([
				N,
				P,
				te / 2,
				L,
				ne
			]), a += te;
		}
	}
	return A;
}
//#endregion
//#region node_modules/ol/render/canvas/Executor.js
var Qs = At(), $s = [], ec = [], tc = [], nc = [];
function rc(e) {
	return e[3].declutterBox;
}
var ic = /* @__PURE__ */ RegExp("[֑-ࣿיִ-﷿ﹰ-ﻼࠀ-࿿-]");
function ac(e, t) {
	return (t === "start" || t === "end") && !ic.test(e) && (t = t === "start" ? "left" : "right"), Js[t];
}
var oc = function() {
	function e(e, t, n, r) {
		this.overlaps = n, this.pixelRatio = t, this.resolution = e, this.alignFill_, this.instructions = r.instructions, this.coordinates = r.coordinates, this.coordinateCache_ = {}, this.renderedTransform_ = Rn(), this.hitDetectionInstructions = r.hitDetectionInstructions, this.pixelCoordinates_ = null, this.viewRotation_ = 0, this.fillStates = r.fillStates || {}, this.strokeStates = r.strokeStates || {}, this.textStates = r.textStates || {}, this.widths_ = {}, this.labels_ = {};
	}
	return e.prototype.createLabel = function(e, t, n, r) {
		var i = e + t + n + r;
		if (this.labels_[i]) return this.labels_[i];
		var a = r ? this.strokeStates[r] : null, o = n ? this.fillStates[n] : null, s = this.textStates[t], c = this.pixelRatio, l = [s.scale[0] * c, s.scale[1] * c], u = ac(e, s.textAlign || "center"), d = r && a.lineWidth ? a.lineWidth : 0, f = e.split("\n"), p = f.length, m = [], h = Vr(s.font, f, m), g = Lr(s.font), _ = g * p, v = h + d, y = [], b = (v + 2) * l[0], x = (_ + d) * l[1], S = {
			width: b < 0 ? Math.floor(b) : Math.ceil(b),
			height: x < 0 ? Math.floor(x) : Math.ceil(x),
			contextInstructions: y
		};
		(l[0] != 1 || l[1] != 1) && y.push("scale", l), y.push("font", s.font), r && (y.push("strokeStyle", a.strokeStyle), y.push("lineWidth", d), y.push("lineCap", a.lineCap), y.push("lineJoin", a.lineJoin), y.push("miterLimit", a.miterLimit), (ce ? OffscreenCanvasRenderingContext2D : CanvasRenderingContext2D).prototype.setLineDash && (y.push("setLineDash", [a.lineDash]), y.push("lineDashOffset", a.lineDashOffset))), n && y.push("fillStyle", o.fillStyle), y.push("textBaseline", "middle"), y.push("textAlign", "center");
		var C = .5 - u, w = u * v + C * d, T;
		if (r) for (T = 0; T < p; ++T) y.push("strokeText", [
			f[T],
			w + C * m[T],
			.5 * (d + g) + T * g
		]);
		if (n) for (T = 0; T < p; ++T) y.push("fillText", [
			f[T],
			w + C * m[T],
			.5 * (d + g) + T * g
		]);
		return this.labels_[i] = S, S;
	}, e.prototype.replayTextBackground_ = function(e, t, n, r, i, a, o) {
		e.beginPath(), e.moveTo.apply(e, t), e.lineTo.apply(e, n), e.lineTo.apply(e, r), e.lineTo.apply(e, i), e.lineTo.apply(e, t), a && (this.alignFill_ = a[2], this.fill_(e)), o && (this.setStrokeStyle_(e, o), e.stroke());
	}, e.prototype.calculateImageOrLabelDimensions_ = function(e, t, n, r, i, a, o, s, c, l, u, d, f, p, m, h) {
		o *= d[0], s *= d[1];
		var g = n - o, _ = r - s, v = i + c > e ? e - c : i, y = a + l > t ? t - l : a, b = p[3] + v * d[0] + p[1], x = p[0] + y * d[1] + p[2], S = g - p[3], C = _ - p[0];
		(m || u !== 0) && ($s[0] = S, nc[0] = S, $s[1] = C, ec[1] = C, ec[0] = S + b, tc[0] = ec[0], tc[1] = C + x, nc[1] = tc[1]);
		var w;
		return u === 0 ? jt(Math.min(S, S + b), Math.min(C, C + x), Math.max(S, S + b), Math.max(C, C + x), Qs) : (w = qn(Rn(), n, r, 1, 1, u, -n, -r), G(w, $s), G(w, ec), G(w, tc), G(w, nc), jt(Math.min($s[0], ec[0], tc[0], nc[0]), Math.min($s[1], ec[1], tc[1], nc[1]), Math.max($s[0], ec[0], tc[0], nc[0]), Math.max($s[1], ec[1], tc[1], nc[1]), Qs)), f && (g = Math.round(g), _ = Math.round(_)), {
			drawImageX: g,
			drawImageY: _,
			drawImageW: v,
			drawImageH: y,
			originX: c,
			originY: l,
			declutterBox: {
				minX: Qs[0],
				minY: Qs[1],
				maxX: Qs[2],
				maxY: Qs[3],
				value: h
			},
			canvasTransform: w,
			scale: d
		};
	}, e.prototype.replayImageOrLabel_ = function(e, t, n, r, i, a, o) {
		var s = !!(a || o), c = r.declutterBox, l = e.canvas, u = o ? o[2] * r.scale[0] / 2 : 0;
		return c.minX - u <= l.width / t && c.maxX + u >= 0 && c.minY - u <= l.height / t && c.maxY + u >= 0 && (s && this.replayTextBackground_(e, $s, ec, tc, nc, a, o), Hr(e, r.canvasTransform, i, n, r.originX, r.originY, r.drawImageW, r.drawImageH, r.drawImageX, r.drawImageY, r.scale)), !0;
	}, e.prototype.fill_ = function(e) {
		if (this.alignFill_) {
			var t = G(this.renderedTransform_, [0, 0]), n = 512 * this.pixelRatio;
			e.save(), e.translate(t[0] % n, t[1] % n), e.rotate(this.viewRotation_);
		}
		e.fill(), this.alignFill_ && e.restore();
	}, e.prototype.setStrokeStyle_ = function(e, t) {
		e.strokeStyle = t[1], e.lineWidth = t[2], e.lineCap = t[3], e.lineJoin = t[4], e.miterLimit = t[5], e.setLineDash && (e.lineDashOffset = t[7], e.setLineDash(t[6]));
	}, e.prototype.drawLabelWithPointPlacement_ = function(e, t, n, r) {
		var i = this.textStates[t], a = this.createLabel(e, t, r, n), o = this.strokeStates[n], s = this.pixelRatio, c = ac(e, i.textAlign || "center"), l = Js[i.textBaseline || "middle"], u = o && o.lineWidth ? o.lineWidth : 0;
		return {
			label: a,
			anchorX: c * (a.width / s - 2 * i.scale[0]) + 2 * (.5 - c) * u,
			anchorY: l * a.height / s + 2 * (.5 - l) * u
		};
	}, e.prototype.execute_ = function(e, t, n, r, i, a, o, s) {
		var c;
		this.pixelCoordinates_ && g(n, this.renderedTransform_) ? c = this.pixelCoordinates_ : (this.pixelCoordinates_ ||= [], c = Ni(this.coordinates, 0, this.coordinates.length, 2, n, this.pixelCoordinates_), Hn(this.renderedTransform_, n));
		for (var l = 0, u = r.length, d = 0, f, p, m, h, _, v, y, b, x, S, C, w, T = 0, E = 0, D = null, O = null, k = this.coordinateCache_, A = this.viewRotation_, j = Math.round(Math.atan2(-n[1], n[0]) * 0xe8d4a51000) / 0xe8d4a51000, M = {
			context: e,
			pixelRatio: this.pixelRatio,
			resolution: this.resolution,
			rotation: A
		}, N = this.instructions != r || this.overlaps ? 0 : 200, P, F, I, L; l < u;) {
			var R = r[l];
			switch (R[0]) {
				case Z.BEGIN_GEOMETRY:
					P = R[1], L = R[3], P.getGeometry() ? o !== void 0 && !Zt(o, L.getExtent()) ? l = R[2] + 1 : ++l : l = R[2];
					break;
				case Z.BEGIN_PATH:
					T > N && (this.fill_(e), T = 0), E > N && (e.stroke(), E = 0), !T && !E && (e.beginPath(), h = NaN, _ = NaN), ++l;
					break;
				case Z.CIRCLE:
					d = R[1];
					var ee = c[d], te = c[d + 1], z = c[d + 2], ne = c[d + 3], re = z - ee, ie = ne - te, ae = Math.sqrt(re * re + ie * ie);
					e.moveTo(ee + ae, te), e.arc(ee, te, ae, 0, 2 * Math.PI, !0), ++l;
					break;
				case Z.CLOSE_PATH:
					e.closePath(), ++l;
					break;
				case Z.CUSTOM:
					d = R[1], f = R[2];
					var oe = R[3], se = R[4], ce = R.length == 6 ? R[5] : void 0;
					M.geometry = oe, M.feature = P, l in k || (k[l] = []);
					var le = k[l];
					ce ? ce(c, d, f, 2, le) : (le[0] = c[d], le[1] = c[d + 1], le.length = 2), se(le, M), ++l;
					break;
				case Z.DRAW_IMAGE:
					d = R[1], f = R[2], b = R[3], p = R[4], m = R[5];
					var ue = R[6], de = R[7], fe = R[8], pe = R[9], me = R[10], he = R[11], ge = R[12], _e = R[13], ve = R[14];
					if (!b && R.length >= 19) {
						x = R[18], S = R[19], C = R[20], w = R[21];
						var ye = this.drawLabelWithPointPlacement_(x, S, C, w);
						b = ye.label, R[3] = b;
						var be = R[22];
						p = (ye.anchorX - be) * this.pixelRatio, R[4] = p;
						var xe = R[23];
						m = (ye.anchorY - xe) * this.pixelRatio, R[5] = m, ue = b.height, R[6] = ue, _e = b.width, R[13] = _e;
					}
					var Se = void 0;
					R.length > 24 && (Se = R[24]);
					var Ce = void 0, we = void 0, Te = void 0;
					R.length > 16 ? (Ce = R[15], we = R[16], Te = R[17]) : (Ce = Ar, we = !1, Te = !1), me && j ? he += A : !me && !j && (he -= A);
					for (var Ee = 0; d < f; d += 2) if (!(Se && Se[Ee++] < _e / this.pixelRatio)) {
						var De = this.calculateImageOrLabelDimensions_(b.width, b.height, c[d], c[d + 1], _e, ue, p, m, fe, pe, he, ge, i, Ce, we || Te, P), Oe = [
							e,
							t,
							b,
							De,
							de,
							we ? D : null,
							Te ? O : null
						], ke = void 0, Ae = void 0;
						if (s && ve) {
							var je = f - d;
							if (!ve[je]) {
								ve[je] = Oe;
								continue;
							}
							if (ke = ve[je], delete ve[je], Ae = rc(ke), s.collides(Ae)) continue;
						}
						s && s.collides(De.declutterBox) || (ke && (s && s.insert(Ae), this.replayImageOrLabel_.apply(this, ke)), s && s.insert(De.declutterBox), this.replayImageOrLabel_.apply(this, Oe));
					}
					++l;
					break;
				case Z.DRAW_CHARS:
					var Me = R[1], Ne = R[2], Pe = R[3], Fe = R[4];
					w = R[5];
					var Ie = R[6], Le = R[7], Re = R[8];
					C = R[9];
					var ze = R[10];
					x = R[11], S = R[12];
					var B = [R[13], R[13]], Be = this.textStates[S], Ve = Be.font, V = [Be.scale[0] * Le, Be.scale[1] * Le], He = void 0;
					Ve in this.widths_ ? He = this.widths_[Ve] : (He = {}, this.widths_[Ve] = He);
					var Ue = ps(c, Me, Ne, 2), We = Math.abs(V[0]) * Br(Ve, x, He);
					if (Fe || We <= Ue) {
						var Ge = this.textStates[S].textAlign, Ke = (Ue - We) * Js[Ge], qe = Zs(c, Me, Ne, 2, x, Ke, Ie, Math.abs(V[0]), Br, Ve, He, j ? 0 : this.viewRotation_);
						drawChars: if (qe) {
							var Je = [], Ye = void 0, Xe = void 0, Ze = void 0, Qe = void 0, $e = void 0;
							if (C) for (Ye = 0, Xe = qe.length; Ye < Xe; ++Ye) {
								$e = qe[Ye], Ze = $e[4], Qe = this.createLabel(Ze, S, "", C), p = $e[2] + (V[0] < 0 ? -ze : ze), m = Pe * Qe.height + (.5 - Pe) * 2 * ze * V[1] / V[0] - Re;
								var De = this.calculateImageOrLabelDimensions_(Qe.width, Qe.height, $e[0], $e[1], Qe.width, Qe.height, p, m, 0, 0, $e[3], B, !1, Ar, !1, P);
								if (s && s.collides(De.declutterBox)) break drawChars;
								Je.push([
									e,
									t,
									Qe,
									De,
									1,
									null,
									null
								]);
							}
							if (w) for (Ye = 0, Xe = qe.length; Ye < Xe; ++Ye) {
								$e = qe[Ye], Ze = $e[4], Qe = this.createLabel(Ze, S, w, ""), p = $e[2], m = Pe * Qe.height - Re;
								var De = this.calculateImageOrLabelDimensions_(Qe.width, Qe.height, $e[0], $e[1], Qe.width, Qe.height, p, m, 0, 0, $e[3], B, !1, Ar, !1, P);
								if (s && s.collides(De.declutterBox)) break drawChars;
								Je.push([
									e,
									t,
									Qe,
									De,
									1,
									null,
									null
								]);
							}
							s && s.load(Je.map(rc));
							for (var et = 0, tt = Je.length; et < tt; ++et) this.replayImageOrLabel_.apply(this, Je[et]);
						}
					}
					++l;
					break;
				case Z.END_GEOMETRY:
					if (a !== void 0) {
						P = R[1];
						var nt = a(P, L);
						if (nt) return nt;
					}
					++l;
					break;
				case Z.FILL:
					N ? T++ : this.fill_(e), ++l;
					break;
				case Z.MOVE_TO_LINE_TO:
					for (d = R[1], f = R[2], F = c[d], I = c[d + 1], v = F + .5 | 0, y = I + .5 | 0, (v !== h || y !== _) && (e.moveTo(F, I), h = v, _ = y), d += 2; d < f; d += 2) F = c[d], I = c[d + 1], v = F + .5 | 0, y = I + .5 | 0, (d == f - 2 || v !== h || y !== _) && (e.lineTo(F, I), h = v, _ = y);
					++l;
					break;
				case Z.SET_FILL_STYLE:
					D = R, this.alignFill_ = R[2], T && (this.fill_(e), T = 0, E &&= (e.stroke(), 0)), e.fillStyle = R[1], ++l;
					break;
				case Z.SET_STROKE_STYLE:
					O = R, E &&= (e.stroke(), 0), this.setStrokeStyle_(e, R), ++l;
					break;
				case Z.STROKE:
					N ? E++ : e.stroke(), ++l;
					break;
				default:
					++l;
					break;
			}
		}
		T && this.fill_(e), E && e.stroke();
	}, e.prototype.execute = function(e, t, n, r, i, a) {
		this.viewRotation_ = r, this.execute_(e, t, n, this.instructions, i, void 0, void 0, a);
	}, e.prototype.executeHitDetection = function(e, t, n, r, i) {
		return this.viewRotation_ = n, this.execute_(e, 1, t, this.hitDetectionInstructions, !0, r, i);
	}, e;
}(), sc = [
	Q.POLYGON,
	Q.CIRCLE,
	Q.LINE_STRING,
	Q.IMAGE,
	Q.TEXT,
	Q.DEFAULT
], cc = function() {
	function e(e, t, n, r, i, a) {
		this.maxExtent_ = e, this.overlaps_ = r, this.pixelRatio_ = n, this.resolution_ = t, this.renderBuffer_ = a, this.executorsByZIndex_ = {}, this.hitDetectionContext_ = null, this.hitDetectionTransform_ = Rn(), this.createExecutors_(i);
	}
	return e.prototype.clip = function(e, t) {
		var n = this.getClipCoords(t);
		e.beginPath(), e.moveTo(n[0], n[1]), e.lineTo(n[2], n[3]), e.lineTo(n[4], n[5]), e.lineTo(n[6], n[7]), e.clip();
	}, e.prototype.createExecutors_ = function(e) {
		for (var t in e) {
			var n = this.executorsByZIndex_[t];
			n === void 0 && (n = {}, this.executorsByZIndex_[t] = n);
			var r = e[t];
			for (var i in r) {
				var a = r[i];
				n[i] = new oc(this.resolution_, this.pixelRatio_, this.overlaps_, a);
			}
		}
	}, e.prototype.hasExecutors = function(e) {
		for (var t in this.executorsByZIndex_) for (var n = this.executorsByZIndex_[t], r = 0, i = e.length; r < i; ++r) if (e[r] in n) return !0;
		return !1;
	}, e.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i, a) {
		r = Math.round(r);
		var o = r * 2 + 1, s = qn(this.hitDetectionTransform_, r + .5, r + .5, 1 / t, -1 / t, -n, -e[0], -e[1]), c = !this.hitDetectionContext_;
		c && (this.hitDetectionContext_ = de(o, o));
		var l = this.hitDetectionContext_;
		l.canvas.width !== o || l.canvas.height !== o ? (l.canvas.width = o, l.canvas.height = o) : c || l.clearRect(0, 0, o, o);
		var u;
		this.renderBuffer_ !== void 0 && (u = At(), Lt(u, e), Ct(u, t * (this.renderBuffer_ + r), u));
		var d = uc(r), p;
		function m(e, t) {
			for (var n = l.getImageData(0, 0, o, o).data, s = 0, c = d.length; s < c; s++) if (n[d[s]] > 0) {
				if (!a || p !== Q.IMAGE && p !== Q.TEXT || a.indexOf(e) !== -1) {
					var u = (d[s] - 3) / 4, f = r - u % o, m = r - (u / o | 0), h = i(e, t, f * f + m * m);
					if (h) return h;
				}
				l.clearRect(0, 0, o, o);
				break;
			}
		}
		var h = Object.keys(this.executorsByZIndex_).map(Number);
		h.sort(f);
		var g, _, v, y, b;
		for (g = h.length - 1; g >= 0; --g) {
			var x = h[g].toString();
			for (v = this.executorsByZIndex_[x], _ = sc.length - 1; _ >= 0; --_) if (p = sc[_], y = v[p], y !== void 0 && (b = y.executeHitDetection(l, s, n, m, u), b)) return b;
		}
	}, e.prototype.getClipCoords = function(e) {
		var t = this.maxExtent_;
		if (!t) return null;
		var n = t[0], r = t[1], i = t[2], a = t[3], o = [
			n,
			r,
			n,
			a,
			i,
			a,
			i,
			r
		];
		return Ni(o, 0, 8, 2, e, o), o;
	}, e.prototype.isEmpty = function() {
		return T(this.executorsByZIndex_);
	}, e.prototype.execute = function(e, t, n, r, i, a, o) {
		var s = Object.keys(this.executorsByZIndex_).map(Number);
		s.sort(f), this.maxExtent_ && (e.save(), this.clip(e, n));
		var c = a || sc, l, u, d, p, m, h;
		for (o && s.reverse(), l = 0, u = s.length; l < u; ++l) {
			var g = s[l].toString();
			for (m = this.executorsByZIndex_[g], d = 0, p = c.length; d < p; ++d) {
				var _ = c[d];
				h = m[_], h !== void 0 && h.execute(e, t, n, r, i, o);
			}
		}
		this.maxExtent_ && e.restore();
	}, e;
}(), lc = {};
function uc(e) {
	if (lc[e] !== void 0) return lc[e];
	for (var t = e * 2 + 1, n = e * e, r = Array(n + 1), i = 0; i <= e; ++i) for (var a = 0; a <= e; ++a) {
		var o = i * i + a * a;
		if (o > n) break;
		var s = r[o];
		s || (s = [], r[o] = s), s.push(((e + i) * t + (e + a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e + a)) * 4 + 3), a > 0 && (s.push(((e + i) * t + (e - a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e - a)) * 4 + 3));
	}
	for (var c = [], i = 0, l = r.length; i < l; ++i) r[i] && c.push.apply(c, r[i]);
	return lc[e] = c, c;
}
//#endregion
//#region node_modules/ol/render/canvas/Immediate.js
var dc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), fc = function(e) {
	dc(t, e);
	function t(t, n, r, i, a, o, s) {
		var c = e.call(this) || this;
		return c.context_ = t, c.pixelRatio_ = n, c.extent_ = r, c.transform_ = i, c.viewRotation_ = a, c.squaredTolerance_ = o, c.userTransform_ = s, c.contextFillState_ = null, c.contextStrokeState_ = null, c.contextTextState_ = null, c.fillState_ = null, c.strokeState_ = null, c.image_ = null, c.imageAnchorX_ = 0, c.imageAnchorY_ = 0, c.imageHeight_ = 0, c.imageOpacity_ = 0, c.imageOriginX_ = 0, c.imageOriginY_ = 0, c.imageRotateWithView_ = !1, c.imageRotation_ = 0, c.imageScale_ = [0, 0], c.imageWidth_ = 0, c.text_ = "", c.textOffsetX_ = 0, c.textOffsetY_ = 0, c.textRotateWithView_ = !1, c.textRotation_ = 0, c.textScale_ = [0, 0], c.textFillState_ = null, c.textStrokeState_ = null, c.textState_ = null, c.pixelCoordinates_ = [], c.tmpLocalTransform_ = Rn(), c;
	}
	return t.prototype.drawImages_ = function(e, t, n, r) {
		if (this.image_) {
			var i = Ni(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.tmpLocalTransform_, s = a.globalAlpha;
			this.imageOpacity_ != 1 && (a.globalAlpha = s * this.imageOpacity_);
			var c = this.imageRotation_;
			this.imageRotateWithView_ && (c += this.viewRotation_);
			for (var l = 0, u = i.length; l < u; l += 2) {
				var d = i[l] - this.imageAnchorX_, f = i[l + 1] - this.imageAnchorY_;
				if (c !== 0 || this.imageScale_[0] != 1 || this.imageScale_[1] != 1) {
					var p = d + this.imageAnchorX_, m = f + this.imageAnchorY_;
					qn(o, p, m, 1, 1, c, -p, -m), a.setTransform.apply(a, o), a.translate(p, m), a.scale(this.imageScale_[0], this.imageScale_[1]), a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, -this.imageAnchorX_, -this.imageAnchorY_, this.imageWidth_, this.imageHeight_), a.setTransform(1, 0, 0, 1, 0, 0);
				} else a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, d, f, this.imageWidth_, this.imageHeight_);
			}
			this.imageOpacity_ != 1 && (a.globalAlpha = s);
		}
	}, t.prototype.drawText_ = function(e, t, n, r) {
		if (!(!this.textState_ || this.text_ === "")) {
			this.textFillState_ && this.setContextFillState_(this.textFillState_), this.textStrokeState_ && this.setContextStrokeState_(this.textStrokeState_), this.setContextTextState_(this.textState_);
			var i = Ni(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.textRotation_;
			for (this.textRotateWithView_ && (o += this.viewRotation_); t < n; t += r) {
				var s = i[t] + this.textOffsetX_, c = i[t + 1] + this.textOffsetY_;
				if (o !== 0 || this.textScale_[0] != 1 || this.textScale_[1] != 1) {
					var l = qn(this.tmpLocalTransform_, s, c, 1, 1, o, -s, -c);
					a.setTransform.apply(a, l), a.translate(s, c), a.scale(this.textScale_[0], this.textScale_[1]), this.textStrokeState_ && a.strokeText(this.text_, 0, 0), this.textFillState_ && a.fillText(this.text_, 0, 0), a.setTransform(1, 0, 0, 1, 0, 0);
				} else this.textStrokeState_ && a.strokeText(this.text_, s, c), this.textFillState_ && a.fillText(this.text_, s, c);
			}
		}
	}, t.prototype.moveToLineTo_ = function(e, t, n, r, i) {
		var a = this.context_, o = Ni(e, t, n, r, this.transform_, this.pixelCoordinates_);
		a.moveTo(o[0], o[1]);
		var s = o.length;
		i && (s -= 2);
		for (var c = 2; c < s; c += 2) a.lineTo(o[c], o[c + 1]);
		return i && a.closePath(), n;
	}, t.prototype.drawRings_ = function(e, t, n, r) {
		for (var i = 0, a = n.length; i < a; ++i) t = this.moveToLineTo_(e, t, n[i], r, !0);
		return t;
	}, t.prototype.drawCircle = function(e) {
		if (Zt(this.extent_, e.getExtent())) {
			if (this.fillState_ || this.strokeState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				var t = Wi(e, this.transform_, this.pixelCoordinates_), n = t[2] - t[0], r = t[3] - t[1], i = Math.sqrt(n * n + r * r), a = this.context_;
				a.beginPath(), a.arc(t[0], t[1], i, 0, 2 * Math.PI), this.fillState_ && a.fill(), this.strokeState_ && a.stroke();
			}
			this.text_ !== "" && this.drawText_(e.getCenter(), 0, 2, 2);
		}
	}, t.prototype.setStyle = function(e) {
		this.setFillStrokeStyle(e.getFill(), e.getStroke()), this.setImageStyle(e.getImage()), this.setTextStyle(e.getText());
	}, t.prototype.setTransform = function(e) {
		this.transform_ = e;
	}, t.prototype.drawGeometry = function(e) {
		switch (e.getType()) {
			case W.POINT:
				this.drawPoint(e);
				break;
			case W.LINE_STRING:
				this.drawLineString(e);
				break;
			case W.POLYGON:
				this.drawPolygon(e);
				break;
			case W.MULTI_POINT:
				this.drawMultiPoint(e);
				break;
			case W.MULTI_LINE_STRING:
				this.drawMultiLineString(e);
				break;
			case W.MULTI_POLYGON:
				this.drawMultiPolygon(e);
				break;
			case W.GEOMETRY_COLLECTION:
				this.drawGeometryCollection(e);
				break;
			case W.CIRCLE:
				this.drawCircle(e);
				break;
			default:
		}
	}, t.prototype.drawFeature = function(e, t) {
		var n = t.getGeometryFunction()(e);
		!n || !Zt(this.extent_, n.getExtent()) || (this.setStyle(t), this.drawGeometry(n));
	}, t.prototype.drawGeometryCollection = function(e) {
		for (var t = e.getGeometriesArray(), n = 0, r = t.length; n < r; ++n) this.drawGeometry(t[n]);
	}, t.prototype.drawPoint = function(e) {
		this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_));
		var t = e.getFlatCoordinates(), n = e.getStride();
		this.image_ && this.drawImages_(t, 0, t.length, n), this.text_ !== "" && this.drawText_(t, 0, t.length, n);
	}, t.prototype.drawMultiPoint = function(e) {
		this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_));
		var t = e.getFlatCoordinates(), n = e.getStride();
		this.image_ && this.drawImages_(t, 0, t.length, n), this.text_ !== "" && this.drawText_(t, 0, t.length, n);
	}, t.prototype.drawLineString = function(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), Zt(this.extent_, e.getExtent())) {
			if (this.strokeState_) {
				this.setContextStrokeState_(this.strokeState_);
				var t = this.context_, n = e.getFlatCoordinates();
				t.beginPath(), this.moveToLineTo_(n, 0, n.length, e.getStride(), !1), t.stroke();
			}
			if (this.text_ !== "") {
				var r = e.getFlatMidpoint();
				this.drawText_(r, 0, 2, 2);
			}
		}
	}, t.prototype.drawMultiLineString = function(e) {
		this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_));
		var t = e.getExtent();
		if (Zt(this.extent_, t)) {
			if (this.strokeState_) {
				this.setContextStrokeState_(this.strokeState_);
				var n = this.context_, r = e.getFlatCoordinates(), i = 0, a = e.getEnds(), o = e.getStride();
				n.beginPath();
				for (var s = 0, c = a.length; s < c; ++s) i = this.moveToLineTo_(r, i, a[s], o, !1);
				n.stroke();
			}
			if (this.text_ !== "") {
				var l = e.getFlatMidpoints();
				this.drawText_(l, 0, l.length, 2);
			}
		}
	}, t.prototype.drawPolygon = function(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), Zt(this.extent_, e.getExtent())) {
			if (this.strokeState_ || this.fillState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				var t = this.context_;
				t.beginPath(), this.drawRings_(e.getOrientedFlatCoordinates(), 0, e.getEnds(), e.getStride()), this.fillState_ && t.fill(), this.strokeState_ && t.stroke();
			}
			if (this.text_ !== "") {
				var n = e.getFlatInteriorPoint();
				this.drawText_(n, 0, 2, 2);
			}
		}
	}, t.prototype.drawMultiPolygon = function(e) {
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), Zt(this.extent_, e.getExtent())) {
			if (this.strokeState_ || this.fillState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				var t = this.context_, n = e.getOrientedFlatCoordinates(), r = 0, i = e.getEndss(), a = e.getStride();
				t.beginPath();
				for (var o = 0, s = i.length; o < s; ++o) {
					var c = i[o];
					r = this.drawRings_(n, r, c, a);
				}
				this.fillState_ && t.fill(), this.strokeState_ && t.stroke();
			}
			if (this.text_ !== "") {
				var l = e.getFlatInteriorPoints();
				this.drawText_(l, 0, l.length, 2);
			}
		}
	}, t.prototype.setContextFillState_ = function(e) {
		var t = this.context_, n = this.contextFillState_;
		n ? n.fillStyle != e.fillStyle && (n.fillStyle = e.fillStyle, t.fillStyle = e.fillStyle) : (t.fillStyle = e.fillStyle, this.contextFillState_ = { fillStyle: e.fillStyle });
	}, t.prototype.setContextStrokeState_ = function(e) {
		var t = this.context_, n = this.contextStrokeState_;
		n ? (n.lineCap != e.lineCap && (n.lineCap = e.lineCap, t.lineCap = e.lineCap), t.setLineDash && (g(n.lineDash, e.lineDash) || t.setLineDash(n.lineDash = e.lineDash), n.lineDashOffset != e.lineDashOffset && (n.lineDashOffset = e.lineDashOffset, t.lineDashOffset = e.lineDashOffset)), n.lineJoin != e.lineJoin && (n.lineJoin = e.lineJoin, t.lineJoin = e.lineJoin), n.lineWidth != e.lineWidth && (n.lineWidth = e.lineWidth, t.lineWidth = e.lineWidth), n.miterLimit != e.miterLimit && (n.miterLimit = e.miterLimit, t.miterLimit = e.miterLimit), n.strokeStyle != e.strokeStyle && (n.strokeStyle = e.strokeStyle, t.strokeStyle = e.strokeStyle)) : (t.lineCap = e.lineCap, t.setLineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.lineWidth = e.lineWidth, t.miterLimit = e.miterLimit, t.strokeStyle = e.strokeStyle, this.contextStrokeState_ = {
			lineCap: e.lineCap,
			lineDash: e.lineDash,
			lineDashOffset: e.lineDashOffset,
			lineJoin: e.lineJoin,
			lineWidth: e.lineWidth,
			miterLimit: e.miterLimit,
			strokeStyle: e.strokeStyle
		});
	}, t.prototype.setContextTextState_ = function(e) {
		var t = this.context_, n = this.contextTextState_, r = e.textAlign ? e.textAlign : Or;
		n ? (n.font != e.font && (n.font = e.font, t.font = e.font), n.textAlign != r && (n.textAlign = r, t.textAlign = r), n.textBaseline != e.textBaseline && (n.textBaseline = e.textBaseline, t.textBaseline = e.textBaseline)) : (t.font = e.font, t.textAlign = r, t.textBaseline = e.textBaseline, this.contextTextState_ = {
			font: e.font,
			textAlign: r,
			textBaseline: e.textBaseline
		});
	}, t.prototype.setFillStrokeStyle = function(e, t) {
		var n = this;
		if (!e) this.fillState_ = null;
		else {
			var r = e.getColor();
			this.fillState_ = { fillStyle: gs(r || Cr) };
		}
		if (!t) this.strokeState_ = null;
		else {
			var i = t.getColor(), a = t.getLineCap(), o = t.getLineDash(), s = t.getLineDashOffset(), c = t.getLineJoin(), l = t.getWidth(), u = t.getMiterLimit(), d = o || Tr;
			this.strokeState_ = {
				lineCap: a === void 0 ? wr : a,
				lineDash: this.pixelRatio_ === 1 ? d : d.map(function(e) {
					return e * n.pixelRatio_;
				}),
				lineDashOffset: (s || 0) * this.pixelRatio_,
				lineJoin: c === void 0 ? Er : c,
				lineWidth: (l === void 0 ? 1 : l) * this.pixelRatio_,
				miterLimit: u === void 0 ? 10 : u,
				strokeStyle: gs(i || Dr)
			};
		}
	}, t.prototype.setImageStyle = function(e) {
		var t;
		if (!e || !(t = e.getSize())) {
			this.image_ = null;
			return;
		}
		var n = e.getAnchor(), r = e.getOrigin();
		this.image_ = e.getImage(this.pixelRatio_), this.imageAnchorX_ = n[0] * this.pixelRatio_, this.imageAnchorY_ = n[1] * this.pixelRatio_, this.imageHeight_ = t[1] * this.pixelRatio_, this.imageOpacity_ = e.getOpacity(), this.imageOriginX_ = r[0], this.imageOriginY_ = r[1], this.imageRotateWithView_ = e.getRotateWithView(), this.imageRotation_ = e.getRotation(), this.imageScale_ = e.getScaleArray(), this.imageWidth_ = t[0] * this.pixelRatio_;
	}, t.prototype.setTextStyle = function(e) {
		if (!e) this.text_ = "";
		else {
			var t = e.getFill();
			if (!t) this.textFillState_ = null;
			else {
				var n = t.getColor();
				this.textFillState_ = { fillStyle: gs(n || Cr) };
			}
			var r = e.getStroke();
			if (!r) this.textStrokeState_ = null;
			else {
				var i = r.getColor(), a = r.getLineCap(), o = r.getLineDash(), s = r.getLineDashOffset(), c = r.getLineJoin(), l = r.getWidth(), u = r.getMiterLimit();
				this.textStrokeState_ = {
					lineCap: a === void 0 ? wr : a,
					lineDash: o || Tr,
					lineDashOffset: s || 0,
					lineJoin: c === void 0 ? Er : c,
					lineWidth: l === void 0 ? 1 : l,
					miterLimit: u === void 0 ? 10 : u,
					strokeStyle: gs(i || Dr)
				};
			}
			var d = e.getFont(), f = e.getOffsetX(), p = e.getOffsetY(), m = e.getRotateWithView(), h = e.getRotation(), g = e.getScaleArray(), _ = e.getText(), v = e.getTextAlign(), y = e.getTextBaseline();
			this.textState_ = {
				font: d === void 0 ? Sr : d,
				textAlign: v === void 0 ? Or : v,
				textBaseline: y === void 0 ? kr : y
			}, this.text_ = _ === void 0 ? "" : _, this.textOffsetX_ = f === void 0 ? 0 : this.pixelRatio_ * f, this.textOffsetY_ = p === void 0 ? 0 : this.pixelRatio_ * p, this.textRotateWithView_ = m !== void 0 && m, this.textRotation_ = h === void 0 ? 0 : h, this.textScale_ = [this.pixelRatio_ * g[0], this.pixelRatio_ * g[1]];
		}
	}, t;
}(Is), pc = {
	FRACTION: "fraction",
	PIXELS: "pixels"
}, mc = {
	BOTTOM_LEFT: "bottom-left",
	BOTTOM_RIGHT: "bottom-right",
	TOP_LEFT: "top-left",
	TOP_RIGHT: "top-right"
}, hc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), gc = function(e) {
	hc(t, e);
	function t(t, n, r, i) {
		var a = e.call(this) || this;
		return a.extent = t, a.pixelRatio_ = r, a.resolution = n, a.state = i, a;
	}
	return t.prototype.changed = function() {
		this.dispatchEvent(O.CHANGE);
	}, t.prototype.getExtent = function() {
		return this.extent;
	}, t.prototype.getImage = function() {
		return F();
	}, t.prototype.getPixelRatio = function() {
		return this.pixelRatio_;
	}, t.prototype.getResolution = function() {
		return this.resolution;
	}, t.prototype.getState = function() {
		return this.state;
	}, t.prototype.load = function() {
		F();
	}, t;
}(D), _c = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})();
(function(e) {
	_c(t, e);
	function t(t, n, r, i, a, o) {
		var s = e.call(this, t, n, r, X.IDLE) || this;
		return s.src_ = i, s.image_ = new Image(), a !== null && (s.image_.crossOrigin = a), s.unlisten_ = null, s.state = X.IDLE, s.imageLoadFunction_ = o, s;
	}
	return t.prototype.getImage = function() {
		return this.image_;
	}, t.prototype.handleImageError_ = function() {
		this.state = X.ERROR, this.unlistenImage_(), this.changed();
	}, t.prototype.handleImageLoad_ = function() {
		this.resolution === void 0 && (this.resolution = qt(this.extent) / this.image_.height), this.state = X.LOADED, this.unlistenImage_(), this.changed();
	}, t.prototype.load = function() {
		(this.state == X.IDLE || this.state == X.ERROR) && (this.state = X.LOADING, this.changed(), this.imageLoadFunction_(this, this.src_), this.unlisten_ = vc(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
	}, t.prototype.setImage = function(e) {
		this.image_ = e, this.resolution = qt(this.extent) / this.image_.height;
	}, t.prototype.unlistenImage_ = function() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}, t;
})(gc);
function vc(e, t, n) {
	var r = e;
	if (r.src && le) {
		var i = r.decode(), a = !0;
		return i.then(function() {
			a && t();
		}).catch(function(e) {
			a && (e.name === "EncodingError" && e.message === "Invalid image type." ? t() : n());
		}), function() {
			a = !1;
		};
	}
	var o = [A(r, O.LOAD, t), A(r, O.ERROR, n)];
	return function() {
		o.forEach(j);
	};
}
//#endregion
//#region node_modules/ol/style/IconImage.js
var yc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), bc = null, xc = function(e) {
	yc(t, e);
	function t(t, n, r, i, a, o) {
		var s = e.call(this) || this;
		return s.hitDetectionImage_ = null, s.image_ = t || new Image(), i !== null && (s.image_.crossOrigin = i), s.canvas_ = {}, s.color_ = o, s.unlisten_ = null, s.imageState_ = a, s.size_ = r, s.src_ = n, s.tainted_, s;
	}
	return t.prototype.isTainted_ = function() {
		if (this.tainted_ === void 0 && this.imageState_ === X.LOADED) {
			bc ||= de(1, 1), bc.drawImage(this.image_, 0, 0);
			try {
				bc.getImageData(0, 0, 1, 1), this.tainted_ = !1;
			} catch {
				bc = null, this.tainted_ = !0;
			}
		}
		return this.tainted_ === !0;
	}, t.prototype.dispatchChangeEvent_ = function() {
		this.dispatchEvent(O.CHANGE);
	}, t.prototype.handleImageError_ = function() {
		this.imageState_ = X.ERROR, this.unlistenImage_(), this.dispatchChangeEvent_();
	}, t.prototype.handleImageLoad_ = function() {
		this.imageState_ = X.LOADED, this.size_ ? (this.image_.width = this.size_[0], this.image_.height = this.size_[1]) : this.size_ = [this.image_.width, this.image_.height], this.unlistenImage_(), this.dispatchChangeEvent_();
	}, t.prototype.getImage = function(e) {
		return this.replaceColor_(e), this.canvas_[e] ? this.canvas_[e] : this.image_;
	}, t.prototype.getPixelRatio = function(e) {
		return this.replaceColor_(e), this.canvas_[e] ? e : 1;
	}, t.prototype.getImageState = function() {
		return this.imageState_;
	}, t.prototype.getHitDetectionImage = function() {
		if (!this.hitDetectionImage_) if (this.isTainted_()) {
			var e = this.size_[0], t = this.size_[1], n = de(e, t);
			n.fillRect(0, 0, e, t), this.hitDetectionImage_ = n.canvas;
		} else this.hitDetectionImage_ = this.image_;
		return this.hitDetectionImage_;
	}, t.prototype.getSize = function() {
		return this.size_;
	}, t.prototype.getSrc = function() {
		return this.src_;
	}, t.prototype.load = function() {
		if (this.imageState_ == X.IDLE) {
			this.imageState_ = X.LOADING;
			try {
				this.image_.src = this.src_;
			} catch {
				this.handleImageError_();
			}
			this.unlisten_ = vc(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this));
		}
	}, t.prototype.replaceColor_ = function(e) {
		if (!(!this.color_ || this.canvas_[e] || this.imageState_ !== X.LOADED)) {
			var t = document.createElement("canvas");
			this.canvas_[e] = t, t.width = Math.ceil(this.image_.width * e), t.height = Math.ceil(this.image_.height * e);
			var n = t.getContext("2d");
			if (n.scale(e, e), n.drawImage(this.image_, 0, 0), n.globalCompositeOperation = "multiply", n.globalCompositeOperation === "multiply" || this.isTainted_()) n.fillStyle = er(this.color_), n.fillRect(0, 0, t.width / e, t.height / e), n.globalCompositeOperation = "destination-in", n.drawImage(this.image_, 0, 0);
			else {
				for (var r = n.getImageData(0, 0, t.width, t.height), i = r.data, a = this.color_[0] / 255, o = this.color_[1] / 255, s = this.color_[2] / 255, c = this.color_[3], l = 0, u = i.length; l < u; l += 4) i[l] *= a, i[l + 1] *= o, i[l + 2] *= s, i[l + 3] *= c;
				n.putImageData(r, 0, 0);
			}
		}
	}, t.prototype.unlistenImage_ = function() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}, t;
}(D);
function Sc(e, t, n, r, i, a) {
	var o = lr.get(t, r, a);
	return o || (o = new xc(e, t, n, r, i, a), lr.set(t, r, a, o)), o;
}
//#endregion
//#region node_modules/ol/style/Icon.js
var Cc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), wc = function(e) {
	Cc(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r.opacity === void 0 ? 1 : r.opacity, a = r.rotation === void 0 ? 0 : r.rotation, o = r.scale === void 0 ? 1 : r.scale, s = r.rotateWithView !== void 0 && r.rotateWithView;
		n = e.call(this, {
			opacity: i,
			rotation: a,
			scale: o,
			displacement: r.displacement === void 0 ? [0, 0] : r.displacement,
			rotateWithView: s
		}) || this, n.anchor_ = r.anchor === void 0 ? [.5, .5] : r.anchor, n.normalizedAnchor_ = null, n.anchorOrigin_ = r.anchorOrigin === void 0 ? mc.TOP_LEFT : r.anchorOrigin, n.anchorXUnits_ = r.anchorXUnits === void 0 ? pc.FRACTION : r.anchorXUnits, n.anchorYUnits_ = r.anchorYUnits === void 0 ? pc.FRACTION : r.anchorYUnits, n.crossOrigin_ = r.crossOrigin === void 0 ? null : r.crossOrigin;
		var c = r.img === void 0 ? null : r.img, l = r.imgSize === void 0 ? null : r.imgSize, u = r.src;
		H(!(u !== void 0 && c), 4), H(!c || c && l, 5), (u === void 0 || u.length === 0) && c && (u = c.src || L(c)), H(u !== void 0 && u.length > 0, 6);
		var d = r.src === void 0 ? X.LOADED : X.IDLE;
		return n.color_ = r.color === void 0 ? null : rr(r.color), n.iconImage_ = Sc(c, u, l, n.crossOrigin_, d, n.color_), n.offset_ = r.offset === void 0 ? [0, 0] : r.offset, n.offsetOrigin_ = r.offsetOrigin === void 0 ? mc.TOP_LEFT : r.offsetOrigin, n.origin_ = null, n.size_ = r.size === void 0 ? null : r.size, n;
	}
	return t.prototype.clone = function() {
		var e = this.getScale();
		return new t({
			anchor: this.anchor_.slice(),
			anchorOrigin: this.anchorOrigin_,
			anchorXUnits: this.anchorXUnits_,
			anchorYUnits: this.anchorYUnits_,
			crossOrigin: this.crossOrigin_,
			color: this.color_ && this.color_.slice ? this.color_.slice() : this.color_ || void 0,
			src: this.getSrc(),
			offset: this.offset_.slice(),
			offsetOrigin: this.offsetOrigin_,
			size: this.size_ === null ? void 0 : this.size_.slice(),
			opacity: this.getOpacity(),
			scale: Array.isArray(e) ? e.slice() : e,
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView()
		});
	}, t.prototype.getAnchor = function() {
		if (this.normalizedAnchor_) return this.normalizedAnchor_;
		var e = this.anchor_, t = this.getSize();
		if (this.anchorXUnits_ == pc.FRACTION || this.anchorYUnits_ == pc.FRACTION) {
			if (!t) return null;
			e = this.anchor_.slice(), this.anchorXUnits_ == pc.FRACTION && (e[0] *= t[0]), this.anchorYUnits_ == pc.FRACTION && (e[1] *= t[1]);
		}
		if (this.anchorOrigin_ != mc.TOP_LEFT) {
			if (!t) return null;
			e === this.anchor_ && (e = this.anchor_.slice()), (this.anchorOrigin_ == mc.TOP_RIGHT || this.anchorOrigin_ == mc.BOTTOM_RIGHT) && (e[0] = -e[0] + t[0]), (this.anchorOrigin_ == mc.BOTTOM_LEFT || this.anchorOrigin_ == mc.BOTTOM_RIGHT) && (e[1] = -e[1] + t[1]);
		}
		var n = this.getDisplacement();
		return e[0] -= n[0], e[1] += n[1], this.normalizedAnchor_ = e, this.normalizedAnchor_;
	}, t.prototype.setAnchor = function(e) {
		this.anchor_ = e, this.normalizedAnchor_ = null;
	}, t.prototype.getColor = function() {
		return this.color_;
	}, t.prototype.getImage = function(e) {
		return this.iconImage_.getImage(e);
	}, t.prototype.getPixelRatio = function(e) {
		return this.iconImage_.getPixelRatio(e);
	}, t.prototype.getImageSize = function() {
		return this.iconImage_.getSize();
	}, t.prototype.getImageState = function() {
		return this.iconImage_.getImageState();
	}, t.prototype.getHitDetectionImage = function() {
		return this.iconImage_.getHitDetectionImage();
	}, t.prototype.getOrigin = function() {
		if (this.origin_) return this.origin_;
		var e = this.offset_;
		if (this.offsetOrigin_ != mc.TOP_LEFT) {
			var t = this.getSize(), n = this.iconImage_.getSize();
			if (!t || !n) return null;
			e = e.slice(), (this.offsetOrigin_ == mc.TOP_RIGHT || this.offsetOrigin_ == mc.BOTTOM_RIGHT) && (e[0] = n[0] - t[0] - e[0]), (this.offsetOrigin_ == mc.BOTTOM_LEFT || this.offsetOrigin_ == mc.BOTTOM_RIGHT) && (e[1] = n[1] - t[1] - e[1]);
		}
		return this.origin_ = e, this.origin_;
	}, t.prototype.getSrc = function() {
		return this.iconImage_.getSrc();
	}, t.prototype.getSize = function() {
		return this.size_ ? this.size_ : this.iconImage_.getSize();
	}, t.prototype.listenImageChange = function(e) {
		this.iconImage_.addEventListener(O.CHANGE, e);
	}, t.prototype.load = function() {
		this.iconImage_.load();
	}, t.prototype.unlistenImageChange = function(e) {
		this.iconImage_.removeEventListener(O.CHANGE, e);
	}, t;
}(hs), Tc = "#333", Ec = function() {
	function e(e) {
		var t = e || {};
		this.font_ = t.font, this.rotation_ = t.rotation, this.rotateWithView_ = t.rotateWithView, this.scale_ = t.scale, this.scaleArray_ = Ra(t.scale === void 0 ? 1 : t.scale), this.text_ = t.text, this.textAlign_ = t.textAlign, this.textBaseline_ = t.textBaseline, this.fill_ = t.fill === void 0 ? new xs({ color: Tc }) : t.fill, this.maxAngle_ = t.maxAngle === void 0 ? Math.PI / 4 : t.maxAngle, this.placement_ = t.placement === void 0 ? Gs.POINT : t.placement, this.overflow_ = !!t.overflow, this.stroke_ = t.stroke === void 0 ? null : t.stroke, this.offsetX_ = t.offsetX === void 0 ? 0 : t.offsetX, this.offsetY_ = t.offsetY === void 0 ? 0 : t.offsetY, this.backgroundFill_ = t.backgroundFill ? t.backgroundFill : null, this.backgroundStroke_ = t.backgroundStroke ? t.backgroundStroke : null, this.padding_ = t.padding === void 0 ? null : t.padding;
	}
	return e.prototype.clone = function() {
		var t = this.getScale();
		return new e({
			font: this.getFont(),
			placement: this.getPlacement(),
			maxAngle: this.getMaxAngle(),
			overflow: this.getOverflow(),
			rotation: this.getRotation(),
			rotateWithView: this.getRotateWithView(),
			scale: Array.isArray(t) ? t.slice() : t,
			text: this.getText(),
			textAlign: this.getTextAlign(),
			textBaseline: this.getTextBaseline(),
			fill: this.getFill() ? this.getFill().clone() : void 0,
			stroke: this.getStroke() ? this.getStroke().clone() : void 0,
			offsetX: this.getOffsetX(),
			offsetY: this.getOffsetY(),
			backgroundFill: this.getBackgroundFill() ? this.getBackgroundFill().clone() : void 0,
			backgroundStroke: this.getBackgroundStroke() ? this.getBackgroundStroke().clone() : void 0,
			padding: this.getPadding()
		});
	}, e.prototype.getOverflow = function() {
		return this.overflow_;
	}, e.prototype.getFont = function() {
		return this.font_;
	}, e.prototype.getMaxAngle = function() {
		return this.maxAngle_;
	}, e.prototype.getPlacement = function() {
		return this.placement_;
	}, e.prototype.getOffsetX = function() {
		return this.offsetX_;
	}, e.prototype.getOffsetY = function() {
		return this.offsetY_;
	}, e.prototype.getFill = function() {
		return this.fill_;
	}, e.prototype.getRotateWithView = function() {
		return this.rotateWithView_;
	}, e.prototype.getRotation = function() {
		return this.rotation_;
	}, e.prototype.getScale = function() {
		return this.scale_;
	}, e.prototype.getScaleArray = function() {
		return this.scaleArray_;
	}, e.prototype.getStroke = function() {
		return this.stroke_;
	}, e.prototype.getText = function() {
		return this.text_;
	}, e.prototype.getTextAlign = function() {
		return this.textAlign_;
	}, e.prototype.getTextBaseline = function() {
		return this.textBaseline_;
	}, e.prototype.getBackgroundFill = function() {
		return this.backgroundFill_;
	}, e.prototype.getBackgroundStroke = function() {
		return this.backgroundStroke_;
	}, e.prototype.getPadding = function() {
		return this.padding_;
	}, e.prototype.setOverflow = function(e) {
		this.overflow_ = e;
	}, e.prototype.setFont = function(e) {
		this.font_ = e;
	}, e.prototype.setMaxAngle = function(e) {
		this.maxAngle_ = e;
	}, e.prototype.setOffsetX = function(e) {
		this.offsetX_ = e;
	}, e.prototype.setOffsetY = function(e) {
		this.offsetY_ = e;
	}, e.prototype.setPlacement = function(e) {
		this.placement_ = e;
	}, e.prototype.setRotateWithView = function(e) {
		this.rotateWithView_ = e;
	}, e.prototype.setFill = function(e) {
		this.fill_ = e;
	}, e.prototype.setRotation = function(e) {
		this.rotation_ = e;
	}, e.prototype.setScale = function(e) {
		this.scale_ = e, this.scaleArray_ = Ra(e === void 0 ? 1 : e);
	}, e.prototype.setStroke = function(e) {
		this.stroke_ = e;
	}, e.prototype.setText = function(e) {
		this.text_ = e;
	}, e.prototype.setTextAlign = function(e) {
		this.textAlign_ = e;
	}, e.prototype.setTextBaseline = function(e) {
		this.textBaseline_ = e;
	}, e.prototype.setBackgroundFill = function(e) {
		this.backgroundFill_ = e;
	}, e.prototype.setBackgroundStroke = function(e) {
		this.backgroundStroke_ = e;
	}, e.prototype.setPadding = function(e) {
		this.padding_ = e;
	}, e;
}(), Dc = .5;
function Oc(e, t, n, r, i, a, o) {
	var s = de(e[0] * Dc, e[1] * Dc);
	s.imageSmoothingEnabled = !1;
	for (var c = s.canvas, l = new fc(s, Dc, i, null, o), u = n.length, d = Math.floor((256 * 256 * 256 - 1) / u), p = {}, m = 1; m <= u; ++m) {
		var h = n[m - 1], g = h.getStyleFunction() || r;
		if (r) {
			var _ = g(h, a);
			if (_) {
				Array.isArray(_) || (_ = [_]);
				for (var v = "#" + ("000000" + (m * d).toString(16)).slice(-6), y = 0, b = _.length; y < b; ++y) {
					var x = _[y], S = x.getGeometryFunction()(h);
					if (!(!S || !Zt(i, S.getExtent()))) {
						var C = x.clone(), w = C.getFill();
						w && w.setColor(v);
						var T = C.getStroke();
						T && (T.setColor(v), T.setLineDash(null)), C.setText(void 0);
						var E = x.getImage();
						if (E && E.getOpacity() !== 0) {
							var D = E.getImageSize();
							if (!D) continue;
							var O = de(D[0], D[1], void 0, { alpha: !1 }), k = O.canvas;
							O.fillStyle = v, O.fillRect(0, 0, k.width, k.height), C.setImage(new wc({
								img: k,
								imgSize: D,
								anchor: E.getAnchor(),
								anchorXUnits: pc.PIXELS,
								anchorYUnits: pc.PIXELS,
								offset: E.getOrigin(),
								opacity: 1,
								size: E.getSize(),
								scale: E.getScale(),
								rotation: E.getRotation(),
								rotateWithView: E.getRotateWithView()
							}));
						}
						var A = C.getZIndex() || 0, j = p[A];
						j || (j = {}, p[A] = j, j[W.POLYGON] = [], j[W.CIRCLE] = [], j[W.LINE_STRING] = [], j[W.POINT] = []), j[S.getType().replace("Multi", "")].push(S, C);
					}
				}
			}
		}
	}
	for (var M = Object.keys(p).map(Number).sort(f), m = 0, N = M.length; m < N; ++m) {
		var j = p[M[m]];
		for (var P in j) for (var F = j[P], y = 0, b = F.length; y < b; y += 2) {
			l.setStyle(F[y + 1]);
			for (var I = 0, L = t.length; I < L; ++I) l.setTransform(t[I]), l.drawGeometry(F[y]);
		}
	}
	return s.getImageData(0, 0, c.width, c.height);
}
function kc(e, t, n) {
	var r = [];
	if (n) {
		var i = Math.floor(Math.round(e[0]) * Dc), a = Math.floor(Math.round(e[1]) * Dc), o = (V(i, 0, n.width - 1) + V(a, 0, n.height - 1) * n.width) * 4, s = n.data[o], c = n.data[o + 1], l = n.data[o + 2] + 256 * (c + 256 * s), u = Math.floor((256 * 256 * 256 - 1) / t.length);
		l && l % u === 0 && r.push(t[l / u - 1]);
	}
	return r;
}
//#endregion
//#region node_modules/ol/renderer/vector.js
var Ac = .5, jc = {
	Point: Uc,
	LineString: Bc,
	Polygon: Gc,
	MultiPoint: Wc,
	MultiLineString: Vc,
	MultiPolygon: Hc,
	GeometryCollection: zc,
	Circle: Fc
};
function Mc(e, t) {
	return parseInt(L(e), 10) - parseInt(L(t), 10);
}
function Nc(e, t) {
	var n = Pc(e, t);
	return n * n;
}
function Pc(e, t) {
	return Ac * e / t;
}
function Fc(e, t, n, r, i) {
	var a = n.getFill(), o = n.getStroke();
	if (a || o) {
		var s = e.getBuilder(n.getZIndex(), Q.CIRCLE);
		s.setFillStrokeStyle(a, o), s.drawCircle(t, r);
	}
	var c = n.getText();
	if (c && c.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Q.TEXT);
		l.setTextStyle(c), l.drawText(t, r);
	}
}
function Ic(e, t, n, r, i, a, o) {
	var s = !1, c = n.getImage();
	if (c) {
		var l = c.getImageState();
		l == X.LOADED || l == X.ERROR ? c.unlistenImageChange(i) : (l == X.IDLE && c.load(), l = c.getImageState(), c.listenImageChange(i), s = !0);
	}
	return Lc(e, t, n, r, a, o), s;
}
function Lc(e, t, n, r, i, a) {
	var o = n.getGeometryFunction()(t);
	if (o) {
		var s = o.simplifyTransformed(r, i);
		if (n.getRenderer()) Rc(e, s, n, t);
		else {
			var c = jc[s.getType()];
			c(e, s, n, t, a);
		}
	}
}
function Rc(e, t, n, r) {
	if (t.getType() == W.GEOMETRY_COLLECTION) {
		for (var i = t.getGeometries(), a = 0, o = i.length; a < o; ++a) Rc(e, i[a], n, r);
		return;
	}
	e.getBuilder(n.getZIndex(), Q.DEFAULT).drawCustom(t, r, n.getRenderer(), n.getHitDetectionRenderer());
}
function zc(e, t, n, r, i) {
	var a = t.getGeometriesArray(), o, s;
	for (o = 0, s = a.length; o < s; ++o) {
		var c = jc[a[o].getType()];
		c(e, a[o], n, r, i);
	}
}
function Bc(e, t, n, r, i) {
	var a = n.getStroke();
	if (a) {
		var o = e.getBuilder(n.getZIndex(), Q.LINE_STRING);
		o.setFillStrokeStyle(null, a), o.drawLineString(t, r);
	}
	var s = n.getText();
	if (s && s.getText()) {
		var c = (i || e).getBuilder(n.getZIndex(), Q.TEXT);
		c.setTextStyle(s), c.drawText(t, r);
	}
}
function Vc(e, t, n, r, i) {
	var a = n.getStroke();
	if (a) {
		var o = e.getBuilder(n.getZIndex(), Q.LINE_STRING);
		o.setFillStrokeStyle(null, a), o.drawMultiLineString(t, r);
	}
	var s = n.getText();
	if (s && s.getText()) {
		var c = (i || e).getBuilder(n.getZIndex(), Q.TEXT);
		c.setTextStyle(s), c.drawText(t, r);
	}
}
function Hc(e, t, n, r, i) {
	var a = n.getFill(), o = n.getStroke();
	if (o || a) {
		var s = e.getBuilder(n.getZIndex(), Q.POLYGON);
		s.setFillStrokeStyle(a, o), s.drawMultiPolygon(t, r);
	}
	var c = n.getText();
	if (c && c.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Q.TEXT);
		l.setTextStyle(c), l.drawText(t, r);
	}
}
function Uc(e, t, n, r, i) {
	var a = n.getImage(), o = n.getText(), s;
	if (i && (e = i, s = a && o && o.getText() ? {} : void 0), a) {
		if (a.getImageState() != X.LOADED) return;
		var c = e.getBuilder(n.getZIndex(), Q.IMAGE);
		c.setImageStyle(a, s), c.drawPoint(t, r);
	}
	if (o && o.getText()) {
		var l = e.getBuilder(n.getZIndex(), Q.TEXT);
		l.setTextStyle(o, s), l.drawText(t, r);
	}
}
function Wc(e, t, n, r, i) {
	var a = n.getImage(), o = n.getText(), s;
	if (i && (e = i, s = a && o && o.getText() ? {} : void 0), a) {
		if (a.getImageState() != X.LOADED) return;
		var c = e.getBuilder(n.getZIndex(), Q.IMAGE);
		c.setImageStyle(a, s), c.drawMultiPoint(t, r);
	}
	if (o && o.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Q.TEXT);
		l.setTextStyle(o, s), l.drawText(t, r);
	}
}
function Gc(e, t, n, r, i) {
	var a = n.getFill(), o = n.getStroke();
	if (a || o) {
		var s = e.getBuilder(n.getZIndex(), Q.POLYGON);
		s.setFillStrokeStyle(a, o), s.drawPolygon(t, r);
	}
	var c = n.getText();
	if (c && c.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Q.TEXT);
		l.setTextStyle(c), l.drawText(t, r);
	}
}
//#endregion
//#region node_modules/ol/renderer/canvas/VectorLayer.js
var Kc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), qc = function(e) {
	Kc(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.boundHandleStyleImageChange_ = n.handleStyleImageChange_.bind(n), n.animatingOrInteracting_, n.dirty_ = !1, n.hitDetectionImageData_ = null, n.renderedFeatures_ = null, n.renderedRevision_ = -1, n.renderedResolution_ = NaN, n.renderedExtent_ = At(), n.wrappedRenderedExtent_ = At(), n.renderedRotation_, n.renderedCenter_ = null, n.renderedProjection_ = null, n.renderedRenderOrder_ = null, n.replayGroup_ = null, n.replayGroupChanged = !0, n.declutterExecutorGroup = null, n.clipping = !0, n;
	}
	return t.prototype.useContainer = function(t, n, r) {
		r < 1 && (t = null), e.prototype.useContainer.call(this, t, n, r);
	}, t.prototype.renderWorlds = function(e, t, n) {
		var r = t.extent, i = t.viewState, a = i.center, o = i.resolution, s = i.projection, c = i.rotation, l = s.getExtent(), u = this.getLayer().getSource(), d = t.pixelRatio, f = t.viewHints, p = !(f[Y.ANIMATING] || f[Y.INTERACTING]), m = this.context, h = Math.round(t.size[0] * d), g = Math.round(t.size[1] * d), _ = u.getWrapX() && s.canWrapX(), v = _ ? U(l) : null, y = _ ? Math.ceil((r[2] - l[2]) / v) + 1 : 1, b = _ ? Math.floor((r[0] - l[0]) / v) : 0;
		do {
			var x = this.getRenderTransform(a, o, c, d, h, g, b * v);
			e.execute(m, 1, x, c, p, void 0, n);
		} while (++b < y);
	}, t.prototype.renderDeclutter = function(e) {
		this.declutterExecutorGroup && this.renderWorlds(this.declutterExecutorGroup, e, e.declutterTree);
	}, t.prototype.renderFrame = function(e, t) {
		var n = e.pixelRatio, r = e.layerStatesArray[e.layerIndex];
		Gn(this.pixelTransform, 1 / n, 1 / n), Jn(this.inversePixelTransform, this.pixelTransform);
		var i = Zn(this.pixelTransform);
		this.useContainer(t, i, r.opacity);
		var a = this.context, o = a.canvas, s = this.replayGroup_, c = this.declutterExecutorGroup;
		if ((!s || s.isEmpty()) && (!c || c.isEmpty())) return null;
		var l = Math.round(e.size[0] * n), u = Math.round(e.size[1] * n);
		o.width != l || o.height != u ? (o.width = l, o.height = u, o.style.transform !== i && (o.style.transform = i)) : this.containerReused || a.clearRect(0, 0, l, u), this.preRender(a, e);
		var d = e.viewState, f = d.projection, p = !1, m = !0;
		if (r.extent && this.clipping) {
			var h = An(r.extent, f);
			m = Zt(h, e.extent), p = m && !Dt(h, e.extent), p && this.clipUnrotated(a, e, h);
		}
		m && this.renderWorlds(s, e), p && a.restore(), this.postRender(a, e);
		var g = ke(r.opacity), _ = this.container;
		return g !== _.style.opacity && (_.style.opacity = g), this.renderedRotation_ !== d.rotation && (this.renderedRotation_ = d.rotation, this.hitDetectionImageData_ = null), this.container;
	}, t.prototype.getFeatures = function(e) {
		return new Promise(function(t) {
			if (!this.hitDetectionImageData_ && !this.animatingOrInteracting_) {
				var n = [this.context.canvas.width, this.context.canvas.height];
				G(this.pixelTransform, n);
				var r = this.renderedCenter_, i = this.renderedResolution_, a = this.renderedRotation_, o = this.renderedProjection_, s = this.wrappedRenderedExtent_, c = this.getLayer(), l = [], u = n[0] * Dc, d = n[1] * Dc;
				l.push(this.getRenderTransform(r, i, a, Dc, u, d, 0).slice());
				var f = c.getSource(), p = o.getExtent();
				if (f.getWrapX() && o.canWrapX() && !Dt(p, s)) {
					for (var m = s[0], h = U(p), g = 0, _ = void 0; m < p[0];) --g, _ = h * g, l.push(this.getRenderTransform(r, i, a, Dc, u, d, _).slice()), m += h;
					for (g = 0, m = s[2]; m > p[2];) ++g, _ = h * g, l.push(this.getRenderTransform(r, i, a, Dc, u, d, _).slice()), m -= h;
				}
				this.hitDetectionImageData_ = Oc(n, l, this.renderedFeatures_, c.getStyleFunction(), s, i, a);
			}
			t(kc(e, this.renderedFeatures_, this.hitDetectionImageData_));
		}.bind(this));
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {
		var a = this;
		if (this.replayGroup_) {
			var o = t.viewState.resolution, s = t.viewState.rotation, c = this.getLayer(), l = {}, u = function(e, t, n) {
				var a = L(e), o = l[a];
				if (!o) {
					if (n === 0) return l[a] = !0, r(e, c, t);
					i.push(l[a] = {
						feature: e,
						layer: c,
						geometry: t,
						distanceSq: n,
						callback: r
					});
				} else if (o !== !0 && n < o.distanceSq) {
					if (n === 0) return l[a] = !0, i.splice(i.lastIndexOf(o), 1), r(e, c, t);
					o.geometry = t, o.distanceSq = n;
				}
			}, d, f = [this.replayGroup_];
			return this.declutterExecutorGroup && f.push(this.declutterExecutorGroup), f.some(function(r) {
				return d = r.forEachFeatureAtCoordinate(e, o, s, n, u, r === a.declutterExecutorGroup ? t.declutterTree.all().map(function(e) {
					return e.value;
				}) : null);
			}), d;
		}
	}, t.prototype.handleFontsChanged = function() {
		var e = this.getLayer();
		e.getVisible() && this.replayGroup_ && e.changed();
	}, t.prototype.handleStyleImageChange_ = function(e) {
		this.renderIfReadyAndVisible();
	}, t.prototype.prepareFrame = function(e) {
		var t = this.getLayer(), n = t.getSource();
		if (!n) return !1;
		var r = e.viewHints[Y.ANIMATING], i = e.viewHints[Y.INTERACTING], a = t.getUpdateWhileAnimating(), o = t.getUpdateWhileInteracting();
		if (!this.dirty_ && !a && r || !o && i) return this.animatingOrInteracting_ = !0, !0;
		this.animatingOrInteracting_ = !1;
		var s = e.extent, c = e.viewState, l = c.projection, u = c.resolution, d = e.pixelRatio, f = t.getRevision(), p = t.getRenderBuffer(), m = t.getRenderOrder();
		m === void 0 && (m = Mc);
		var h = c.center.slice(), _ = Ct(s, p * u), v = _.slice(), y = [_.slice()], b = l.getExtent();
		if (n.getWrapX() && l.canWrapX() && !Dt(b, e.extent)) {
			var x = U(b), S = Math.max(U(_) / 2, x);
			_[0] = b[0] - S, _[2] = b[2] + S, un(h, l);
			var C = nn(y[0], l);
			C[0] < b[0] && C[2] < b[2] ? y.push([
				C[0] + x,
				C[1],
				C[2] + x,
				C[3]
			]) : C[0] > b[0] && C[2] > b[2] && y.push([
				C[0] - x,
				C[1],
				C[2] - x,
				C[3]
			]);
		}
		if (!this.dirty_ && this.renderedResolution_ == u && this.renderedRevision_ == f && this.renderedRenderOrder_ == m && Dt(this.wrappedRenderedExtent_, _)) return g(this.renderedExtent_, v) || (this.hitDetectionImageData_ = null, this.renderedExtent_ = v), this.renderedCenter_ = h, this.replayGroupChanged = !1, !0;
		this.replayGroup_ = null, this.dirty_ = !1;
		var w = new Xs(Pc(u, d), _, u, d), T;
		this.getLayer().getDeclutter() && (T = new Xs(Pc(u, d), _, u, d));
		var E = En(), D;
		if (E) {
			for (var O = 0, k = y.length; O < k; ++O) {
				var A = y[O], j = kn(A, l);
				n.loadFeatures(j, jn(u, l), E);
			}
			D = Sn(E, l);
		} else for (var O = 0, k = y.length; O < k; ++O) n.loadFeatures(y[O], u, l);
		var M = Nc(u, d), N = function(e) {
			var n, r = e.getStyleFunction() || t.getStyleFunction();
			if (r && (n = r(e, u)), n) {
				var i = this.renderFeature(e, M, n, w, D, T);
				this.dirty_ = this.dirty_ || i;
			}
		}.bind(this), P = kn(_, l), F = n.getFeaturesInExtent(P);
		m && F.sort(m);
		for (var O = 0, k = F.length; O < k; ++O) N(F[O]);
		this.renderedFeatures_ = F;
		var I = w.finish(), L = new cc(_, u, d, n.getOverlaps(), I, t.getRenderBuffer());
		return T && (this.declutterExecutorGroup = new cc(_, u, d, n.getOverlaps(), T.finish(), t.getRenderBuffer())), this.renderedResolution_ = u, this.renderedRevision_ = f, this.renderedRenderOrder_ = m, this.renderedExtent_ = v, this.wrappedRenderedExtent_ = _, this.renderedCenter_ = h, this.renderedProjection_ = l, this.replayGroup_ = L, this.hitDetectionImageData_ = null, this.replayGroupChanged = !0, !0;
	}, t.prototype.renderFeature = function(e, t, n, r, i, a) {
		if (!n) return !1;
		var o = !1;
		if (Array.isArray(n)) for (var s = 0, c = n.length; s < c; ++s) o = Ic(r, e, n[s], t, this.boundHandleStyleImageChange_, i, a) || o;
		else o = Ic(r, e, n, t, this.boundHandleStyleImageChange_, i, a);
		return o;
	}, t;
}(so), Jc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Yc = function(e) {
	Jc(t, e);
	function t(t) {
		return e.call(this, t) || this;
	}
	return t.prototype.createRenderer = function() {
		return new qc(this);
	}, t;
}(js), Xc = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Zc = function(e) {
	Xc(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.projection = gn(t.projection), n.attributions_ = Qc(t.attributions), n.attributionsCollapsible_ = t.attributionsCollapsible === void 0 || t.attributionsCollapsible, n.loading = !1, n.state_ = t.state === void 0 ? pr.READY : t.state, n.wrapX_ = t.wrapX !== void 0 && t.wrapX, n.viewResolver = null, n.viewRejector = null;
		var r = n;
		return n.viewPromise_ = new Promise(function(e, t) {
			r.viewResolver = e, r.viewRejector = t;
		}), n;
	}
	return t.prototype.getAttributions = function() {
		return this.attributions_;
	}, t.prototype.getAttributionsCollapsible = function() {
		return this.attributionsCollapsible_;
	}, t.prototype.getProjection = function() {
		return this.projection;
	}, t.prototype.getResolutions = function() {
		return F();
	}, t.prototype.getView = function() {
		return this.viewPromise_;
	}, t.prototype.getState = function() {
		return this.state_;
	}, t.prototype.getWrapX = function() {
		return this.wrapX_;
	}, t.prototype.getContextOptions = function() {}, t.prototype.refresh = function() {
		this.changed();
	}, t.prototype.setAttributions = function(e) {
		this.attributions_ = Qc(e), this.changed();
	}, t.prototype.setState = function(e) {
		this.state_ = e, this.changed();
	}, t;
}(z);
function Qc(e) {
	return e ? Array.isArray(e) ? function(t) {
		return e;
	} : typeof e == "function" ? e : function(t) {
		return [e];
	} : null;
}
//#endregion
//#region node_modules/ol/source/VectorEventType.js
var $c = {
	ADDFEATURE: "addfeature",
	CHANGEFEATURE: "changefeature",
	CLEAR: "clear",
	REMOVEFEATURE: "removefeature",
	FEATURESLOADSTART: "featuresloadstart",
	FEATURESLOADEND: "featuresloadend",
	FEATURESLOADERROR: "featuresloaderror"
};
//#endregion
//#region node_modules/ol/interaction.js
function el(e) {
	var t = e || {}, n = new ti(), r = new as(-.005, .05, 100);
	return (t.altShiftDragRotate === void 0 || t.altShiftDragRotate) && n.push(new Go()), (t.doubleClickZoom === void 0 || t.doubleClickZoom) && n.push(new Eo({
		delta: t.zoomDelta,
		duration: t.zoomDuration
	})), (t.dragPan === void 0 || t.dragPan) && n.push(new Uo({
		onFocusOnly: t.onFocusOnly,
		kinetic: r
	})), (t.pinchRotate === void 0 || t.pinchRotate) && n.push(new us()), (t.pinchZoom === void 0 || t.pinchZoom) && n.push(new fs({ duration: t.zoomDuration })), (t.keyboard === void 0 || t.keyboard) && (n.push(new ns()), n.push(new is({
		delta: t.zoomDelta,
		duration: t.zoomDuration
	}))), (t.mouseWheelZoom === void 0 || t.mouseWheelZoom) && n.push(new cs({
		onFocusOnly: t.onFocusOnly,
		duration: t.zoomDuration
	})), (t.shiftDragZoom === void 0 || t.shiftDragZoom) && n.push(new $o({ duration: t.zoomDuration })), n;
}
//#endregion
//#region node_modules/ol/Map.js
var tl = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), nl = function(e) {
	tl(t, e);
	function t(t) {
		var n = this;
		return t = S({}, t), t.controls ||= yo(), t.interactions ||= el({ onFocusOnly: !0 }), n = e.call(this, t) || this, n;
	}
	return t.prototype.createRenderer = function() {
		return new Gr(this);
	}, t;
}(Ba), rl = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), il = function(e) {
	rl(t, e);
	function t(t, n, r) {
		var i = e.call(this) || this, a = r || {};
		return i.tileCoord = t, i.state = n, i.interimTile = null, i.key = "", i.transition_ = a.transition === void 0 ? 250 : a.transition, i.transitionStarts_ = {}, i;
	}
	return t.prototype.changed = function() {
		this.dispatchEvent(O.CHANGE);
	}, t.prototype.release = function() {}, t.prototype.getKey = function() {
		return this.key + "/" + this.tileCoord;
	}, t.prototype.getInterimTile = function() {
		if (!this.interimTile) return this;
		var e = this.interimTile;
		do {
			if (e.getState() == J.LOADED) return this.transition_ = 0, e;
			e = e.interimTile;
		} while (e);
		return this;
	}, t.prototype.refreshInterimChain = function() {
		if (this.interimTile) {
			var e = this.interimTile, t = this;
			do {
				if (e.getState() == J.LOADED) {
					e.interimTile = null;
					break;
				} else e.getState() == J.LOADING ? t = e : e.getState() == J.IDLE ? t.interimTile = e.interimTile : t = e;
				e = t.interimTile;
			} while (e);
		}
	}, t.prototype.getTileCoord = function() {
		return this.tileCoord;
	}, t.prototype.getState = function() {
		return this.state;
	}, t.prototype.setState = function(e) {
		if (this.state !== J.ERROR && this.state > e) throw Error("Tile load sequence violation");
		this.state = e, this.changed();
	}, t.prototype.load = function() {
		F();
	}, t.prototype.getAlpha = function(e, t) {
		if (!this.transition_) return 1;
		var n = this.transitionStarts_[e];
		if (!n) n = t, this.transitionStarts_[e] = n;
		else if (n === -1) return 1;
		var r = t - n + 1e3 / 60;
		return r >= this.transition_ ? 1 : Oi(r / this.transition_);
	}, t.prototype.inTransition = function(e) {
		return this.transition_ ? this.transitionStarts_[e] !== -1 : !1;
	}, t.prototype.endTransition = function(e) {
		this.transition_ && (this.transitionStarts_[e] = -1);
	}, t;
}(D), al = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), ol = function(e) {
	al(t, e);
	function t(t, n, r, i, a, o) {
		var s = e.call(this, t, n, o) || this;
		return s.crossOrigin_ = i, s.src_ = r, s.key = r, s.image_ = new Image(), i !== null && (s.image_.crossOrigin = i), s.unlisten_ = null, s.tileLoadFunction_ = a, s;
	}
	return t.prototype.getImage = function() {
		return this.image_;
	}, t.prototype.setImage = function(e) {
		this.image_ = e, this.state = J.LOADED, this.unlistenImage_(), this.changed();
	}, t.prototype.handleImageError_ = function() {
		this.state = J.ERROR, this.unlistenImage_(), this.image_ = sl(), this.changed();
	}, t.prototype.handleImageLoad_ = function() {
		var e = this.image_;
		e.naturalWidth && e.naturalHeight ? this.state = J.LOADED : this.state = J.EMPTY, this.unlistenImage_(), this.changed();
	}, t.prototype.load = function() {
		this.state == J.ERROR && (this.state = J.IDLE, this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_)), this.state == J.IDLE && (this.state = J.LOADING, this.changed(), this.tileLoadFunction_(this, this.src_), this.unlisten_ = vc(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
	}, t.prototype.unlistenImage_ = function() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}, t;
}(il);
function sl() {
	var e = de(1, 1);
	return e.fillStyle = "rgba(0,0,0,0)", e.fillRect(0, 0, 1, 1), e.canvas;
}
//#endregion
//#region node_modules/ol/tilecoord.js
function cl(e, t, n, r) {
	return r === void 0 ? [
		e,
		t,
		n
	] : (r[0] = e, r[1] = t, r[2] = n, r);
}
function ll(e, t, n) {
	return e + "/" + t + "/" + n;
}
function ul(e) {
	return ll(e[0], e[1], e[2]);
}
function dl(e) {
	return e.split("/").map(Number);
}
function fl(e) {
	return (e[1] << e[0]) + e[2];
}
function pl(e, t) {
	var n = e[0], r = e[1], i = e[2];
	if (t.getMinZoom() > n || n > t.getMaxZoom()) return !1;
	var a = t.getFullTileRange(n);
	return !a || a.containsXY(r, i);
}
//#endregion
//#region node_modules/ol/tilegrid/TileGrid.js
var ml = [
	0,
	0,
	0
], hl = function() {
	function e(e) {
		this.minZoom = e.minZoom === void 0 ? 0 : e.minZoom, this.resolutions_ = e.resolutions, H(_(this.resolutions_, function(e, t) {
			return t - e;
		}, !0), 17);
		var t;
		if (!e.origins) {
			for (var n = 0, r = this.resolutions_.length - 1; n < r; ++n) if (!t) t = this.resolutions_[n] / this.resolutions_[n + 1];
			else if (this.resolutions_[n] / this.resolutions_[n + 1] !== t) {
				t = void 0;
				break;
			}
		}
		this.zoomFactor_ = t, this.maxZoom = this.resolutions_.length - 1, this.origin_ = e.origin === void 0 ? null : e.origin, this.origins_ = null, e.origins !== void 0 && (this.origins_ = e.origins, H(this.origins_.length == this.resolutions_.length, 20));
		var i = e.extent;
		i !== void 0 && !this.origin_ && !this.origins_ && (this.origin_ = Yt(i)), H(!this.origin_ && this.origins_ || this.origin_ && !this.origins_, 18), this.tileSizes_ = null, e.tileSizes !== void 0 && (this.tileSizes_ = e.tileSizes, H(this.tileSizes_.length == this.resolutions_.length, 19)), this.tileSize_ = e.tileSize === void 0 ? this.tileSizes_ ? null : 256 : e.tileSize, H(!this.tileSize_ && this.tileSizes_ || this.tileSize_ && !this.tileSizes_, 22), this.extent_ = i === void 0 ? null : i, this.fullTileRanges_ = null, this.tmpSize_ = [0, 0], this.tmpExtent_ = [
			0,
			0,
			0,
			0
		], e.sizes === void 0 ? i && this.calculateTileRanges_(i) : this.fullTileRanges_ = e.sizes.map(function(e, t) {
			var n = new co(Math.min(0, e[0]), Math.max(e[0] - 1, -1), Math.min(0, e[1]), Math.max(e[1] - 1, -1));
			if (i) {
				var r = this.getTileRangeForExtentAndZ(i, t);
				n.minX = Math.max(r.minX, n.minX), n.maxX = Math.min(r.maxX, n.maxX), n.minY = Math.max(r.minY, n.minY), n.maxY = Math.min(r.maxY, n.maxY);
			}
			return n;
		}, this);
	}
	return e.prototype.forEachTileCoord = function(e, t, n) {
		for (var r = this.getTileRangeForExtentAndZ(e, t), i = r.minX, a = r.maxX; i <= a; ++i) for (var o = r.minY, s = r.maxY; o <= s; ++o) n([
			t,
			i,
			o
		]);
	}, e.prototype.forEachTileCoordParentTileRange = function(e, t, n, r) {
		var i, a, o, s = null, c = e[0] - 1;
		for (this.zoomFactor_ === 2 ? (a = e[1], o = e[2]) : s = this.getTileCoordExtent(e, r); c >= this.minZoom;) {
			if (this.zoomFactor_ === 2 ? (a = Math.floor(a / 2), o = Math.floor(o / 2), i = lo(a, a, o, o, n)) : i = this.getTileRangeForExtentAndZ(s, c, n), t(c, i)) return !0;
			--c;
		}
		return !1;
	}, e.prototype.getExtent = function() {
		return this.extent_;
	}, e.prototype.getMaxZoom = function() {
		return this.maxZoom;
	}, e.prototype.getMinZoom = function() {
		return this.minZoom;
	}, e.prototype.getOrigin = function(e) {
		return this.origin_ ? this.origin_ : this.origins_[e];
	}, e.prototype.getResolution = function(e) {
		return this.resolutions_[e];
	}, e.prototype.getResolutions = function() {
		return this.resolutions_;
	}, e.prototype.getTileCoordChildTileRange = function(e, t, n) {
		if (e[0] < this.maxZoom) {
			if (this.zoomFactor_ === 2) {
				var r = e[1] * 2, i = e[2] * 2;
				return lo(r, r + 1, i, i + 1, t);
			}
			var a = this.getTileCoordExtent(e, n || this.tmpExtent_);
			return this.getTileRangeForExtentAndZ(a, e[0] + 1, t);
		}
		return null;
	}, e.prototype.getTileRangeForTileCoordAndZ = function(e, t, n) {
		if (t > this.maxZoom || t < this.minZoom) return null;
		var r = e[0], i = e[1], a = e[2];
		if (t === r) return lo(i, a, i, a, n);
		if (this.zoomFactor_) {
			var o = this.zoomFactor_ ** +(t - r), s = Math.floor(i * o), c = Math.floor(a * o);
			return t < r ? lo(s, s, c, c, n) : lo(s, Math.floor(o * (i + 1)) - 1, c, Math.floor(o * (a + 1)) - 1, n);
		}
		var l = this.getTileCoordExtent(e, this.tmpExtent_);
		return this.getTileRangeForExtentAndZ(l, t, n);
	}, e.prototype.getTileRangeExtent = function(e, t, n) {
		var r = this.getOrigin(e), i = this.getResolution(e), a = Ra(this.getTileSize(e), this.tmpSize_), o = r[0] + t.minX * a[0] * i, s = r[0] + (t.maxX + 1) * a[0] * i;
		return jt(o, r[1] + t.minY * a[1] * i, s, r[1] + (t.maxY + 1) * a[1] * i, n);
	}, e.prototype.getTileRangeForExtentAndZ = function(e, t, n) {
		var r = ml;
		this.getTileCoordForXYAndZ_(e[0], e[3], t, !1, r);
		var i = r[1], a = r[2];
		return this.getTileCoordForXYAndZ_(e[2], e[1], t, !0, r), lo(i, r[1], a, r[2], n);
	}, e.prototype.getTileCoordCenter = function(e) {
		var t = this.getOrigin(e[0]), n = this.getResolution(e[0]), r = Ra(this.getTileSize(e[0]), this.tmpSize_);
		return [t[0] + (e[1] + .5) * r[0] * n, t[1] - (e[2] + .5) * r[1] * n];
	}, e.prototype.getTileCoordExtent = function(e, t) {
		var n = this.getOrigin(e[0]), r = this.getResolution(e[0]), i = Ra(this.getTileSize(e[0]), this.tmpSize_), a = n[0] + e[1] * i[0] * r, o = n[1] - (e[2] + 1) * i[1] * r;
		return jt(a, o, a + i[0] * r, o + i[1] * r, t);
	}, e.prototype.getTileCoordForCoordAndResolution = function(e, t, n) {
		return this.getTileCoordForXYAndResolution_(e[0], e[1], t, !1, n);
	}, e.prototype.getTileCoordForXYAndResolution_ = function(e, t, n, r, i) {
		var a = this.getZForResolution(n), o = n / this.getResolution(a), s = this.getOrigin(a), c = Ra(this.getTileSize(a), this.tmpSize_), l = r ? .5 : 0, u = r ? .5 : 0, d = Math.floor((e - s[0]) / n + l), f = Math.floor((s[1] - t) / n + u), p = o * d / c[0], m = o * f / c[1];
		return r ? (p = Math.ceil(p) - 1, m = Math.ceil(m) - 1) : (p = Math.floor(p), m = Math.floor(m)), cl(a, p, m, i);
	}, e.prototype.getTileCoordForXYAndZ_ = function(e, t, n, r, i) {
		var a = this.getOrigin(n), o = this.getResolution(n), s = Ra(this.getTileSize(n), this.tmpSize_), c = r ? .5 : 0, l = r ? .5 : 0, u = Math.floor((e - a[0]) / o + c), d = Math.floor((a[1] - t) / o + l), f = u / s[0], p = d / s[1];
		return r ? (f = Math.ceil(f) - 1, p = Math.ceil(p) - 1) : (f = Math.floor(f), p = Math.floor(p)), cl(n, f, p, i);
	}, e.prototype.getTileCoordForCoordAndZ = function(e, t, n) {
		return this.getTileCoordForXYAndZ_(e[0], e[1], t, !1, n);
	}, e.prototype.getTileCoordResolution = function(e) {
		return this.resolutions_[e[0]];
	}, e.prototype.getTileSize = function(e) {
		return this.tileSize_ ? this.tileSize_ : this.tileSizes_[e];
	}, e.prototype.getFullTileRange = function(e) {
		return this.fullTileRanges_ ? this.fullTileRanges_[e] : this.extent_ ? this.getTileRangeForExtentAndZ(this.extent_, e) : null;
	}, e.prototype.getZForResolution = function(e, t) {
		return V(p(this.resolutions_, e, t || 0), this.minZoom, this.maxZoom);
	}, e.prototype.calculateTileRanges_ = function(e) {
		for (var t = this.resolutions_.length, n = Array(t), r = this.minZoom; r < t; ++r) n[r] = this.getTileRangeForExtentAndZ(e, r);
		this.fullTileRanges_ = n;
	}, e;
}(), gl = .5, _l = 10, vl = .25, yl = function() {
	function e(e, t, n, r, i, a) {
		this.sourceProj_ = e, this.targetProj_ = t;
		var o = {}, s = Cn(this.targetProj_, this.sourceProj_);
		this.transformInv_ = function(e) {
			var t = e[0] + "/" + e[1];
			return o[t] || (o[t] = s(e)), o[t];
		}, this.maxSourceExtent_ = r, this.errorThresholdSquared_ = i * i, this.triangles_ = [], this.wrapsXInSource_ = !1, this.canWrapXInSource_ = this.sourceProj_.canWrapX() && !!r && !!this.sourceProj_.getExtent() && U(r) == U(this.sourceProj_.getExtent()), this.sourceWorldWidth_ = this.sourceProj_.getExtent() ? U(this.sourceProj_.getExtent()) : null, this.targetWorldWidth_ = this.targetProj_.getExtent() ? U(this.targetProj_.getExtent()) : null;
		var c = Yt(n), l = Xt(n), u = Ut(n), d = Ht(n), f = this.transformInv_(c), p = this.transformInv_(l), m = this.transformInv_(u), h = this.transformInv_(d), g = _l + (a ? Math.max(0, Math.ceil(Ue(Vt(n) / (a * a * 256 * 256)))) : 0);
		if (this.addQuad_(c, l, u, d, f, p, m, h, g), this.wrapsXInSource_) {
			var _ = Infinity;
			this.triangles_.forEach(function(e, t, n) {
				_ = Math.min(_, e.source[0][0], e.source[1][0], e.source[2][0]);
			}), this.triangles_.forEach(function(e) {
				if (Math.max(e.source[0][0], e.source[1][0], e.source[2][0]) - _ > this.sourceWorldWidth_ / 2) {
					var t = [
						[e.source[0][0], e.source[0][1]],
						[e.source[1][0], e.source[1][1]],
						[e.source[2][0], e.source[2][1]]
					];
					t[0][0] - _ > this.sourceWorldWidth_ / 2 && (t[0][0] -= this.sourceWorldWidth_), t[1][0] - _ > this.sourceWorldWidth_ / 2 && (t[1][0] -= this.sourceWorldWidth_), t[2][0] - _ > this.sourceWorldWidth_ / 2 && (t[2][0] -= this.sourceWorldWidth_);
					var n = Math.min(t[0][0], t[1][0], t[2][0]);
					Math.max(t[0][0], t[1][0], t[2][0]) - n < this.sourceWorldWidth_ / 2 && (e.source = t);
				}
			}.bind(this));
		}
		o = {};
	}
	return e.prototype.addTriangle_ = function(e, t, n, r, i, a) {
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
	}, e.prototype.addQuad_ = function(e, t, n, r, i, a, o, s, c) {
		var l = St([
			i,
			a,
			o,
			s
		]), u = this.sourceWorldWidth_ ? U(l) / this.sourceWorldWidth_ : null, d = this.sourceWorldWidth_, f = this.sourceProj_.canWrapX() && u > .5 && u < 1, p = !1;
		if (c > 0 && (this.targetProj_.isGlobal() && this.targetWorldWidth_ && (p = U(St([
			e,
			t,
			n,
			r
		])) / this.targetWorldWidth_ > vl || p), !f && this.sourceProj_.isGlobal() && u && (p = u > vl || p)), !(!p && this.maxSourceExtent_ && isFinite(l[0]) && isFinite(l[1]) && isFinite(l[2]) && isFinite(l[3]) && !Zt(l, this.maxSourceExtent_))) {
			var m = 0;
			if (!p && (!isFinite(i[0]) || !isFinite(i[1]) || !isFinite(a[0]) || !isFinite(a[1]) || !isFinite(o[0]) || !isFinite(o[1]) || !isFinite(s[0]) || !isFinite(s[1]))) {
				if (c > 0) p = !0;
				else if (m = (!isFinite(i[0]) || !isFinite(i[1]) ? 8 : 0) + (!isFinite(a[0]) || !isFinite(a[1]) ? 4 : 0) + (!isFinite(o[0]) || !isFinite(o[1]) ? 2 : 0) + +(!isFinite(s[0]) || !isFinite(s[1])), m != 1 && m != 2 && m != 4 && m != 8) return;
			}
			if (c > 0) {
				if (!p) {
					var h = [(e[0] + n[0]) / 2, (e[1] + n[1]) / 2], g = this.transformInv_(h), _ = void 0;
					_ = f ? (Je(i[0], d) + Je(o[0], d)) / 2 - Je(g[0], d) : (i[0] + o[0]) / 2 - g[0];
					var v = (i[1] + o[1]) / 2 - g[1];
					p = _ * _ + v * v > this.errorThresholdSquared_;
				}
				if (p) {
					if (Math.abs(e[0] - n[0]) <= Math.abs(e[1] - n[1])) {
						var y = [(t[0] + n[0]) / 2, (t[1] + n[1]) / 2], b = this.transformInv_(y), x = [(r[0] + e[0]) / 2, (r[1] + e[1]) / 2], S = this.transformInv_(x);
						this.addQuad_(e, t, y, x, i, a, b, S, c - 1), this.addQuad_(x, y, n, r, S, b, o, s, c - 1);
					} else {
						var C = [(e[0] + t[0]) / 2, (e[1] + t[1]) / 2], w = this.transformInv_(C), T = [(n[0] + r[0]) / 2, (n[1] + r[1]) / 2], E = this.transformInv_(T);
						this.addQuad_(e, C, T, r, i, w, E, s, c - 1), this.addQuad_(C, t, n, T, w, a, o, E, c - 1);
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
	}, e.prototype.calculateSourceExtent = function() {
		var e = At();
		return this.triangles_.forEach(function(t, n, r) {
			var i = t.source;
			Lt(e, i[0]), Lt(e, i[1]), Lt(e, i[2]);
		}), e;
	}, e.prototype.getTriangles = function() {
		return this.triangles_;
	}, e;
}(), bl = {
	imageSmoothingEnabled: !1,
	msImageSmoothingEnabled: !1
}, xl;
function Sl(e, t, n, r, i) {
	e.beginPath(), e.moveTo(0, 0), e.lineTo(t, n), e.lineTo(r, i), e.closePath(), e.save(), e.clip(), e.fillRect(0, 0, Math.max(t, r) + 1, Math.max(n, i)), e.restore();
}
function Cl(e, t) {
	return Math.abs(e[t * 4] - 210) > 2 || Math.abs(e[t * 4 + 3] - .75 * 255) > 2;
}
function wl() {
	if (xl === void 0) {
		var e = document.createElement("canvas").getContext("2d");
		e.globalCompositeOperation = "lighter", e.fillStyle = "rgba(210, 0, 0, 0.75)", Sl(e, 4, 5, 4, 0), Sl(e, 4, 5, 0, 5);
		var t = e.getImageData(0, 0, 3, 3).data;
		xl = Cl(t, 0) || Cl(t, 4) || Cl(t, 8);
	}
	return xl;
}
function Tl(e, t, n, r) {
	var i = wn(n, t, e), a = _n(t, r, n), o = t.getMetersPerUnit();
	o !== void 0 && (a *= o);
	var s = e.getMetersPerUnit();
	s !== void 0 && (a /= s);
	var c = e.getExtent();
	if (!c || Et(c, i)) {
		var l = _n(e, a, i) / a;
		isFinite(l) && l > 0 && (a /= l);
	}
	return a;
}
function El(e, t, n, r) {
	var i = Tl(e, t, Wt(n), r);
	return (!isFinite(i) || i <= 0) && Bt(n, function(n) {
		return i = Tl(e, t, n, r), isFinite(i) && i > 0;
	}), i;
}
function Dl(e, t, n, r, i, a, o, s, c, l, u, d) {
	var f = de(Math.round(n * e), Math.round(n * t));
	if (S(f, d), c.length === 0) return f.canvas;
	f.scale(n, n);
	function p(e) {
		return Math.round(e * n) / n;
	}
	f.globalCompositeOperation = "lighter";
	var m = At();
	c.forEach(function(e, t, n) {
		It(m, e.extent);
	});
	var h = U(m), g = qt(m), _ = de(Math.round(n * h / r), Math.round(n * g / r));
	S(_, d);
	var v = n / r;
	c.forEach(function(e, t, n) {
		var r = e.extent[0] - m[0], i = -(e.extent[3] - m[3]), a = U(e.extent), o = qt(e.extent);
		e.image.width > 0 && e.image.height > 0 && _.drawImage(e.image, l, l, e.image.width - 2 * l, e.image.height - 2 * l, r * v, i * v, a * v, o * v);
	});
	var y = Yt(o);
	return s.getTriangles().forEach(function(e, t, i) {
		var o = e.source, s = e.target, c = o[0][0], l = o[0][1], u = o[1][0], h = o[1][1], g = o[2][0], v = o[2][1], b = p((s[0][0] - y[0]) / a), x = p(-(s[0][1] - y[1]) / a), S = p((s[1][0] - y[0]) / a), C = p(-(s[1][1] - y[1]) / a), w = p((s[2][0] - y[0]) / a), T = p(-(s[2][1] - y[1]) / a), E = c, D = l;
		c = 0, l = 0, u -= E, h -= D, g -= E, v -= D;
		var O = Ke([
			[
				u,
				h,
				0,
				0,
				S - b
			],
			[
				g,
				v,
				0,
				0,
				w - b
			],
			[
				0,
				0,
				u,
				h,
				C - x
			],
			[
				0,
				0,
				g,
				v,
				T - x
			]
		]);
		if (O) {
			if (f.save(), f.beginPath(), wl() || d === bl) {
				f.moveTo(S, C);
				for (var k = 4, A = b - S, j = x - C, M = 0; M < k; M++) f.lineTo(S + p((M + 1) * A / k), C + p(M * j / (k - 1))), M != k - 1 && f.lineTo(S + p((M + 1) * A / k), C + p((M + 1) * j / (k - 1)));
				f.lineTo(w, T);
			} else f.moveTo(S, C), f.lineTo(b, x), f.lineTo(w, T);
			f.clip(), f.transform(O[0], O[2], O[1], O[3], b, x), f.translate(m[0] - E, m[3] - D), f.scale(r / n, -r / n), f.drawImage(_.canvas, 0, 0), f.restore();
		}
	}), u && (f.save(), f.globalCompositeOperation = "source-over", f.strokeStyle = "black", f.lineWidth = 1, s.getTriangles().forEach(function(e, t, n) {
		var r = e.target, i = (r[0][0] - y[0]) / a, o = -(r[0][1] - y[1]) / a, s = (r[1][0] - y[0]) / a, c = -(r[1][1] - y[1]) / a, l = (r[2][0] - y[0]) / a, u = -(r[2][1] - y[1]) / a;
		f.beginPath(), f.moveTo(s, c), f.lineTo(i, o), f.lineTo(l, u), f.closePath(), f.stroke();
	}), f.restore()), f.canvas;
}
//#endregion
//#region node_modules/ol/reproj/Tile.js
var Ol = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), kl = function(e) {
	Ol(t, e);
	function t(t, n, r, i, a, o, s, c, l, u, d, f) {
		var p = e.call(this, a, J.IDLE) || this;
		p.renderEdges_ = d !== void 0 && d, p.contextOptions_ = f, p.pixelRatio_ = s, p.gutter_ = c, p.canvas_ = null, p.sourceTileGrid_ = n, p.targetTileGrid_ = i, p.wrappedTileCoord_ = o || a, p.sourceTiles_ = [], p.sourcesListenerKeys_ = null, p.sourceZ_ = 0;
		var m = i.getTileCoordExtent(p.wrappedTileCoord_), h = p.targetTileGrid_.getExtent(), g = p.sourceTileGrid_.getExtent(), _ = h ? Jt(m, h) : m;
		if (Vt(_) === 0) return p.state = J.EMPTY, p;
		var v = t.getExtent();
		v && (g = g ? Jt(g, v) : v);
		var y = i.getResolution(p.wrappedTileCoord_[0]), b = El(t, r, _, y);
		if (!isFinite(b) || b <= 0 || (p.triangulation_ = new yl(t, r, _, g, b * (u === void 0 ? gl : u), y), p.triangulation_.getTriangles().length === 0)) return p.state = J.EMPTY, p;
		p.sourceZ_ = n.getZForResolution(b);
		var x = p.triangulation_.calculateSourceExtent();
		if (g && (t.canWrapX() ? (x[1] = V(x[1], g[1], g[3]), x[3] = V(x[3], g[1], g[3])) : x = Jt(x, g)), !Vt(x)) p.state = J.EMPTY;
		else {
			for (var S = n.getTileRangeForExtentAndZ(x, p.sourceZ_), C = S.minX; C <= S.maxX; C++) for (var w = S.minY; w <= S.maxY; w++) {
				var T = l(p.sourceZ_, C, w, s);
				T && p.sourceTiles_.push(T);
			}
			p.sourceTiles_.length === 0 && (p.state = J.EMPTY);
		}
		return p;
	}
	return t.prototype.getImage = function() {
		return this.canvas_;
	}, t.prototype.reproject_ = function() {
		var e = [];
		if (this.sourceTiles_.forEach(function(t, n, r) {
			t && t.getState() == J.LOADED && e.push({
				extent: this.sourceTileGrid_.getTileCoordExtent(t.tileCoord),
				image: t.getImage()
			});
		}.bind(this)), this.sourceTiles_.length = 0, e.length === 0) this.state = J.ERROR;
		else {
			var t = this.wrappedTileCoord_[0], n = this.targetTileGrid_.getTileSize(t), r = typeof n == "number" ? n : n[0], i = typeof n == "number" ? n : n[1], a = this.targetTileGrid_.getResolution(t), o = this.sourceTileGrid_.getResolution(this.sourceZ_), s = this.targetTileGrid_.getTileCoordExtent(this.wrappedTileCoord_);
			this.canvas_ = Dl(r, i, this.pixelRatio_, o, this.sourceTileGrid_.getExtent(), a, s, this.triangulation_, e, this.gutter_, this.renderEdges_, this.contextOptions_), this.state = J.LOADED;
		}
		this.changed();
	}, t.prototype.load = function() {
		if (this.state == J.IDLE) {
			this.state = J.LOADING, this.changed();
			var e = 0;
			this.sourcesListenerKeys_ = [], this.sourceTiles_.forEach(function(t, n, r) {
				var i = t.getState();
				if (i == J.IDLE || i == J.LOADING) {
					e++;
					var a = k(t, O.CHANGE, function(n) {
						var r = t.getState();
						(r == J.LOADED || r == J.ERROR || r == J.EMPTY) && (j(a), e--, e === 0 && (this.unlistenSources_(), this.reproject_()));
					}, this);
					this.sourcesListenerKeys_.push(a);
				}
			}.bind(this)), e === 0 ? setTimeout(this.reproject_.bind(this), 0) : this.sourceTiles_.forEach(function(e, t, n) {
				e.getState() == J.IDLE && e.load();
			});
		}
	}, t.prototype.unlistenSources_ = function() {
		this.sourcesListenerKeys_.forEach(j), this.sourcesListenerKeys_ = null;
	}, t;
}(il), Al = function() {
	function e(e) {
		this.highWaterMark = e === void 0 ? 2048 : e, this.count_ = 0, this.entries_ = {}, this.oldest_ = null, this.newest_ = null;
	}
	return e.prototype.canExpireCache = function() {
		return this.highWaterMark > 0 && this.getCount() > this.highWaterMark;
	}, e.prototype.clear = function() {
		this.count_ = 0, this.entries_ = {}, this.oldest_ = null, this.newest_ = null;
	}, e.prototype.containsKey = function(e) {
		return this.entries_.hasOwnProperty(e);
	}, e.prototype.forEach = function(e) {
		for (var t = this.oldest_; t;) e(t.value_, t.key_, this), t = t.newer;
	}, e.prototype.get = function(e, t) {
		var n = this.entries_[e];
		return H(n !== void 0, 15), n === this.newest_ ? n.value_ : (n === this.oldest_ ? (this.oldest_ = this.oldest_.newer, this.oldest_.older = null) : (n.newer.older = n.older, n.older.newer = n.newer), n.newer = null, n.older = this.newest_, this.newest_.newer = n, this.newest_ = n, n.value_);
	}, e.prototype.remove = function(e) {
		var t = this.entries_[e];
		return H(t !== void 0, 15), t === this.newest_ ? (this.newest_ = t.older, this.newest_ && (this.newest_.newer = null)) : t === this.oldest_ ? (this.oldest_ = t.newer, this.oldest_ && (this.oldest_.older = null)) : (t.newer.older = t.older, t.older.newer = t.newer), delete this.entries_[e], --this.count_, t.value_;
	}, e.prototype.getCount = function() {
		return this.count_;
	}, e.prototype.getKeys = function() {
		var e = Array(this.count_), t = 0, n;
		for (n = this.newest_; n; n = n.older) e[t++] = n.key_;
		return e;
	}, e.prototype.getValues = function() {
		var e = Array(this.count_), t = 0, n;
		for (n = this.newest_; n; n = n.older) e[t++] = n.value_;
		return e;
	}, e.prototype.peekLast = function() {
		return this.oldest_.value_;
	}, e.prototype.peekLastKey = function() {
		return this.oldest_.key_;
	}, e.prototype.peekFirstKey = function() {
		return this.newest_.key_;
	}, e.prototype.pop = function() {
		var e = this.oldest_;
		return delete this.entries_[e.key_], e.newer && (e.newer.older = null), this.oldest_ = e.newer, this.oldest_ || (this.newest_ = null), --this.count_, e.value_;
	}, e.prototype.replace = function(e, t) {
		this.get(e), this.entries_[e].value_ = t;
	}, e.prototype.set = function(e, t) {
		H(!(e in this.entries_), 16);
		var n = {
			key_: e,
			newer: null,
			older: this.newest_,
			value_: t
		};
		this.newest_ ? this.newest_.newer = n : this.oldest_ = n, this.newest_ = n, this.entries_[e] = n, ++this.count_;
	}, e.prototype.setSize = function(e) {
		this.highWaterMark = e;
	}, e;
}(), jl = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Ml = function(e) {
	jl(t, e);
	function t() {
		return e !== null && e.apply(this, arguments) || this;
	}
	return t.prototype.expireCache = function(e) {
		for (; this.canExpireCache() && !(this.peekLast().getKey() in e);) this.pop().release();
	}, t.prototype.pruneExceptNewestZ = function() {
		if (this.getCount() !== 0) {
			var e = dl(this.peekFirstKey())[0];
			this.forEach(function(t) {
				t.tileCoord[0] !== e && (this.remove(ul(t.tileCoord)), t.release());
			}.bind(this));
		}
	}, t;
}(Al), Nl = {
	TILELOADSTART: "tileloadstart",
	TILELOADEND: "tileloadend",
	TILELOADERROR: "tileloaderror"
};
//#endregion
//#region node_modules/ol/tilegrid.js
function Pl(e) {
	var t = e.getDefaultTileGrid();
	return t || (t = Rl(e), e.setDefaultTileGrid(t)), t;
}
function Fl(e, t, n) {
	var r = t[0], i = e.getTileCoordCenter(t), a = zl(n);
	if (Et(a, i)) return t;
	var o = U(a), s = Math.ceil((a[0] - i[0]) / o);
	return i[0] += o * s, e.getTileCoordForCoordAndZ(i, r);
}
function Il(e, t, n, r) {
	var i = r === void 0 ? vt.TOP_LEFT : r, a = Ll(e, t, n);
	return new hl({
		extent: e,
		origin: Gt(e, i),
		resolutions: a,
		tileSize: n
	});
}
function Ll(e, t, n, r) {
	for (var i = t === void 0 ? 42 : t, a = qt(e), o = U(e), s = Ra(n === void 0 ? 256 : n), c = r > 0 ? r : Math.max(o / s[0], a / s[1]), l = i + 1, u = Array(l), d = 0; d < l; ++d) u[d] = c / 2 ** d;
	return u;
}
function Rl(e, t, n, r) {
	return Il(zl(e), t, n, r);
}
function zl(e) {
	e = gn(e);
	var t = e.getExtent();
	if (!t) {
		var n = 180 * Be[B.DEGREES] / e.getMetersPerUnit();
		t = jt(-n, -n, n, n);
	}
	return t;
}
//#endregion
//#region node_modules/ol/source/Tile.js
var Bl = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Vl = function(e) {
	Bl(t, e);
	function t(t) {
		var n = e.call(this, {
			attributions: t.attributions,
			attributionsCollapsible: t.attributionsCollapsible,
			projection: t.projection,
			state: t.state,
			wrapX: t.wrapX
		}) || this;
		n.on, n.once, n.un, n.opaque_ = t.opaque !== void 0 && t.opaque, n.tilePixelRatio_ = t.tilePixelRatio === void 0 ? 1 : t.tilePixelRatio, n.tileGrid = t.tileGrid === void 0 ? null : t.tileGrid;
		var r = [256, 256], i = t.tileGrid;
		return i && Ra(i.getTileSize(i.getMinZoom()), r), n.tileCache = new Ml(t.cacheSize || 0), n.tmpSize = [0, 0], n.key_ = t.key || "", n.tileOptions = { transition: t.transition }, n.zDirection = t.zDirection ? t.zDirection : 0, n;
	}
	return t.prototype.canExpireCache = function() {
		return this.tileCache.canExpireCache();
	}, t.prototype.expireCache = function(e, t) {
		var n = this.getTileCacheForProjection(e);
		n && n.expireCache(t);
	}, t.prototype.forEachLoadedTile = function(e, t, n, r) {
		var i = this.getTileCacheForProjection(e);
		if (!i) return !1;
		for (var a = !0, o, s, c, l = n.minX; l <= n.maxX; ++l) for (var u = n.minY; u <= n.maxY; ++u) s = ll(t, l, u), c = !1, i.containsKey(s) && (o = i.get(s), c = o.getState() === J.LOADED, c &&= r(o) !== !1), c || (a = !1);
		return a;
	}, t.prototype.getGutterForProjection = function(e) {
		return 0;
	}, t.prototype.getKey = function() {
		return this.key_;
	}, t.prototype.setKey = function(e) {
		this.key_ !== e && (this.key_ = e, this.changed());
	}, t.prototype.getOpaque = function(e) {
		return this.opaque_;
	}, t.prototype.getResolutions = function() {
		return this.tileGrid.getResolutions();
	}, t.prototype.getTile = function(e, t, n, r, i) {
		return F();
	}, t.prototype.getTileGrid = function() {
		return this.tileGrid;
	}, t.prototype.getTileGridForProjection = function(e) {
		return this.tileGrid ? this.tileGrid : Pl(e);
	}, t.prototype.getTileCacheForProjection = function(e) {
		return H(xn(this.getProjection(), e), 68), this.tileCache;
	}, t.prototype.getTilePixelRatio = function(e) {
		return this.tilePixelRatio_;
	}, t.prototype.getTilePixelSize = function(e, t, n) {
		var r = this.getTileGridForProjection(n), i = this.getTilePixelRatio(t), a = Ra(r.getTileSize(e), this.tmpSize);
		return i == 1 ? a : La(a, i, this.tmpSize);
	}, t.prototype.getTileCoordForTileUrlFunction = function(e, t) {
		var n = t === void 0 ? this.getProjection() : t, r = this.getTileGridForProjection(n);
		return this.getWrapX() && n.isGlobal() && (e = Fl(r, e, n)), pl(e, r) ? e : null;
	}, t.prototype.clear = function() {
		this.tileCache.clear();
	}, t.prototype.refresh = function() {
		this.clear(), e.prototype.refresh.call(this);
	}, t.prototype.updateCacheSize = function(e, t) {
		var n = this.getTileCacheForProjection(t);
		e > n.highWaterMark && (n.highWaterMark = e);
	}, t.prototype.useTile = function(e, t, n, r) {}, t;
}(Zc), Hl = function(e) {
	Bl(t, e);
	function t(t, n) {
		var r = e.call(this, t) || this;
		return r.tile = n, r;
	}
	return t;
}(l);
//#endregion
//#region node_modules/ol/tileurlfunction.js
function Ul(e, t) {
	var n = /\{z\}/g, r = /\{x\}/g, i = /\{y\}/g, a = /\{-y\}/g;
	return (function(o, s, c) {
		if (o) return e.replace(n, o[0].toString()).replace(r, o[1].toString()).replace(i, o[2].toString()).replace(a, function() {
			var e = o[0], n = t.getFullTileRange(e);
			return H(n, 55), (n.getHeight() - o[2] - 1).toString();
		});
	});
}
function Wl(e, t) {
	for (var n = e.length, r = Array(n), i = 0; i < n; ++i) r[i] = Ul(e[i], t);
	return Gl(r);
}
function Gl(e) {
	return e.length === 1 ? e[0] : (function(t, n, r) {
		if (t) return e[Je(fl(t), e.length)](t, n, r);
	});
}
function Kl(e) {
	var t = [], n = /\{([a-z])-([a-z])\}/.exec(e);
	if (n) {
		var r = n[1].charCodeAt(0), i = n[2].charCodeAt(0), a = void 0;
		for (a = r; a <= i; ++a) t.push(e.replace(n[0], String.fromCharCode(a)));
		return t;
	}
	if (n = /\{(\d+)-(\d+)\}/.exec(e), n) {
		for (var o = parseInt(n[2], 10), s = parseInt(n[1], 10); s <= o; s++) t.push(e.replace(n[0], s.toString()));
		return t;
	}
	return t.push(e), t;
}
//#endregion
//#region node_modules/ol/source/UrlTile.js
var ql = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Jl = function(e) {
	ql(t, e);
	function t(n) {
		var r = e.call(this, {
			attributions: n.attributions,
			cacheSize: n.cacheSize,
			opaque: n.opaque,
			projection: n.projection,
			state: n.state,
			tileGrid: n.tileGrid,
			tilePixelRatio: n.tilePixelRatio,
			wrapX: n.wrapX,
			transition: n.transition,
			key: n.key,
			attributionsCollapsible: n.attributionsCollapsible,
			zDirection: n.zDirection
		}) || this;
		return r.generateTileUrlFunction_ = r.tileUrlFunction === t.prototype.tileUrlFunction, r.tileLoadFunction = n.tileLoadFunction, n.tileUrlFunction && (r.tileUrlFunction = n.tileUrlFunction), r.urls = null, n.urls ? r.setUrls(n.urls) : n.url && r.setUrl(n.url), r.tileLoadingKeys_ = {}, r;
	}
	return t.prototype.getTileLoadFunction = function() {
		return this.tileLoadFunction;
	}, t.prototype.getTileUrlFunction = function() {
		return Object.getPrototypeOf(this).tileUrlFunction === this.tileUrlFunction ? this.tileUrlFunction.bind(this) : this.tileUrlFunction;
	}, t.prototype.getUrls = function() {
		return this.urls;
	}, t.prototype.handleTileChange = function(e) {
		var t = e.target, n = L(t), r = t.getState(), i;
		r == J.LOADING ? (this.tileLoadingKeys_[n] = !0, i = Nl.TILELOADSTART) : n in this.tileLoadingKeys_ && (delete this.tileLoadingKeys_[n], i = r == J.ERROR ? Nl.TILELOADERROR : r == J.LOADED ? Nl.TILELOADEND : void 0), i != null && this.dispatchEvent(new Hl(i, t));
	}, t.prototype.setTileLoadFunction = function(e) {
		this.tileCache.clear(), this.tileLoadFunction = e, this.changed();
	}, t.prototype.setTileUrlFunction = function(e, t) {
		this.tileUrlFunction = e, this.tileCache.pruneExceptNewestZ(), t === void 0 ? this.changed() : this.setKey(t);
	}, t.prototype.setUrl = function(e) {
		var t = Kl(e);
		this.urls = t, this.setUrls(t);
	}, t.prototype.setUrls = function(e) {
		this.urls = e;
		var t = e.join("\n");
		this.generateTileUrlFunction_ ? this.setTileUrlFunction(Wl(e, this.tileGrid), t) : this.setKey(t);
	}, t.prototype.tileUrlFunction = function(e, t, n) {}, t.prototype.useTile = function(e, t, n) {
		var r = ll(e, t, n);
		this.tileCache.containsKey(r) && this.tileCache.get(r);
	}, t;
}(Vl), Yl = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Xl = function(e) {
	Yl(t, e);
	function t(t) {
		var n = e.call(this, {
			attributions: t.attributions,
			cacheSize: t.cacheSize,
			opaque: t.opaque,
			projection: t.projection,
			state: t.state,
			tileGrid: t.tileGrid,
			tileLoadFunction: t.tileLoadFunction ? t.tileLoadFunction : Zl,
			tilePixelRatio: t.tilePixelRatio,
			tileUrlFunction: t.tileUrlFunction,
			url: t.url,
			urls: t.urls,
			wrapX: t.wrapX,
			transition: t.transition,
			key: t.key,
			attributionsCollapsible: t.attributionsCollapsible,
			zDirection: t.zDirection
		}) || this;
		return n.crossOrigin = t.crossOrigin === void 0 ? null : t.crossOrigin, n.tileClass = t.tileClass === void 0 ? ol : t.tileClass, n.tileCacheForProjection = {}, n.tileGridForProjection = {}, n.reprojectionErrorThreshold_ = t.reprojectionErrorThreshold, n.contextOptions_ = t.imageSmoothing === !1 ? bl : void 0, n.renderReprojectionEdges_ = !1, n;
	}
	return t.prototype.canExpireCache = function() {
		if (this.tileCache.canExpireCache()) return !0;
		for (var e in this.tileCacheForProjection) if (this.tileCacheForProjection[e].canExpireCache()) return !0;
		return !1;
	}, t.prototype.expireCache = function(e, t) {
		var n = this.getTileCacheForProjection(e);
		for (var r in this.tileCache.expireCache(this.tileCache == n ? t : {}), this.tileCacheForProjection) {
			var i = this.tileCacheForProjection[r];
			i.expireCache(i == n ? t : {});
		}
	}, t.prototype.getContextOptions = function() {
		return this.contextOptions_;
	}, t.prototype.getGutterForProjection = function(e) {
		return this.getProjection() && e && !xn(this.getProjection(), e) ? 0 : this.getGutter();
	}, t.prototype.getGutter = function() {
		return 0;
	}, t.prototype.getKey = function() {
		return e.prototype.getKey.call(this) + (this.contextOptions_ ? "\n" + JSON.stringify(this.contextOptions_) : "");
	}, t.prototype.getOpaque = function(t) {
		return this.getProjection() && t && !xn(this.getProjection(), t) ? !1 : e.prototype.getOpaque.call(this, t);
	}, t.prototype.getTileGridForProjection = function(e) {
		var t = this.getProjection();
		if (this.tileGrid && (!t || xn(t, e))) return this.tileGrid;
		var n = L(e);
		return n in this.tileGridForProjection || (this.tileGridForProjection[n] = Pl(e)), this.tileGridForProjection[n];
	}, t.prototype.getTileCacheForProjection = function(e) {
		var t = this.getProjection();
		if (!t || xn(t, e)) return this.tileCache;
		var n = L(e);
		return n in this.tileCacheForProjection || (this.tileCacheForProjection[n] = new Ml(this.tileCache.highWaterMark)), this.tileCacheForProjection[n];
	}, t.prototype.createTile_ = function(e, t, n, r, i, a) {
		var o = [
			e,
			t,
			n
		], s = this.getTileCoordForTileUrlFunction(o, i), c = s ? this.tileUrlFunction(s, r, i) : void 0, l = new this.tileClass(o, c === void 0 ? J.EMPTY : J.IDLE, c === void 0 ? "" : c, this.crossOrigin, this.tileLoadFunction, this.tileOptions);
		return l.key = a, l.addEventListener(O.CHANGE, this.handleTileChange.bind(this)), l;
	}, t.prototype.getTile = function(e, t, n, r, i) {
		var a = this.getProjection();
		if (!a || !i || xn(a, i)) return this.getTileInternal(e, t, n, r, a || i);
		var o = this.getTileCacheForProjection(i), s = [
			e,
			t,
			n
		], c = void 0, l = ul(s);
		o.containsKey(l) && (c = o.get(l));
		var u = this.getKey();
		if (c && c.key == u) return c;
		var d = new kl(a, this.getTileGridForProjection(a), i, this.getTileGridForProjection(i), s, this.getTileCoordForTileUrlFunction(s, i), this.getTilePixelRatio(r), this.getGutter(), function(e, t, n, r) {
			return this.getTileInternal(e, t, n, r, a);
		}.bind(this), this.reprojectionErrorThreshold_, this.renderReprojectionEdges_, this.contextOptions_);
		return d.key = u, c ? (d.interimTile = c, d.refreshInterimChain(), o.replace(l, d)) : o.set(l, d), d;
	}, t.prototype.getTileInternal = function(e, t, n, r, i) {
		var a = null, o = ll(e, t, n), s = this.getKey();
		if (!this.tileCache.containsKey(o)) a = this.createTile_(e, t, n, r, i, s), this.tileCache.set(o, a);
		else if (a = this.tileCache.get(o), a.key != s) {
			var c = a;
			a = this.createTile_(e, t, n, r, i, s), c.getState() == J.IDLE ? a.interimTile = c.interimTile : a.interimTile = c, a.refreshInterimChain(), this.tileCache.replace(o, a);
		}
		return a;
	}, t.prototype.setRenderReprojectionEdges = function(e) {
		if (this.renderReprojectionEdges_ != e) {
			for (var t in this.renderReprojectionEdges_ = e, this.tileCacheForProjection) this.tileCacheForProjection[t].clear();
			this.changed();
		}
	}, t.prototype.setTileGridForProjection = function(e, t) {
		var n = gn(e);
		if (n) {
			var r = L(n);
			r in this.tileGridForProjection || (this.tileGridForProjection[r] = t);
		}
	}, t;
}(Jl);
function Zl(e, t) {
	e.getImage().src = t;
}
//#endregion
//#region node_modules/ol/source/Zoomify.js
var Ql = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), $l = {
	DEFAULT: "default",
	TRUNCATED: "truncated"
}, eu = function(e) {
	Ql(t, e);
	function t(t, n, r, i, a, o, s) {
		var c = e.call(this, n, r, i, a, o, s) || this;
		return c.zoomifyImage_ = null, c.tileSize_ = t, c;
	}
	return t.prototype.getImage = function() {
		if (this.zoomifyImage_) return this.zoomifyImage_;
		var t = e.prototype.getImage.call(this);
		if (this.state == J.LOADED) {
			var n = this.tileSize_;
			if (t.width == n[0] && t.height == n[1]) return this.zoomifyImage_ = t, t;
			var r = de(n[0], n[1]);
			return r.drawImage(t, 0, 0), this.zoomifyImage_ = r.canvas, r.canvas;
		} else return t;
	}, t;
}(ol), tu = function(e) {
	Ql(t, e);
	function t(t) {
		var n = this, r = t, i = r.size, a = r.tierSizeCalculation === void 0 ? $l.DEFAULT : r.tierSizeCalculation, o = r.tilePixelRatio || 1, s = i[0], c = i[1], l = [], u = r.tileSize || 256, d = u * o;
		switch (a) {
			case $l.DEFAULT:
				for (; s > d || c > d;) l.push([Math.ceil(s / d), Math.ceil(c / d)]), d += d;
				break;
			case $l.TRUNCATED:
				for (var f = s, p = c; f > d || p > d;) l.push([Math.ceil(f / d), Math.ceil(p / d)]), f >>= 1, p >>= 1;
				break;
			default:
				H(!1, 53);
				break;
		}
		l.push([1, 1]), l.reverse();
		for (var m = [o], h = [0], g = 1, _ = l.length; g < _; g++) m.push(o << g), h.push(l[g - 1][0] * l[g - 1][1] + h[g - 1]);
		m.reverse();
		var v = new hl({
			tileSize: u,
			extent: r.extent || [
				0,
				-c,
				s,
				0
			],
			resolutions: m
		}), y = r.url;
		y && y.indexOf("{TileGroup}") == -1 && y.indexOf("{tileIndex}") == -1 && (y += "{TileGroup}/{z}-{x}-{y}.jpg");
		var b = Kl(y), x = u * o;
		function S(e) {
			return (function(t, n, r) {
				if (t) {
					var i = t[0], a = t[1], o = t[2], s = a + o * l[i][0], c = {
						z: i,
						x: a,
						y: o,
						tileIndex: s,
						TileGroup: "TileGroup" + ((s + h[i]) / x | 0)
					};
					return e.replace(/\{(\w+?)\}/g, function(e, t) {
						return c[t];
					});
				} else return;
			});
		}
		var C = Gl(b.map(S)), w = eu.bind(null, Ra(u * o));
		n = e.call(this, {
			attributions: r.attributions,
			cacheSize: r.cacheSize,
			crossOrigin: r.crossOrigin,
			imageSmoothing: r.imageSmoothing,
			projection: r.projection,
			tilePixelRatio: o,
			reprojectionErrorThreshold: r.reprojectionErrorThreshold,
			tileClass: w,
			tileGrid: v,
			tileUrlFunction: C,
			transition: r.transition
		}) || this, n.zDirection = r.zDirection;
		var T = C(v.getTileCoordForCoordAndResolution(Wt(v.getExtent()), m[m.length - 1]), 1, null), E = new Image();
		return E.addEventListener("error", function() {
			x = u, this.changed();
		}.bind(n)), E.src = T, n;
	}
	return t;
}(Xl), nu = function(e) {
	if (!e) return null;
	var t = e.getViewport().getElementsByClassName("ol-fixedoverlay")[0];
	return t || (e.getViewport().querySelector(".ol-layers") ? (t = document.createElement("canvas"), t.className = "ol-fixedoverlay", e.getViewport().querySelector(".ol-layers").after(t), e.on("precompose", function(n) {
		t.width = e.getSize()[0] * n.frameState.pixelRatio, t.height = e.getSize()[1] * n.frameState.pixelRatio;
	})) : t = e.getViewport().querySelector("canvas")), t;
}, ru = class extends ye {
	constructor(e) {
		e ||= {}, super(e), this.setStyle(e.style);
	}
	setMap(e) {
		this.getCanvas(e);
		var t = this.getMap();
		if (this._listener &&= (P(this._listener), null), super.setMap(e), t) try {
			t.renderSync();
		} catch {}
		e && (this._listener = e.on("postcompose", this._draw.bind(this)));
	}
	getCanvas(e) {
		return nu(e);
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
		this._style = e || new Cs({});
	}
	getStyle() {
		return this._style;
	}
	getStroke() {
		return this._style.getStroke() || this._style.setStroke(new Ss({
			color: "#000",
			width: 1.25
		})), this._style.getStroke();
	}
	getFill() {
		return this._style.getFill() || this._style.setFill(new xs({ color: "#fff" })), this._style.getFill();
	}
	getTextStroke() {
		var e = this._style.getText();
		return e ||= new Ec({}), e.getStroke() || e.setStroke(new Ss({
			color: "#fff",
			width: 3
		})), e.getStroke();
	}
	getTextFill() {
		var e = this._style.getText();
		return e ||= new Ec({}), e.getFill() || e.setFill(new xs({ color: "#fff" })), e.getFill();
	}
	getTextFont() {
		var e = this._style.getText();
		return e ||= new Ec({}), e.getFont() || e.setFont("12px sans-serif"), e.getFont();
	}
	_draw() {
		console.warn("[CanvasBase] draw function not implemented.");
	}
}, iu = class extends ru {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = "ol-graticule ol-unselectable ol-hidden", super({ element: t }), this.set("projection", e.projection || "EPSG:4326");
		var n = new Ve({ code: this.get("projection") }).getMetersPerUnit();
		for (this.fac = 1; n / this.fac > 10;) this.fac *= 10;
		this.fac = 1e4 / this.fac, this.set("maxResolution", e.maxResolution || Infinity), this.set("step", e.step || .1), this.set("stepCoord", e.stepCoord || 1), this.set("spacing", e.spacing || 40), this.set("intervals", e.intervals), this.set("precision", e.precision), this.set("margin", e.margin || 0), this.set("borderWidth", e.borderWidth || 5), this.set("stroke", e.stroke !== !1), this.formatCoord = e.formatCoord || function(e) {
			return e;
		}, e.style instanceof Cs ? this.setStyle(e.style) : this.setStyle(new Cs({
			stroke: new Ss({
				color: "#000",
				width: 1
			}),
			fill: new xs({ color: "#fff" }),
			text: new Ec({
				stroke: new Ss({
					color: "#fff",
					width: 2
				}),
				fill: new xs({ color: "#000" })
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
			], l = -Infinity, u = Infinity, d = -Infinity, f = Infinity, p = 0, m; m = c[p]; p++) c[p] = wn(m, s.getView().getProjection(), o), l = Math.max(l, c[p][0]), u = Math.min(u, c[p][0]), d = Math.max(d, c[p][1]), f = Math.min(f, c[p][1]);
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
			var T = gn(o).getExtent();
			T && (u < T[0] && (u = T[0]), f < T[1] && (f = T[1]), l > T[2] && (l = T[2] + g), d > T[3] && (d = T[3] + g));
			var E = this.getStyle().getStroke() && this.get("stroke"), D = this.getStyle().getText(), O = this.getStyle().getFill();
			t.save(), t.scale(r, r), t.beginPath(), t.rect(y, y, i - 2 * y, a - 2 * y), t.clip(), t.beginPath();
			var k = {
				top: [],
				left: [],
				bottom: [],
				right: []
			}, A, j, M, N, P;
			for (A = u; A < l; A += g) for (N = wn([A, f], o, s.getView().getProjection()), N = s.getPixelFromCoordinate(N), E && t.moveTo(N[0], N[1]), M = N, j = f + w; j <= d; j += w) P = wn([A, j], o, s.getView().getProjection()), P = s.getPixelFromCoordinate(P), E && t.lineTo(P[0], P[1]), M[1] > 0 && P[1] < 0 && k.top.push([A, M]), M[1] > a && P[1] < a && k.bottom.push([A, M]), M = P;
			for (j = f; j < d; j += g) for (N = wn([u, j], o, s.getView().getProjection()), N = s.getPixelFromCoordinate(N), E && t.moveTo(N[0], N[1]), M = N, A = u + w; A <= l; A += w) P = wn([A, j], o, s.getView().getProjection()), P = s.getPixelFromCoordinate(P), E && t.lineTo(P[0], P[1]), M[0] < 0 && P[0] > 0 && k.left.push([j, M]), M[0] < i && P[0] > i && k.right.push([j, M]), M = P;
			if (E && (t.strokeStyle = this.getStyle().getStroke().getColor(), t.lineWidth = this.getStyle().getStroke().getWidth(), t.stroke()), D) {
				t.fillStyle = this.getStyle().getText().getFill().getColor(), t.strokeStyle = this.getStyle().getText().getStroke().getColor(), t.lineWidth = this.getStyle().getText().getStroke().getWidth(), t.font = this.getStyle().getText().getFont(), t.textAlign = "center", t.textBaseline = "hanging";
				var F, I, L = (O ? v : 0) + y + 2;
				for (p = 0; F = k.top[p]; p++) Math.round(F[0] / this.get("step")) % _ || (I = this.formatCoord(F[0], "top"), t.strokeText(I, F[1][0], L), t.fillText(I, F[1][0], L));
				for (t.textBaseline = "alphabetic", p = 0; F = k.bottom[p]; p++) Math.round(F[0] / this.get("step")) % _ || (I = this.formatCoord(F[0], "bottom"), t.strokeText(I, F[1][0], a - L), t.fillText(I, F[1][0], a - L));
				for (t.textBaseline = "middle", t.textAlign = "left", p = 0; F = k.left[p]; p++) Math.round(F[0] / this.get("step")) % _ || (I = this.formatCoord(F[0], "left"), t.strokeText(I, L, F[1][1]), t.fillText(I, L, F[1][1]));
				for (t.textAlign = "right", p = 0; F = k.right[p]; p++) Math.round(F[0] / this.get("step")) % _ || (I = this.formatCoord(F[0], "right"), t.strokeText(I, i - L, F[1][1]), t.fillText(I, i - L, F[1][1]));
			}
			if (O) {
				var R = this.getStyle().getFill().getColor(), ee, te;
				for ((te = this.getStyle().getStroke()) ? ee = this.getStyle().getStroke().getColor() : (ee = R, R = "#fff"), t.strokeStyle = ee, t.lineWidth = te ? te.getWidth() : 1, p = 1; p < k.top.length; p++) t.beginPath(), t.rect(k.top[p - 1][1][0], y, k.top[p][1][0] - k.top[p - 1][1][0], v), t.fillStyle = Math.round(k.top[p][0] / g) % 2 ? ee : R, t.fill(), t.stroke();
				for (p = 1; p < k.bottom.length; p++) t.beginPath(), t.rect(k.bottom[p - 1][1][0], a - v - y, k.bottom[p][1][0] - k.bottom[p - 1][1][0], v), t.fillStyle = Math.round(k.bottom[p][0] / g) % 2 ? ee : R, t.fill(), t.stroke();
				for (p = 1; p < k.left.length; p++) t.beginPath(), t.rect(y, k.left[p - 1][1][1], v, k.left[p][1][1] - k.left[p - 1][1][1]), t.fillStyle = Math.round(k.left[p][0] / g) % 2 ? ee : R, t.fill(), t.stroke();
				for (p = 1; p < k.right.length; p++) t.beginPath(), t.rect(i - v - y, k.right[p - 1][1][1], v, k.right[p][1][1] - k.right[p - 1][1][1]), t.fillStyle = Math.round(k.right[p][0] / g) % 2 ? ee : R, t.fill(), t.stroke();
				t.beginPath(), t.fillStyle = ee, t.rect(y, y, v, v), t.rect(y, a - v - y, v, v), t.rect(i - v - y, y, v, v), t.rect(i - v - y, a - v - y, v, v), t.fill();
			}
			t.restore();
		}
	}
}, au = {
	IMAGE: "image",
	HYBRID: "hybrid",
	VECTOR: "vector"
}, ou = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), su = {
	image: [
		Q.POLYGON,
		Q.CIRCLE,
		Q.LINE_STRING,
		Q.IMAGE,
		Q.TEXT
	],
	hybrid: [Q.POLYGON, Q.LINE_STRING],
	vector: []
}, cu = {
	hybrid: [
		Q.IMAGE,
		Q.TEXT,
		Q.DEFAULT
	],
	vector: [
		Q.POLYGON,
		Q.CIRCLE,
		Q.LINE_STRING,
		Q.IMAGE,
		Q.TEXT,
		Q.DEFAULT
	]
}, lu = function(e) {
	ou(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.boundHandleStyleImageChange_ = n.handleStyleImageChange_.bind(n), n.dirty_ = !1, n.renderedLayerRevision_, n.renderedPixelToCoordinateTransform_ = null, n.renderedRotation_, n.tmpTransform_ = Rn(), n;
	}
	return t.prototype.prepareTile = function(e, t, n) {
		var r, i = e.getState();
		return (i === J.LOADED || i === J.ERROR) && (this.updateExecutorGroup_(e, t, n), this.tileImageNeedsRender_(e) && (r = !0)), r;
	}, t.prototype.getTile = function(t, n, r, i) {
		var a = i.pixelRatio, o = i.viewState, s = o.resolution, c = o.projection, l = this.getLayer(), u = l.getSource().getTile(t, n, r, a, c), d = i.viewHints, f = !(d[Y.ANIMATING] || d[Y.INTERACTING]);
		return (f || !u.wantedResolution) && (u.wantedResolution = s), this.prepareTile(u, a, c) && (f || Date.now() - i.time < 8) && l.getRenderMode() !== au.VECTOR && this.renderTileImage_(u, i), e.prototype.getTile.call(this, t, n, r, i);
	}, t.prototype.isDrawableTile = function(t) {
		var n = this.getLayer();
		return e.prototype.isDrawableTile.call(this, t) && (n.getRenderMode() === au.VECTOR ? L(n) in t.executorGroups : t.hasContext(n));
	}, t.prototype.getTileImage = function(e) {
		return e.getImage(this.getLayer());
	}, t.prototype.prepareFrame = function(t) {
		var n = this.getLayer().getRevision();
		return this.renderedLayerRevision_ != n && (this.renderedTiles.length = 0), this.renderedLayerRevision_ = n, e.prototype.prepareFrame.call(this, t);
	}, t.prototype.updateExecutorGroup_ = function(e, t, n) {
		var r = this.getLayer(), i = r.getRevision(), a = r.getRenderOrder() || null, o = e.wantedResolution, s = e.getReplayState(r);
		if (!(!s.dirty && s.renderedResolution === o && s.renderedRevision == i && s.renderedRenderOrder == a)) {
			var c = r.getSource(), l = r.getDeclutter(), u = c.getTileGrid(), d = c.getTileGridForProjection(n).getTileCoordExtent(e.wrappedTileCoord), f = c.getSourceTiles(t, n, e), p = L(r);
			delete e.hitDetectionImageData[p], e.executorGroups[p] = [], l && (e.declutterExecutorGroups[p] = []);
			for (var m = function(n, i) {
				var m = f[n];
				if (m.getState() != J.LOADED) return "continue";
				var g = m.tileCoord, _ = u.getTileCoordExtent(g), v = Jt(d, _), y = Ct(v, r.getRenderBuffer() * o, h.tmpExtent), b = Ft(_, v) ? null : y;
				s.dirty = !1;
				var x = new Xs(0, y, o, t), S = l ? new Xs(0, v, o, t) : void 0, C = Nc(o, t), w = function(e) {
					var t, n = e.getStyleFunction() || r.getStyleFunction();
					if (n && (t = n(e, o)), t) {
						var i = this.renderFeature(e, C, t, x, S);
						this.dirty_ = this.dirty_ || i, s.dirty = s.dirty || i;
					}
				}, T = m.getFeatures();
				a && a !== s.renderedRenderOrder && T.sort(a);
				for (var E = 0, D = T.length; E < D; ++E) {
					var O = T[E];
					(!b || Zt(b, O.getGeometry().getExtent())) && w.call(h, O);
				}
				var k = x.finish(), A = new cc(r.getRenderMode() !== au.VECTOR && l && f.length === 1 ? null : v, o, t, c.getOverlaps(), k, r.getRenderBuffer());
				if (e.executorGroups[p].push(A), S) {
					var j = new cc(null, o, t, c.getOverlaps(), S.finish(), r.getRenderBuffer());
					e.declutterExecutorGroups[p].push(j);
				}
			}, h = this, g = 0, _ = f.length; g < _; ++g) m(g, _);
			s.renderedRevision = i, s.renderedRenderOrder = a, s.renderedResolution = o;
		}
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {
		var a = t.viewState.resolution, o = t.viewState.rotation;
		n ??= 0;
		var s = this.getLayer(), c = s.getSource().getTileGridForProjection(t.viewState.projection), l = St([e]);
		Ct(l, a * n, l);
		for (var u = {}, d = function(e, t, n) {
			var a = e.getId();
			a === void 0 && (a = L(e));
			var o = u[a];
			if (!o) {
				if (n === 0) return u[a] = !0, r(e, s, t);
				i.push(u[a] = {
					feature: e,
					layer: s,
					geometry: t,
					distanceSq: n,
					callback: r
				});
			} else if (o !== !0 && n < o.distanceSq) {
				if (n === 0) return u[a] = !0, i.splice(i.lastIndexOf(o), 1), r(e, s, t);
				o.geometry = t, o.distanceSq = n;
			}
		}, f = this.renderedTiles, p, m = function(r, i) {
			var u = f[r];
			if (!Zt(c.getTileCoordExtent(u.wrappedTileCoord), l)) return "continue";
			var m = L(s), h = [u.executorGroups[m]], g = u.declutterExecutorGroups[m];
			g && h.push(g), h.some(function(r) {
				for (var i = r === g ? t.declutterTree.all().map(function(e) {
					return e.value;
				}) : null, s = 0, c = r.length; s < c; ++s) if (p = r[s].forEachFeatureAtCoordinate(e, a, o, n, d, i), p) return !0;
			});
		}, h = 0, g = f.length; !p && h < g; ++h) m(h, g);
		return p;
	}, t.prototype.getFeatures = function(e) {
		return new Promise(function(t, n) {
			for (var r = this.getLayer(), i = L(r), a = r.getSource(), o = this.renderedProjection, s = o.getExtent(), c = this.renderedResolution, l = a.getTileGridForProjection(o), u = G(this.renderedPixelToCoordinateTransform_, e.slice()), d = l.getTileCoordForCoordAndResolution(u, c), f, p = 0, m = this.renderedTiles.length; p < m; ++p) if (d.toString() === this.renderedTiles[p].tileCoord.toString()) {
				if (f = this.renderedTiles[p], f.getState() === J.LOADED) {
					var h = l.getTileCoordExtent(f.tileCoord);
					a.getWrapX() && o.canWrapX() && !Dt(s, h) && un(u, o);
					break;
				}
				f = void 0;
			}
			if (!f || f.loadingSourceTiles > 0) {
				t([]);
				return;
			}
			var g = Yt(l.getTileCoordExtent(f.wrappedTileCoord)), _ = [(u[0] - g[0]) / c, (g[1] - u[1]) / c], v = f.getSourceTiles().reduce(function(e, t) {
				return e.concat(t.getFeatures());
			}, []), y = f.hitDetectionImageData[i];
			if (!y && !this.animatingOrInteracting_) {
				var b = Ra(l.getTileSize(l.getZForResolution(c))), x = this.renderedRotation_;
				y = Oc(b, [this.getRenderTransform(l.getTileCoordCenter(f.wrappedTileCoord), c, 0, Dc, b[0] * Dc, b[1] * Dc, 0)], v, r.getStyleFunction(), l.getTileCoordExtent(f.wrappedTileCoord), f.getReplayState(r).renderedResolution, x), f.hitDetectionImageData[i] = y;
			}
			t(kc(_, v, y));
		}.bind(this));
	}, t.prototype.handleFontsChanged = function() {
		var e = this.getLayer();
		e.getVisible() && this.renderedLayerRevision_ !== void 0 && e.changed();
	}, t.prototype.handleStyleImageChange_ = function(e) {
		this.renderIfReadyAndVisible();
	}, t.prototype.renderDeclutter = function(e) {
		var t = this.context, n = t.globalAlpha;
		t.globalAlpha = this.getLayer().getOpacity();
		for (var r = e.viewHints, i = !(r[Y.ANIMATING] || r[Y.INTERACTING]), a = this.renderedTiles, o = 0, s = a.length; o < s; ++o) {
			var c = a[o], l = c.declutterExecutorGroups[L(this.getLayer())];
			if (l) for (var u = l.length - 1; u >= 0; --u) l[u].execute(this.context, 1, this.getTileRenderTransform(c, e), e.viewState.rotation, i, void 0, e.declutterTree);
		}
		t.globalAlpha = n;
	}, t.prototype.getTileRenderTransform = function(e, t) {
		var n = t.pixelRatio, r = t.viewState, i = r.center, a = r.resolution, o = r.rotation, s = t.size, c = Math.round(s[0] * n), l = Math.round(s[1] * n), u = this.getLayer().getSource().getTileGridForProjection(t.viewState.projection), d = e.tileCoord, f = u.getTileCoordExtent(e.wrappedTileCoord), p = u.getTileCoordExtent(d, this.tmpExtent)[0] - f[0];
		return Bn(Wn(this.inversePixelTransform.slice(), 1 / n, 1 / n), this.getRenderTransform(i, a, o, n, c, l, p));
	}, t.prototype.renderFrame = function(t, n) {
		var r = t.viewHints, i = !(r[Y.ANIMATING] || r[Y.INTERACTING]);
		e.prototype.renderFrame.call(this, t, n), this.renderedPixelToCoordinateTransform_ = t.pixelToCoordinateTransform.slice(), this.renderedRotation_ = t.viewState.rotation;
		var a = this.getLayer(), o = a.getRenderMode(), s = this.context, c = s.globalAlpha;
		s.globalAlpha = a.getOpacity();
		for (var l = cu[o], u = t.viewState.rotation, d = this.renderedTiles, f = [], p = [], m = d.length - 1; m >= 0; --m) for (var h = d[m], g = this.getTileRenderTransform(h, t), _ = h.executorGroups[L(a)], v = !1, y = 0, b = _.length; y < b; ++y) {
			var x = _[y];
			if (x.hasExecutors(l)) {
				var S = h.tileCoord[0], C = void 0;
				if (!v && (C = x.getClipCoords(g), C)) {
					s.save();
					for (var w = 0, T = f.length; w < T; ++w) {
						var E = f[w];
						S < p[w] && (s.beginPath(), s.moveTo(C[0], C[1]), s.lineTo(C[2], C[3]), s.lineTo(C[4], C[5]), s.lineTo(C[6], C[7]), s.moveTo(E[6], E[7]), s.lineTo(E[4], E[5]), s.lineTo(E[2], E[3]), s.lineTo(E[0], E[1]), s.clip());
					}
				}
				x.execute(s, 1, g, u, i, l), !v && C && (s.restore(), f.push(C), p.push(S), v = !0);
			}
		}
		return s.globalAlpha = c, this.container;
	}, t.prototype.renderFeature = function(e, t, n, r, i) {
		if (!n) return !1;
		var a = !1;
		if (Array.isArray(n)) for (var o = 0, s = n.length; o < s; ++o) a = Ic(r, e, n[o], t, this.boundHandleStyleImageChange_, void 0, i) || a;
		else a = Ic(r, e, n, t, this.boundHandleStyleImageChange_, void 0, i);
		return a;
	}, t.prototype.tileImageNeedsRender_ = function(e) {
		var t = this.getLayer();
		if (t.getRenderMode() === au.VECTOR) return !1;
		var n = e.getReplayState(t), r = t.getRevision(), i = e.wantedResolution;
		return n.renderedTileResolution !== i || n.renderedTileRevision !== r;
	}, t.prototype.renderTileImage_ = function(e, t) {
		var n = this.getLayer(), r = e.getReplayState(n), i = n.getRevision(), a = e.executorGroups[L(n)];
		r.renderedTileRevision = i;
		var o = e.wrappedTileCoord, s = o[0], c = n.getSource(), l = t.pixelRatio, u = t.viewState.projection, d = c.getTileGridForProjection(u), f = d.getResolution(e.tileCoord[0]), p = t.pixelRatio / e.wantedResolution * f, m = d.getResolution(s), h = e.getContext(n);
		l = Math.round(Math.max(l, p / l));
		var g = c.getTilePixelSize(s, l, u);
		h.canvas.width = g[0], h.canvas.height = g[1];
		var _ = l / p;
		if (_ !== 1) {
			var v = zn(this.tmpTransform_);
			Wn(v, _, _), h.setTransform.apply(h, v);
		}
		var y = d.getTileCoordExtent(o, this.tmpExtent), b = p / m, x = zn(this.tmpTransform_);
		Wn(x, b, -b), Kn(x, -y[0], -y[3]);
		for (var S = 0, C = a.length; S < C; ++S) a[S].execute(h, _, x, 0, !0, su[n.getRenderMode()]);
		r.renderedTileResolution = e.wantedResolution;
	}, t;
}(fo), uu = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), du = function(e) {
	uu(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		delete i.preload, delete i.useInterimTilesOnError, n = e.call(this, i) || this, n.on, n.once, n.un, r.renderMode === au.IMAGE && (console.warn("renderMode: \"image\" is deprecated. Option ignored."), r.renderMode = void 0);
		var a = r.renderMode || au.HYBRID;
		return H(a == au.HYBRID || a == au.VECTOR, 28), n.renderMode_ = a, n.setPreload(r.preload ? r.preload : 0), n.setUseInterimTilesOnError(r.useInterimTilesOnError === void 0 || r.useInterimTilesOnError), n;
	}
	return t.prototype.createRenderer = function() {
		return new lu(this);
	}, t.prototype.getFeatures = function(t) {
		return e.prototype.getFeatures.call(this, t);
	}, t.prototype.getRenderMode = function() {
		return this.renderMode_;
	}, t.prototype.getPreload = function() {
		return this.get(to.PRELOAD);
	}, t.prototype.getUseInterimTilesOnError = function() {
		return this.get(to.USE_INTERIM_TILES_ON_ERROR);
	}, t.prototype.setPreload = function(e) {
		this.set(to.PRELOAD, e);
	}, t.prototype.setUseInterimTilesOnError = function(e) {
		this.set(to.USE_INTERIM_TILES_ON_ERROR, e);
	}, t;
}(js), fu = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), pu = function(e) {
	fu(t, e);
	function t(t) {
		var n = this, r = t || {};
		return n = e.call(this, r) || this, n;
	}
	return t;
}(hr), mu = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), hu = function(e) {
	mu(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.image_ = null, n;
	}
	return t.prototype.getImage = function() {
		return this.image_ ? this.image_.getImage() : null;
	}, t.prototype.prepareFrame = function(e) {
		var t = e.layerStatesArray[e.layerIndex], n = e.pixelRatio, r = e.viewState, i = r.resolution, a = this.getLayer().getSource(), o = e.viewHints, s = e.extent;
		if (t.extent !== void 0 && (s = Jt(s, An(t.extent, r.projection))), !o[Y.ANIMATING] && !o[Y.INTERACTING] && !Qt(s)) if (a) {
			var c = r.projection, l = a.getImage(s, i, n, c);
			l && this.loadImage(l) && (this.image_ = l);
		} else this.image_ = null;
		return !!this.image_;
	}, t.prototype.renderFrame = function(e, t) {
		var n = this.image_, r = n.getExtent(), i = n.getResolution(), a = n.getPixelRatio(), o = e.layerStatesArray[e.layerIndex], s = e.pixelRatio, c = e.viewState, l = c.center, u = c.resolution, d = e.size, f = s * i / (u * a), p = Math.round(d[0] * s), m = Math.round(d[1] * s), h = c.rotation;
		if (h) {
			var g = Math.round(Math.sqrt(p * p + m * m));
			p = g, m = g;
		}
		qn(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / s, 1 / s, h, -p / 2, -m / 2), Jn(this.inversePixelTransform, this.pixelTransform);
		var _ = Zn(this.pixelTransform);
		this.useContainer(t, _, o.opacity);
		var v = this.context, y = v.canvas;
		y.width != p || y.height != m ? (y.width = p, y.height = m) : this.containerReused || v.clearRect(0, 0, p, m);
		var b = !1, x = !0;
		if (o.extent) {
			var C = An(o.extent, c.projection);
			x = Zt(C, e.extent), b = x && !Dt(C, e.extent), b && this.clipUnrotated(v, e, C);
		}
		var w = n.getImage(), T = qn(this.tempTransform, p / 2, m / 2, f, f, 0, a * (r[0] - l[0]) / i, a * (l[1] - r[3]) / i);
		this.renderedResolution = i * s / a;
		var E = w.width * T[0], D = w.height * T[3];
		if (S(v, this.getLayer().getSource().getContextOptions()), this.preRender(v, e), x && E >= .5 && D >= .5) {
			var O = T[4], k = T[5], A = o.opacity, j = void 0;
			A !== 1 && (j = v.globalAlpha, v.globalAlpha = A), v.drawImage(w, 0, 0, +w.width, +w.height, Math.round(O), Math.round(k), Math.round(E), Math.round(D)), A !== 1 && (v.globalAlpha = j);
		}
		return this.postRender(v, e), b && v.restore(), _ !== y.style.transform && (y.style.transform = _), this.container;
	}, t;
}(so), gu = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), _u = function(e) {
	gu(t, e);
	function t(t) {
		return e.call(this, t) || this;
	}
	return t.prototype.createRenderer = function() {
		return new hu(this);
	}, t;
}(pu), vu = 34962, yu = 34963, bu = 35040, xu = 35044, Su = 35048, Cu = 5121, wu = 5123, Tu = 5125, Eu = 5126, Du = [
	"experimental-webgl",
	"webgl",
	"webkit-3d",
	"moz-webgl"
];
function Ou(e, t) {
	for (var n = Du.length, r = 0; r < n; ++r) try {
		var i = e.getContext(Du[r], t);
		if (i) return i;
	} catch {}
	return null;
}
//#endregion
//#region node_modules/ol/webgl/Buffer.js
var ku = {
	STATIC_DRAW: xu,
	STREAM_DRAW: bu,
	DYNAMIC_DRAW: Su
}, Au = function() {
	function e(e, t) {
		this.array = null, this.type = e, H(e === 34962 || e === 34963, 62), this.usage = t === void 0 ? ku.STATIC_DRAW : t;
	}
	return e.prototype.ofSize = function(e) {
		this.array = new (ju(this.type))(e);
	}, e.prototype.fromArray = function(e) {
		var t = ju(this.type);
		this.array = t.from ? t.from(e) : new t(e);
	}, e.prototype.fromArrayBuffer = function(e) {
		this.array = new (ju(this.type))(e);
	}, e.prototype.getType = function() {
		return this.type;
	}, e.prototype.getArray = function() {
		return this.array;
	}, e.prototype.getUsage = function() {
		return this.usage;
	}, e.prototype.getSize = function() {
		return this.array ? this.array.length : 0;
	}, e;
}();
function ju(e) {
	switch (e) {
		case vu: return Float32Array;
		case yu: return Uint32Array;
		default: return Float32Array;
	}
}
//#endregion
//#region node_modules/ol/webgl/ContextEventType.js
var Mu = {
	LOST: "webglcontextlost",
	RESTORED: "webglcontextrestored"
}, Nu = "\n  precision mediump float;\n  \n  attribute vec2 a_position;\n  varying vec2 v_texCoord;\n  varying vec2 v_screenCoord;\n  \n  uniform vec2 u_screenSize;\n   \n  void main() {\n    v_texCoord = a_position * 0.5 + 0.5;\n    v_screenCoord = v_texCoord * u_screenSize;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n  }\n", Pu = "\n  precision mediump float;\n   \n  uniform sampler2D u_image;\n   \n  varying vec2 v_texCoord;\n   \n  void main() {\n    gl_FragColor = texture2D(u_image, v_texCoord);\n  }\n", Fu = function() {
	function e(e) {
		this.gl_ = e.webGlContext;
		var t = this.gl_;
		this.scaleRatio_ = e.scaleRatio || 1, this.renderTargetTexture_ = t.createTexture(), this.renderTargetTextureSize_ = null, this.frameBuffer_ = t.createFramebuffer();
		var n = t.createShader(t.VERTEX_SHADER);
		t.shaderSource(n, e.vertexShader || Nu), t.compileShader(n);
		var r = t.createShader(t.FRAGMENT_SHADER);
		t.shaderSource(r, e.fragmentShader || Pu), t.compileShader(r), this.renderTargetProgram_ = t.createProgram(), t.attachShader(this.renderTargetProgram_, n), t.attachShader(this.renderTargetProgram_, r), t.linkProgram(this.renderTargetProgram_), this.renderTargetVerticesBuffer_ = t.createBuffer(), t.bindBuffer(t.ARRAY_BUFFER, this.renderTargetVerticesBuffer_), t.bufferData(t.ARRAY_BUFFER, new Float32Array([
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
		]), t.STATIC_DRAW), this.renderTargetAttribLocation_ = t.getAttribLocation(this.renderTargetProgram_, "a_position"), this.renderTargetUniformLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_screenSize"), this.renderTargetTextureLocation_ = t.getUniformLocation(this.renderTargetProgram_, "u_image"), this.uniforms_ = [], e.uniforms && Object.keys(e.uniforms).forEach(function(n) {
			this.uniforms_.push({
				value: e.uniforms[n],
				location: t.getUniformLocation(this.renderTargetProgram_, n)
			});
		}.bind(this));
	}
	return e.prototype.getGL = function() {
		return this.gl_;
	}, e.prototype.init = function(e) {
		var t = this.getGL(), n = [t.drawingBufferWidth * this.scaleRatio_, t.drawingBufferHeight * this.scaleRatio_];
		if (t.bindFramebuffer(t.FRAMEBUFFER, this.getFrameBuffer()), t.viewport(0, 0, n[0], n[1]), !this.renderTargetTextureSize_ || this.renderTargetTextureSize_[0] !== n[0] || this.renderTargetTextureSize_[1] !== n[1]) {
			this.renderTargetTextureSize_ = n;
			var r = 0, i = t.RGBA, a = 0, o = t.RGBA, s = t.UNSIGNED_BYTE;
			t.bindTexture(t.TEXTURE_2D, this.renderTargetTexture_), t.texImage2D(t.TEXTURE_2D, r, i, n[0], n[1], a, o, s, null), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, this.renderTargetTexture_, 0);
		}
	}, e.prototype.apply = function(e, t) {
		var n = this.getGL(), r = e.size;
		n.bindFramebuffer(n.FRAMEBUFFER, t ? t.getFrameBuffer() : null), n.activeTexture(n.TEXTURE0), n.bindTexture(n.TEXTURE_2D, this.renderTargetTexture_), n.clearColor(0, 0, 0, 0), n.clear(n.COLOR_BUFFER_BIT), n.enable(n.BLEND), n.blendFunc(n.ONE, n.ONE_MINUS_SRC_ALPHA), n.viewport(0, 0, n.drawingBufferWidth, n.drawingBufferHeight), n.bindBuffer(n.ARRAY_BUFFER, this.renderTargetVerticesBuffer_), n.useProgram(this.renderTargetProgram_), n.enableVertexAttribArray(this.renderTargetAttribLocation_), n.vertexAttribPointer(this.renderTargetAttribLocation_, 2, n.FLOAT, !1, 0, 0), n.uniform2f(this.renderTargetUniformLocation_, r[0], r[1]), n.uniform1i(this.renderTargetTextureLocation_, 0), this.applyUniforms(e), n.drawArrays(n.TRIANGLES, 0, 6);
	}, e.prototype.getFrameBuffer = function() {
		return this.frameBuffer_;
	}, e.prototype.applyUniforms = function(e) {
		var t = this.getGL(), n, r = 1;
		this.uniforms_.forEach(function(i) {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof ImageData) i.texture ||= t.createTexture(), t.activeTexture(t["TEXTURE" + r]), t.bindTexture(t.TEXTURE_2D, i.texture), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), n instanceof ImageData ? t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, n.width, n.height, 0, t.UNSIGNED_BYTE, new Uint8Array(n.data)) : t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n), t.uniform1i(i.location, r++);
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
				default: return;
			}
			else typeof n == "number" && t.uniform1f(i.location, n);
		});
	}, e;
}();
//#endregion
//#region node_modules/ol/vec/mat4.js
function Iu() {
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
function Lu(e, t) {
	return e[0] = t[0], e[1] = t[1], e[4] = t[2], e[5] = t[3], e[12] = t[4], e[13] = t[5], e;
}
//#endregion
//#region node_modules/ol/webgl/Helper.js
var Ru = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), zu = {
	PROJECTION_MATRIX: "u_projectionMatrix",
	OFFSET_SCALE_MATRIX: "u_offsetScaleMatrix",
	OFFSET_ROTATION_MATRIX: "u_offsetRotateMatrix",
	TIME: "u_time",
	ZOOM: "u_zoom",
	RESOLUTION: "u_resolution"
}, Bu = {
	UNSIGNED_BYTE: Cu,
	UNSIGNED_SHORT: wu,
	UNSIGNED_INT: Tu,
	FLOAT: Eu
}, Vu = function(e) {
	Ru(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		n.boundHandleWebGLContextLost_ = n.handleWebGLContextLost.bind(n), n.boundHandleWebGLContextRestored_ = n.handleWebGLContextRestored.bind(n), n.canvas_ = document.createElement("canvas"), n.canvas_.style.position = "absolute", n.canvas_.style.left = "0", n.gl_ = Ou(n.canvas_);
		var i = n.getGL();
		if (n.bufferCache_ = {}, n.extensionCache_ = {}, n.currentProgram_ = null, n.canvas_.addEventListener(Mu.LOST, n.boundHandleWebGLContextLost_), n.canvas_.addEventListener(Mu.RESTORED, n.boundHandleWebGLContextRestored_), n.offsetRotateMatrix_ = Rn(), n.offsetScaleMatrix_ = Rn(), n.tmpMat4_ = Iu(), n.uniformLocations_ = {}, n.attribLocations_ = {}, n.uniforms_ = [], r.uniforms) for (var a in r.uniforms) n.uniforms_.push({
			name: a,
			value: r.uniforms[a]
		});
		return n.postProcessPasses_ = r.postProcesses ? r.postProcesses.map(function(e) {
			return new Fu({
				webGlContext: i,
				scaleRatio: e.scaleRatio,
				vertexShader: e.vertexShader,
				fragmentShader: e.fragmentShader,
				uniforms: e.uniforms
			});
		}) : [new Fu({ webGlContext: i })], n.shaderCompileErrors_ = null, n.startTime_ = Date.now(), n;
	}
	return t.prototype.getExtension = function(e) {
		if (e in this.extensionCache_) return this.extensionCache_[e];
		var t = this.gl_.getExtension(e);
		return this.extensionCache_[e] = t, t;
	}, t.prototype.bindBuffer = function(e) {
		var t = this.getGL(), n = L(e), r = this.bufferCache_[n];
		r || (r = {
			buffer: e,
			webGlBuffer: t.createBuffer()
		}, this.bufferCache_[n] = r), t.bindBuffer(e.getType(), r.webGlBuffer);
	}, t.prototype.flushBufferData = function(e) {
		var t = this.getGL();
		this.bindBuffer(e), t.bufferData(e.getType(), e.getArray(), e.getUsage());
	}, t.prototype.deleteBuffer = function(e) {
		var t = this.getGL(), n = L(e), r = this.bufferCache_[n];
		r && !t.isContextLost() && t.deleteBuffer(r.webGlBuffer), delete this.bufferCache_[n];
	}, t.prototype.disposeInternal = function() {
		this.canvas_.removeEventListener(Mu.LOST, this.boundHandleWebGLContextLost_), this.canvas_.removeEventListener(Mu.RESTORED, this.boundHandleWebGLContextRestored_);
		var e = this.gl_.getExtension("WEBGL_lose_context");
		e && e.loseContext(), delete this.gl_, delete this.canvas_;
	}, t.prototype.prepareDraw = function(e, t) {
		var n = this.getGL(), r = this.getCanvas(), i = e.size, a = e.pixelRatio;
		r.width = i[0] * a, r.height = i[1] * a, r.style.width = i[0] + "px", r.style.height = i[1] + "px", n.useProgram(this.currentProgram_);
		for (var o = this.postProcessPasses_.length - 1; o >= 0; o--) this.postProcessPasses_[o].init(e);
		n.bindTexture(n.TEXTURE_2D, null), n.clearColor(0, 0, 0, 0), n.clear(n.COLOR_BUFFER_BIT), n.enable(n.BLEND), n.blendFunc(n.ONE, t ? n.ZERO : n.ONE_MINUS_SRC_ALPHA), n.useProgram(this.currentProgram_), this.applyFrameState(e), this.applyUniforms(e);
	}, t.prototype.prepareDrawToRenderTarget = function(e, t, n) {
		var r = this.getGL(), i = t.getSize();
		r.bindFramebuffer(r.FRAMEBUFFER, t.getFramebuffer()), r.viewport(0, 0, i[0], i[1]), r.bindTexture(r.TEXTURE_2D, t.getTexture()), r.clearColor(0, 0, 0, 0), r.clear(r.COLOR_BUFFER_BIT), r.enable(r.BLEND), r.blendFunc(r.ONE, n ? r.ZERO : r.ONE_MINUS_SRC_ALPHA), r.useProgram(this.currentProgram_), this.applyFrameState(e), this.applyUniforms(e);
	}, t.prototype.drawElements = function(e, t) {
		var n = this.getGL();
		this.getExtension("OES_element_index_uint");
		var r = n.UNSIGNED_INT, i = 4, a = t - e, o = e * i;
		n.drawElements(n.TRIANGLES, a, r, o);
	}, t.prototype.finalizeDraw = function(e) {
		for (var t = 0; t < this.postProcessPasses_.length; t++) this.postProcessPasses_[t].apply(e, this.postProcessPasses_[t + 1] || null);
	}, t.prototype.getCanvas = function() {
		return this.canvas_;
	}, t.prototype.getGL = function() {
		return this.gl_;
	}, t.prototype.applyFrameState = function(e) {
		var t = e.size, n = e.viewState.rotation, r = zn(this.offsetScaleMatrix_);
		Wn(r, 2 / t[0], 2 / t[1]);
		var i = zn(this.offsetRotateMatrix_);
		n !== 0 && Un(i, -n), this.setUniformMatrixValue(zu.OFFSET_SCALE_MATRIX, Lu(this.tmpMat4_, r)), this.setUniformMatrixValue(zu.OFFSET_ROTATION_MATRIX, Lu(this.tmpMat4_, i)), this.setUniformFloatValue(zu.TIME, (Date.now() - this.startTime_) * .001), this.setUniformFloatValue(zu.ZOOM, e.viewState.zoom), this.setUniformFloatValue(zu.RESOLUTION, e.viewState.resolution);
	}, t.prototype.applyUniforms = function(e) {
		var t = this.getGL(), n, r = 0;
		this.uniforms_.forEach(function(i) {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof HTMLImageElement || n instanceof ImageData) i.texture ||= (i.prevValue = void 0, t.createTexture()), t.activeTexture(t["TEXTURE" + r]), t.bindTexture(t.TEXTURE_2D, i.texture), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), (!(n instanceof HTMLImageElement) || n.complete) && i.prevValue !== n && (i.prevValue = n, t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n)), t.uniform1i(this.getUniformLocation(i.name), r++);
			else if (Array.isArray(n) && n.length === 6) this.setUniformMatrixValue(i.name, Lu(this.tmpMat4_, n));
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
		}.bind(this));
	}, t.prototype.useProgram = function(e) {
		return e == this.currentProgram_ ? !1 : (this.getGL().useProgram(e), this.currentProgram_ = e, this.uniformLocations_ = {}, this.attribLocations_ = {}, !0);
	}, t.prototype.compileShader = function(e, t) {
		var n = this.getGL(), r = n.createShader(t);
		return n.shaderSource(r, e), n.compileShader(r), r;
	}, t.prototype.getProgram = function(e, t) {
		var n = this.getGL(), r = this.compileShader(e, n.FRAGMENT_SHADER), i = this.compileShader(t, n.VERTEX_SHADER), a = n.createProgram();
		if (n.attachShader(a, r), n.attachShader(a, i), n.linkProgram(a), !n.getShaderParameter(r, n.COMPILE_STATUS)) {
			var o = "Fragment shader compliation failed: " + n.getShaderInfoLog(r);
			throw Error(o);
		}
		if (n.deleteShader(r), !n.getShaderParameter(i, n.COMPILE_STATUS)) {
			var o = "Vertex shader compilation failed: " + n.getShaderInfoLog(i);
			throw Error(o);
		}
		if (n.deleteShader(i), !n.getProgramParameter(a, n.LINK_STATUS)) {
			var o = "GL program linking failed: " + n.getShaderInfoLog(i);
			throw Error(o);
		}
		return a;
	}, t.prototype.getUniformLocation = function(e) {
		return this.uniformLocations_[e] === void 0 && (this.uniformLocations_[e] = this.getGL().getUniformLocation(this.currentProgram_, e)), this.uniformLocations_[e];
	}, t.prototype.getAttributeLocation = function(e) {
		return this.attribLocations_[e] === void 0 && (this.attribLocations_[e] = this.getGL().getAttribLocation(this.currentProgram_, e)), this.attribLocations_[e];
	}, t.prototype.makeProjectionTransform = function(e, t) {
		var n = e.size, r = e.viewState.rotation, i = e.viewState.resolution, a = e.viewState.center;
		return zn(t), qn(t, 0, 0, 2 / (i * n[0]), 2 / (i * n[1]), -r, -a[0], -a[1]), t;
	}, t.prototype.setUniformFloatValue = function(e, t) {
		this.getGL().uniform1f(this.getUniformLocation(e), t);
	}, t.prototype.setUniformMatrixValue = function(e, t) {
		this.getGL().uniformMatrix4fv(this.getUniformLocation(e), !1, t);
	}, t.prototype.enableAttributeArray_ = function(e, t, n, r, i) {
		var a = this.getAttributeLocation(e);
		a < 0 || (this.getGL().enableVertexAttribArray(a), this.getGL().vertexAttribPointer(a, t, n, !1, r, i));
	}, t.prototype.enableAttributes = function(e) {
		for (var t = Hu(e), n = 0, r = 0; r < e.length; r++) {
			var i = e[r];
			this.enableAttributeArray_(i.name, i.size, i.type || 5126, t, n), n += i.size * Uu(i.type);
		}
	}, t.prototype.handleWebGLContextLost = function() {
		C(this.bufferCache_), this.currentProgram_ = null;
	}, t.prototype.handleWebGLContextRestored = function() {}, t.prototype.createTexture = function(e, t, n) {
		var r = this.getGL(), i = n || r.createTexture(), a = 0, o = r.RGBA, s = 0, c = r.RGBA, l = r.UNSIGNED_BYTE;
		return r.bindTexture(r.TEXTURE_2D, i), t ? r.texImage2D(r.TEXTURE_2D, a, o, c, l, t) : r.texImage2D(r.TEXTURE_2D, a, o, e[0], e[1], s, c, l, null), r.texParameteri(r.TEXTURE_2D, r.TEXTURE_MIN_FILTER, r.LINEAR), r.texParameteri(r.TEXTURE_2D, r.TEXTURE_WRAP_S, r.CLAMP_TO_EDGE), r.texParameteri(r.TEXTURE_2D, r.TEXTURE_WRAP_T, r.CLAMP_TO_EDGE), i;
	}, t;
}(d);
function Hu(e) {
	for (var t = 0, n = 0; n < e.length; n++) {
		var r = e[n];
		t += r.size * Uu(r.type);
	}
	return t;
}
function Uu(e) {
	switch (e) {
		case Bu.UNSIGNED_BYTE: return Uint8Array.BYTES_PER_ELEMENT;
		case Bu.UNSIGNED_SHORT: return Uint16Array.BYTES_PER_ELEMENT;
		case Bu.UNSIGNED_INT: return Uint32Array.BYTES_PER_ELEMENT;
		case Bu.FLOAT:
		default: return Float32Array.BYTES_PER_ELEMENT;
	}
}
//#endregion
//#region node_modules/ol/renderer/webgl/Layer.js
var Wu = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), Gu = { GENERATE_BUFFERS: "GENERATE_BUFFERS" }, Ku = function(e) {
	Wu(t, e);
	function t(t, n) {
		var r = e.call(this, t) || this, i = n || {};
		return r.helper = new Vu({
			postProcesses: i.postProcesses,
			uniforms: i.uniforms
		}), i.className !== void 0 && (r.helper.getCanvas().className = i.className), r;
	}
	return t.prototype.disposeInternal = function() {
		this.helper.dispose(), delete this.helper, e.prototype.disposeInternal.call(this);
	}, t.prototype.dispatchRenderEvent_ = function(e, t) {
		var n = this.getLayer();
		if (n.hasListener(e)) {
			var r = new xr(e, null, t, null);
			n.dispatchEvent(r);
		}
	}, t.prototype.preRender = function(e) {
		this.dispatchRenderEvent_(fr.PRERENDER, e);
	}, t.prototype.postRender = function(e) {
		this.dispatchRenderEvent_(fr.POSTRENDER, e);
	}, t;
}(ao);
function qu(e, t) {
	var n = t || [], r = 256, i = r - 1;
	return n[0] = Math.floor(e / r / r / r) / i, n[1] = Math.floor(e / r / r) % r / i, n[2] = Math.floor(e / r) % r / i, n[3] = e % r / i, n;
}
function Ju(e) {
	var t = 0, n = 256, r = n - 1;
	return t += Math.round(e[0] * n * n * n * r), t += Math.round(e[1] * n * n * r), t += Math.round(e[2] * n * r), t += Math.round(e[3] * r), t;
}
//#endregion
//#region node_modules/ol/webgl/RenderTarget.js
var Yu = /* @__PURE__ */ new Uint8Array(4), Xu = function() {
	function e(e, t) {
		this.helper_ = e;
		var n = e.getGL();
		this.texture_ = n.createTexture(), this.framebuffer_ = n.createFramebuffer(), this.size_ = t || [1, 1], this.data_ = /* @__PURE__ */ new Uint8Array(), this.dataCacheDirty_ = !0, this.updateSize_();
	}
	return e.prototype.setSize = function(e) {
		g(e, this.size_) || (this.size_[0] = e[0], this.size_[1] = e[1], this.updateSize_());
	}, e.prototype.getSize = function() {
		return this.size_;
	}, e.prototype.clearCachedData = function() {
		this.dataCacheDirty_ = !0;
	}, e.prototype.readAll = function() {
		if (this.dataCacheDirty_) {
			var e = this.size_, t = this.helper_.getGL();
			t.bindFramebuffer(t.FRAMEBUFFER, this.framebuffer_), t.readPixels(0, 0, e[0], e[1], t.RGBA, t.UNSIGNED_BYTE, this.data_), this.dataCacheDirty_ = !1;
		}
		return this.data_;
	}, e.prototype.readPixel = function(e, t) {
		if (e < 0 || t < 0 || e > this.size_[0] || t >= this.size_[1]) return Yu[0] = 0, Yu[1] = 0, Yu[2] = 0, Yu[3] = 0, Yu;
		this.readAll();
		var n = Math.floor(e) + (this.size_[1] - Math.floor(t) - 1) * this.size_[0];
		return Yu[0] = this.data_[n * 4], Yu[1] = this.data_[n * 4 + 1], Yu[2] = this.data_[n * 4 + 2], Yu[3] = this.data_[n * 4 + 3], Yu;
	}, e.prototype.getTexture = function() {
		return this.texture_;
	}, e.prototype.getFramebuffer = function() {
		return this.framebuffer_;
	}, e.prototype.updateSize_ = function() {
		var e = this.size_, t = this.helper_.getGL();
		this.texture_ = this.helper_.createTexture(e, null, this.texture_), t.bindFramebuffer(t.FRAMEBUFFER, this.framebuffer_), t.viewport(0, 0, e[0], e[1]), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, this.texture_, 0), this.data_ = new Uint8Array(e[0] * e[1] * 4);
	}, e;
}(), Zu = new Blob(["var e=\"function\"==typeof Object.assign?Object.assign:function(e,n){if(null==e)throw new TypeError(\"Cannot convert undefined or null to object\");for(var t=Object(e),r=1,o=arguments.length;r<o;++r){var i=arguments[r];if(null!=i)for(var f in i)i.hasOwnProperty(f)&&(t[f]=i[f])}return t},n=\"GENERATE_BUFFERS\",t=[],r={vertexPosition:0,indexPosition:0};function o(e,n,t,r,o){e[n+0]=t,e[n+1]=r,e[n+2]=o}function i(e,n,i,f,s,u){var a=3+s,l=e[n+0],v=e[n+1],c=t;c.length=s;for(var g=0;g<c.length;g++)c[g]=e[n+2+g];var b=u?u.vertexPosition:0,h=u?u.indexPosition:0,d=b/a;return o(i,b,l,v,0),c.length&&i.set(c,b+3),o(i,b+=a,l,v,1),c.length&&i.set(c,b+3),o(i,b+=a,l,v,2),c.length&&i.set(c,b+3),o(i,b+=a,l,v,3),c.length&&i.set(c,b+3),b+=a,f[h++]=d,f[h++]=d+1,f[h++]=d+3,f[h++]=d+1,f[h++]=d+2,f[h++]=d+3,r.vertexPosition=b,r.indexPosition=h,r}var f=self;f.onmessage=function(t){var r=t.data;if(r.type===n){for(var o=r.customAttributesCount,s=2+o,u=new Float32Array(r.renderInstructions),a=u.length/s,l=4*a*(o+3),v=new Uint32Array(6*a),c=new Float32Array(l),g=null,b=0;b<u.length;b+=s)g=i(u,b,c,v,o,g);var h=e({vertexBuffer:c.buffer,indexBuffer:v.buffer,renderInstructions:u.buffer},r);f.postMessage(h,[c.buffer,v.buffer,u.buffer])}};"], { type: "application/javascript" }), Qu = URL.createObjectURL(Zu);
function $u() {
	return new Worker(Qu);
}
//#endregion
//#region node_modules/ol/renderer/webgl/PointsLayer.js
var ed = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), td = function(e) {
	ed(t, e);
	function t(t, n) {
		var r = this, i = n.uniforms || {}, a = Rn();
		i[zu.PROJECTION_MATRIX] = a, r = e.call(this, t, {
			className: n.className,
			uniforms: i,
			postProcesses: n.postProcesses
		}) || this, r.sourceRevision_ = -1, r.verticesBuffer_ = new Au(vu, Su), r.hitVerticesBuffer_ = new Au(vu, Su), r.indicesBuffer_ = new Au(yu, Su), r.program_ = r.helper.getProgram(n.fragmentShader, n.vertexShader), r.hitDetectionEnabled_ = !!(n.hitFragmentShader && n.hitVertexShader), r.hitProgram_ = r.hitDetectionEnabled_ && r.helper.getProgram(n.hitFragmentShader, n.hitVertexShader);
		var o = n.attributes ? n.attributes.map(function(e) {
			return {
				name: "a_" + e.name,
				size: 1,
				type: Bu.FLOAT
			};
		}) : [];
		r.attributes = [{
			name: "a_position",
			size: 2,
			type: Bu.FLOAT
		}, {
			name: "a_index",
			size: 1,
			type: Bu.FLOAT
		}].concat(o), r.hitDetectionAttributes = [
			{
				name: "a_position",
				size: 2,
				type: Bu.FLOAT
			},
			{
				name: "a_index",
				size: 1,
				type: Bu.FLOAT
			},
			{
				name: "a_hitColor",
				size: 4,
				type: Bu.FLOAT
			},
			{
				name: "a_featureUid",
				size: 1,
				type: Bu.FLOAT
			}
		].concat(o), r.customAttributes = n.attributes ? n.attributes : [], r.previousExtent_ = At(), r.currentTransform_ = a, r.renderTransform_ = Rn(), r.invertRenderTransform_ = Rn(), r.renderInstructions_ = /* @__PURE__ */ new Float32Array(), r.hitRenderInstructions_ = /* @__PURE__ */ new Float32Array(), r.hitRenderTarget_ = r.hitDetectionEnabled_ && new Xu(r.helper), r.worker_ = $u(), r.worker_.addEventListener("message", function(e) {
			var t = e.data;
			if (t.type === Gu.GENERATE_BUFFERS) {
				var n = t.projectionTransform;
				t.hitDetection ? (this.hitVerticesBuffer_.fromArrayBuffer(t.vertexBuffer), this.helper.flushBufferData(this.hitVerticesBuffer_)) : (this.verticesBuffer_.fromArrayBuffer(t.vertexBuffer), this.helper.flushBufferData(this.verticesBuffer_)), this.indicesBuffer_.fromArrayBuffer(t.indexBuffer), this.helper.flushBufferData(this.indicesBuffer_), this.renderTransform_ = n, Jn(this.invertRenderTransform_, this.renderTransform_), t.hitDetection ? this.hitRenderInstructions_ = new Float32Array(e.data.renderInstructions) : this.renderInstructions_ = new Float32Array(e.data.renderInstructions), this.getLayer().changed();
			}
		}.bind(r)), r.featureCache_ = {}, r.featureCount_ = 0;
		var s = r.getLayer().getSource();
		return r.sourceListenKeys_ = [
			k(s, $c.ADDFEATURE, r.handleSourceFeatureAdded_, r),
			k(s, $c.CHANGEFEATURE, r.handleSourceFeatureChanged_, r),
			k(s, $c.REMOVEFEATURE, r.handleSourceFeatureDelete_, r),
			k(s, $c.CLEAR, r.handleSourceFeatureClear_, r)
		], s.forEachFeature(function(e) {
			this.featureCache_[L(e)] = {
				feature: e,
				properties: e.getProperties(),
				geometry: e.getGeometry()
			}, this.featureCount_++;
		}.bind(r)), r;
	}
	return t.prototype.handleSourceFeatureAdded_ = function(e) {
		var t = e.feature;
		this.featureCache_[L(t)] = {
			feature: t,
			properties: t.getProperties(),
			geometry: t.getGeometry()
		}, this.featureCount_++;
	}, t.prototype.handleSourceFeatureChanged_ = function(e) {
		var t = e.feature;
		this.featureCache_[L(t)] = {
			feature: t,
			properties: t.getProperties(),
			geometry: t.getGeometry()
		};
	}, t.prototype.handleSourceFeatureDelete_ = function(e) {
		var t = e.feature;
		delete this.featureCache_[L(t)], this.featureCount_--;
	}, t.prototype.handleSourceFeatureClear_ = function() {
		this.featureCache_ = {}, this.featureCount_ = 0;
	}, t.prototype.renderFrame = function(e) {
		this.preRender(e);
		var t = this.indicesBuffer_.getSize();
		this.helper.drawElements(0, t), this.helper.finalizeDraw(e);
		var n = this.helper.getCanvas(), r = e.layerStatesArray[e.layerIndex].opacity;
		return r !== parseFloat(n.style.opacity) && (n.style.opacity = String(r)), this.hitDetectionEnabled_ && (this.renderHitDetection(e), this.hitRenderTarget_.clearCachedData()), this.postRender(e), n;
	}, t.prototype.prepareFrame = function(e) {
		var t = this.getLayer(), n = t.getSource(), r = e.viewState, i = !e.viewHints[Y.ANIMATING] && !e.viewHints[Y.INTERACTING], a = !Ft(this.previousExtent_, e.extent), o = this.sourceRevision_ < n.getRevision();
		if (o && (this.sourceRevision_ = n.getRevision()), i && (a || o)) {
			var s = r.projection, c = r.resolution, l = t instanceof js ? t.getRenderBuffer() : 0, u = Ct(e.extent, l * c);
			n.loadFeatures(u, c, s), this.rebuildBuffers_(e), this.previousExtent_ = e.extent.slice();
		}
		return this.helper.makeProjectionTransform(e, this.currentTransform_), Bn(this.currentTransform_, this.invertRenderTransform_), this.helper.useProgram(this.program_), this.helper.prepareDraw(e), this.helper.bindBuffer(this.verticesBuffer_), this.helper.bindBuffer(this.indicesBuffer_), this.helper.enableAttributes(this.attributes), !0;
	}, t.prototype.rebuildBuffers_ = function(e) {
		var t = Rn();
		this.helper.makeProjectionTransform(e, t);
		var n = (2 + this.customAttributes.length) * this.featureCount_;
		if ((!this.renderInstructions_ || this.renderInstructions_.length !== n) && (this.renderInstructions_ = new Float32Array(n)), this.hitDetectionEnabled_) {
			var r = (7 + this.customAttributes.length) * this.featureCount_;
			(!this.hitRenderInstructions_ || this.hitRenderInstructions_.length !== r) && (this.hitRenderInstructions_ = new Float32Array(r));
		}
		var i, a, o = [], s = [], c = 0, l = 0, u;
		for (var d in this.featureCache_) if (i = this.featureCache_[d], a = i.geometry, !(!a || a.getType() !== W.POINT)) {
			o[0] = a.getFlatCoordinates()[0], o[1] = a.getFlatCoordinates()[1], G(t, o), u = qu(l + 6, s), this.renderInstructions_[c++] = o[0], this.renderInstructions_[c++] = o[1], this.hitDetectionEnabled_ && (this.hitRenderInstructions_[l++] = o[0], this.hitRenderInstructions_[l++] = o[1], this.hitRenderInstructions_[l++] = u[0], this.hitRenderInstructions_[l++] = u[1], this.hitRenderInstructions_[l++] = u[2], this.hitRenderInstructions_[l++] = u[3], this.hitRenderInstructions_[l++] = Number(d));
			for (var f = void 0, p = 0; p < this.customAttributes.length; p++) f = this.customAttributes[p].callback(i.feature, i.properties), this.renderInstructions_[c++] = f, this.hitDetectionEnabled_ && (this.hitRenderInstructions_[l++] = f);
		}
		var m = {
			type: Gu.GENERATE_BUFFERS,
			renderInstructions: this.renderInstructions_.buffer,
			customAttributesCount: this.customAttributes.length
		};
		if (m.projectionTransform = t, this.worker_.postMessage(m, [this.renderInstructions_.buffer]), this.renderInstructions_ = null, this.hitDetectionEnabled_) {
			var h = {
				type: Gu.GENERATE_BUFFERS,
				renderInstructions: this.hitRenderInstructions_.buffer,
				customAttributesCount: 5 + this.customAttributes.length
			};
			h.projectionTransform = t, h.hitDetection = !0, this.worker_.postMessage(h, [this.hitRenderInstructions_.buffer]), this.hitRenderInstructions_ = null;
		}
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {
		if (H(this.hitDetectionEnabled_, 66), this.hitRenderInstructions_) {
			var a = G(t.coordinateToPixelTransform, e.slice()), o = this.hitRenderTarget_.readPixel(a[0] / 2, a[1] / 2), s = Ju([
				o[0] / 255,
				o[1] / 255,
				o[2] / 255,
				o[3] / 255
			]), c = this.hitRenderInstructions_[s], l = Math.floor(c).toString(), u = this.getLayer().getSource().getFeatureByUid(l);
			if (u) return r(u, this.getLayer(), null);
		}
	}, t.prototype.renderHitDetection = function(e) {
		if (this.hitVerticesBuffer_.getSize()) {
			this.hitRenderTarget_.setSize([Math.floor(e.size[0] / 2), Math.floor(e.size[1] / 2)]), this.helper.useProgram(this.hitProgram_), this.helper.prepareDrawToRenderTarget(e, this.hitRenderTarget_, !0), this.helper.bindBuffer(this.hitVerticesBuffer_), this.helper.bindBuffer(this.indicesBuffer_), this.helper.enableAttributes(this.hitDetectionAttributes);
			var t = this.indicesBuffer_.getSize();
			this.helper.drawElements(0, t);
		}
	}, t.prototype.disposeInternal = function() {
		this.worker_.terminate(), this.layer_ = null, this.sourceListenKeys_.forEach(function(e) {
			j(e);
		}), this.sourceListenKeys_ = null, e.prototype.disposeInternal.call(this);
	}, t;
}(Ku), nd = (function() {
	var e = function(t, n) {
		return e = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(e, t) {
			e.__proto__ = t;
		} || function(e, t) {
			for (var n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
		}, e(t, n);
	};
	return function(t, n) {
		if (typeof n != "function" && n !== null) throw TypeError("Class extends value " + String(n) + " is not a constructor or null");
		e(t, n);
		function r() {
			this.constructor = t;
		}
		t.prototype = n === null ? Object.create(n) : (r.prototype = n.prototype, new r());
	};
})(), rd = {
	BLUR: "blur",
	GRADIENT: "gradient",
	RADIUS: "radius"
}, id = [
	"#00f",
	"#0ff",
	"#0f0",
	"#ff0",
	"#f00"
], ad = function(e) {
	nd(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		delete i.gradient, delete i.radius, delete i.blur, delete i.weight, n = e.call(this, i) || this, n.gradient_ = null, n.addChangeListener(rd.GRADIENT, n.handleGradientChanged_), n.setGradient(r.gradient ? r.gradient : id), n.setBlur(r.blur === void 0 ? 15 : r.blur), n.setRadius(r.radius === void 0 ? 8 : r.radius);
		var a = r.weight ? r.weight : "weight";
		return typeof a == "string" ? n.weightFunction_ = function(e) {
			return e.get(a);
		} : n.weightFunction_ = a, n.setRenderOrder(null), n;
	}
	return t.prototype.getBlur = function() {
		return this.get(rd.BLUR);
	}, t.prototype.getGradient = function() {
		return this.get(rd.GRADIENT);
	}, t.prototype.getRadius = function() {
		return this.get(rd.RADIUS);
	}, t.prototype.handleGradientChanged_ = function() {
		this.gradient_ = od(this.getGradient());
	}, t.prototype.setBlur = function(e) {
		this.set(rd.BLUR, e);
	}, t.prototype.setGradient = function(e) {
		this.set(rd.GRADIENT, e);
	}, t.prototype.setRadius = function(e) {
		this.set(rd.RADIUS, e);
	}, t.prototype.createRenderer = function() {
		return new td(this, {
			className: this.getClassName(),
			attributes: [{
				name: "weight",
				callback: function(e) {
					var t = this.weightFunction_(e);
					return t === void 0 ? 1 : V(t, 0, 1);
				}.bind(this)
			}],
			vertexShader: "\n        precision mediump float;\n        uniform mat4 u_projectionMatrix;\n        uniform mat4 u_offsetScaleMatrix;\n        uniform float u_size;\n        attribute vec2 a_position;\n        attribute float a_index;\n        attribute float a_weight;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n\n        void main(void) {\n          mat4 offsetMatrix = u_offsetScaleMatrix;\n          float offsetX = a_index == 0.0 || a_index == 3.0 ? -u_size / 2.0 : u_size / 2.0;\n          float offsetY = a_index == 0.0 || a_index == 1.0 ? -u_size / 2.0 : u_size / 2.0;\n          vec4 offsets = offsetMatrix * vec4(offsetX, offsetY, 0.0, 0.0);\n          gl_Position = u_projectionMatrix * vec4(a_position, 0.0, 1.0) + offsets;\n          float u = a_index == 0.0 || a_index == 3.0 ? 0.0 : 1.0;\n          float v = a_index == 0.0 || a_index == 1.0 ? 0.0 : 1.0;\n          v_texCoord = vec2(u, v);\n          v_weight = a_weight;\n        }",
			fragmentShader: "\n        precision mediump float;\n        uniform float u_blurSlope;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n\n        void main(void) {\n          vec2 texCoord = v_texCoord * 2.0 - vec2(1.0, 1.0);\n          float sqRadius = texCoord.x * texCoord.x + texCoord.y * texCoord.y;\n          float value = (1.0 - sqrt(sqRadius)) * u_blurSlope;\n          float alpha = smoothstep(0.0, 1.0, value) * v_weight;\n          gl_FragColor = vec4(alpha, alpha, alpha, alpha);\n        }",
			hitVertexShader: "\n        precision mediump float;\n        uniform mat4 u_projectionMatrix;\n        uniform mat4 u_offsetScaleMatrix;\n        uniform float u_size;\n        attribute vec2 a_position;\n        attribute float a_index;\n        attribute float a_weight;\n        attribute vec4 a_hitColor;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n        varying vec4 v_hitColor;\n\n        void main(void) {\n          mat4 offsetMatrix = u_offsetScaleMatrix;\n          float offsetX = a_index == 0.0 || a_index == 3.0 ? -u_size / 2.0 : u_size / 2.0;\n          float offsetY = a_index == 0.0 || a_index == 1.0 ? -u_size / 2.0 : u_size / 2.0;\n          vec4 offsets = offsetMatrix * vec4(offsetX, offsetY, 0.0, 0.0);\n          gl_Position = u_projectionMatrix * vec4(a_position, 0.0, 1.0) + offsets;\n          float u = a_index == 0.0 || a_index == 3.0 ? 0.0 : 1.0;\n          float v = a_index == 0.0 || a_index == 1.0 ? 0.0 : 1.0;\n          v_texCoord = vec2(u, v);\n          v_hitColor = a_hitColor;\n          v_weight = a_weight;\n        }",
			hitFragmentShader: "\n        precision mediump float;\n        uniform float u_blurSlope;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n        varying vec4 v_hitColor;\n\n        void main(void) {\n          vec2 texCoord = v_texCoord * 2.0 - vec2(1.0, 1.0);\n          float sqRadius = texCoord.x * texCoord.x + texCoord.y * texCoord.y;\n          float value = (1.0 - sqrt(sqRadius)) * u_blurSlope;\n          float alpha = smoothstep(0.0, 1.0, value) * v_weight;\n          if (alpha < 0.05) {\n            discard;\n          }\n\n          gl_FragColor = v_hitColor;\n        }",
			uniforms: {
				u_size: function() {
					return (this.get(rd.RADIUS) + this.get(rd.BLUR)) * 2;
				}.bind(this),
				u_blurSlope: function() {
					return this.get(rd.RADIUS) / Math.max(1, this.get(rd.BLUR));
				}.bind(this)
			},
			postProcesses: [{
				fragmentShader: "\n            precision mediump float;\n\n            uniform sampler2D u_image;\n            uniform sampler2D u_gradientTexture;\n\n            varying vec2 v_texCoord;\n\n            void main() {\n              vec4 color = texture2D(u_image, v_texCoord);\n              gl_FragColor.a = color.a;\n              gl_FragColor.rgb = texture2D(u_gradientTexture, vec2(0.5, color.a)).rgb;\n              gl_FragColor.rgb *= gl_FragColor.a;\n            }",
				uniforms: { u_gradientTexture: function() {
					return this.gradient_;
				}.bind(this) }
			}]
		});
	}, t.prototype.renderDeclutter = function() {}, t;
}(Yc);
function od(e) {
	for (var t = 1, n = 256, r = de(t, n), i = r.createLinearGradient(0, 0, t, n), a = 1 / (e.length - 1), o = 0, s = e.length; o < s; ++o) i.addColorStop(o * a, e[o]);
	return r.fillStyle = i, r.fillRect(0, 0, t, n), r.canvas;
}
//#endregion
//#region node_modules/ol-ext/util/input/Base.js
var sd = class extends z {
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
}, cd = class extends sd {
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
}, ld = class extends cd {
	constructor(e) {
		e ||= {}, super(e), this.element.className = ("ol-ext-toggle-switch " + (e.className || "")).trim();
	}
}, ud = class extends cd {
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
		default:
			n.setAttribute(r, t[r]);
			break;
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
	return new ld(Object.assign({ input: t }, e || {})), t;
}, $.createCheck = function(e) {
	var t = $.create("INPUT", {
		name: e.name,
		type: e.type === "radio" ? "radio" : "checkbox",
		on: e.on,
		parent: e.parent
	}), n = Object.assign({ input: t }, e || {});
	return e.type === "radio" ? new ud(n) : new cd(n), t;
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
	e instanceof nl && (e = e.getTargetElement()), !("ontouchstart" in window) && e instanceof Element && (e.style.cursor = t);
};
//#endregion
//#region node_modules/ol-ext/control/LayerSwitcher.js
var dd = class extends ye {
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
		if (super.setMap(e), this.drawPanel(), this._listener) for (var t in this._listener) P(this._listener[t]);
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
					case "-50%":
						i -= Math.round(t / 2);
						break;
					default: break;
				}
				return i + n <= t - 3 * r / 2 ? (i = t - 3 * r / 2 - n, $.hide(this.botv)) : $.show(this.botv), i >= 0 ? (i = 0, $.hide(this.topv)) : $.show(this.topv), $.setStyle(this.panel_, { top: i + "px" }), !0;
			} else return $.setStyle(this.element, { height: "auto" }), $.setStyle(this.panel_, { top: 0 }), $.hide(this.botv), $.hide(this.topv), !1;
		} else return !1;
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
				P(e);
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
		return !r || Zt(this.getMap().getView().calculateExtent(this.getMap().getSize()), r);
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
			if (this.show_progress && c instanceof mo) {
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
		return e ? e.getLayers ? "ol-layer-group" : e instanceof Yc ? "ol-layer-vector" : e instanceof du ? "ol-layer-vectortile" : e instanceof mo ? "ol-layer-tile" : e instanceof _u ? "ol-layer-image" : e instanceof ad ? "ol-layer-heatmap" : e.getFeatures ? "ol-layer-vectorimage" : "unknown" : "none";
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
dd.prototype.tip = {
	up: "up/down",
	down: "down",
	info: "informations...",
	extent: "zoom to extent",
	trash: "remove layer",
	plus: "expand/shrink"
};
//#endregion
//#region node_modules/ol-ext/control/Button.js
var fd = class extends ye {
	constructor(e) {
		e ||= {};
		var t = document.createElement("div");
		t.className = (e.className || "") + " ol-button ol-unselectable ol-control", super({
			element: t,
			target: e.target
		});
		var n = this, r = this.button_ = document.createElement(/ol-text-button/.test(e.className) ? "div" : "button");
		for (var i in e.id ? r.setAttribute("id", e.id) : r.setAttribute("id", "ol-button-" + L(this)), this.button_.className = e.classButton || "", r.type = "button", e.title && (r.title = e.title), e.name && (r.name = e.name), e.html instanceof Element ? r.appendChild(e.html) : r.innerHTML = e.html || "", r.addEventListener("click", function(t) {
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
}, pd = class extends fd {
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
}, md = document.getElementById("map");
if (md === null) throw Error("The OpenLayers map element could not be found.");
var hd = JSON.parse(md.dataset.layers ?? "[]");
if (hd.length === 0) throw Error("No layers were supplied to the OpenLayers viewer.");
var gd = hd.map((e) => {
	let t = new tu({
		url: e.url,
		size: e.size,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	return new mo({
		title: e.name,
		source: t
	});
}), _d = gd[0].getSource(), vd = _d.getTileGrid(), yd = vd.getResolutions(), bd = vd.getExtent(), xd = new Ve({
	code: "ZoomifyProjection",
	units: "pixels",
	extent: bd,
	metersPerUnit: hd[0].mpp * 1e-6,
	getPointResolution(e) {
		return e;
	}
});
mn(xd);
var Sd = new ka({
	projection: xd,
	resolutions: yd,
	constrainOnlyCenter: !0
}), Cd = new nl({
	target: md,
	layers: gd,
	view: Sd
}), wd = new eo({
	units: "metric",
	bar: !0,
	steps: 10,
	minWidth: 256
});
Cd.addControl(wd);
var Td = new Ka({
	className: "ol-overviewmap ol-custom-overviewmap",
	layers: [new mo({ source: _d })]
});
Cd.addControl(Td);
var Ed = new In({
	coordinateFormat: (e) => on([e[0], -e[1]], "{x}, {y}", 0),
	projection: xd,
	className: "ol-mouse-position",
	placeholder: "\xA0"
});
Cd.addControl(Ed);
var Dd = new Ja({
	autoHide: !1,
	className: "ol-rotate"
});
Cd.addControl(Dd);
var Od = new Ne();
Cd.addControl(Od);
var kd = new dd();
Cd.addControl(kd);
var Ad = 64, jd = 64, Md = new Cs({
	stroke: new Ss({
		color: "rgba(0, 0, 0, 0.5)",
		width: 1
	}),
	text: new Ec({
		font: "12px Calibri,sans-serif",
		fill: new xs({ color: "rgba(0, 0, 0, 1)" }),
		stroke: new Ss({
			color: "rgba(255, 255, 255, 1)",
			width: 3
		})
	})
}), Nd = new iu({
	projection: xd.getCode(),
	margin: jd,
	style: Md,
	spacing: Ad,
	formatCoord(e, t) {
		let n;
		return n = t === "left" || t === "right" ? -Math.floor(e) : Math.floor(e), n >= 1e6 && (n = n.toExponential(3), n = n.replace("+", "")), n;
	}
}), Pd = Ad, Fd = jd, Id = new iu({
	projection: xd.getCode(),
	spacing: Pd,
	margin: Fd,
	style: Md,
	formatCoord(e, t) {
		let n = Cd.getView().calculateExtent(Cd.getSize()), r = Cd.getView().getResolution(), i = n[0] + r * Fd, a = n[3] - r * Fd, o;
		if (o = t === "left" || t === "right" ? -(e - a) : e - i, o = Math.floor(o / r / Pd), t === "left" || t === "right") {
			let e = "";
			do
				e += String.fromCharCode(65 + o % 26), o = Math.floor(o / 26);
			while (o > 0);
			return e.split("").reverse().join("");
		}
		return o;
	}
}), Ld = new pd({
	html: "<i class=\"fas fa-ruler-combined\"></i>",
	className: "ol-graticule",
	title: "Toggle Graticule",
	onToggle(e) {
		e ? (Rd.setActive(!1), Id.setMap(null), Nd.setMap(Cd)) : Nd.setMap(null);
	}
}), Rd;
Cd.addControl(Ld), Rd = new pd({
	html: "<i class=\"fas fa-border-all\"></i>",
	className: "ol-screen-space-graticule",
	title: "Toggle Screen Space Graticule",
	onToggle(e) {
		e ? (Ld.setActive(!1), Nd.setMap(null), Id.setMap(Cd)) : Id.setMap(null);
	}
}), Cd.addControl(Rd), Cd.getView().fit(bd), Object.assign(window, {
	extent: bd,
	fullscreen: Od,
	graticule: Nd,
	graticuleToggle: Ld,
	layerSwitcher: kd,
	layers: gd,
	layersData: hd,
	map: Cd,
	mousePositionControl: Ed,
	overviewMapControl: Td,
	projection: xd,
	resolutions: yd,
	rotate: Dd,
	scaleLineControl: wd,
	screenSpaceGraticule: Id,
	screenSpaceGraticuleToggle: Rd,
	view: Sd
});
//#endregion
