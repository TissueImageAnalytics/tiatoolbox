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
var ee = 0;
function I(e) {
	return e.ol_uid ||= String(++ee);
}
var L = "6.9.0", te = (function() {
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
})(), ne = function(e) {
	te(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.key = n, i.oldValue = r, i;
	}
	return t;
}(l), R = function(e) {
	te(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.on, n.once, n.un, I(n), n.values_ = null, t !== void 0 && n.setProperties(t), n;
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
		this.dispatchEvent(new ne(n, e, t)), n = u.PROPERTYCHANGE, this.dispatchEvent(new ne(n, e, t));
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
}(N), re = {
	POSTRENDER: "postrender",
	MOVESTART: "movestart",
	MOVEEND: "moveend"
}, ie = typeof navigator < "u" && navigator.userAgent !== void 0 ? navigator.userAgent.toLowerCase() : "", ae = ie.indexOf("firefox") !== -1;
ie.indexOf("safari") !== -1 && ie.indexOf("chrom");
var oe = ie.indexOf("webkit") !== -1 && ie.indexOf("edge") == -1, se = ie.indexOf("macintosh") !== -1, ce = typeof devicePixelRatio < "u" ? devicePixelRatio : 1, le = typeof WorkerGlobalScope < "u" && typeof OffscreenCanvas < "u" && self instanceof WorkerGlobalScope, ue = typeof Image < "u" && Image.prototype.decode, de = (function() {
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
function fe(e, t, n, r) {
	var i;
	return n && n.length ? i = n.shift() : le ? i = new OffscreenCanvas(e || 300, t || 300) : (i = document.createElement("canvas"), i.style.all = "unset"), e && (i.width = e), t && (i.height = t), i.getContext("2d", r);
}
function pe(e) {
	var t = e.offsetWidth, n = getComputedStyle(e);
	return t += parseInt(n.marginLeft, 10) + parseInt(n.marginRight, 10), t;
}
function me(e) {
	var t = e.offsetHeight, n = getComputedStyle(e);
	return t += parseInt(n.marginTop, 10) + parseInt(n.marginBottom, 10), t;
}
function he(e, t) {
	var n = t.parentNode;
	n && n.replaceChild(e, t);
}
function ge(e) {
	return e && e.parentNode ? e.parentNode.removeChild(e) : null;
}
function _e(e) {
	for (; e.lastChild;) e.removeChild(e.lastChild);
}
function ve(e, t) {
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
var ye = (function() {
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
})(), be = function(e) {
	ye(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t.element;
		return r && !t.target && !r.style.pointerEvents && (r.style.pointerEvents = "auto"), n.element = r || null, n.target_ = null, n.map_ = null, n.listenerKeys = [], t.render && (n.render = t.render), t.target && n.setTarget(t.target), n;
	}
	return t.prototype.disposeInternal = function() {
		ge(this.element), e.prototype.disposeInternal.call(this);
	}, t.prototype.getMap = function() {
		return this.map_;
	}, t.prototype.setMap = function(e) {
		this.map_ && ge(this.element);
		for (var t = 0, n = this.listenerKeys.length; t < n; ++t) j(this.listenerKeys[t]);
		this.listenerKeys.length = 0, this.map_ = e, this.map_ && ((this.target_ ? this.target_ : e.getOverlayContainerStopEvent()).appendChild(this.element), this.render !== b && this.listenerKeys.push(k(e, re.POSTRENDER, this.render, this)), e.render());
	}, t.prototype.render = function(e) {}, t.prototype.setTarget = function(e) {
		this.target_ = typeof e == "string" ? document.getElementById(e) : e;
	}, t;
}(R), xe = "ol-hidden", Se = "ol-selectable", Ce = "ol-unselectable", we = "ol-unsupported", Te = "ol-control", Ee = "ol-collapsed", De = new RegExp([
	"^\\s*(?=(?:(?:[-a-z]+\\s*){0,2}(italic|oblique))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(small-caps))?)",
	"(?=(?:(?:[-a-z]+\\s*){0,2}(bold(?:er)?|lighter|[1-9]00 ))?)",
	"(?:(?:normal|\\1|\\2|\\3)\\s*){0,3}((?:xx?-)?",
	"(?:small|large)|medium|smaller|larger|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx]))",
	"(?:\\s*\\/\\s*(normal|[\\.\\d]+(?:\\%|in|[cem]m|ex|p[ctx])?))",
	"?\\s*([-,\\\"\\'\\sa-z]+?)\\s*$"
].join(""), "i"), Oe = [
	"style",
	"variant",
	"weight",
	"size",
	"lineHeight",
	"family"
], ke = function(e) {
	var t = e.match(De);
	if (!t) return null;
	for (var n = {
		lineHeight: "normal",
		size: "1.2em",
		style: "normal",
		weight: "normal",
		variant: "normal"
	}, r = 0, i = Oe.length; r < i; ++r) {
		var a = t[r + 1];
		a !== void 0 && (n[Oe[r]] = a);
	}
	return n.families = n.family.split(/,\s?/), n;
};
function Ae(e) {
	return e === 1 ? "" : String(Math.round(e * 100) / 100);
}
//#endregion
//#region node_modules/ol/control/FullScreen.js
var je = (function() {
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
})(), Me = [
	"fullscreenchange",
	"webkitfullscreenchange",
	"MSFullscreenChange"
], Ne = {
	ENTERFULLSCREEN: "enterfullscreen",
	LEAVEFULLSCREEN: "leavefullscreen"
}, Pe = function(e) {
	je(t, e);
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
		n.setClassName_(n.button_, Ie()), n.button_.setAttribute("type", "button"), n.button_.title = o, n.button_.appendChild(n.labelNode_), n.button_.addEventListener(O.CLICK, n.handleClick_.bind(n), !1);
		var s = n.cssClassName_ + " " + Ce + " " + Te + " " + (Fe() ? "" : we), c = n.element;
		return c.className = s, c.appendChild(n.button_), n.keys_ = r.keys !== void 0 && r.keys, n.source_ = r.source, n;
	}
	return t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.handleFullScreen_();
	}, t.prototype.handleFullScreen_ = function() {
		if (Fe()) {
			var e = this.getMap();
			if (e) if (Ie()) ze();
			else {
				var t = void 0;
				t = this.source_ ? typeof this.source_ == "string" ? document.getElementById(this.source_) : this.source_ : e.getTargetElement(), this.keys_ ? Re(t) : Le(t);
			}
		}
	}, t.prototype.handleFullScreenChange_ = function() {
		var e = this.getMap();
		Ie() ? (this.setClassName_(this.button_, !0), he(this.labelActiveNode_, this.labelNode_), this.dispatchEvent(Ne.ENTERFULLSCREEN)) : (this.setClassName_(this.button_, !1), he(this.labelNode_, this.labelActiveNode_), this.dispatchEvent(Ne.LEAVEFULLSCREEN)), e && e.updateSize();
	}, t.prototype.setClassName_ = function(e, t) {
		var n, r, i, a = this.activeClassName_, o = this.inactiveClassName_, s = t ? a : o;
		(n = e.classList).remove.apply(n, a), (r = e.classList).remove.apply(r, o), (i = e.classList).add.apply(i, s);
	}, t.prototype.setMap = function(t) {
		if (e.prototype.setMap.call(this, t), t) for (var n = 0, r = Me.length; n < r; ++n) this.listenerKeys.push(k(document, Me[n], this.handleFullScreenChange_, this));
	}, t;
}(be);
function Fe() {
	var e = document.body;
	return !!(e.webkitRequestFullscreen || e.msRequestFullscreen && document.msFullscreenEnabled || e.requestFullscreen && document.fullscreenEnabled);
}
function Ie() {
	return !!(document.webkitIsFullScreen || document.msFullscreenElement || document.fullscreenElement);
}
function Le(e) {
	e.requestFullscreen ? e.requestFullscreen() : e.msRequestFullscreen ? e.msRequestFullscreen() : e.webkitRequestFullscreen && e.webkitRequestFullscreen();
}
function Re(e) {
	e.webkitRequestFullscreen ? e.webkitRequestFullscreen() : Le(e);
}
function ze() {
	document.exitFullscreen ? document.exitFullscreen() : document.msExitFullscreen ? document.msExitFullscreen() : document.webkitExitFullscreen && document.webkitExitFullscreen();
}
//#endregion
//#region node_modules/ol/pointer/EventType.js
var Be = {
	POINTERMOVE: "pointermove",
	POINTERDOWN: "pointerdown",
	POINTERUP: "pointerup",
	POINTEROVER: "pointerover",
	POINTEROUT: "pointerout",
	POINTERENTER: "pointerenter",
	POINTERLEAVE: "pointerleave",
	POINTERCANCEL: "pointercancel"
}, z = {
	RADIANS: "radians",
	DEGREES: "degrees",
	FEET: "ft",
	METERS: "m",
	PIXELS: "pixels",
	TILE_PIXELS: "tile-pixels",
	USFEET: "us-ft"
};
z.METERS, z.FEET, z.USFEET, z.RADIANS, z.DEGREES;
var Ve = {};
Ve[z.RADIANS] = 6370997 / (2 * Math.PI), Ve[z.DEGREES] = 2 * Math.PI * 6370997 / 360, Ve[z.FEET] = .3048, Ve[z.METERS] = 1, Ve[z.USFEET] = 1200 / 3937;
//#endregion
//#region node_modules/ol/proj/Projection.js
var He = function() {
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
		return this.metersPerUnit_ || Ve[this.units_];
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
function B(e, t, n) {
	return Math.min(Math.max(e, t), n);
}
var Ue = (function() {
	return "cosh" in Math ? Math.cosh : function(e) {
		var t = Math.exp(e);
		return (t + 1 / t) / 2;
	};
})(), We = (function() {
	return "log2" in Math ? Math.log2 : function(e) {
		return Math.log(e) * Math.LOG2E;
	};
})();
function Ge(e, t, n, r, i, a) {
	var o = i - n, s = a - r;
	if (o !== 0 || s !== 0) {
		var c = ((e - n) * o + (t - r) * s) / (o * o + s * s);
		c > 1 ? (n = i, r = a) : c > 0 && (n += o * c, r += s * c);
	}
	return Ke(e, t, n, r);
}
function Ke(e, t, n, r) {
	var i = n - e, a = r - t;
	return i * i + a * a;
}
function qe(e) {
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
function Je(e) {
	return e * Math.PI / 180;
}
function Ye(e, t) {
	var n = e % t;
	return n * t < 0 ? n + t : n;
}
function Xe(e, t, n) {
	return e + n * (t - e);
}
//#endregion
//#region node_modules/ol/proj/epsg3857.js
var Ze = (function() {
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
})(), Qe = 6378137, $e = Math.PI * Qe, et = [
	-$e,
	-$e,
	$e,
	$e
], tt = [
	-180,
	-85,
	180,
	85
], nt = Qe * Math.log(Math.tan(Math.PI / 2)), rt = function(e) {
	Ze(t, e);
	function t(t) {
		return e.call(this, {
			code: t,
			units: z.METERS,
			extent: et,
			global: !0,
			worldExtent: tt,
			getPointResolution: function(e, t) {
				return e / Ue(t[1] / 6378137);
			}
		}) || this;
	}
	return t;
}(He), it = [
	new rt("EPSG:3857"),
	new rt("EPSG:102100"),
	new rt("EPSG:102113"),
	new rt("EPSG:900913"),
	new rt("http://www.opengis.net/def/crs/EPSG/0/3857"),
	new rt("http://www.opengis.net/gml/srs/epsg.xml#3857")
];
function at(e, t, n) {
	var r = e.length, i = n > 1 ? n : 2, a = t;
	a === void 0 && (a = i > 2 ? e.slice() : Array(r));
	for (var o = 0; o < r; o += i) {
		a[o] = $e * e[o] / 180;
		var s = Qe * Math.log(Math.tan(Math.PI * (+e[o + 1] + 90) / 360));
		s > nt ? s = nt : s < -nt && (s = -nt), a[o + 1] = s;
	}
	return a;
}
function ot(e, t, n) {
	var r = e.length, i = n > 1 ? n : 2, a = t;
	a === void 0 && (a = i > 2 ? e.slice() : Array(r));
	for (var o = 0; o < r; o += i) a[o] = 180 * e[o] / $e, a[o + 1] = 360 * Math.atan(Math.exp(e[o + 1] / Qe)) / Math.PI - 90;
	return a;
}
//#endregion
//#region node_modules/ol/proj/epsg4326.js
var st = (function() {
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
})(), ct = 6378137, lt = [
	-180,
	-90,
	180,
	90
], ut = Math.PI * ct / 180, dt = function(e) {
	st(t, e);
	function t(t, n) {
		return e.call(this, {
			code: t,
			units: z.DEGREES,
			extent: lt,
			axisOrientation: n,
			global: !0,
			metersPerUnit: ut,
			worldExtent: lt
		}) || this;
	}
	return t;
}(He), ft = [
	new dt("CRS:84"),
	new dt("EPSG:4326", "neu"),
	new dt("urn:ogc:def:crs:OGC:1.3:CRS84"),
	new dt("urn:ogc:def:crs:OGC:2:84"),
	new dt("http://www.opengis.net/def/crs/OGC/1.3/CRS84", "neu"),
	new dt("http://www.opengis.net/gml/srs/epsg.xml#4326", "neu"),
	new dt("http://www.opengis.net/def/crs/EPSG/0/4326", "neu")
], pt = {};
function mt(e) {
	return pt[e] || pt[e.replace(/urn:(x-)?ogc:def:crs:EPSG:(.*:)?(\w+)$/, "EPSG:$3")] || null;
}
function ht(e, t) {
	pt[e] = t;
}
//#endregion
//#region node_modules/ol/proj/transforms.js
var gt = {};
function _t(e, t, n) {
	var r = e.getCode(), i = t.getCode();
	r in gt || (gt[r] = {}), gt[r][i] = n;
}
function vt(e, t) {
	var n;
	return e in gt && t in gt[e] && (n = gt[e][t]), n;
}
//#endregion
//#region node_modules/ol/extent/Corner.js
var yt = {
	BOTTOM_LEFT: "bottom-left",
	BOTTOM_RIGHT: "bottom-right",
	TOP_LEFT: "top-left",
	TOP_RIGHT: "top-right"
}, bt = {
	UNKNOWN: 0,
	INTERSECTING: 1,
	ABOVE: 2,
	RIGHT: 4,
	BELOW: 8,
	LEFT: 16
}, xt = (function() {
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
})(), St = function(e) {
	xt(t, e);
	function t(t) {
		var n = this, r = "Assertion failed. See https://openlayers.org/en/" + ("v" + L.split("-")[0]) + "/doc/errors/#" + t + " for details.";
		return n = e.call(this, r) || this, n.code = t, n.name = "AssertionError", n.message = r, n;
	}
	return t;
}(Error);
//#endregion
//#region node_modules/ol/asserts.js
function V(e, t) {
	if (!e) throw new St(t);
}
//#endregion
//#region node_modules/ol/extent.js
function Ct(e) {
	for (var t = jt(), n = 0, r = e.length; n < r; ++n) Rt(t, e[n]);
	return t;
}
function wt(e, t, n) {
	return n ? (n[0] = e[0] - t, n[1] = e[1] - t, n[2] = e[2] + t, n[3] = e[3] + t, n) : [
		e[0] - t,
		e[1] - t,
		e[2] + t,
		e[3] + t
	];
}
function Tt(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e.slice();
}
function Et(e, t, n) {
	var r = t < e[0] ? e[0] - t : e[2] < t ? t - e[2] : 0, i = n < e[1] ? e[1] - n : e[3] < n ? n - e[3] : 0;
	return r * r + i * i;
}
function Dt(e, t) {
	return kt(e, t[0], t[1]);
}
function Ot(e, t) {
	return e[0] <= t[0] && t[2] <= e[2] && e[1] <= t[1] && t[3] <= e[3];
}
function kt(e, t, n) {
	return e[0] <= t && t <= e[2] && e[1] <= n && n <= e[3];
}
function At(e, t) {
	var n = e[0], r = e[1], i = e[2], a = e[3], o = t[0], s = t[1], c = bt.UNKNOWN;
	return o < n ? c |= bt.LEFT : o > i && (c |= bt.RIGHT), s < r ? c |= bt.BELOW : s > a && (c |= bt.ABOVE), c === bt.UNKNOWN && (c = bt.INTERSECTING), c;
}
function jt() {
	return [
		Infinity,
		Infinity,
		-Infinity,
		-Infinity
	];
}
function Mt(e, t, n, r, i) {
	return i ? (i[0] = e, i[1] = t, i[2] = n, i[3] = r, i) : [
		e,
		t,
		n,
		r
	];
}
function Nt(e) {
	return Mt(Infinity, Infinity, -Infinity, -Infinity, e);
}
function Pt(e, t) {
	var n = e[0], r = e[1];
	return Mt(n, r, n, r, t);
}
function Ft(e, t, n, r, i) {
	return zt(Nt(i), e, t, n, r);
}
function It(e, t) {
	return e[0] == t[0] && e[2] == t[2] && e[1] == t[1] && e[3] == t[3];
}
function Lt(e, t) {
	return t[0] < e[0] && (e[0] = t[0]), t[2] > e[2] && (e[2] = t[2]), t[1] < e[1] && (e[1] = t[1]), t[3] > e[3] && (e[3] = t[3]), e;
}
function Rt(e, t) {
	t[0] < e[0] && (e[0] = t[0]), t[0] > e[2] && (e[2] = t[0]), t[1] < e[1] && (e[1] = t[1]), t[1] > e[3] && (e[3] = t[1]);
}
function zt(e, t, n, r, i) {
	for (; n < r; n += i) Bt(e, t[n], t[n + 1]);
	return e;
}
function Bt(e, t, n) {
	e[0] = Math.min(e[0], t), e[1] = Math.min(e[1], n), e[2] = Math.max(e[2], t), e[3] = Math.max(e[3], n);
}
function Vt(e, t) {
	var n = t(Ut(e));
	return n || (n = t(Wt(e)), n) || (n = t(Zt(e)), n) || (n = t(Xt(e)), n) ? n : !1;
}
function Ht(e) {
	var t = 0;
	return $t(e) || (t = H(e) * Jt(e)), t;
}
function Ut(e) {
	return [e[0], e[1]];
}
function Wt(e) {
	return [e[2], e[1]];
}
function Gt(e) {
	return [(e[0] + e[2]) / 2, (e[1] + e[3]) / 2];
}
function Kt(e, t) {
	var n;
	return t === yt.BOTTOM_LEFT ? n = Ut(e) : t === yt.BOTTOM_RIGHT ? n = Wt(e) : t === yt.TOP_LEFT ? n = Xt(e) : t === yt.TOP_RIGHT ? n = Zt(e) : V(!1, 13), n;
}
function qt(e, t, n, r, i) {
	var a = t * r[0] / 2, o = t * r[1] / 2, s = Math.cos(n), c = Math.sin(n), l = a * s, u = a * c, d = o * s, f = o * c, p = e[0], m = e[1], h = p - l + f, g = p - l - f, _ = p + l - f, v = p + l + f, y = m - u - d, b = m - u + d, x = m + u + d, S = m + u - d;
	return Mt(Math.min(h, g, _, v), Math.min(y, b, x, S), Math.max(h, g, _, v), Math.max(y, b, x, S), i);
}
function Jt(e) {
	return e[3] - e[1];
}
function Yt(e, t, n) {
	var r = n || jt();
	return Qt(e, t) ? (e[0] > t[0] ? r[0] = e[0] : r[0] = t[0], e[1] > t[1] ? r[1] = e[1] : r[1] = t[1], e[2] < t[2] ? r[2] = e[2] : r[2] = t[2], e[3] < t[3] ? r[3] = e[3] : r[3] = t[3]) : Nt(r), r;
}
function Xt(e) {
	return [e[0], e[3]];
}
function Zt(e) {
	return [e[2], e[3]];
}
function H(e) {
	return e[2] - e[0];
}
function Qt(e, t) {
	return e[0] <= t[2] && e[2] >= t[0] && e[1] <= t[3] && e[3] >= t[1];
}
function $t(e) {
	return e[2] < e[0] || e[3] < e[1];
}
function en(e, t) {
	return t ? (t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t) : e;
}
function tn(e, t) {
	var n = (e[2] - e[0]) / 2 * (t - 1), r = (e[3] - e[1]) / 2 * (t - 1);
	e[0] -= n, e[2] += n, e[1] -= r, e[3] += r;
}
function nn(e, t, n) {
	var r = !1, i = At(e, t), a = At(e, n);
	if (i === bt.INTERSECTING || a === bt.INTERSECTING) r = !0;
	else {
		var o = e[0], s = e[1], c = e[2], l = e[3], u = t[0], d = t[1], f = n[0], p = n[1], m = (p - d) / (f - u), h = void 0, g = void 0;
		a & bt.ABOVE && !(i & bt.ABOVE) && (h = f - (p - l) / m, r = h >= o && h <= c), !r && a & bt.RIGHT && !(i & bt.RIGHT) && (g = p - (f - c) * m, r = g >= s && g <= l), !r && a & bt.BELOW && !(i & bt.BELOW) && (h = f - (p - s) / m, r = h >= o && h <= c), !r && a & bt.LEFT && !(i & bt.LEFT) && (g = p - (f - o) * m, r = g >= s && g <= l);
	}
	return r;
}
function rn(e, t) {
	var n = t.getExtent(), r = Gt(e);
	if (t.canWrapX() && (r[0] < n[0] || r[0] >= n[2])) {
		var i = H(n), a = Math.floor((r[0] - n[0]) / i) * i;
		e[0] -= a, e[2] -= a;
	}
	return e;
}
//#endregion
//#region node_modules/ol/geom/GeometryType.js
var U = {
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
function an(e, t, n) {
	var r = n || 6371008.8, i = Je(e[1]), a = Je(t[1]), o = (a - i) / 2, s = Je(t[0] - e[0]) / 2, c = Math.sin(o) * Math.sin(o) + Math.sin(s) * Math.sin(s) * Math.cos(i) * Math.cos(a);
	return 2 * r * Math.atan2(Math.sqrt(c), Math.sqrt(1 - c));
}
//#endregion
//#region node_modules/ol/coordinate.js
function on(e, t) {
	return e[0] += +t[0], e[1] += +t[1], e;
}
function sn(e, t, n) {
	return e ? t.replace("{x}", e[0].toFixed(n)).replace("{y}", e[1].toFixed(n)) : "";
}
function cn(e, t) {
	for (var n = !0, r = e.length - 1; r >= 0; --r) if (e[r] != t[r]) {
		n = !1;
		break;
	}
	return n;
}
function ln(e, t) {
	var n = Math.cos(t), r = Math.sin(t), i = e[0] * n - e[1] * r, a = e[1] * n + e[0] * r;
	return e[0] = i, e[1] = a, e;
}
function un(e, t) {
	return e[0] *= t, e[1] *= t, e;
}
function dn(e, t) {
	if (t.canWrapX()) {
		var n = H(t.getExtent()), r = fn(e, t, n);
		r && (e[0] -= r * n);
	}
	return e;
}
function fn(e, t, n) {
	var r = t.getExtent(), i = 0;
	if (t.canWrapX() && (e[0] < r[0] || e[0] > r[2])) {
		var a = n || H(r);
		i = Math.floor((e[0] - r[0]) / a);
	}
	return i;
}
//#endregion
//#region node_modules/ol/proj.js
function pn(e, t, n) {
	var r;
	if (t !== void 0) {
		for (var i = 0, a = e.length; i < a; ++i) t[i] = e[i];
		r = t;
	} else r = e.slice();
	return r;
}
function mn(e, t, n) {
	if (t !== void 0 && e !== t) {
		for (var r = 0, i = e.length; r < i; ++r) t[r] = e[r];
		e = t;
	}
	return e;
}
function hn(e) {
	ht(e.getCode(), e), _t(e, e, pn);
}
function gn(e) {
	e.forEach(hn);
}
function _n(e) {
	return typeof e == "string" ? mt(e) : e || null;
}
function vn(e, t, n, r) {
	e = _n(e);
	var i, a = e.getPointResolutionFunc();
	if (a) {
		if (i = a(t, n), r && r !== e.getUnits()) {
			var o = e.getMetersPerUnit();
			o && (i = i * o / Ve[r]);
		}
	} else {
		var s = e.getUnits();
		if (s == z.DEGREES && !r || r == z.DEGREES) i = t;
		else {
			var c = Cn(e, _n("EPSG:4326"));
			if (c === mn && s !== z.DEGREES) i = t * e.getMetersPerUnit();
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
				l = c(l, l, 2), i = (an(l.slice(0, 2), l.slice(2, 4)) + an(l.slice(4, 6), l.slice(6, 8))) / 2;
			}
			var o = r ? Ve[r] : e.getMetersPerUnit();
			o !== void 0 && (i /= o);
		}
	}
	return i;
}
function yn(e) {
	gn(e), e.forEach(function(t) {
		e.forEach(function(e) {
			t !== e && _t(t, e, pn);
		});
	});
}
function bn(e, t, n, r) {
	e.forEach(function(e) {
		t.forEach(function(t) {
			_t(e, t, n), _t(t, e, r);
		});
	});
}
function xn(e, t) {
	return e ? typeof e == "string" ? _n(e) : e : _n(t);
}
function Sn(e, t) {
	if (e === t) return !0;
	var n = e.getUnits() === t.getUnits();
	return (e.getCode() === t.getCode() || Cn(e, t) === pn) && n;
}
function Cn(e, t) {
	var n = vt(e.getCode(), t.getCode());
	return n ||= mn, n;
}
function wn(e, t) {
	return Cn(_n(e), _n(t));
}
function Tn(e, t, n) {
	return wn(t, n)(e, void 0, e.length);
}
var En = null;
function Dn() {
	return En;
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
function Mn(e, t) {
	return e;
}
function Nn() {
	yn(it), yn(ft), bn(ft, it, at, ot);
}
Nn();
//#endregion
//#region node_modules/ol/control/MousePosition.js
var Pn = (function() {
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
})(), Fn = "projection", In = "coordinateFormat", Ln = function(e) {
	Pn(t, e);
	function t(t) {
		var n = this, r = t || {}, i = document.createElement("div");
		i.className = r.className === void 0 ? "ol-mouse-position" : r.className, n = e.call(this, {
			element: i,
			render: r.render,
			target: r.target
		}) || this, n.on, n.once, n.un, n.addChangeListener(Fn, n.handleProjectionChanged_), r.coordinateFormat && n.setCoordinateFormat(r.coordinateFormat), r.projection && n.setProjection(r.projection);
		var a = !0, o = "&#160;";
		return "undefinedHTML" in r ? (r.undefinedHTML !== void 0 && (o = r.undefinedHTML), a = !!o) : "placeholder" in r && (r.placeholder === !1 ? a = !1 : o = String(r.placeholder)), n.placeholder_ = o, n.renderOnMouseOut_ = a, n.renderedHTML_ = i.innerHTML, n.mapProjection_ = null, n.transform_ = null, n;
	}
	return t.prototype.handleProjectionChanged_ = function() {
		this.transform_ = null;
	}, t.prototype.getCoordinateFormat = function() {
		return this.get(In);
	}, t.prototype.getProjection = function() {
		return this.get(Fn);
	}, t.prototype.handleMouseMove = function(e) {
		var t = this.getMap();
		this.updateHTML_(t.getEventPixel(e));
	}, t.prototype.handleMouseOut = function(e) {
		this.updateHTML_(null);
	}, t.prototype.setMap = function(t) {
		if (e.prototype.setMap.call(this, t), t) {
			var n = t.getViewport();
			this.listenerKeys.push(k(n, Be.POINTERMOVE, this.handleMouseMove, this)), this.renderOnMouseOut_ && this.listenerKeys.push(k(n, Be.POINTEROUT, this.handleMouseOut, this)), this.updateHTML_(null);
		}
	}, t.prototype.setCoordinateFormat = function(e) {
		this.set(In, e);
	}, t.prototype.setProjection = function(e) {
		this.set(Fn, _n(e));
	}, t.prototype.updateHTML_ = function(e) {
		var t = this.placeholder_;
		if (e && this.mapProjection_) {
			if (!this.transform_) {
				var n = this.getProjection();
				n ? this.transform_ = Cn(this.mapProjection_, n) : this.transform_ = mn;
			}
			var r = this.getMap().getCoordinateFromPixelInternal(e);
			if (r) {
				var i = Dn();
				i && (this.transform_ = Cn(this.mapProjection_, i)), this.transform_(r, r);
				var a = this.getCoordinateFormat();
				t = a ? a(r) : r.toString();
			}
		}
		(!this.renderedHTML_ || t !== this.renderedHTML_) && (this.element.innerHTML = t, this.renderedHTML_ = t);
	}, t.prototype.render = function(e) {
		var t = e.frameState;
		t ? this.mapProjection_ != t.viewState.projection && (this.mapProjection_ = t.viewState.projection, this.transform_ = null) : this.mapProjection_ = null;
	}, t;
}(be), Rn = [
	,
	,
	,
	,
	,
	,
];
function zn() {
	return [
		1,
		0,
		0,
		1,
		0,
		0
	];
}
function Bn(e) {
	return Hn(e, 1, 0, 0, 1, 0, 0);
}
function Vn(e, t) {
	var n = e[0], r = e[1], i = e[2], a = e[3], o = e[4], s = e[5], c = t[0], l = t[1], u = t[2], d = t[3], f = t[4], p = t[5];
	return e[0] = n * c + i * l, e[1] = r * c + a * l, e[2] = n * u + i * d, e[3] = r * u + a * d, e[4] = n * f + i * p + o, e[5] = r * f + a * p + s, e;
}
function Hn(e, t, n, r, i, a, o) {
	return e[0] = t, e[1] = n, e[2] = r, e[3] = i, e[4] = a, e[5] = o, e;
}
function Un(e, t) {
	return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e;
}
function W(e, t) {
	var n = t[0], r = t[1];
	return t[0] = e[0] * n + e[2] * r + e[4], t[1] = e[1] * n + e[3] * r + e[5], t;
}
function Wn(e, t) {
	var n = Math.cos(t), r = Math.sin(t);
	return Vn(e, Hn(Rn, n, r, -r, n, 0, 0));
}
function Gn(e, t, n) {
	return Vn(e, Hn(Rn, t, 0, 0, n, 0, 0));
}
function Kn(e, t, n) {
	return Hn(e, t, 0, 0, n, 0, 0);
}
function qn(e, t, n) {
	return Vn(e, Hn(Rn, 1, 0, 0, 1, t, n));
}
function Jn(e, t, n, r, i, a, o, s) {
	var c = Math.sin(a), l = Math.cos(a);
	return e[0] = r * l, e[1] = i * c, e[2] = -r * c, e[3] = i * l, e[4] = o * r * l - s * r * c + t, e[5] = o * i * c + s * i * l + n, e;
}
function Yn(e, t) {
	var n = Xn(t);
	V(n !== 0, 32);
	var r = t[0], i = t[1], a = t[2], o = t[3], s = t[4], c = t[5];
	return e[0] = o / n, e[1] = -i / n, e[2] = -a / n, e[3] = r / n, e[4] = (a * c - o * s) / n, e[5] = -(r * c - i * s) / n, e;
}
function Xn(e) {
	return e[0] * e[3] - e[1] * e[2];
}
var Zn;
function Qn(e) {
	var t = "matrix(" + e.join(", ") + ")";
	if (le) return t;
	var n = Zn ||= document.createElement("div");
	return n.style.transform = t, n.style.transform;
}
//#endregion
//#region node_modules/ol/color.js
var $n = /^#([a-f0-9]{3}|[a-f0-9]{4}(?:[a-f0-9]{2}){0,2})$/i, er = /^([a-z]*)$|^hsla?\(.*\)$/i;
function tr(e) {
	return typeof e == "string" ? e : sr(e);
}
function nr(e) {
	var t = document.createElement("div");
	if (t.style.color = e, t.style.color !== "") {
		document.body.appendChild(t);
		var n = getComputedStyle(t).color;
		return document.body.removeChild(t), n;
	} else return "";
}
var rr = (function() {
	var e = {}, t = 0;
	return (function(n) {
		var r;
		if (e.hasOwnProperty(n)) r = e[n];
		else {
			if (t >= 1024) {
				var i = 0;
				for (var a in e) i++ & 3 || (delete e[a], --t);
			}
			r = ar(n), e[n] = r, ++t;
		}
		return r;
	});
})();
function ir(e) {
	return Array.isArray(e) ? e : rr(e);
}
function ar(e) {
	var t, n, r, i, a;
	if (er.exec(e) && (e = nr(e)), $n.exec(e)) {
		var o = e.length - 1, s = void 0;
		s = o <= 4 ? 1 : 2;
		var c = o === 4 || o === 8;
		t = parseInt(e.substr(1 + 0 * s, s), 16), n = parseInt(e.substr(1 + 1 * s, s), 16), r = parseInt(e.substr(1 + 2 * s, s), 16), i = c ? parseInt(e.substr(1 + 3 * s, s), 16) : 255, s == 1 && (t = (t << 4) + t, n = (n << 4) + n, r = (r << 4) + r, c && (i = (i << 4) + i)), a = [
			t,
			n,
			r,
			i / 255
		];
	} else e.indexOf("rgba(") == 0 ? (a = e.slice(5, -1).split(",").map(Number), or(a)) : e.indexOf("rgb(") == 0 ? (a = e.slice(4, -1).split(",").map(Number), a.push(1), or(a)) : V(!1, 14);
	return a;
}
function or(e) {
	return e[0] = B(e[0] + .5 | 0, 0, 255), e[1] = B(e[1] + .5 | 0, 0, 255), e[2] = B(e[2] + .5 | 0, 0, 255), e[3] = B(e[3], 0, 1), e;
}
function sr(e) {
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
var cr = function() {
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
		var r = lr(e, t, n);
		return r in this.cache_ ? this.cache_[r] : null;
	}, e.prototype.set = function(e, t, n, r) {
		var i = lr(e, t, n);
		this.cache_[i] = r, ++this.cacheSize_;
	}, e.prototype.setSize = function(e) {
		this.maxCacheSize_ = e, this.expire();
	}, e;
}();
function lr(e, t, n) {
	var r = n ? tr(n) : "null";
	return t + ":" + e + ":" + r;
}
var ur = new cr(), G = {
	OPACITY: "opacity",
	VISIBLE: "visible",
	EXTENT: "extent",
	Z_INDEX: "zIndex",
	MAX_RESOLUTION: "maxResolution",
	MIN_RESOLUTION: "minResolution",
	MAX_ZOOM: "maxZoom",
	MIN_ZOOM: "minZoom",
	SOURCE: "source"
}, dr = (function() {
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
})(), fr = function(e) {
	dr(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = S({}, t);
		return typeof t.properties == "object" && (delete r.properties, S(r, t.properties)), r[G.OPACITY] = t.opacity === void 0 ? 1 : t.opacity, V(typeof r[G.OPACITY] == "number", 64), r[G.VISIBLE] = t.visible === void 0 || t.visible, r[G.Z_INDEX] = t.zIndex, r[G.MAX_RESOLUTION] = t.maxResolution === void 0 ? Infinity : t.maxResolution, r[G.MIN_RESOLUTION] = t.minResolution === void 0 ? 0 : t.minResolution, r[G.MIN_ZOOM] = t.minZoom === void 0 ? -Infinity : t.minZoom, r[G.MAX_ZOOM] = t.maxZoom === void 0 ? Infinity : t.maxZoom, n.className_ = r.className === void 0 ? "ol-layer" : t.className, delete r.className, n.setProperties(r), n.state_ = null, n;
	}
	return t.prototype.getClassName = function() {
		return this.className_;
	}, t.prototype.getLayerState = function(e) {
		var t = this.state_ || {
			layer: this,
			managed: e === void 0 || e
		}, n = this.getZIndex();
		return t.opacity = B(Math.round(this.getOpacity() * 100) / 100, 0, 1), t.sourceState = this.getSourceState(), t.visible = this.getVisible(), t.extent = this.getExtent(), t.zIndex = n === void 0 && !t.managed ? Infinity : n, t.maxResolution = this.getMaxResolution(), t.minResolution = Math.max(this.getMinResolution(), 0), t.minZoom = this.getMinZoom(), t.maxZoom = this.getMaxZoom(), this.state_ = t, t;
	}, t.prototype.getLayersArray = function(e) {
		return F();
	}, t.prototype.getLayerStatesArray = function(e) {
		return F();
	}, t.prototype.getExtent = function() {
		return this.get(G.EXTENT);
	}, t.prototype.getMaxResolution = function() {
		return this.get(G.MAX_RESOLUTION);
	}, t.prototype.getMinResolution = function() {
		return this.get(G.MIN_RESOLUTION);
	}, t.prototype.getMinZoom = function() {
		return this.get(G.MIN_ZOOM);
	}, t.prototype.getMaxZoom = function() {
		return this.get(G.MAX_ZOOM);
	}, t.prototype.getOpacity = function() {
		return this.get(G.OPACITY);
	}, t.prototype.getSourceState = function() {
		return F();
	}, t.prototype.getVisible = function() {
		return this.get(G.VISIBLE);
	}, t.prototype.getZIndex = function() {
		return this.get(G.Z_INDEX);
	}, t.prototype.setExtent = function(e) {
		this.set(G.EXTENT, e);
	}, t.prototype.setMaxResolution = function(e) {
		this.set(G.MAX_RESOLUTION, e);
	}, t.prototype.setMinResolution = function(e) {
		this.set(G.MIN_RESOLUTION, e);
	}, t.prototype.setMaxZoom = function(e) {
		this.set(G.MAX_ZOOM, e);
	}, t.prototype.setMinZoom = function(e) {
		this.set(G.MIN_ZOOM, e);
	}, t.prototype.setOpacity = function(e) {
		V(typeof e == "number", 64), this.set(G.OPACITY, e);
	}, t.prototype.setVisible = function(e) {
		this.set(G.VISIBLE, e);
	}, t.prototype.setZIndex = function(e) {
		this.set(G.Z_INDEX, e);
	}, t.prototype.disposeInternal = function() {
		this.state_ &&= (this.state_.layer = null, null), e.prototype.disposeInternal.call(this);
	}, t;
}(R), pr = {
	PRERENDER: "prerender",
	POSTRENDER: "postrender",
	PRECOMPOSE: "precompose",
	POSTCOMPOSE: "postcompose",
	RENDERCOMPLETE: "rendercomplete"
}, mr = {
	UNDEFINED: "undefined",
	LOADING: "loading",
	READY: "ready",
	ERROR: "error"
}, hr = (function() {
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
})(), gr = function(e) {
	hr(t, e);
	function t(t) {
		var n = this, r = S({}, t);
		delete r.source, n = e.call(this, r) || this, n.on, n.once, n.un, n.mapPrecomposeKey_ = null, n.mapRenderKey_ = null, n.sourceChangeKey_ = null, n.renderer_ = null, t.render && (n.render = t.render), t.map && n.setMap(t.map), n.addChangeListener(G.SOURCE, n.handleSourcePropertyChange_);
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
		return this.get(G.SOURCE) || null;
	}, t.prototype.getSourceState = function() {
		var e = this.getSource();
		return e ? e.getState() : mr.UNDEFINED;
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
		this.mapPrecomposeKey_ &&= (j(this.mapPrecomposeKey_), null), e || this.changed(), this.mapRenderKey_ &&= (j(this.mapRenderKey_), null), e && (this.mapPrecomposeKey_ = k(e, pr.PRECOMPOSE, function(e) {
			var t = e.frameState.layerStatesArray, n = this.getLayerState(!1);
			V(!t.some(function(e) {
				return e.layer === n.layer;
			}), 67), t.push(n);
		}, this), this.mapRenderKey_ = k(this, O.CHANGE, e.render, e), this.changed());
	}, t.prototype.setSource = function(e) {
		this.set(G.SOURCE, e);
	}, t.prototype.getRenderer = function() {
		return this.renderer_ ||= this.createRenderer(), this.renderer_;
	}, t.prototype.hasRenderer = function() {
		return !!this.renderer_;
	}, t.prototype.createRenderer = function() {
		return null;
	}, t.prototype.disposeInternal = function() {
		this.renderer_ && (this.renderer_.dispose(), delete this.renderer_), this.setSource(null), e.prototype.disposeInternal.call(this);
	}, t;
}(fr);
function _r(e, t) {
	if (!e.visible) return !1;
	var n = t.resolution;
	if (n < e.minResolution || n >= e.maxResolution) return !1;
	var r = t.zoom;
	return r > e.minZoom && r <= e.maxZoom;
}
//#endregion
//#region node_modules/ol/renderer/Map.js
var vr = (function() {
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
})(), yr = function(e) {
	vr(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.map_ = t, n;
	}
	return t.prototype.dispatchRenderEvent = function(e, t) {
		F();
	}, t.prototype.calculateMatrices2D = function(e) {
		var t = e.viewState, n = e.coordinateToPixelTransform, r = e.pixelToCoordinateTransform;
		Jn(n, e.size[0] / 2, e.size[1] / 2, 1 / t.resolution, -1 / t.resolution, -t.rotation, -t.center[0], -t.center[1]), Yn(r, n);
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i, a, o, s) {
		var c, l = t.viewState;
		function u(e, t, n, r) {
			return i.call(a, t, e ? n : null, r);
		}
		var d = l.projection, f = dn(e.slice(), d), p = [[0, 0]];
		if (d.canWrapX() && r) {
			var m = H(d.getExtent());
			p.push([-m, 0], [m, 0]);
		}
		for (var h = t.layerStatesArray, g = h.length, _ = [], v = [], y = 0; y < p.length; y++) for (var b = g - 1; b >= 0; --b) {
			var x = h[b], S = x.layer;
			if (S.hasRenderer() && _r(x, l) && o.call(s, S)) {
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
		ur.canExpireCache() && e.postRenderFunctions.push(br);
	}, t;
}(d);
function br(e, t) {
	ur.expire();
}
//#endregion
//#region node_modules/ol/render/Event.js
var xr = (function() {
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
})(), Sr = function(e) {
	xr(t, e);
	function t(t, n, r, i) {
		var a = e.call(this, t) || this;
		return a.inversePixelTransform = n, a.frameState = r, a.context = i, a;
	}
	return t;
}(l), Cr = "10px sans-serif", wr = "#000", Tr = "round", Er = [], Dr = "round", Or = "#000", kr = "center", Ar = "middle", jr = [
	0,
	0,
	0,
	0
], Mr = new R(), Nr = new D();
Nr.setSize = function() {
	console.warn("labelCache is deprecated.");
};
var Pr = null, Fr, Ir = {}, Lr = (function() {
	var e = ["monospace", "serif"], t = e.length, n = "wmytzilWMYTZIL@#/&?$%10", r, i;
	function a(r, a, o) {
		for (var s = !0, c = 0; c < t; ++c) {
			var l = e[c];
			if (i = Br(r + " " + a + " 32px " + l, n), o != l) {
				var u = Br(r + " " + a + " 32px " + o + "," + l, n);
				s &&= u != i;
			}
		}
		return !!s;
	}
	function o() {
		for (var e = !0, t = Mr.getKeys(), n = 0, i = t.length; n < i; ++n) {
			var o = t[n];
			Mr.get(o) < 100 && (a.apply(this, o.split("\n")) ? (C(Ir), Pr = null, Fr = void 0, Mr.set(o, 100)) : (Mr.set(o, Mr.get(o) + 1, !0), e = !1));
		}
		e && (clearInterval(r), r = void 0);
	}
	return function(e) {
		var t = ke(e);
		if (t) for (var n = t.families, i = 0, s = n.length; i < s; ++i) {
			var c = n[i], l = t.style + "\n" + t.weight + "\n" + c;
			Mr.get(l) === void 0 && (Mr.set(l, 100, !0), a(t.style, t.weight, c) || (Mr.set(l, 0, !0), r === void 0 && (r = setInterval(o, 32))));
		}
	};
})(), Rr = (function() {
	var e;
	return function(t) {
		var n = Ir[t];
		if (n == null) {
			if (le) {
				var r = ke(t), i = zr(t, "Žg");
				n = (isNaN(Number(r.lineHeight)) ? 1.2 : Number(r.lineHeight)) * (i.actualBoundingBoxAscent + i.actualBoundingBoxDescent);
			} else e || (e = document.createElement("div"), e.innerHTML = "M", e.style.minHeight = "0", e.style.maxHeight = "none", e.style.height = "auto", e.style.padding = "0", e.style.border = "none", e.style.position = "absolute", e.style.display = "block", e.style.left = "-99999px"), e.style.font = t, document.body.appendChild(e), n = e.offsetHeight, document.body.removeChild(e);
			Ir[t] = n;
		}
		return n;
	};
})();
function zr(e, t) {
	return Pr ||= fe(1, 1), e != Fr && (Pr.font = e, Fr = Pr.font), Pr.measureText(t);
}
function Br(e, t) {
	return zr(e, t).width;
}
function Vr(e, t, n) {
	if (t in n) return n[t];
	var r = Br(e, t);
	return n[t] = r, r;
}
function Hr(e, t, n) {
	for (var r = t.length, i = 0, a = 0; a < r; ++a) {
		var o = Br(e, t[a]);
		i = Math.max(i, o), n.push(o);
	}
	return i;
}
function Ur(e, t, n, r, i, a, o, s, c, l, u) {
	e.save(), n !== 1 && (e.globalAlpha *= n), t && e.setTransform.apply(e, t), r.contextInstructions ? (e.translate(c, l), e.scale(u[0], u[1]), Wr(r, e)) : u[0] < 0 || u[1] < 0 ? (e.translate(c, l), e.scale(u[0], u[1]), e.drawImage(r, i, a, o, s, 0, 0, o, s)) : e.drawImage(r, i, a, o, s, c, l, o * u[0], s * u[1]), e.restore();
}
function Wr(e, t) {
	for (var n = e.contextInstructions, r = 0, i = n.length; r < i; r += 2) Array.isArray(n[r + 1]) ? t[n[r]].apply(t, n[r + 1]) : t[n[r]] = n[r + 1];
}
//#endregion
//#region node_modules/ol/renderer/Composite.js
var Gr = (function() {
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
})(), Kr = function(e) {
	Gr(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		n.fontChangeListenerKey_ = k(Mr, u.PROPERTYCHANGE, t.redrawText.bind(t)), n.element_ = document.createElement("div");
		var r = n.element_.style;
		r.position = "absolute", r.width = "100%", r.height = "100%", r.zIndex = "0", n.element_.className = Ce + " ol-layers";
		var i = t.getViewport();
		return i.insertBefore(n.element_, i.firstChild || null), n.children_ = [], n.renderedVisible_ = !0, n;
	}
	return t.prototype.dispatchRenderEvent = function(e, t) {
		var n = this.getMap();
		if (n.hasListener(e)) {
			var r = new Sr(e, void 0, t);
			n.dispatchEvent(r);
		}
	}, t.prototype.disposeInternal = function() {
		j(this.fontChangeListenerKey_), this.element_.parentNode.removeChild(this.element_), e.prototype.disposeInternal.call(this);
	}, t.prototype.renderFrame = function(e) {
		if (!e) {
			this.renderedVisible_ &&= (this.element_.style.display = "none", !1);
			return;
		}
		this.calculateMatrices2D(e), this.dispatchRenderEvent(pr.PRECOMPOSE, e);
		var t = e.layerStatesArray.sort(function(e, t) {
			return e.zIndex - t.zIndex;
		}), n = e.viewState;
		this.children_.length = 0;
		for (var r = [], i = null, a = 0, o = t.length; a < o; ++a) {
			var s = t[a];
			if (e.layerIndex = a, !(!_r(s, n) || s.sourceState != mr.READY && s.sourceState != mr.UNDEFINED)) {
				var c = s.layer, l = c.render(e, i);
				l && (l !== i && (this.children_.push(l), i = l), "getDeclutter" in c && r.push(c));
			}
		}
		for (var a = r.length - 1; a >= 0; --a) r[a].renderDeclutter(e);
		ve(this.element_, this.children_), this.dispatchRenderEvent(pr.POSTCOMPOSE, e), this.renderedVisible_ ||= (this.element_.style.display = "", !0), this.scheduleExpireIconCache(e);
	}, t.prototype.forEachLayerAtPixel = function(e, t, n, r, i) {
		for (var a = t.viewState, o = t.layerStatesArray, s = o.length - 1; s >= 0; --s) {
			var c = o[s], l = c.layer;
			if (l.hasRenderer() && _r(c, a) && i(l)) {
				var u = l.getRenderer().getDataAtPixel(e, t, n);
				if (u) {
					var d = r(l, u);
					if (d) return d;
				}
			}
		}
	}, t;
}(yr), qr = {
	LAYERGROUP: "layergroup",
	SIZE: "size",
	TARGET: "target",
	VIEW: "view"
}, Jr = {
	BOTTOM_LEFT: "bottom-left",
	BOTTOM_CENTER: "bottom-center",
	BOTTOM_RIGHT: "bottom-right",
	CENTER_LEFT: "center-left",
	CENTER_CENTER: "center-center",
	CENTER_RIGHT: "center-right",
	TOP_LEFT: "top-left",
	TOP_CENTER: "top-center",
	TOP_RIGHT: "top-right"
}, Yr = (function() {
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
})(), Xr = {
	ELEMENT: "element",
	MAP: "map",
	OFFSET: "offset",
	POSITION: "position",
	POSITIONING: "positioning"
}, Zr = function(e) {
	Yr(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un, n.options = t, n.id = t.id, n.insertFirst = t.insertFirst === void 0 || t.insertFirst, n.stopEvent = t.stopEvent === void 0 || t.stopEvent, n.element = document.createElement("div"), n.element.className = t.className === void 0 ? "ol-overlay-container " + Se : t.className, n.element.style.position = "absolute", n.element.style.pointerEvents = "auto";
		var r = t.autoPan;
		return r && typeof r != "object" && (r = {
			animation: t.autoPanAnimation,
			margin: t.autoPanMargin
		}), n.autoPan = r || !1, n.rendered = {
			transform_: "",
			visible: !0
		}, n.mapPostrenderListenerKey = null, n.addChangeListener(Xr.ELEMENT, n.handleElementChanged), n.addChangeListener(Xr.MAP, n.handleMapChanged), n.addChangeListener(Xr.OFFSET, n.handleOffsetChanged), n.addChangeListener(Xr.POSITION, n.handlePositionChanged), n.addChangeListener(Xr.POSITIONING, n.handlePositioningChanged), t.element !== void 0 && n.setElement(t.element), n.setOffset(t.offset === void 0 ? [0, 0] : t.offset), n.setPositioning(t.positioning === void 0 ? Jr.TOP_LEFT : t.positioning), t.position !== void 0 && n.setPosition(t.position), n;
	}
	return t.prototype.getElement = function() {
		return this.get(Xr.ELEMENT);
	}, t.prototype.getId = function() {
		return this.id;
	}, t.prototype.getMap = function() {
		return this.get(Xr.MAP);
	}, t.prototype.getOffset = function() {
		return this.get(Xr.OFFSET);
	}, t.prototype.getPosition = function() {
		return this.get(Xr.POSITION);
	}, t.prototype.getPositioning = function() {
		return this.get(Xr.POSITIONING);
	}, t.prototype.handleElementChanged = function() {
		_e(this.element);
		var e = this.getElement();
		e && this.element.appendChild(e);
	}, t.prototype.handleMapChanged = function() {
		this.mapPostrenderListenerKey &&= (ge(this.element), j(this.mapPostrenderListenerKey), null);
		var e = this.getMap();
		if (e) {
			this.mapPostrenderListenerKey = k(e, re.POSTRENDER, this.render, this), this.updatePixelPosition();
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
		this.set(Xr.ELEMENT, e);
	}, t.prototype.setMap = function(e) {
		this.set(Xr.MAP, e);
	}, t.prototype.setOffset = function(e) {
		this.set(Xr.OFFSET, e);
	}, t.prototype.setPosition = function(e) {
		this.set(Xr.POSITION, e);
	}, t.prototype.performAutoPan = function() {
		this.autoPan && this.panIntoView(this.autoPan);
	}, t.prototype.panIntoView = function(e) {
		var t = this.getMap();
		if (!(!t || !t.getTargetElement() || !this.get(Xr.POSITION))) {
			var n = this.getRect(t.getTargetElement(), t.getSize()), r = this.getElement(), i = this.getRect(r, [pe(r), me(r)]), a = e || {}, o = a.margin === void 0 ? 20 : a.margin;
			if (!Ot(n, i)) {
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
		this.set(Xr.POSITIONING, e);
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
		i == Jr.BOTTOM_RIGHT || i == Jr.CENTER_RIGHT || i == Jr.TOP_RIGHT ? s = "-100%" : (i == Jr.BOTTOM_CENTER || i == Jr.CENTER_CENTER || i == Jr.TOP_CENTER) && (s = "-50%"), i == Jr.BOTTOM_LEFT || i == Jr.BOTTOM_CENTER || i == Jr.BOTTOM_RIGHT ? c = "-100%" : (i == Jr.CENTER_LEFT || i == Jr.CENTER_CENTER || i == Jr.CENTER_RIGHT) && (c = "-50%");
		var l = "translate(" + s + ", " + c + ") translate(" + a + ", " + o + ")";
		this.rendered.transform_ != l && (this.rendered.transform_ = l, n.transform = l, n.msTransform = l);
	}, t.prototype.getOptions = function() {
		return this.options;
	}, t;
}(R), Qr = {
	ADD: "add",
	REMOVE: "remove"
}, $r = (function() {
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
})(), ei = { LENGTH: "length" }, ti = function(e) {
	$r(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.element = n, i.index = r, i;
	}
	return t;
}(l), ni = function(e) {
	$r(t, e);
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
		return this.get(ei.LENGTH);
	}, t.prototype.insertAt = function(e, t) {
		this.unique_ && this.assertUnique_(t), this.array_.splice(e, 0, t), this.updateLength_(), this.dispatchEvent(new ti(Qr.ADD, t, e));
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
		return this.array_.splice(e, 1), this.updateLength_(), this.dispatchEvent(new ti(Qr.REMOVE, t, e)), t;
	}, t.prototype.setAt = function(e, t) {
		var n = this.getLength();
		if (e < n) {
			this.unique_ && this.assertUnique_(t, e);
			var r = this.array_[e];
			this.array_[e] = t, this.dispatchEvent(new ti(Qr.REMOVE, r, e)), this.dispatchEvent(new ti(Qr.ADD, t, e));
		} else {
			for (var i = n; i < e; ++i) this.insertAt(i, void 0);
			this.insertAt(e, t);
		}
	}, t.prototype.updateLength_ = function() {
		this.set(ei.LENGTH, this.array_.length);
	}, t.prototype.assertUnique_ = function(e, t) {
		for (var n = 0, r = this.array_.length; n < r; ++n) if (this.array_[n] === e && n !== t) throw new St(58);
	}, t;
}(R), ri = (function() {
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
})(), ii = { LAYERS: "layers" }, ai = function(e) {
	ri(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		delete i.layers;
		var a = r.layers;
		return n = e.call(this, i) || this, n.on, n.once, n.un, n.layersListenerKeys_ = [], n.listenerKeys_ = {}, n.addChangeListener(ii.LAYERS, n.handleLayersChanged_), a ? Array.isArray(a) ? a = new ni(a.slice(), { unique: !0 }) : V(typeof a.getArray == "function", 43) : a = new ni(void 0, { unique: !0 }), n.setLayers(a), n;
	}
	return t.prototype.handleLayerChange_ = function() {
		this.changed();
	}, t.prototype.handleLayersChanged_ = function() {
		this.layersListenerKeys_.forEach(j), this.layersListenerKeys_.length = 0;
		var e = this.getLayers();
		for (var t in this.layersListenerKeys_.push(k(e, Qr.ADD, this.handleLayersAdd_, this), k(e, Qr.REMOVE, this.handleLayersRemove_, this)), this.listenerKeys_) this.listenerKeys_[t].forEach(j);
		C(this.listenerKeys_);
		for (var n = e.getArray(), r = 0, i = n.length; r < i; r++) {
			var a = n[r];
			this.listenerKeys_[I(a)] = [k(a, u.PROPERTYCHANGE, this.handleLayerChange_, this), k(a, O.CHANGE, this.handleLayerChange_, this)];
		}
		this.changed();
	}, t.prototype.handleLayersAdd_ = function(e) {
		var t = e.element;
		this.listenerKeys_[I(t)] = [k(t, u.PROPERTYCHANGE, this.handleLayerChange_, this), k(t, O.CHANGE, this.handleLayerChange_, this)], this.changed();
	}, t.prototype.handleLayersRemove_ = function(e) {
		var t = e.element, n = I(t);
		this.listenerKeys_[n].forEach(j), delete this.listenerKeys_[n], this.changed();
	}, t.prototype.getLayers = function() {
		return this.get(ii.LAYERS);
	}, t.prototype.setLayers = function(e) {
		this.set(ii.LAYERS, e);
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
			s.opacity *= r.opacity, s.visible = s.visible && r.visible, s.maxResolution = Math.min(s.maxResolution, r.maxResolution), s.minResolution = Math.max(s.minResolution, r.minResolution), s.minZoom = Math.max(s.minZoom, r.minZoom), s.maxZoom = Math.min(s.maxZoom, r.maxZoom), r.extent !== void 0 && (s.extent === void 0 ? s.extent = r.extent : s.extent = Yt(s.extent, r.extent)), s.zIndex === void 0 && (s.zIndex = i);
		}
		return t;
	}, t.prototype.getSourceState = function() {
		return mr.READY;
	}, t;
}(fr), oi = (function() {
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
})(), si = function(e) {
	oi(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.map = n, i.frameState = r === void 0 ? null : r, i;
	}
	return t;
}(l), ci = (function() {
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
})(), li = function(e) {
	ci(t, e);
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
}(si), K = {
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
}, ui = (function() {
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
})(), di = function(e) {
	ui(t, e);
	function t(t, n) {
		var r = e.call(this, t) || this;
		r.map_ = t, r.clickTimeoutId_, r.emulateClicks_ = !1, r.dragging_ = !1, r.dragListenerKeys_ = [], r.moveTolerance_ = n === void 0 ? 1 : n, r.down_ = null;
		var i = r.map_.getViewport();
		return r.activePointers_ = 0, r.trackedTouches_ = {}, r.element_ = i, r.pointerdownListenerKey_ = k(i, Be.POINTERDOWN, r.handlePointerDown_, r), r.originalPointerMoveEvent_, r.relayedListenerKey_ = k(i, Be.POINTERMOVE, r.relayEvent_, r), r.boundHandleTouchMove_ = r.handleTouchMove_.bind(r), r.element_.addEventListener(O.TOUCHMOVE, r.boundHandleTouchMove_, de ? { passive: !1 } : !1), r;
	}
	return t.prototype.emulateClick_ = function(e) {
		var t = new li(K.CLICK, this.map_, e);
		this.dispatchEvent(t), this.clickTimeoutId_ === void 0 ? this.clickTimeoutId_ = setTimeout(function() {
			this.clickTimeoutId_ = void 0;
			var t = new li(K.SINGLECLICK, this.map_, e);
			this.dispatchEvent(t);
		}.bind(this), 250) : (clearTimeout(this.clickTimeoutId_), this.clickTimeoutId_ = void 0, t = new li(K.DBLCLICK, this.map_, e), this.dispatchEvent(t));
	}, t.prototype.updateActivePointers_ = function(e) {
		var t = e;
		t.type == K.POINTERUP || t.type == K.POINTERCANCEL ? delete this.trackedTouches_[t.pointerId] : t.type == K.POINTERDOWN && (this.trackedTouches_[t.pointerId] = !0), this.activePointers_ = Object.keys(this.trackedTouches_).length;
	}, t.prototype.handlePointerUp_ = function(e) {
		this.updateActivePointers_(e);
		var t = new li(K.POINTERUP, this.map_, e);
		this.dispatchEvent(t), this.emulateClicks_ && !t.defaultPrevented && !this.dragging_ && this.isMouseActionButton_(e) && this.emulateClick_(this.down_), this.activePointers_ === 0 && (this.dragListenerKeys_.forEach(j), this.dragListenerKeys_.length = 0, this.dragging_ = !1, this.down_ = null);
	}, t.prototype.isMouseActionButton_ = function(e) {
		return e.button === 0;
	}, t.prototype.handlePointerDown_ = function(e) {
		this.emulateClicks_ = this.activePointers_ === 0, this.updateActivePointers_(e);
		var t = new li(K.POINTERDOWN, this.map_, e);
		for (var n in this.dispatchEvent(t), this.down_ = {}, e) {
			var r = e[n];
			this.down_[n] = typeof r == "function" ? b : r;
		}
		if (this.dragListenerKeys_.length === 0) {
			var i = this.map_.getOwnerDocument();
			this.dragListenerKeys_.push(k(i, K.POINTERMOVE, this.handlePointerMove_, this), k(i, K.POINTERUP, this.handlePointerUp_, this), k(this.element_, K.POINTERCANCEL, this.handlePointerUp_, this)), this.element_.getRootNode && this.element_.getRootNode() !== i && this.dragListenerKeys_.push(k(this.element_.getRootNode(), K.POINTERUP, this.handlePointerUp_, this));
		}
	}, t.prototype.handlePointerMove_ = function(e) {
		if (this.isMoving_(e)) {
			this.dragging_ = !0;
			var t = new li(K.POINTERDRAG, this.map_, e, this.dragging_);
			this.dispatchEvent(t);
		}
	}, t.prototype.relayEvent_ = function(e) {
		this.originalPointerMoveEvent_ = e;
		var t = !!(this.down_ && this.isMoving_(e));
		this.dispatchEvent(new li(e.type, this.map_, e, t));
	}, t.prototype.handleTouchMove_ = function(e) {
		var t = this.originalPointerMoveEvent_;
		(!t || t.defaultPrevented) && (typeof e.cancelable != "boolean" || e.cancelable === !0) && e.preventDefault();
	}, t.prototype.isMoving_ = function(e) {
		return this.dragging_ || Math.abs(e.clientX - this.down_.clientX) > this.moveTolerance_ || Math.abs(e.clientY - this.down_.clientY) > this.moveTolerance_;
	}, t.prototype.disposeInternal = function() {
		this.relayedListenerKey_ &&= (j(this.relayedListenerKey_), null), this.element_.removeEventListener(O.TOUCHMOVE, this.boundHandleTouchMove_), this.pointerdownListenerKey_ &&= (j(this.pointerdownListenerKey_), null), this.dragListenerKeys_.forEach(j), this.dragListenerKeys_.length = 0, this.element_ = null, e.prototype.disposeInternal.call(this);
	}, t;
}(D), fi = Infinity, pi = function() {
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
		V(!(this.keyFunction_(e) in this.queuedElements_), 31);
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
}(), q = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
}, mi = (function() {
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
})(), hi = function(e) {
	mi(t, e);
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
		if (n === q.LOADED || n === q.ERROR || n === q.EMPTY) {
			t.removeEventListener(O.CHANGE, this.boundHandleTileChange_);
			var r = t.getKey();
			r in this.tilesLoadingKeys_ && (delete this.tilesLoadingKeys_[r], --this.tilesLoading_), this.tileChangeCallback_();
		}
	}, t.prototype.loadMoreTiles = function(e, t) {
		for (var n = 0, r, i, a; this.tilesLoading_ < e && n < t && this.getCount() > 0;) i = this.dequeue()[0], a = i.getKey(), r = i.getState(), r === q.IDLE && !(a in this.tilesLoadingKeys_) && (this.tilesLoadingKeys_[a] = !0, ++this.tilesLoading_, ++n, i.load());
	}, t;
}(pi);
function gi(e, t, n, r, i) {
	if (!e || !(n in e.wantedTiles) || !e.wantedTiles[n][t.getKey()]) return fi;
	var a = e.viewState.center, o = r[0] - a[0], s = r[1] - a[1];
	return 65536 * Math.log(i) + Math.sqrt(o * o + s * s) / i;
}
//#endregion
//#region node_modules/ol/ViewHint.js
var J = {
	ANIMATING: 0,
	INTERACTING: 1
}, _i = {
	CENTER: "center",
	RESOLUTION: "resolution",
	ROTATION: "rotation"
};
//#endregion
//#region node_modules/ol/centerconstraint.js
function vi(e, t, n) {
	return (function(r, i, a, o, s) {
		if (r) {
			var c = t ? 0 : a[0] * i, l = t ? 0 : a[1] * i, u = s ? s[0] : 0, d = s ? s[1] : 0, f = e[0] + c / 2 + u, p = e[2] - c / 2 + u, m = e[1] + l / 2 + d, h = e[3] - l / 2 + d;
			f > p && (f = (p + f) / 2, p = f), m > h && (m = (h + m) / 2, h = m);
			var g = B(r[0], f, p), _ = B(r[1], m, h), v = 30 * i;
			return o && n && (g += -v * Math.log(1 + Math.max(0, f - r[0]) / v) + v * Math.log(1 + Math.max(0, r[0] - p) / v), _ += -v * Math.log(1 + Math.max(0, m - r[1]) / v) + v * Math.log(1 + Math.max(0, r[1] - h) / v)), [g, _];
		} else return;
	});
}
function yi(e) {
	return e;
}
//#endregion
//#region node_modules/ol/resolutionconstraint.js
function bi(e, t, n, r) {
	var i = H(t) / n[0], a = Jt(t) / n[1];
	return r ? Math.min(e, Math.max(i, a)) : Math.min(e, Math.min(i, a));
}
function xi(e, t, n) {
	var r = Math.min(e, t), i = 50;
	return r *= Math.log(1 + i * Math.max(0, e / t - 1)) / i + 1, n && (r = Math.max(r, n), r /= Math.log(1 + i * Math.max(0, n / e - 1)) / i + 1), B(r, n / 2, t * 2);
}
function Si(e, t, n, r) {
	return (function(i, a, o, s) {
		if (i !== void 0) {
			var c = e[0], l = e[e.length - 1], u = n ? bi(c, n, o, r) : c;
			if (s) return t === void 0 || t ? xi(i, u, l) : B(i, l, u);
			var d = Math.floor(p(e, Math.min(u, i), a));
			return e[d] > u && d < e.length - 1 ? e[d + 1] : e[d];
		} else return;
	});
}
function Ci(e, t, n, r, i, a) {
	return (function(o, s, c, l) {
		if (o !== void 0) {
			var u = i ? bi(t, i, c, a) : t, d = n === void 0 ? 0 : n;
			if (l) return r === void 0 || r ? xi(o, u, d) : B(o, d, u);
			var f = 1e-9, p = Math.ceil(Math.log(t / u) / Math.log(e) - f), m = -s * (.5 - f) + .5, h = Math.floor(Math.log(t / Math.min(u, o)) / Math.log(e) + m);
			return B(t / e ** +Math.max(p, h), d, u);
		} else return;
	});
}
function wi(e, t, n, r, i) {
	return (function(a, o, s, c) {
		if (a !== void 0) {
			var l = r ? bi(e, r, s, i) : e;
			return !(n === void 0 || n) || !c ? B(a, t, l) : xi(a, l, t);
		} else return;
	});
}
//#endregion
//#region node_modules/ol/rotationconstraint.js
function Ti(e) {
	if (e !== void 0) return 0;
}
function Ei(e) {
	if (e !== void 0) return e;
}
function Di(e) {
	var t = 2 * Math.PI / e;
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return e = Math.floor(e / t + .5) * t, e;
	});
}
function Oi(e) {
	var t = e || Je(5);
	return (function(e, n) {
		if (n) return e;
		if (e !== void 0) return Math.abs(e) <= t ? 0 : e;
	});
}
//#endregion
//#region node_modules/ol/easing.js
function ki(e) {
	return e ** 3;
}
function Ai(e) {
	return 1 - ki(1 - e);
}
function ji(e) {
	return 3 * e * e - 2 * e * e * e;
}
function Mi(e) {
	return e;
}
//#endregion
//#region node_modules/ol/geom/GeometryLayout.js
var Ni = {
	XY: "XY",
	XYZ: "XYZ",
	XYM: "XYM",
	XYZM: "XYZM"
};
//#endregion
//#region node_modules/ol/geom/flat/transform.js
function Pi(e, t, n, r, i, a) {
	for (var o = a || [], s = 0, c = t; c < n; c += r) {
		var l = e[c], u = e[c + 1];
		o[s++] = i[0] * l + i[2] * u + i[4], o[s++] = i[1] * l + i[3] * u + i[5];
	}
	return a && o.length != s && (o.length = s), o;
}
function Fi(e, t, n, r, i, a, o) {
	for (var s = o || [], c = Math.cos(i), l = Math.sin(i), u = a[0], d = a[1], f = 0, p = t; p < n; p += r) {
		var m = e[p] - u, h = e[p + 1] - d;
		s[f++] = u + m * c - h * l, s[f++] = d + m * l + h * c;
		for (var g = p + 2; g < p + r; ++g) s[f++] = e[g];
	}
	return o && s.length != f && (s.length = f), s;
}
function Ii(e, t, n, r, i, a, o, s) {
	for (var c = s || [], l = o[0], u = o[1], d = 0, f = t; f < n; f += r) {
		var p = e[f] - l, m = e[f + 1] - u;
		c[d++] = l + i * p, c[d++] = u + a * m;
		for (var h = f + 2; h < f + r; ++h) c[d++] = e[h];
	}
	return s && c.length != d && (c.length = d), c;
}
function Li(e, t, n, r, i, a, o) {
	for (var s = o || [], c = 0, l = t; l < n; l += r) {
		s[c++] = e[l] + i, s[c++] = e[l + 1] + a;
		for (var u = l + 2; u < l + r; ++u) s[c++] = e[u];
	}
	return o && s.length != c && (s.length = c), s;
}
//#endregion
//#region node_modules/ol/geom/Geometry.js
var Ri = (function() {
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
})(), zi = zn(), Bi = function(e) {
	Ri(t, e);
	function t() {
		var t = e.call(this) || this;
		return t.extent_ = jt(), t.extentRevision_ = -1, t.simplifiedGeometryMaxMinSquaredTolerance = 0, t.simplifiedGeometryRevision = 0, t.simplifyTransformedInternal = x(function(e, t, n) {
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
			(isNaN(t[0]) || isNaN(t[1])) && Nt(t), this.extentRevision_ = this.getRevision();
		}
		return en(this.extent_, e);
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
		var n = _n(e), r = n.getUnits() == z.TILE_PIXELS ? function(e, r, i) {
			var a = n.getExtent(), o = n.getWorldExtent(), s = Jt(o) / Jt(a);
			return Jn(zi, o[0], o[3], s, -s, 0, 0, 0), Pi(e, 0, e.length, i, zi, r), wn(n, t)(e, r, i);
		} : wn(n, t);
		return this.applyTransform(r), this;
	}, t;
}(R), Vi = (function() {
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
})(), Hi = function(e) {
	Vi(t, e);
	function t() {
		var t = e.call(this) || this;
		return t.layout = Ni.XY, t.stride = 2, t.flatCoordinates = null, t;
	}
	return t.prototype.computeExtent = function(e) {
		return Ft(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e);
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
		this.stride = Wi(e), this.layout = e, this.flatCoordinates = t;
	}, t.prototype.setCoordinates = function(e, t) {
		F();
	}, t.prototype.setLayout = function(e, t, n) {
		var r;
		if (e) r = Wi(e);
		else {
			for (var i = 0; i < n; ++i) if (t.length === 0) {
				this.layout = Ni.XY, this.stride = 2;
				return;
			} else t = t[0];
			r = t.length, e = Ui(r);
		}
		this.layout = e, this.stride = r;
	}, t.prototype.applyTransform = function(e) {
		this.flatCoordinates && (e(this.flatCoordinates, this.flatCoordinates, this.stride), this.changed());
	}, t.prototype.rotate = function(e, t) {
		var n = this.getFlatCoordinates();
		if (n) {
			var r = this.getStride();
			Fi(n, 0, n.length, r, e, t, n), this.changed();
		}
	}, t.prototype.scale = function(e, t, n) {
		var r = t;
		r === void 0 && (r = e);
		var i = n;
		i ||= Gt(this.getExtent());
		var a = this.getFlatCoordinates();
		if (a) {
			var o = this.getStride();
			Ii(a, 0, a.length, o, e, r, i, a), this.changed();
		}
	}, t.prototype.translate = function(e, t) {
		var n = this.getFlatCoordinates();
		if (n) {
			var r = this.getStride();
			Li(n, 0, n.length, r, e, t, n), this.changed();
		}
	}, t;
}(Bi);
function Ui(e) {
	var t;
	return e == 2 ? t = Ni.XY : e == 3 ? t = Ni.XYZ : e == 4 && (t = Ni.XYZM), t;
}
function Wi(e) {
	var t;
	return e == Ni.XY ? t = 2 : e == Ni.XYZ || e == Ni.XYM ? t = 3 : e == Ni.XYZM && (t = 4), t;
}
function Gi(e, t, n) {
	var r = e.getFlatCoordinates();
	if (r) {
		var i = e.getStride();
		return Pi(r, 0, r.length, i, t, n);
	} else return null;
}
//#endregion
//#region node_modules/ol/geom/flat/closest.js
function Ki(e, t, n, r, i, a, o) {
	var s = e[t], c = e[t + 1], l = e[n] - s, u = e[n + 1] - c, d;
	if (l === 0 && u === 0) d = t;
	else {
		var f = ((i - s) * l + (a - c) * u) / (l * l + u * u);
		if (f > 1) d = n;
		else if (f > 0) {
			for (var p = 0; p < r; ++p) o[p] = Xe(e[t + p], e[n + p], f);
			o.length = r;
			return;
		} else d = t;
	}
	for (var p = 0; p < r; ++p) o[p] = e[d + p];
	o.length = r;
}
function qi(e, t, n, r, i) {
	var a = e[t], o = e[t + 1];
	for (t += r; t < n; t += r) {
		var s = e[t], c = e[t + 1], l = Ke(a, o, s, c);
		l > i && (i = l), a = s, o = c;
	}
	return i;
}
function Ji(e, t, n, r, i) {
	for (var a = 0, o = n.length; a < o; ++a) {
		var s = n[a];
		i = qi(e, t, s, r, i), t = s;
	}
	return i;
}
function Yi(e, t, n, r, i, a, o, s, c, l, u) {
	if (t == n) return l;
	var d, f;
	if (i === 0) if (f = Ke(o, s, e[t], e[t + 1]), f < l) {
		for (d = 0; d < r; ++d) c[d] = e[t + d];
		return c.length = r, f;
	} else return l;
	for (var p = u || [NaN, NaN], m = t + r; m < n;) if (Ki(e, m - r, m, r, o, s, p), f = Ke(o, s, p[0], p[1]), f < l) {
		for (l = f, d = 0; d < r; ++d) c[d] = p[d];
		c.length = r, m += r;
	} else m += r * Math.max((Math.sqrt(f) - Math.sqrt(l)) / i | 0, 1);
	if (a && (Ki(e, n - r, t, r, o, s, p), f = Ke(o, s, p[0], p[1]), f < l)) {
		for (l = f, d = 0; d < r; ++d) c[d] = p[d];
		c.length = r;
	}
	return l;
}
function Xi(e, t, n, r, i, a, o, s, c, l, u) {
	for (var d = u || [NaN, NaN], f = 0, p = n.length; f < p; ++f) {
		var m = n[f];
		l = Yi(e, t, m, r, i, a, o, s, c, l, d), t = m;
	}
	return l;
}
//#endregion
//#region node_modules/ol/geom/flat/deflate.js
function Zi(e, t, n, r) {
	for (var i = 0, a = n.length; i < a; ++i) e[t++] = n[i];
	return t;
}
function Qi(e, t, n, r) {
	for (var i = 0, a = n.length; i < a; ++i) for (var o = n[i], s = 0; s < r; ++s) e[t++] = o[s];
	return t;
}
function $i(e, t, n, r, i) {
	for (var a = i || [], o = 0, s = 0, c = n.length; s < c; ++s) {
		var l = Qi(e, t, n[s], r);
		a[o++] = l, t = l;
	}
	return a.length = o, a;
}
//#endregion
//#region node_modules/ol/geom/flat/simplify.js
function ea(e, t, n, r, i, a, o) {
	var s = (n - t) / r;
	if (s < 3) {
		for (; t < n; t += r) a[o++] = e[t], a[o++] = e[t + 1];
		return o;
	}
	var c = Array(s);
	c[0] = 1, c[s - 1] = 1;
	for (var l = [t, n - r], u = 0; l.length > 0;) {
		for (var d = l.pop(), f = l.pop(), p = 0, m = e[f], h = e[f + 1], g = e[d], _ = e[d + 1], v = f + r; v < d; v += r) {
			var y = e[v], b = e[v + 1], x = Ge(y, b, m, h, g, _);
			x > p && (u = v, p = x);
		}
		p > i && (c[(u - t) / r] = 1, f + r < u && l.push(f, u), u + r < d && l.push(u, d));
	}
	for (var v = 0; v < s; ++v) c[v] && (a[o++] = e[t + v * r], a[o++] = e[t + v * r + 1]);
	return o;
}
function ta(e, t) {
	return t * Math.round(e / t);
}
function na(e, t, n, r, i, a, o) {
	if (t == n) return o;
	var s = ta(e[t], i), c = ta(e[t + 1], i);
	t += r, a[o++] = s, a[o++] = c;
	var l, u;
	do
		if (l = ta(e[t], i), u = ta(e[t + 1], i), t += r, t == n) return a[o++] = l, a[o++] = u, o;
	while (l == s && u == c);
	for (; t < n;) {
		var d = ta(e[t], i), f = ta(e[t + 1], i);
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
function ra(e, t, n, r, i, a, o, s) {
	for (var c = 0, l = n.length; c < l; ++c) {
		var u = n[c];
		o = na(e, t, u, r, i, a, o), s.push(o), t = u;
	}
	return o;
}
//#endregion
//#region node_modules/ol/geom/flat/inflate.js
function ia(e, t, n, r, i) {
	for (var a = i === void 0 ? [] : i, o = 0, s = t; s < n; s += r) a[o++] = e.slice(s, s + r);
	return a.length = o, a;
}
function aa(e, t, n, r, i) {
	for (var a = i === void 0 ? [] : i, o = 0, s = 0, c = n.length; s < c; ++s) {
		var l = n[s];
		a[o++] = ia(e, t, l, r, a[o]), t = l;
	}
	return a.length = o, a;
}
function oa(e, t, n, r, i) {
	for (var a = i === void 0 ? [] : i, o = 0, s = 0, c = n.length; s < c; ++s) {
		var l = n[s];
		a[o++] = aa(e, t, l, r, a[o]), t = l[l.length - 1];
	}
	return a.length = o, a;
}
//#endregion
//#region node_modules/ol/geom/flat/area.js
function sa(e, t, n, r) {
	for (var i = 0, a = e[n - r], o = e[n - r + 1]; t < n; t += r) {
		var s = e[t], c = e[t + 1];
		i += o * s - a * c, a = s, o = c;
	}
	return i / 2;
}
function ca(e, t, n, r) {
	for (var i = 0, a = 0, o = n.length; a < o; ++a) {
		var s = n[a];
		i += sa(e, t, s, r), t = s;
	}
	return i;
}
//#endregion
//#region node_modules/ol/geom/LinearRing.js
var la = (function() {
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
})(), ua = function(e) {
	la(t, e);
	function t(t, n) {
		var r = e.call(this) || this;
		return r.maxDelta_ = -1, r.maxDeltaRevision_ = -1, n !== void 0 && !Array.isArray(t[0]) ? r.setFlatCoordinates(n, t) : r.setCoordinates(t, n), r;
	}
	return t.prototype.clone = function() {
		return new t(this.flatCoordinates.slice(), this.layout);
	}, t.prototype.closestPointXY = function(e, t, n, r) {
		return r < Et(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(qi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), Yi(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, this.maxDelta_, !0, e, t, n, r));
	}, t.prototype.getArea = function() {
		return sa(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}, t.prototype.getCoordinates = function() {
		return ia(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride);
	}, t.prototype.getSimplifiedGeometryInternal = function(e) {
		var n = [];
		return n.length = ea(this.flatCoordinates, 0, this.flatCoordinates.length, this.stride, e, n, 0), new t(n, Ni.XY);
	}, t.prototype.getType = function() {
		return U.LINEAR_RING;
	}, t.prototype.intersectsExtent = function(e) {
		return !1;
	}, t.prototype.setCoordinates = function(e, t) {
		this.setLayout(t, e, 1), this.flatCoordinates ||= [], this.flatCoordinates.length = Qi(this.flatCoordinates, 0, e, this.stride), this.changed();
	}, t;
}(Hi), da = (function() {
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
})(), fa = function(e) {
	da(t, e);
	function t(t, n) {
		var r = e.call(this) || this;
		return r.setCoordinates(t, n), r;
	}
	return t.prototype.clone = function() {
		var e = new t(this.flatCoordinates.slice(), this.layout);
		return e.applyProperties(this), e;
	}, t.prototype.closestPointXY = function(e, t, n, r) {
		var i = this.flatCoordinates, a = Ke(e, t, i[0], i[1]);
		if (a < r) {
			for (var o = this.stride, s = 0; s < o; ++s) n[s] = i[s];
			return n.length = o, a;
		} else return r;
	}, t.prototype.getCoordinates = function() {
		return this.flatCoordinates ? this.flatCoordinates.slice() : [];
	}, t.prototype.computeExtent = function(e) {
		return Pt(this.flatCoordinates, e);
	}, t.prototype.getType = function() {
		return U.POINT;
	}, t.prototype.intersectsExtent = function(e) {
		return kt(e, this.flatCoordinates[0], this.flatCoordinates[1]);
	}, t.prototype.setCoordinates = function(e, t) {
		this.setLayout(t, e, 0), this.flatCoordinates ||= [], this.flatCoordinates.length = Zi(this.flatCoordinates, 0, e, this.stride), this.changed();
	}, t;
}(Hi);
//#endregion
//#region node_modules/ol/geom/flat/contains.js
function pa(e, t, n, r, i) {
	return !Vt(i, function(i) {
		return !ma(e, t, n, r, i[0], i[1]);
	});
}
function ma(e, t, n, r, i, a) {
	for (var o = 0, s = e[n - r], c = e[n - r + 1]; t < n; t += r) {
		var l = e[t], u = e[t + 1];
		c <= a ? u > a && (l - s) * (a - c) - (i - s) * (u - c) > 0 && o++ : u <= a && (l - s) * (a - c) - (i - s) * (u - c) < 0 && o--, s = l, c = u;
	}
	return o !== 0;
}
function ha(e, t, n, r, i, a) {
	if (n.length === 0 || !ma(e, t, n[0], r, i, a)) return !1;
	for (var o = 1, s = n.length; o < s; ++o) if (ma(e, n[o - 1], n[o], r, i, a)) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/interiorpoint.js
function ga(e, t, n, r, i, a, o) {
	for (var s, c, l, u, d, p, m, h = i[a + 1], g = [], _ = 0, v = n.length; _ < v; ++_) {
		var y = n[_];
		for (u = e[y - r], p = e[y - r + 1], s = t; s < y; s += r) d = e[s], m = e[s + 1], (h <= p && m <= h || p <= h && h <= m) && (l = (h - p) / (m - p) * (d - u) + u, g.push(l)), u = d, p = m;
	}
	var b = NaN, x = -Infinity;
	for (g.sort(f), u = g[0], s = 1, c = g.length; s < c; ++s) {
		d = g[s];
		var S = Math.abs(d - u);
		S > x && (l = (u + d) / 2, ha(e, t, n, r, l, h) && (b = l, x = S)), u = d;
	}
	return isNaN(b) && (b = i[a]), o ? (o.push(b, h, x), o) : [
		b,
		h,
		x
	];
}
//#endregion
//#region node_modules/ol/geom/flat/segments.js
function _a(e, t, n, r, i) {
	var a;
	for (t += r; t < n; t += r) if (a = i(e.slice(t - r, t), e.slice(t, t + r)), a) return a;
	return !1;
}
//#endregion
//#region node_modules/ol/geom/flat/intersectsextent.js
function va(e, t, n, r, i) {
	var a = zt(jt(), e, t, n, r);
	return Qt(i, a) ? Ot(i, a) || a[0] >= i[0] && a[2] <= i[2] || a[1] >= i[1] && a[3] <= i[3] ? !0 : _a(e, t, n, r, function(e, t) {
		return nn(i, e, t);
	}) : !1;
}
function ya(e, t, n, r, i) {
	return !!(va(e, t, n, r, i) || ma(e, t, n, r, i[0], i[1]) || ma(e, t, n, r, i[0], i[3]) || ma(e, t, n, r, i[2], i[1]) || ma(e, t, n, r, i[2], i[3]));
}
function ba(e, t, n, r, i) {
	if (!ya(e, t, n[0], r, i)) return !1;
	if (n.length === 1) return !0;
	for (var a = 1, o = n.length; a < o; ++a) if (pa(e, n[a - 1], n[a], r, i) && !va(e, n[a - 1], n[a], r, i)) return !1;
	return !0;
}
//#endregion
//#region node_modules/ol/geom/flat/reverse.js
function xa(e, t, n, r) {
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
function Sa(e, t, n, r) {
	for (var i = 0, a = e[n - r], o = e[n - r + 1]; t < n; t += r) {
		var s = e[t], c = e[t + 1];
		i += (s - a) * (c + o), a = s, o = c;
	}
	return i === 0 ? void 0 : i > 0;
}
function Ca(e, t, n, r, i) {
	for (var a = i !== void 0 && i, o = 0, s = n.length; o < s; ++o) {
		var c = n[o], l = Sa(e, t, c, r);
		if (o === 0) {
			if (a && l || !a && !l) return !1;
		} else if (a && !l || !a && l) return !1;
		t = c;
	}
	return !0;
}
function wa(e, t, n, r, i) {
	for (var a = i !== void 0 && i, o = 0, s = n.length; o < s; ++o) {
		var c = n[o], l = Sa(e, t, c, r);
		(o === 0 ? a && l || !a && !l : a && !l || !a && l) && xa(e, t, c, r), t = c;
	}
	return t;
}
//#endregion
//#region node_modules/ol/geom/Polygon.js
var Ta = (function() {
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
})(), Ea = function(e) {
	Ta(t, e);
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
		return r < Et(this.getExtent(), e, t) ? r : (this.maxDeltaRevision_ != this.getRevision() && (this.maxDelta_ = Math.sqrt(Ji(this.flatCoordinates, 0, this.ends_, this.stride, 0)), this.maxDeltaRevision_ = this.getRevision()), Xi(this.flatCoordinates, 0, this.ends_, this.stride, this.maxDelta_, !0, e, t, n, r));
	}, t.prototype.containsXY = function(e, t) {
		return ha(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, t);
	}, t.prototype.getArea = function() {
		return ca(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride);
	}, t.prototype.getCoordinates = function(e) {
		var t;
		return e === void 0 ? t = this.flatCoordinates : (t = this.getOrientedFlatCoordinates().slice(), wa(t, 0, this.ends_, this.stride, e)), aa(t, 0, this.ends_, this.stride);
	}, t.prototype.getEnds = function() {
		return this.ends_;
	}, t.prototype.getFlatInteriorPoint = function() {
		if (this.flatInteriorPointRevision_ != this.getRevision()) {
			var e = Gt(this.getExtent());
			this.flatInteriorPoint_ = ga(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e, 0), this.flatInteriorPointRevision_ = this.getRevision();
		}
		return this.flatInteriorPoint_;
	}, t.prototype.getInteriorPoint = function() {
		return new fa(this.getFlatInteriorPoint(), Ni.XYM);
	}, t.prototype.getLinearRingCount = function() {
		return this.ends_.length;
	}, t.prototype.getLinearRing = function(e) {
		return e < 0 || this.ends_.length <= e ? null : new ua(this.flatCoordinates.slice(e === 0 ? 0 : this.ends_[e - 1], this.ends_[e]), this.layout);
	}, t.prototype.getLinearRings = function() {
		for (var e = this.layout, t = this.flatCoordinates, n = this.ends_, r = [], i = 0, a = 0, o = n.length; a < o; ++a) {
			var s = n[a], c = new ua(t.slice(i, s), e);
			r.push(c), i = s;
		}
		return r;
	}, t.prototype.getOrientedFlatCoordinates = function() {
		if (this.orientedRevision_ != this.getRevision()) {
			var e = this.flatCoordinates;
			Ca(e, 0, this.ends_, this.stride) ? this.orientedFlatCoordinates_ = e : (this.orientedFlatCoordinates_ = e.slice(), this.orientedFlatCoordinates_.length = wa(this.orientedFlatCoordinates_, 0, this.ends_, this.stride)), this.orientedRevision_ = this.getRevision();
		}
		return this.orientedFlatCoordinates_;
	}, t.prototype.getSimplifiedGeometryInternal = function(e) {
		var n = [], r = [];
		return n.length = ra(this.flatCoordinates, 0, this.ends_, this.stride, Math.sqrt(e), n, 0, r), new t(n, Ni.XY, r);
	}, t.prototype.getType = function() {
		return U.POLYGON;
	}, t.prototype.intersectsExtent = function(e) {
		return ba(this.getOrientedFlatCoordinates(), 0, this.ends_, this.stride, e);
	}, t.prototype.setCoordinates = function(e, t) {
		this.setLayout(t, e, 2), this.flatCoordinates ||= [];
		var n = $i(this.flatCoordinates, 0, e, this.stride, this.ends_);
		this.flatCoordinates.length = n.length === 0 ? 0 : n[n.length - 1], this.changed();
	}, t;
}(Hi);
function Da(e) {
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
	return new Ea(a, Ni.XY, [a.length]);
}
//#endregion
//#region node_modules/ol/View.js
var Oa = (function() {
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
})(), ka = 0, Aa = function(e) {
	Oa(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = S({}, t);
		return n.hints_ = [0, 0], n.animations_ = [], n.updateAnimationKey_, n.projection_ = xn(r.projection, "EPSG:3857"), n.viewportSize_ = [100, 100], n.targetCenter_ = null, n.targetResolution_, n.targetRotation_, n.nextCenter_ = null, n.nextResolution_, n.nextRotation_, n.cancelAnchor_ = void 0, r.center &&= kn(r.center, n.projection_), r.extent &&= jn(r.extent, n.projection_), n.applyOptions_(r), n;
	}
	return t.prototype.applyOptions_ = function(e) {
		var t = {}, n = Na(e);
		this.maxResolution_ = n.maxResolution, this.minResolution_ = n.minResolution, this.zoomFactor_ = n.zoomFactor, this.resolutions_ = e.resolutions, this.padding_ = e.padding, this.minZoom_ = n.minZoom;
		var r = Ma(e), i = n.constraint, a = Pa(e);
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
			r.center && (r = S({}, r), r.center = kn(r.center, this.getProjection())), r.anchor && (r = S({}, r), r.anchor = kn(r.anchor, this.getProjection())), t[n] = r;
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
			n && ja(n, !0);
			return;
		}
		for (var a = Date.now(), o = this.targetCenter_.slice(), s = this.targetResolution_, c = this.targetRotation_, l = []; r < t; ++r) {
			var u = arguments[r], d = {
				start: a,
				complete: !1,
				anchor: u.anchor,
				duration: u.duration === void 0 ? 1e3 : u.duration,
				easing: u.easing || ji,
				callback: n
			};
			if (u.center && (d.sourceCenter = o, d.targetCenter = u.center.slice(), o = d.targetCenter), u.zoom === void 0 ? u.resolution && (d.sourceResolution = s, d.targetResolution = u.resolution, s = d.targetResolution) : (d.sourceResolution = s, d.targetResolution = this.getResolutionForZoom(u.zoom), s = d.targetResolution), u.rotation !== void 0) {
				d.sourceRotation = c;
				var f = Ye(u.rotation - c + Math.PI, 2 * Math.PI) - Math.PI;
				d.targetRotation = c + f, c = d.targetRotation;
			}
			Fa(d) ? d.complete = !0 : a += d.duration, l.push(d);
		}
		this.animations_.push(l), this.setHint(J.ANIMATING, 1), this.updateAnimations_();
	}, t.prototype.getAnimating = function() {
		return this.hints_[J.ANIMATING] > 0;
	}, t.prototype.getInteracting = function() {
		return this.hints_[J.INTERACTING] > 0;
	}, t.prototype.cancelAnimations = function() {
		this.setHint(J.ANIMATING, -this.hints_[J.ANIMATING]);
		for (var e, t = 0, n = this.animations_.length; t < n; ++t) {
			var r = this.animations_[t];
			if (r[0].callback && ja(r[0].callback, !1), !e) for (var i = 0, a = r.length; i < a; ++i) {
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
							var b = u === 1 ? Ye(s.targetRotation + Math.PI, 2 * Math.PI) - Math.PI : s.sourceRotation + u * (s.targetRotation - s.sourceRotation);
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
					this.animations_[n] = null, this.setHint(J.ANIMATING, -1), this.nextCenter_ = null, this.nextResolution_ = NaN, this.nextRotation_ = NaN;
					var S = r[0].callback;
					S && ja(S, !0);
				}
			}
			this.animations_ = this.animations_.filter(Boolean), t && this.updateAnimationKey_ === void 0 && (this.updateAnimationKey_ = requestAnimationFrame(this.updateAnimations_.bind(this)));
		}
	}, t.prototype.calculateCenterRotate = function(e, t) {
		var n, r = this.getCenterInternal();
		return r !== void 0 && (n = [r[0] - t[0], r[1] - t[1]], ln(n, e - this.getRotation()), on(n, t)), n;
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
		return e && On(e, this.getProjection());
	}, t.prototype.getCenterInternal = function() {
		return this.get(_i.CENTER);
	}, t.prototype.getConstraints = function() {
		return this.constraints_;
	}, t.prototype.getConstrainResolution = function() {
		return this.options_.constrainResolution;
	}, t.prototype.getHints = function(e) {
		return e === void 0 ? this.hints_.slice() : (e[0] = this.hints_[0], e[1] = this.hints_[1], e);
	}, t.prototype.calculateExtent = function(e) {
		return An(this.calculateExtentInternal(e), this.getProjection());
	}, t.prototype.calculateExtentInternal = function(e) {
		var t = e || this.getViewportSizeMinusPadding_(), n = this.getCenterInternal();
		V(n, 1);
		var r = this.getResolution();
		V(r !== void 0, 2);
		var i = this.getRotation();
		return V(i !== void 0, 3), qt(n, r, i, t);
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
		return this.get(_i.RESOLUTION);
	}, t.prototype.getResolutions = function() {
		return this.resolutions_;
	}, t.prototype.getResolutionForExtent = function(e, t) {
		return this.getResolutionForExtentInternal(jn(e, this.getProjection()), t);
	}, t.prototype.getResolutionForExtentInternal = function(e, t) {
		var n = t || this.getViewportSizeMinusPadding_(), r = H(e) / n[0], i = Jt(e) / n[1];
		return Math.max(r, i);
	}, t.prototype.getResolutionForValueFunction = function(e) {
		var t = e || 2, n = this.getConstrainedResolution(this.maxResolution_), r = this.minResolution_, i = Math.log(n / r) / Math.log(t);
		return (function(e) {
			return n / t ** +(e * i);
		});
	}, t.prototype.getRotation = function() {
		return this.get(_i.ROTATION);
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
			r = Ia(r, this.getViewportSize_(), [a[0] / 2 + i[3], a[1] / 2 + i[0]], t, n);
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
			var t = B(Math.floor(e), 0, this.resolutions_.length - 2), n = this.resolutions_[t] / this.resolutions_[t + 1];
			return this.resolutions_[t] / n ** +B(e - t, 0, 1);
		} else return this.maxResolution_ / this.zoomFactor_ ** +(e - this.minZoom_);
	}, t.prototype.fit = function(e, t) {
		var n;
		if (V(Array.isArray(e) || typeof e.getSimplifiedGeometry == "function", 24), Array.isArray(e)) {
			V(!$t(e), 25);
			var r = jn(e, this.getProjection());
			n = Da(r);
		} else if (e.getType() === U.CIRCLE) {
			var r = jn(e.getExtent(), this.getProjection());
			n = Da(r), n.rotate(this.getRotation(), Gt(r));
		} else {
			var i = Dn();
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
		var l = this.getRotation(), u = Math.sin(l), d = Math.cos(l), f = Gt(s);
		f[0] += (i[1] - i[3]) / 2 * c, f[1] += (i[0] - i[2]) / 2 * c;
		var p = f[0] * d - f[1] * u, m = f[1] * d + f[0] * u, h = this.getConstrainedCenter([p, m], c), g = n.callback ? n.callback : b;
		n.duration === void 0 ? (this.targetResolution_ = c, this.targetCenter_ = h, this.applyTargetState_(!1, !0), ja(g, !0)) : this.animateInternal({
			resolution: c,
			center: h,
			duration: n.duration,
			easing: n.easing
		}, g);
	}, t.prototype.centerOn = function(e, t, n) {
		this.centerOnInternal(kn(e, this.getProjection()), t, n);
	}, t.prototype.centerOnInternal = function(e, t, n) {
		this.setCenterInternal(Ia(e, t, n, this.getResolution(), this.getRotation()));
	}, t.prototype.calculateCenterShift = function(e, t, n, r) {
		var i, a = this.padding_;
		if (a && e) {
			var o = this.getViewportSizeMinusPadding_(-n), s = Ia(e, r, [o[0] / 2 + a[3], o[1] / 2 + a[0]], t, n);
			i = [e[0] - s[0], e[1] - s[1]];
		}
		return i;
	}, t.prototype.isDef = function() {
		return !!this.getCenterInternal() && this.getResolution() !== void 0;
	}, t.prototype.adjustCenter = function(e) {
		var t = On(this.targetCenter_, this.getProjection());
		this.setCenter([t[0] + e[0], t[1] + e[1]]);
	}, t.prototype.adjustCenterInternal = function(e) {
		var t = this.targetCenter_;
		this.setCenterInternal([t[0] + e[0], t[1] + e[1]]);
	}, t.prototype.adjustResolution = function(e, t) {
		var n = t && kn(t, this.getProjection());
		this.adjustResolutionInternal(e, n);
	}, t.prototype.adjustResolutionInternal = function(e, t) {
		var n = this.getAnimating() || this.getInteracting(), r = this.getViewportSize_(this.getRotation()), i = this.constraints_.resolution(this.targetResolution_ * e, 0, r, n);
		t && (this.targetCenter_ = this.calculateCenterZoom(i, t)), this.targetResolution_ *= e, this.applyTargetState_();
	}, t.prototype.adjustZoom = function(e, t) {
		this.adjustResolution(this.zoomFactor_ ** +-e, t);
	}, t.prototype.adjustRotation = function(e, t) {
		t &&= kn(t, this.getProjection()), this.adjustRotationInternal(e, t);
	}, t.prototype.adjustRotationInternal = function(e, t) {
		var n = this.getAnimating() || this.getInteracting(), r = this.constraints_.rotation(this.targetRotation_ + e, n);
		t && (this.targetCenter_ = this.calculateCenterRotate(r, t)), this.targetRotation_ += e, this.applyTargetState_();
	}, t.prototype.setCenter = function(e) {
		this.setCenterInternal(kn(e, this.getProjection()));
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
		this.get(_i.ROTATION) !== r && this.set(_i.ROTATION, r), this.get(_i.RESOLUTION) !== a && this.set(_i.RESOLUTION, a), (!this.get(_i.CENTER) || !cn(this.get(_i.CENTER), o)) && this.set(_i.CENTER, o), this.getAnimating() && !e && this.cancelAnimations(), this.cancelAnchor_ = void 0;
	}, t.prototype.resolveConstraints = function(e, t, n) {
		var r = e === void 0 ? 200 : e, i = t || 0, a = this.constraints_.rotation(this.targetRotation_), o = this.getViewportSize_(a), s = this.constraints_.resolution(this.targetResolution_, i, o), c = this.constraints_.center(this.targetCenter_, s, o, !1, this.calculateCenterShift(this.targetCenter_, s, a, o));
		if (r === 0 && !this.cancelAnchor_) {
			this.targetResolution_ = s, this.targetRotation_ = a, this.targetCenter_ = c, this.applyTargetState_();
			return;
		}
		var l = n || (r === 0 ? this.cancelAnchor_ : void 0);
		this.cancelAnchor_ = void 0, (this.getResolution() !== s || this.getRotation() !== a || !this.getCenterInternal() || !cn(this.getCenterInternal(), c)) && (this.getAnimating() && this.cancelAnimations(), this.animateInternal({
			rotation: a,
			center: c,
			resolution: s,
			duration: r,
			easing: Ai,
			anchor: l
		}));
	}, t.prototype.beginInteraction = function() {
		this.resolveConstraints(0), this.setHint(J.INTERACTING, 1);
	}, t.prototype.endInteraction = function(e, t, n) {
		var r = n && kn(n, this.getProjection());
		this.endInteractionInternal(e, t, r);
	}, t.prototype.endInteractionInternal = function(e, t, n) {
		this.setHint(J.INTERACTING, -1), this.resolveConstraints(e, t, n);
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
}(R);
function ja(e, t) {
	setTimeout(function() {
		e(t);
	}, 0);
}
function Ma(e) {
	if (e.extent !== void 0) {
		var t = e.smoothExtentConstraint === void 0 || e.smoothExtentConstraint;
		return vi(e.extent, e.constrainOnlyCenter, t);
	}
	var n = xn(e.projection, "EPSG:3857");
	if (e.multiWorld !== !0 && n.isGlobal()) {
		var r = n.getExtent().slice();
		return r[0] = -Infinity, r[2] = Infinity, vi(r, !1, !1);
	}
	return yi;
}
function Na(e) {
	var t, n, r, i = 28, a = 2, o = e.minZoom === void 0 ? ka : e.minZoom, s = e.maxZoom === void 0 ? i : e.maxZoom, c = e.zoomFactor === void 0 ? a : e.zoomFactor, l = e.multiWorld !== void 0 && e.multiWorld, u = e.smoothResolutionConstraint === void 0 || e.smoothResolutionConstraint, d = e.showFullExtent !== void 0 && e.showFullExtent, f = xn(e.projection, "EPSG:3857"), p = f.getExtent(), m = e.constrainOnlyCenter, h = e.extent;
	if (!l && !h && f.isGlobal() && (m = !1, h = p), e.resolutions !== void 0) {
		var g = e.resolutions;
		n = g[o], r = g[s] === void 0 ? g[g.length - 1] : g[s], t = e.constrainResolution ? Si(g, u, !m && h, d) : wi(n, r, u, !m && h, d);
	} else {
		var _ = (p ? Math.max(H(p), Jt(p)) : 360 * Ve[z.DEGREES] / f.getMetersPerUnit()) / 256 / a ** +ka, v = _ / a ** +(i - ka);
		n = e.maxResolution, n === void 0 ? n = _ / c ** +o : o = 0, r = e.minResolution, r === void 0 && (r = e.maxZoom === void 0 ? v : e.maxResolution === void 0 ? _ / c ** +s : n / c ** +s), s = o + Math.floor(Math.log(n / r) / Math.log(c)), r = n / c ** +(s - o), t = e.constrainResolution ? Ci(c, n, r, u, !m && h, d) : wi(n, r, u, !m && h, d);
	}
	return {
		constraint: t,
		maxResolution: n,
		minResolution: r,
		minZoom: o,
		zoomFactor: c
	};
}
function Pa(e) {
	if (e.enableRotation === void 0 || e.enableRotation) {
		var t = e.constrainRotation;
		return t === void 0 || t === !0 ? Oi() : t === !1 ? Ei : typeof t == "number" ? Di(t) : Ei;
	} else return Ti;
}
function Fa(e) {
	return !(e.sourceCenter && e.targetCenter && !cn(e.sourceCenter, e.targetCenter) || e.sourceResolution !== e.targetResolution || e.sourceRotation !== e.targetRotation);
}
function Ia(e, t, n, r, i) {
	var a = Math.cos(-i), o = Math.sin(-i), s = e[0] * a - e[1] * o, c = e[1] * a + e[0] * o;
	return s += (t[0] / 2 - n[0]) * r, c += (n[1] - t[1] / 2) * r, o = -o, [s * a - c * o, c * a + s * o];
}
//#endregion
//#region node_modules/ol/size.js
function La(e) {
	return e[0] > 0 && e[1] > 0;
}
function Ra(e, t, n) {
	return n === void 0 && (n = [0, 0]), n[0] = e[0] * t + .5 | 0, n[1] = e[1] * t + .5 | 0, n;
}
function za(e, t) {
	return Array.isArray(e) ? e : (t === void 0 ? t = [e, e] : (t[0] = e, t[1] = e), t);
}
//#endregion
//#region node_modules/ol/PluggableMap.js
var Ba = (function() {
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
})(), Va = function(e) {
	Ba(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = Ha(t);
		n.boundHandleBrowserEvent_ = n.handleBrowserEvent.bind(n), n.maxTilesLoading_ = t.maxTilesLoading === void 0 ? 16 : t.maxTilesLoading, n.pixelRatio_ = t.pixelRatio === void 0 ? ce : t.pixelRatio, n.postRenderTimeoutHandle_, n.animationDelayKey_, n.animationDelay_ = function() {
			this.animationDelayKey_ = void 0, this.renderFrame_(Date.now());
		}.bind(n), n.coordinateToPixelTransform_ = zn(), n.pixelToCoordinateTransform_ = zn(), n.frameIndex_ = 0, n.frameState_ = null, n.previousExtent_ = null, n.viewPropertyListenerKey_ = null, n.viewChangeListenerKey_ = null, n.layerGroupPropertyListenerKeys_ = null, n.viewport_ = document.createElement("div"), n.viewport_.className = "ol-viewport" + ("ontouchstart" in window ? " ol-touch" : ""), n.viewport_.style.position = "relative", n.viewport_.style.overflow = "hidden", n.viewport_.style.width = "100%", n.viewport_.style.height = "100%", n.overlayContainer_ = document.createElement("div"), n.overlayContainer_.style.position = "absolute", n.overlayContainer_.style.zIndex = "0", n.overlayContainer_.style.width = "100%", n.overlayContainer_.style.height = "100%", n.overlayContainer_.style.pointerEvents = "none", n.overlayContainer_.className = "ol-overlaycontainer", n.viewport_.appendChild(n.overlayContainer_), n.overlayContainerStopEvent_ = document.createElement("div"), n.overlayContainerStopEvent_.style.position = "absolute", n.overlayContainerStopEvent_.style.zIndex = "0", n.overlayContainerStopEvent_.style.width = "100%", n.overlayContainerStopEvent_.style.height = "100%", n.overlayContainerStopEvent_.style.pointerEvents = "none", n.overlayContainerStopEvent_.className = "ol-overlaycontainer-stopevent", n.viewport_.appendChild(n.overlayContainerStopEvent_), n.mapBrowserEventHandler_ = null, n.moveTolerance_ = t.moveTolerance, n.keyboardEventTarget_ = r.keyboardEventTarget, n.keyHandlerKeys_ = null, n.controls = r.controls || new ni(), n.interactions = r.interactions || new ni(), n.overlays_ = r.overlays, n.overlayIdIndex_ = {}, n.renderer_ = null, n.handleResize_, n.postRenderFunctions_ = [], n.tileQueue_ = new hi(n.getTilePriority.bind(n), n.handleTileChange_.bind(n)), n.addChangeListener(qr.LAYERGROUP, n.handleLayerGroupChanged_), n.addChangeListener(qr.VIEW, n.handleViewChanged_), n.addChangeListener(qr.SIZE, n.handleSizeChanged_), n.addChangeListener(qr.TARGET, n.handleTargetChanged_), n.setProperties(r.values);
		var i = n;
		return t.view && !(t.view instanceof Aa) && t.view.then(function(e) {
			i.setView(new Aa(e));
		}), n.controls.addEventListener(Qr.ADD, function(e) {
			e.element.setMap(this);
		}.bind(n)), n.controls.addEventListener(Qr.REMOVE, function(e) {
			e.element.setMap(null);
		}.bind(n)), n.interactions.addEventListener(Qr.ADD, function(e) {
			e.element.setMap(this);
		}.bind(n)), n.interactions.addEventListener(Qr.REMOVE, function(e) {
			e.element.setMap(null);
		}.bind(n)), n.overlays_.addEventListener(Qr.ADD, function(e) {
			this.addOverlayInternal_(e.element);
		}.bind(n)), n.overlays_.addEventListener(Qr.REMOVE, function(e) {
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
		return this.get(qr.TARGET);
	}, t.prototype.getTargetElement = function() {
		var e = this.getTarget();
		return e === void 0 ? null : typeof e == "string" ? document.getElementById(e) : e;
	}, t.prototype.getCoordinateFromPixel = function(e) {
		return On(this.getCoordinateFromPixelInternal(e), this.getView().getProjection());
	}, t.prototype.getCoordinateFromPixelInternal = function(e) {
		var t = this.frameState_;
		return t ? W(t.pixelToCoordinateTransform, e.slice()) : null;
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
		return this.get(qr.LAYERGROUP);
	}, t.prototype.setLayers = function(e) {
		var t = this.getLayerGroup();
		if (e instanceof ni) {
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
		var t = kn(e, this.getView().getProjection());
		return this.getPixelFromCoordinateInternal(t);
	}, t.prototype.getPixelFromCoordinateInternal = function(e) {
		var t = this.frameState_;
		return t ? W(t.coordinateToPixelTransform, e.slice(0, 2)) : null;
	}, t.prototype.getRenderer = function() {
		return this.renderer_;
	}, t.prototype.getSize = function() {
		return this.get(qr.SIZE);
	}, t.prototype.getView = function() {
		return this.get(qr.VIEW);
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
		return gi(this.frameState_, e, t, n, r);
	}, t.prototype.handleBrowserEvent = function(e, t) {
		var n = new li(t || e.type, this, e);
		this.handleMapBrowserEvent(n);
	}, t.prototype.handleMapBrowserEvent = function(e) {
		if (this.frameState_) {
			var t = e.originalEvent, n = t.type;
			if (n === Be.POINTERDOWN || n === O.WHEEL || n === O.KEYDOWN) {
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
				if (i[J.ANIMATING] || i[J.INTERACTING]) {
					var a = Date.now() - e.time > 8;
					n = a ? 0 : 8, r = a ? 0 : 2;
				}
			}
			t.getTilesLoading() < n && (t.reprioritize(), t.loadMoreTiles(n, r));
		}
		e && this.hasListener(pr.RENDERCOMPLETE) && !e.animate && !this.tileQueue_.getTilesLoading() && !this.getLoading() && this.renderer_.dispatchRenderEvent(pr.RENDERCOMPLETE, e);
		for (var o = this.postRenderFunctions_, s = 0, c = o.length; s < c; ++s) o[s](this, e);
		o.length = 0;
	}, t.prototype.handleSizeChanged_ = function() {
		this.getView() && !this.getView().getAnimating() && this.getView().resolveConstraints(0), this.render();
	}, t.prototype.handleTargetChanged_ = function() {
		var e;
		if (this.getTarget() && (e = this.getTargetElement()), this.mapBrowserEventHandler_) {
			for (var t = 0, n = this.keyHandlerKeys_.length; t < n; ++t) j(this.keyHandlerKeys_[t]);
			this.keyHandlerKeys_ = null, this.viewport_.removeEventListener(O.CONTEXTMENU, this.boundHandleBrowserEvent_), this.viewport_.removeEventListener(O.WHEEL, this.boundHandleBrowserEvent_), this.handleResize_ !== void 0 && (removeEventListener(O.RESIZE, this.handleResize_, !1), this.handleResize_ = void 0), this.mapBrowserEventHandler_.dispose(), this.mapBrowserEventHandler_ = null, ge(this.viewport_);
		}
		if (!e) this.renderer_ &&= (clearTimeout(this.postRenderTimeoutHandle_), this.postRenderTimeoutHandle_ = void 0, this.postRenderFunctions_.length = 0, this.renderer_.dispose(), null), this.animationDelayKey_ &&= (cancelAnimationFrame(this.animationDelayKey_), void 0);
		else {
			for (var r in e.appendChild(this.viewport_), this.renderer_ ||= this.createRenderer(), this.mapBrowserEventHandler_ = new di(this, this.moveTolerance_), K) this.mapBrowserEventHandler_.addEventListener(K[r], this.handleMapBrowserEvent.bind(this));
			this.viewport_.addEventListener(O.CONTEXTMENU, this.boundHandleBrowserEvent_, !1), this.viewport_.addEventListener(O.WHEEL, this.boundHandleBrowserEvent_, de ? { passive: !1 } : !1);
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
		if (n !== void 0 && La(n) && r && r.isDef()) {
			var o = r.getHints(this.frameState_ ? this.frameState_.viewHints : void 0), s = r.getState();
			if (a = {
				animate: !1,
				coordinateToPixelTransform: this.coordinateToPixelTransform_,
				declutterTree: null,
				extent: qt(s.center, s.resolution, s.rotation, n),
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
				a.nextExtent = qt(s.nextCenter, s.nextResolution, c, n);
			}
		}
		this.frameState_ = a, this.renderer_.renderFrame(a), a && (a.animate && this.render(), Array.prototype.push.apply(this.postRenderFunctions_, a.postRenderFunctions), i && (!this.previousExtent_ || !$t(this.previousExtent_) && !It(a.extent, this.previousExtent_)) && (this.dispatchEvent(new si(re.MOVESTART, this, i)), this.previousExtent_ = Nt(this.previousExtent_)), this.previousExtent_ && !a.viewHints[J.ANIMATING] && !a.viewHints[J.INTERACTING] && !It(a.extent, this.previousExtent_) && (this.dispatchEvent(new si(re.MOVEEND, this, a)), Tt(a.extent, this.previousExtent_))), this.dispatchEvent(new si(re.POSTRENDER, this, a)), this.postRenderTimeoutHandle_ ||= setTimeout(function() {
			t.postRenderTimeoutHandle_ = void 0, t.handlePostRender();
		}, 0);
	}, t.prototype.setLayerGroup = function(e) {
		this.set(qr.LAYERGROUP, e);
	}, t.prototype.setSize = function(e) {
		this.set(qr.SIZE, e);
	}, t.prototype.setTarget = function(e) {
		this.set(qr.TARGET, e);
	}, t.prototype.setView = function(e) {
		if (!e || e instanceof Aa) {
			this.set(qr.VIEW, e);
			return;
		}
		this.set(qr.VIEW, new Aa());
		var t = this;
		e.then(function(e) {
			t.setView(new Aa(e));
		});
	}, t.prototype.updateSize = function() {
		var e = this.getTargetElement(), t = void 0;
		if (e) {
			var n = getComputedStyle(e), r = e.offsetWidth - parseFloat(n.borderLeftWidth) - parseFloat(n.paddingLeft) - parseFloat(n.paddingRight) - parseFloat(n.borderRightWidth), i = e.offsetHeight - parseFloat(n.borderTopWidth) - parseFloat(n.paddingTop) - parseFloat(n.paddingBottom) - parseFloat(n.borderBottomWidth);
			!isNaN(r) && !isNaN(i) && (t = [r, i], !La(t) && (e.offsetWidth || e.offsetHeight || e.getClientRects().length) && console.warn("No map visible because the map container's width or height are 0."));
		}
		this.setSize(t), this.updateViewportSize_();
	}, t.prototype.updateViewportSize_ = function() {
		var e = this.getView();
		if (e) {
			var t = void 0, n = getComputedStyle(this.viewport_);
			n.width && n.height && (t = [parseInt(n.width, 10), parseInt(n.height, 10)]), e.setViewportSize(t);
		}
	}, t;
}(R);
function Ha(e) {
	var t = null;
	e.keyboardEventTarget !== void 0 && (t = typeof e.keyboardEventTarget == "string" ? document.getElementById(e.keyboardEventTarget) : e.keyboardEventTarget);
	var n = {}, r = e.layers && typeof e.layers.getLayers == "function" ? e.layers : new ai({ layers: e.layers });
	n[qr.LAYERGROUP] = r, n[qr.TARGET] = e.target, n[qr.VIEW] = e.view instanceof Aa ? e.view : new Aa();
	var i;
	e.controls !== void 0 && (Array.isArray(e.controls) ? i = new ni(e.controls.slice()) : (V(typeof e.controls.getArray == "function", 47), i = e.controls));
	var a;
	e.interactions !== void 0 && (Array.isArray(e.interactions) ? a = new ni(e.interactions.slice()) : (V(typeof e.interactions.getArray == "function", 48), a = e.interactions));
	var o;
	return e.overlays === void 0 ? o = new ni() : Array.isArray(e.overlays) ? o = new ni(e.overlays.slice()) : (V(typeof e.overlays.getArray == "function", 49), o = e.overlays), {
		controls: i,
		interactions: a,
		keyboardEventTarget: t,
		overlays: o,
		values: n
	};
}
//#endregion
//#region node_modules/ol/control/OverviewMap.js
var Ua = (function() {
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
})(), Wa = .75, Ga = .1, Ka = function(e) {
	Ua(t, e);
	function t() {
		return e !== null && e.apply(this, arguments) || this;
	}
	return t.prototype.createRenderer = function() {
		return new Kr(this);
	}, t;
}(Va), qa = function(e) {
	Ua(t, e);
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
		l.setAttribute("type", "button"), l.title = a, l.appendChild(c), l.addEventListener(O.CLICK, n.handleClick_.bind(n), !1), n.ovmapDiv_ = document.createElement("div"), n.ovmapDiv_.className = "ol-overviewmap-map", n.view_ = r.view, n.ovmap_ = new Ka({ view: r.view });
		var u = n.ovmap_;
		r.layers && r.layers.forEach(function(e) {
			u.addLayer(e);
		});
		var d = document.createElement("div");
		d.className = "ol-overviewmap-box", d.style.boxSizing = "border-box", n.boxOverlay_ = new Zr({
			position: [0, 0],
			positioning: Jr.CENTER_CENTER,
			element: d
		}), n.ovmap_.addOverlay(n.boxOverlay_);
		var f = i + " " + Ce + " " + Te + (n.collapsed_ && n.collapsible_ ? " " + Ee : "") + (n.collapsible_ ? "" : " ol-uncollapsible"), p = n.element;
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
		if (e.key === qr.VIEW) {
			var t = e.oldValue;
			t && this.unbindView_(t);
			var n = this.getMap().getView();
			this.bindView_(n);
		} else !this.ovmap_.isRendered() && (e.key === qr.TARGET || e.key === qr.SIZE) && this.ovmap_.updateSize();
	}, t.prototype.bindView_ = function(e) {
		if (!this.view_) {
			var t = new Aa({ projection: e.getProjection() });
			this.ovmap_.setView(t);
		}
		e.addChangeListener(_i.ROTATION, this.boundHandleRotationChanged_), this.handleRotationChanged_();
	}, t.prototype.unbindView_ = function(e) {
		e.removeChangeListener(_i.ROTATION, this.boundHandleRotationChanged_);
	}, t.prototype.handleRotationChanged_ = function() {
		this.rotateWithView_ && this.ovmap_.getView().setRotation(this.getMap().getView().getRotation());
	}, t.prototype.validateExtent_ = function() {
		var e = this.getMap(), t = this.ovmap_;
		if (!(!e.isRendered() || !t.isRendered())) {
			var n = e.getSize(), r = e.getView().calculateExtentInternal(n);
			if (!(this.viewExtent_ && It(r, this.viewExtent_))) {
				this.viewExtent_ = r;
				var i = t.getSize(), a = t.getView().calculateExtentInternal(i), o = t.getPixelFromCoordinateInternal(Xt(r)), s = t.getPixelFromCoordinateInternal(Wt(r)), c = Math.abs(o[0] - s[0]), l = Math.abs(o[1] - s[1]), u = i[0], d = i[1];
				c < u * Ga || l < d * Ga || c > u * Wa || l > d * Wa ? this.resetExtent_() : Ot(a, r) || this.recenter_();
			}
		}
	}, t.prototype.resetExtent_ = function() {
		if (!(Wa === 0 || Ga === 0)) {
			var e = this.getMap(), t = this.ovmap_, n = e.getSize(), r = e.getView().calculateExtentInternal(n), i = t.getView();
			tn(r, 1 / (2 ** (Math.log(Wa / Ga) / Math.LN2 / 2) * Ga)), i.fitInternal(Da(r));
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
		this.ovmapPostrenderKey_ ||= A(this.ovmap_, re.POSTRENDER, function(e) {
			delete this.ovmapPostrenderKey_, this.updateBox_();
		}, this);
	}, t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.handleToggle_();
	}, t.prototype.handleToggle_ = function() {
		this.element.classList.toggle(Ee), this.collapsed_ ? he(this.collapseLabel_, this.label_) : he(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_;
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
}(be), Ja = (function() {
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
})(), Ya = function(e) {
	Ja(t, e);
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
		var l = i + " " + Ce + " " + Te, u = n.element;
		return u.className = l, u.appendChild(c), n.callResetNorth_ = r.resetNorth ? r.resetNorth : void 0, n.duration_ = r.duration === void 0 ? 250 : r.duration, n.autoHide_ = r.autoHide === void 0 || r.autoHide, n.rotation_ = void 0, n.autoHide_ && n.element.classList.add(xe), n;
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
				easing: Ai
			}) : e.setRotation(0));
		}
	}, t.prototype.render = function(e) {
		var t = e.frameState;
		if (t) {
			var n = t.viewState.rotation;
			if (n != this.rotation_) {
				var r = "rotate(" + n + "rad)";
				if (this.autoHide_) {
					var i = this.element.classList.contains(xe);
					!i && n === 0 ? this.element.classList.add(xe) : i && n !== 0 && this.element.classList.remove(xe);
				}
				this.label_.style.transform = r;
			}
			this.rotation_ = n;
		}
	}, t;
}(be), Xa = (function() {
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
})(), Za = "units", Qa = {
	DEGREES: "degrees",
	IMPERIAL: "imperial",
	NAUTICAL: "nautical",
	METRIC: "metric",
	US: "us"
}, $a = [
	1,
	2,
	5
], eo = 25.4 / .28, to = function(e) {
	Xa(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r.className === void 0 ? r.bar ? "ol-scale-bar" : "ol-scale-line" : r.className;
		return n = e.call(this, {
			element: document.createElement("div"),
			render: r.render,
			target: r.target
		}) || this, n.on, n.once, n.un, n.innerElement_ = document.createElement("div"), n.innerElement_.className = i + "-inner", n.element.className = i + " " + Ce, n.element.appendChild(n.innerElement_), n.viewState_ = null, n.minWidth_ = r.minWidth === void 0 ? 64 : r.minWidth, n.renderedVisible_ = !1, n.renderedWidth_ = void 0, n.renderedHTML_ = "", n.addChangeListener(Za, n.handleUnitsChanged_), n.setUnits(r.units || Qa.METRIC), n.scaleBar_ = r.bar || !1, n.scaleBarSteps_ = r.steps || 4, n.scaleBarText_ = r.text || !1, n.dpi_ = r.dpi || void 0, n;
	}
	return t.prototype.getUnits = function() {
		return this.get(Za);
	}, t.prototype.handleUnitsChanged_ = function() {
		this.updateElement_();
	}, t.prototype.setUnits = function(e) {
		this.set(Za, e);
	}, t.prototype.setDpi = function(e) {
		this.dpi_ = e;
	}, t.prototype.updateElement_ = function() {
		var e = this.viewState_;
		if (!e) {
			this.renderedVisible_ &&= (this.element.style.display = "none", !1);
			return;
		}
		var t = e.center, n = e.projection, r = this.getUnits(), i = r == Qa.DEGREES ? z.DEGREES : z.METERS, a = vn(n, e.resolution, t, i), o = this.minWidth_ * (this.dpi_ || eo) / eo, s = o * a, c = "";
		if (r == Qa.DEGREES) {
			var l = Ve[z.DEGREES];
			s *= l, s < l / 60 ? (c = "″", a *= 3600) : s < l ? (c = "′", a *= 60) : c = "°";
		} else r == Qa.IMPERIAL ? s < .9144 ? (c = "in", a /= .0254) : s < 1609.344 ? (c = "ft", a /= .3048) : (c = "mi", a /= 1609.344) : r == Qa.NAUTICAL ? (a /= 1852, c = "nm") : r == Qa.METRIC ? s < .001 ? (c = "μm", a *= 1e6) : s < 1 ? (c = "mm", a *= 1e3) : s < 1e3 ? c = "m" : (c = "km", a /= 1e3) : r == Qa.US ? s < .9144 ? (c = "in", a *= 39.37) : s < 1609.344 ? (c = "ft", a /= .30480061) : (c = "mi", a /= 1609.3472) : V(!1, 33);
		for (var u = 3 * Math.floor(Math.log(o * a) / Math.log(10)), d, f, p;;) {
			p = Math.floor(u / 3);
			var m = 10 ** p;
			if (d = $a[(u % 3 + 3) % 3] * m, f = Math.round(d / a), isNaN(f)) {
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
		var e = vn(this.viewState_.projection, this.viewState_.resolution, this.viewState_.center), t = this.dpi_ || eo, n = this.viewState_.projection.getMetersPerUnit();
		return parseFloat(e.toString()) * n * (1e3 / 25.4) * t;
	}, t.prototype.render = function(e) {
		var t = e.frameState;
		t ? this.viewState_ = t.viewState : this.viewState_ = null, this.updateElement_();
	}, t;
}(be), no = {
	PRELOAD: "preload",
	USE_INTERIM_TILES_ON_ERROR: "useInterimTilesOnError"
}, ro = (function() {
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
})(), io = function(e) {
	ro(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		return delete i.preload, delete i.useInterimTilesOnError, n = e.call(this, i) || this, n.on, n.once, n.un, n.setPreload(r.preload === void 0 ? 0 : r.preload), n.setUseInterimTilesOnError(r.useInterimTilesOnError === void 0 || r.useInterimTilesOnError), n;
	}
	return t.prototype.getPreload = function() {
		return this.get(no.PRELOAD);
	}, t.prototype.setPreload = function(e) {
		this.set(no.PRELOAD, e);
	}, t.prototype.getUseInterimTilesOnError = function() {
		return this.get(no.USE_INTERIM_TILES_ON_ERROR);
	}, t.prototype.setUseInterimTilesOnError = function(e) {
		this.set(no.USE_INTERIM_TILES_ON_ERROR, e);
	}, t;
}(gr), Y = {
	IDLE: 0,
	LOADING: 1,
	LOADED: 2,
	ERROR: 3,
	EMPTY: 4
}, ao = (function() {
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
})(), oo = function(e) {
	ao(t, e);
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
		e.target.getState() === Y.LOADED && this.renderIfReadyAndVisible();
	}, t.prototype.loadImage = function(e) {
		var t = e.getState();
		return t != Y.LOADED && t != Y.ERROR && e.addEventListener(O.CHANGE, this.boundHandleImageChange_), t == Y.IDLE && (e.load(), t = e.getState()), t == Y.LOADED;
	}, t.prototype.renderIfReadyAndVisible = function() {
		var e = this.getLayer();
		e.getVisible() && e.getSourceState() == mr.READY && e.changed();
	}, t;
}(N), so = (function() {
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
})(), co = function(e) {
	so(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.container = null, n.renderedResolution, n.tempTransform = zn(), n.pixelTransform = zn(), n.inversePixelTransform = zn(), n.context = null, n.containerReused = !1, n;
	}
	return t.prototype.useContainer = function(e, t, n) {
		var r = this.getLayer().getClassName(), i, a;
		if (e && e.style.opacity === Ae(n) && e.className === r) {
			var o = e.firstElementChild;
			o instanceof HTMLCanvasElement && (a = o.getContext("2d"));
		}
		if (a && a.canvas.style.transform === t ? (this.container = e, this.context = a, this.containerReused = !0) : this.containerReused &&= (this.container = null, this.context = null, !1), !this.container) {
			i = document.createElement("div"), i.className = r;
			var s = i.style;
			s.position = "absolute", s.width = "100%", s.height = "100%", a = fe();
			var o = a.canvas;
			i.appendChild(o), s = o.style, s.position = "absolute", s.left = "0", s.transformOrigin = "top left", this.container = i, this.context = a;
		}
	}, t.prototype.clipUnrotated = function(e, t, n) {
		var r = Xt(n), i = Zt(n), a = Wt(n), o = Ut(n);
		W(t.coordinateToPixelTransform, r), W(t.coordinateToPixelTransform, i), W(t.coordinateToPixelTransform, a), W(t.coordinateToPixelTransform, o);
		var s = this.inversePixelTransform;
		W(s, r), W(s, i), W(s, a), W(s, o), e.save(), e.beginPath(), e.moveTo(Math.round(r[0]), Math.round(r[1])), e.lineTo(Math.round(i[0]), Math.round(i[1])), e.lineTo(Math.round(a[0]), Math.round(a[1])), e.lineTo(Math.round(o[0]), Math.round(o[1])), e.clip();
	}, t.prototype.dispatchRenderEvent_ = function(e, t, n) {
		var r = this.getLayer();
		if (r.hasListener(e)) {
			var i = new Sr(e, this.inversePixelTransform, n, t);
			r.dispatchEvent(i);
		}
	}, t.prototype.preRender = function(e, t) {
		this.dispatchRenderEvent_(pr.PRERENDER, e, t);
	}, t.prototype.postRender = function(e, t) {
		this.dispatchRenderEvent_(pr.POSTRENDER, e, t);
	}, t.prototype.getRenderTransform = function(e, t, n, r, i, a, o) {
		var s = i / 2, c = a / 2, l = r / t, u = -l, d = -e[0] + o, f = -e[1];
		return Jn(this.tempTransform, s, c, l, u, -n, d, f);
	}, t.prototype.getDataAtPixel = function(e, t, n) {
		var r = W(this.inversePixelTransform, e.slice()), i = this.context, a = this.getLayer().getExtent();
		if (a && !Dt(a, W(t.pixelToCoordinateTransform, e.slice()))) return null;
		var o;
		try {
			var s = Math.round(r[0]), c = Math.round(r[1]), l = document.createElement("canvas"), u = l.getContext("2d");
			l.width = 1, l.height = 1, u.clearRect(0, 0, 1, 1), u.drawImage(i.canvas, s, c, 1, 1, 0, 0, 1, 1), o = u.getImageData(0, 0, 1, 1).data;
		} catch (e) {
			return e.name === "SecurityError" ? /* @__PURE__ */ new Uint8Array() : o;
		}
		return o[3] === 0 ? null : o;
	}, t;
}(oo), lo = function() {
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
function uo(e, t, n, r, i) {
	return i === void 0 ? new lo(e, t, n, r) : (i.minX = e, i.maxX = t, i.minY = n, i.maxY = r, i);
}
//#endregion
//#region node_modules/ol/renderer/canvas/TileLayer.js
var fo = (function() {
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
})(), po = function(e) {
	fo(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.extentChanged = !0, n.renderedExtent_ = null, n.renderedPixelRatio, n.renderedProjection = null, n.renderedRevision, n.renderedTiles = [], n.newTiles_ = !1, n.tmpExtent = jt(), n.tmpTileRange_ = new lo(0, 0, 0, 0), n;
	}
	return t.prototype.isDrawableTile = function(e) {
		var t = this.getLayer(), n = e.getState(), r = t.getUseInterimTilesOnError();
		return n == q.LOADED || n == q.EMPTY || n == q.ERROR && !r;
	}, t.prototype.getTile = function(e, t, n, r) {
		var i = r.pixelRatio, a = r.viewState.projection, o = this.getLayer(), s = o.getSource().getTile(e, t, n, i, a);
		return s.getState() == q.ERROR && (o.getUseInterimTilesOnError() ? o.getPreload() > 0 && (this.newTiles_ = !0) : s.setState(q.LOADED)), this.isDrawableTile(s) || (s = s.getInterimTile()), s;
	}, t.prototype.loadedTileCallback = function(t, n, r) {
		return this.isDrawableTile(r) ? e.prototype.loadedTileCallback.call(this, t, n, r) : !1;
	}, t.prototype.prepareFrame = function(e) {
		return !!this.getLayer().getSource();
	}, t.prototype.renderFrame = function(e, t) {
		var n = e.layerStatesArray[e.layerIndex], r = e.viewState, i = r.projection, a = r.resolution, o = r.center, s = r.rotation, c = e.pixelRatio, l = this.getLayer(), u = l.getSource(), d = u.getRevision(), p = u.getTileGridForProjection(i), m = p.getZForResolution(a, u.zDirection), h = p.getResolution(m), g = e.extent, _ = n.extent && jn(n.extent, i);
		_ && (g = Yt(g, jn(n.extent, i)));
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
				var P = I(this);
				if (N.getState() == q.LOADED) {
					D[m][N.tileCoord.toString()] = N;
					var F = N.inTransition(P);
					!this.newTiles_ && (F || this.renderedTiles.indexOf(N) === -1) && (this.newTiles_ = !0);
				}
				if (N.getAlpha(P, e.time) === 1) continue;
			}
			var ee = p.getTileCoordChildTileRange(N.tileCoord, A, k), L = !1;
			ee && (L = O(m + 1, ee)), L || p.forEachTileCoordParentTileRange(N.tileCoord, O, A, k);
		}
		var te = h / a;
		Jn(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / v, 1 / v, s, -y / 2, -b / 2);
		var ne = Qn(this.pixelTransform);
		this.useContainer(t, ne, n.opacity);
		var R = this.context, re = R.canvas;
		Yn(this.inversePixelTransform, this.pixelTransform), Jn(this.tempTransform, y / 2, b / 2, te, te, 0, -y / 2, -b / 2), re.width != y || re.height != b ? (re.width = y, re.height = b) : this.containerReused || R.clearRect(0, 0, y, b), _ && this.clipUnrotated(R, e, _), S(R, u.getContextOptions()), this.preRender(R, e), this.renderedTiles.length = 0;
		var ie = Object.keys(D).map(Number);
		ie.sort(f);
		var ae, oe, se;
		n.opacity === 1 && (!this.containerReused || u.getOpaque(e.viewState.projection)) ? ie = ie.reverse() : (ae = [], oe = []);
		for (var ce = ie.length - 1; ce >= 0; --ce) {
			var le = ie[ce], ue = u.getTilePixelSize(le, c, i), de = p.getResolution(le) / h, fe = ue[0] * de * te, pe = ue[1] * de * te, me = p.getTileCoordForCoordAndZ(Xt(T), le), he = p.getTileCoordExtent(me), ge = W(this.tempTransform, [v * (he[0] - T[0]) / h, v * (T[3] - he[3]) / h]), _e = v * u.getGutterForProjection(i), ve = D[le];
			for (var ye in ve) {
				var N = ve[ye], be = N.tileCoord, xe = me[1] - be[1], Se = Math.round(ge[0] - (xe - 1) * fe), Ce = me[2] - be[2], we = Math.round(ge[1] - (Ce - 1) * pe), j = Math.round(ge[0] - xe * fe), M = Math.round(ge[1] - Ce * pe), Te = Se - j, Ee = we - M, De = m === le, F = De && N.getAlpha(I(this), e.time) !== 1;
				if (!F) if (ae) {
					R.save(), se = [
						j,
						M,
						j + Te,
						M,
						j + Te,
						M + Ee,
						j,
						M + Ee
					];
					for (var Oe = 0, ke = ae.length; Oe < ke; ++Oe) if (m !== le && le < oe[Oe]) {
						var je = ae[Oe];
						R.beginPath(), R.moveTo(se[0], se[1]), R.lineTo(se[2], se[3]), R.lineTo(se[4], se[5]), R.lineTo(se[6], se[7]), R.moveTo(je[6], je[7]), R.lineTo(je[4], je[5]), R.lineTo(je[2], je[3]), R.lineTo(je[0], je[1]), R.clip();
					}
					ae.push(se), oe.push(le);
				} else R.clearRect(j, M, Te, Ee);
				this.drawTileImage(N, e, j, M, Te, Ee, _e, De), ae && !F ? (R.restore(), this.renderedTiles.unshift(N)) : this.renderedTiles.push(N), this.updateUsedTiles(e.usedTiles, u, N);
			}
		}
		this.renderedRevision = d, this.renderedResolution = h, this.extentChanged = !this.renderedExtent_ || !It(this.renderedExtent_, T), this.renderedExtent_ = T, this.renderedPixelRatio = c, this.renderedProjection = i, this.manageTilePyramid(e, u, p, c, i, g, m, l.getPreload()), this.scheduleExpireCache(e, u), this.postRender(R, e), n.extent && R.restore(), ne !== re.style.transform && (re.style.transform = ne);
		var Me = Ae(n.opacity), Ne = this.container;
		return Me !== Ne.style.opacity && (Ne.style.opacity = Me), this.container;
	}, t.prototype.drawTileImage = function(e, t, n, r, i, a, o, s) {
		var c = this.getTileImage(e);
		if (c) {
			var l = I(this), u = s ? e.getAlpha(l, t.time) : 1, d = u !== this.context.globalAlpha;
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
				var r = I(e);
				r in n.usedTiles && e.expireCache(n.viewState.projection, n.usedTiles[r]);
			}.bind(null, t);
			e.postRenderFunctions.push(n);
		}
	}, t.prototype.updateUsedTiles = function(e, t, n) {
		var r = I(t);
		r in e || (e[r] = {}), e[r][n.getKey()] = !0;
	}, t.prototype.manageTilePyramid = function(e, t, n, r, i, a, o, s, c) {
		var l = I(t);
		l in e.wantedTiles || (e.wantedTiles[l] = {});
		var u = e.wantedTiles[l], d = e.tileQueue, f = n.getMinZoom(), p = 0, m, h, g, _, v, y;
		for (y = f; y <= o; ++y) for (h = n.getTileRangeForExtentAndZ(a, y, h), g = n.getResolution(y), _ = h.minX; _ <= h.maxX; ++_) for (v = h.minY; v <= h.maxY; ++v) o - y <= s ? (++p, m = t.getTile(y, _, v, r, i), m.getState() == q.IDLE && (u[m.getKey()] = !0, d.isKeyQueued(m.getKey()) || d.enqueue([
			m,
			l,
			n.getTileCoordCenter(m.tileCoord),
			g
		])), c !== void 0 && c(m)) : t.useTile(y, _, v, i);
		t.updateCacheSize(p, i);
	}, t;
}(co);
po.prototype.getLayer;
//#endregion
//#region node_modules/ol/layer/Tile.js
var mo = (function() {
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
})(), ho = function(e) {
	mo(t, e);
	function t(t) {
		return e.call(this, t) || this;
	}
	return t.prototype.createRenderer = function() {
		return new po(this);
	}, t;
}(io), go = (function() {
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
})(), _o = function(e) {
	go(t, e);
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
		var d = i + " " + Ce + " " + Te + (n.collapsed_ && n.collapsible_ ? " " + Ee : "") + (n.collapsible_ ? "" : " ol-uncollapsible"), f = n.element;
		return f.className = d, f.appendChild(n.toggleButton_), f.appendChild(n.ulElement_), n.renderedAttributions_ = [], n.renderedVisible_ = !0, n;
	}
	return t.prototype.collectSourceAttributions_ = function(e) {
		for (var t = {}, n = [], r = !0, i = e.layerStatesArray, a = 0, o = i.length; a < o; ++a) {
			var s = i[a];
			if (_r(s, e.viewState)) {
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
			_e(this.ulElement_);
			for (var r = 0, i = t.length; r < i; ++r) {
				var a = document.createElement("li");
				a.innerHTML = t[r], this.ulElement_.appendChild(a);
			}
			this.renderedAttributions_ = t;
		}
	}, t.prototype.handleClick_ = function(e) {
		e.preventDefault(), this.handleToggle_(), this.userCollapsed_ = this.collapsed_;
	}, t.prototype.handleToggle_ = function() {
		this.element.classList.toggle(Ee), this.collapsed_ ? he(this.collapseLabel_, this.label_) : he(this.label_, this.collapseLabel_), this.collapsed_ = !this.collapsed_, this.toggleButton_.setAttribute("aria-expanded", String(!this.collapsed_));
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
}(be), vo = (function() {
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
})(), yo = function(e) {
	vo(t, e);
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
		var m = i + " " + Ce + " " + Te, h = n.element;
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
					easing: Ai
				})) : t.setZoom(r);
			}
		}
	}, t;
}(be);
//#endregion
//#region node_modules/ol/control.js
function bo(e) {
	var t = e || {}, n = new ni();
	return (t.zoom === void 0 || t.zoom) && n.push(new yo(t.zoomOptions)), (t.rotate === void 0 || t.rotate) && n.push(new Ya(t.rotateOptions)), (t.attribution === void 0 || t.attribution) && n.push(new _o(t.attributionOptions)), n;
}
//#endregion
//#region node_modules/ol/interaction/Property.js
var xo = { ACTIVE: "active" }, So = (function() {
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
})(), Co = function(e) {
	So(t, e);
	function t(t) {
		var n = e.call(this) || this;
		return n.on, n.once, n.un, t && t.handleEvent && (n.handleEvent = t.handleEvent), n.map_ = null, n.setActive(!0), n;
	}
	return t.prototype.getActive = function() {
		return this.get(xo.ACTIVE);
	}, t.prototype.getMap = function() {
		return this.map_;
	}, t.prototype.handleEvent = function(e) {
		return !0;
	}, t.prototype.setActive = function(e) {
		this.set(xo.ACTIVE, e);
	}, t.prototype.setMap = function(e) {
		this.map_ = e;
	}, t;
}(R);
function wo(e, t, n) {
	var r = e.getCenterInternal();
	if (r) {
		var i = [r[0] + t[0], r[1] + t[1]];
		e.animateInternal({
			duration: n === void 0 ? 250 : n,
			easing: Mi,
			center: e.getConstrainedCenter(i)
		});
	}
}
function To(e, t, n, r) {
	var i = e.getZoom();
	if (i !== void 0) {
		var a = e.getConstrainedZoom(i + t), o = e.getResolutionForZoom(a);
		e.getAnimating() && e.cancelAnimations(), e.animate({
			resolution: o,
			anchor: n,
			duration: r === void 0 ? 250 : r,
			easing: Ai
		});
	}
}
//#endregion
//#region node_modules/ol/interaction/DoubleClickZoom.js
var Eo = (function() {
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
})(), Do = function(e) {
	Eo(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		return n.delta_ = r.delta ? r.delta : 1, n.duration_ = r.duration === void 0 ? 250 : r.duration, n;
	}
	return t.prototype.handleEvent = function(e) {
		var t = !1;
		if (e.type == K.DBLCLICK) {
			var n = e.originalEvent, r = e.map, i = e.coordinate, a = n.shiftKey ? -this.delta_ : this.delta_;
			To(r.getView(), a, i, this.duration_), n.preventDefault(), t = !0;
		}
		return !t;
	}, t;
}(Co), Oo = (function() {
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
})(), ko = function(e) {
	Oo(t, e);
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
			if (e.type == K.POINTERDRAG) this.handleDragEvent(e), e.originalEvent.preventDefault();
			else if (e.type == K.POINTERUP) {
				var n = this.handleUpEvent(e);
				this.handlingDownUpSequence = n && this.targetPointers.length > 0;
			}
		} else if (e.type == K.POINTERDOWN) {
			var r = this.handleDownEvent(e);
			this.handlingDownUpSequence = r, t = this.stopDown(r);
		} else e.type == K.POINTERMOVE && this.handleMoveEvent(e);
		return !t;
	}, t.prototype.handleMoveEvent = function(e) {}, t.prototype.handleUpEvent = function(e) {
		return !1;
	}, t.prototype.stopDown = function(e) {
		return e;
	}, t.prototype.updateTrackedPointers_ = function(e) {
		if (jo(e)) {
			var t = e.originalEvent, n = t.pointerId.toString();
			e.type == K.POINTERUP ? delete this.trackedPointers_[n] : (e.type == K.POINTERDOWN || n in this.trackedPointers_) && (this.trackedPointers_[n] = t), this.targetPointers = w(this.trackedPointers_);
		}
	}, t;
}(Co);
function Ao(e) {
	for (var t = e.length, n = 0, r = 0, i = 0; i < t; i++) n += e[i].clientX, r += e[i].clientY;
	return [n / t, r / t];
}
function jo(e) {
	var t = e.type;
	return t === K.POINTERDOWN || t === K.POINTERDRAG || t === K.POINTERUP;
}
//#endregion
//#region node_modules/ol/events/condition.js
function Mo(e) {
	var t = arguments;
	return function(e) {
		for (var n = !0, r = 0, i = t.length; r < i && (n &&= t[r](e), n); ++r);
		return n;
	};
}
var No = function(e) {
	var t = e.originalEvent;
	return t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, Po = function(e) {
	return e.target.getTargetElement().contains(document.activeElement);
}, Fo = function(e) {
	return !e.map.getTargetElement().hasAttribute("tabindex") || Po(e);
}, Io = v, Lo = function(e) {
	var t = e.originalEvent;
	return t.button == 0 && !(oe && se && t.ctrlKey);
}, Ro = function(e) {
	var t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && !t.shiftKey;
}, zo = function(e) {
	var t = e.originalEvent;
	return !t.altKey && !(t.metaKey || t.ctrlKey) && t.shiftKey;
}, Bo = function(e) {
	var t = e.originalEvent.target.tagName;
	return t !== "INPUT" && t !== "SELECT" && t !== "TEXTAREA";
}, Vo = function(e) {
	var t = e.originalEvent;
	return V(t !== void 0, 56), t.pointerType == "mouse";
}, Ho = function(e) {
	var t = e.originalEvent;
	return V(t !== void 0, 56), t.isPrimary && t.button === 0;
}, Uo = (function() {
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
})(), Wo = function(e) {
	Uo(t, e);
	function t(t) {
		var n = e.call(this, { stopDown: y }) || this, r = t || {};
		n.kinetic_ = r.kinetic, n.lastCentroid = null, n.lastPointersCount_, n.panning_ = !1;
		var i = r.condition ? r.condition : Mo(Ro, Ho);
		return n.condition_ = r.onFocusOnly ? Mo(Fo, i) : i, n.noKinetic_ = !1, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		this.panning_ || (this.panning_ = !0, this.getMap().getView().beginInteraction());
		var t = this.targetPointers, n = Ao(t);
		if (t.length == this.lastPointersCount_) {
			if (this.kinetic_ && this.kinetic_.update(n[0], n[1]), this.lastCentroid) {
				var r = [this.lastCentroid[0] - n[0], n[1] - this.lastCentroid[1]], i = e.map.getView();
				un(r, i.getResolution()), ln(r, i.getRotation()), i.adjustCenterInternal(r);
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
					easing: Ai
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
}(ko), Go = (function() {
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
})(), Ko = function(e) {
	Go(t, e);
	function t(t) {
		var n = this, r = t || {};
		return n = e.call(this, { stopDown: y }) || this, n.condition_ = r.condition ? r.condition : No, n.lastAngle_ = void 0, n.duration_ = r.duration === void 0 ? 250 : r.duration, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		if (Vo(e)) {
			var t = e.map, n = t.getView();
			if (n.getConstraints().rotation !== Ti) {
				var r = t.getSize(), i = e.pixel, a = Math.atan2(r[1] / 2 - i[1], i[0] - r[0] / 2);
				if (this.lastAngle_ !== void 0) {
					var o = a - this.lastAngle_;
					n.adjustRotationInternal(-o);
				}
				this.lastAngle_ = a;
			}
		}
	}, t.prototype.handleUpEvent = function(e) {
		return Vo(e) ? (e.map.getView().endInteraction(this.duration_), !1) : !0;
	}, t.prototype.handleDownEvent = function(e) {
		return Vo(e) && Lo(e) && this.condition_(e) ? (e.map.getView().beginInteraction(), this.lastAngle_ = void 0, !0) : !1;
	}, t;
}(ko), qo = (function() {
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
})(), Jo = function(e) {
	qo(t, e);
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
		n[4] = n[0].slice(), this.geometry_ ? this.geometry_.setCoordinates([n]) : this.geometry_ = new Ea([n]);
	}, t.prototype.getGeometry = function() {
		return this.geometry_;
	}, t;
}(d), Yo = (function() {
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
})(), Xo = {
	BOXSTART: "boxstart",
	BOXDRAG: "boxdrag",
	BOXEND: "boxend",
	BOXCANCEL: "boxcancel"
}, Zo = function(e) {
	Yo(t, e);
	function t(t, n, r) {
		var i = e.call(this, t) || this;
		return i.coordinate = n, i.mapBrowserEvent = r, i;
	}
	return t;
}(l), Qo = function(e) {
	Yo(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.on, n.once, n.un;
		var r = t || {};
		return n.box_ = new Jo(r.className || "ol-dragbox"), n.minArea_ = r.minArea === void 0 ? 64 : r.minArea, r.onBoxEnd && (n.onBoxEnd = r.onBoxEnd), n.startPixel_ = null, n.condition_ = r.condition ? r.condition : Lo, n.boxEndCondition_ = r.boxEndCondition ? r.boxEndCondition : n.defaultBoxEndCondition, n;
	}
	return t.prototype.defaultBoxEndCondition = function(e, t, n) {
		var r = n[0] - t[0], i = n[1] - t[1];
		return r * r + i * i >= this.minArea_;
	}, t.prototype.getGeometry = function() {
		return this.box_.getGeometry();
	}, t.prototype.handleDragEvent = function(e) {
		this.box_.setPixels(this.startPixel_, e.pixel), this.dispatchEvent(new Zo(Xo.BOXDRAG, e.coordinate, e));
	}, t.prototype.handleUpEvent = function(e) {
		this.box_.setMap(null);
		var t = this.boxEndCondition_(e, this.startPixel_, e.pixel);
		return t && this.onBoxEnd(e), this.dispatchEvent(new Zo(t ? Xo.BOXEND : Xo.BOXCANCEL, e.coordinate, e)), !1;
	}, t.prototype.handleDownEvent = function(e) {
		return this.condition_(e) ? (this.startPixel_ = e.pixel, this.box_.setMap(e.map), this.box_.setPixels(this.startPixel_, this.startPixel_), this.dispatchEvent(new Zo(Xo.BOXSTART, e.coordinate, e)), !0) : !1;
	}, t.prototype.onBoxEnd = function(e) {}, t;
}(ko), $o = (function() {
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
})(), es = function(e) {
	$o(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r.condition ? r.condition : zo;
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
			easing: Ai
		});
	}, t;
}(Qo), ts = {
	LEFT: 37,
	UP: 38,
	RIGHT: 39,
	DOWN: 40
}, ns = (function() {
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
})(), rs = function(e) {
	ns(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		return n.defaultCondition_ = function(e) {
			return Ro(e) && Bo(e);
		}, n.condition_ = r.condition === void 0 ? n.defaultCondition_ : r.condition, n.duration_ = r.duration === void 0 ? 100 : r.duration, n.pixelDelta_ = r.pixelDelta === void 0 ? 128 : r.pixelDelta, n;
	}
	return t.prototype.handleEvent = function(e) {
		var t = !1;
		if (e.type == O.KEYDOWN) {
			var n = e.originalEvent, r = n.keyCode;
			if (this.condition_(e) && (r == ts.DOWN || r == ts.LEFT || r == ts.RIGHT || r == ts.UP)) {
				var i = e.map.getView(), a = i.getResolution() * this.pixelDelta_, o = 0, s = 0;
				r == ts.DOWN ? s = -a : r == ts.LEFT ? o = -a : r == ts.RIGHT ? o = a : s = a;
				var c = [o, s];
				ln(c, i.getRotation()), wo(i, c, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}, t;
}(Co), is = (function() {
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
})(), as = function(e) {
	is(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		return n.condition_ = r.condition ? r.condition : Bo, n.delta_ = r.delta ? r.delta : 1, n.duration_ = r.duration === void 0 ? 100 : r.duration, n;
	}
	return t.prototype.handleEvent = function(e) {
		var t = !1;
		if (e.type == O.KEYDOWN || e.type == O.KEYPRESS) {
			var n = e.originalEvent, r = n.charCode;
			if (this.condition_(e) && (r == 43 || r == 45)) {
				var i = e.map, a = r == 43 ? this.delta_ : -this.delta_;
				To(i.getView(), a, void 0, this.duration_), n.preventDefault(), t = !0;
			}
		}
		return !t;
	}, t;
}(Co), os = function() {
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
}(), ss = (function() {
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
})(), cs = {
	TRACKPAD: "trackpad",
	WHEEL: "wheel"
}, ls = function(e) {
	ss(t, e);
	function t(t) {
		var n = this, r = t || {};
		n = e.call(this, r) || this, n.totalDelta_ = 0, n.lastDelta_ = 0, n.maxDelta_ = r.maxDelta === void 0 ? 1 : r.maxDelta, n.duration_ = r.duration === void 0 ? 250 : r.duration, n.timeout_ = r.timeout === void 0 ? 80 : r.timeout, n.useAnchor_ = r.useAnchor === void 0 || r.useAnchor, n.constrainResolution_ = r.constrainResolution !== void 0 && r.constrainResolution;
		var i = r.condition ? r.condition : Io;
		return n.condition_ = r.onFocusOnly ? Mo(Fo, i) : i, n.lastAnchor_ = null, n.startTime_ = void 0, n.timeoutId_, n.mode_ = void 0, n.trackpadEventGap_ = 400, n.trackpadTimeoutId_, n.deltaPerZoom_ = 300, n;
	}
	return t.prototype.endInteraction_ = function() {
		this.trackpadTimeoutId_ = void 0, this.getMap().getView().endInteraction(void 0, this.lastDelta_ ? this.lastDelta_ > 0 ? 1 : -1 : 0, this.lastAnchor_);
	}, t.prototype.handleEvent = function(e) {
		if (!this.condition_(e) || e.type !== O.WHEEL) return !0;
		var t = e.map, n = e.originalEvent;
		n.preventDefault(), this.useAnchor_ && (this.lastAnchor_ = e.coordinate);
		var r;
		if (e.type == O.WHEEL && (r = n.deltaY, ae && n.deltaMode === WheelEvent.DOM_DELTA_PIXEL && (r /= ce), n.deltaMode === WheelEvent.DOM_DELTA_LINE && (r *= 40)), r === 0) return !1;
		this.lastDelta_ = r;
		var i = Date.now();
		this.startTime_ === void 0 && (this.startTime_ = i), (!this.mode_ || i - this.startTime_ > this.trackpadEventGap_) && (this.mode_ = Math.abs(r) < 4 ? cs.TRACKPAD : cs.WHEEL);
		var a = t.getView();
		if (this.mode_ === cs.TRACKPAD && !(a.getConstrainResolution() || this.constrainResolution_)) return this.trackpadTimeoutId_ ? clearTimeout(this.trackpadTimeoutId_) : (a.getAnimating() && a.cancelAnimations(), a.beginInteraction()), this.trackpadTimeoutId_ = setTimeout(this.endInteraction_.bind(this), this.timeout_), a.adjustZoom(-r / this.deltaPerZoom_, this.lastAnchor_), this.startTime_ = i, !1;
		this.totalDelta_ += r;
		var o = Math.max(this.timeout_ - (i - this.startTime_), 0);
		return clearTimeout(this.timeoutId_), this.timeoutId_ = setTimeout(this.handleWheelZoom_.bind(this, t), o), !1;
	}, t.prototype.handleWheelZoom_ = function(e) {
		var t = e.getView();
		t.getAnimating() && t.cancelAnimations();
		var n = -B(this.totalDelta_, -this.maxDelta_ * this.deltaPerZoom_, this.maxDelta_ * this.deltaPerZoom_) / this.deltaPerZoom_;
		(t.getConstrainResolution() || this.constrainResolution_) && (n = n ? n > 0 ? 1 : -1 : 0), To(t, n, this.lastAnchor_, this.duration_), this.mode_ = void 0, this.totalDelta_ = 0, this.lastAnchor_ = null, this.startTime_ = void 0, this.timeoutId_ = void 0;
	}, t.prototype.setMouseAnchor = function(e) {
		this.useAnchor_ = e, e || (this.lastAnchor_ = null);
	}, t;
}(Co), us = (function() {
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
})(), ds = function(e) {
	us(t, e);
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
		if (s.getConstraints().rotation !== Ti) {
			var c = o.getViewport().getBoundingClientRect(), l = Ao(this.targetPointers);
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
}(ko), fs = (function() {
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
})(), ps = function(e) {
	fs(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r;
		return i.stopDown ||= y, n = e.call(this, i) || this, n.anchor_ = null, n.duration_ = r.duration === void 0 ? 400 : r.duration, n.lastDistance_ = void 0, n.lastScaleDelta_ = 1, n;
	}
	return t.prototype.handleDragEvent = function(e) {
		var t = 1, n = this.targetPointers[0], r = this.targetPointers[1], i = n.clientX - r.clientX, a = n.clientY - r.clientY, o = Math.sqrt(i * i + a * a);
		this.lastDistance_ !== void 0 && (t = this.lastDistance_ / o), this.lastDistance_ = o;
		var s = e.map, c = s.getView();
		t != 1 && (this.lastScaleDelta_ = t);
		var l = s.getViewport().getBoundingClientRect(), u = Ao(this.targetPointers);
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
}(ko);
//#endregion
//#region node_modules/ol/geom/flat/length.js
function ms(e, t, n, r) {
	for (var i = e[t], a = e[t + 1], o = 0, s = t + r; s < n; s += r) {
		var c = e[s], l = e[s + 1];
		o += Math.sqrt((c - i) * (c - i) + (l - a) * (l - a)), i = c, a = l;
	}
	return o;
}
//#endregion
//#region node_modules/rbush/rbush.min.js
var hs = /* @__PURE__ */ o(((e, t) => {
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
})), gs = function() {
	function e(e) {
		this.opacity_ = e.opacity, this.rotateWithView_ = e.rotateWithView, this.rotation_ = e.rotation, this.scale_ = e.scale, this.scaleArray_ = za(e.scale), this.displacement_ = e.displacement;
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
		this.scale_ = e, this.scaleArray_ = za(e);
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
function _s(e) {
	return Array.isArray(e) ? sr(e) : e;
}
//#endregion
//#region node_modules/ol/style/RegularShape.js
var vs = (function() {
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
})(), ys = function(e) {
	vs(t, e);
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
			var n = this.renderOptions_, r = fe(n.size * e, n.size * e);
			this.draw_(n, r, e), t = r.canvas, this.canvas_[e] = t;
		}
		return t;
	}, t.prototype.getPixelRatio = function(e) {
		return e;
	}, t.prototype.getImageSize = function() {
		return this.size_;
	}, t.prototype.getImageState = function() {
		return Y.LOADED;
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
		var e = Dr, t = 0, n = null, r = 0, i, a = 0;
		this.stroke_ && (i = this.stroke_.getColor(), i === null && (i = Or), i = _s(i), a = this.stroke_.getWidth(), a === void 0 && (a = 1), n = this.stroke_.getLineDash(), r = this.stroke_.getLineDashOffset(), e = this.stroke_.getLineJoin(), e === void 0 && (e = Dr), t = this.stroke_.getMiterLimit(), t === void 0 && (t = 10));
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
			r === null && (r = wr), t.fillStyle = _s(r), t.fill();
		}
		this.stroke_ && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, t.setLineDash && e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}, t.prototype.createHitDetectionCanvas_ = function(e) {
		if (this.fill_) {
			var t = this.fill_.getColor(), n = 0;
			if (typeof t == "string" && (t = ir(t)), t === null ? n = 1 : Array.isArray(t) && (n = t.length === 4 ? t[3] : 1), n === 0) {
				var r = fe(e.size, e.size);
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
		t.translate(e.size / 2, e.size / 2), this.createPath_(t), t.fillStyle = wr, t.fill(), this.stroke_ && (t.strokeStyle = e.strokeStyle, t.lineWidth = e.strokeWidth, e.lineDash && (t.setLineDash(e.lineDash), t.lineDashOffset = e.lineDashOffset), t.lineJoin = e.lineJoin, t.miterLimit = e.miterLimit, t.stroke());
	}, t;
}(gs), bs = (function() {
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
})(), xs = function(e) {
	bs(t, e);
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
}(ys), Ss = function() {
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
}(), Cs = function() {
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
}(), ws = function() {
	function e(e) {
		var t = e || {};
		this.geometry_ = null, this.geometryFunction_ = Os, t.geometry !== void 0 && this.setGeometry(t.geometry), this.fill_ = t.fill === void 0 ? null : t.fill, this.image_ = t.image === void 0 ? null : t.image, this.renderer_ = t.renderer === void 0 ? null : t.renderer, this.hitDetectionRenderer_ = t.hitDetectionRenderer === void 0 ? null : t.hitDetectionRenderer, this.stroke_ = t.stroke === void 0 ? null : t.stroke, this.text_ = t.text === void 0 ? null : t.text, this.zIndex_ = t.zIndex;
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
		}) : this.geometryFunction_ = Os, this.geometry_ = e;
	}, e.prototype.setZIndex = function(e) {
		this.zIndex_ = e;
	}, e;
}();
function Ts(e) {
	var t;
	if (typeof e == "function") t = e;
	else {
		var n;
		Array.isArray(e) ? n = e : (V(typeof e.getZIndex == "function", 41), n = [e]), t = function() {
			return n;
		};
	}
	return t;
}
var Es = null;
function Ds(e, t) {
	if (!Es) {
		var n = new Ss({ color: "rgba(255,255,255,0.4)" }), r = new Cs({
			color: "#3399CC",
			width: 1.25
		});
		Es = [new ws({
			image: new xs({
				fill: n,
				stroke: r,
				radius: 5
			}),
			fill: n,
			stroke: r
		})];
	}
	return Es;
}
function Os(e) {
	return e.getGeometry();
}
//#endregion
//#region node_modules/ol/layer/BaseVector.js
var ks = /* @__PURE__ */ c(hs(), 1), As = (function() {
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
})(), js = { RENDER_ORDER: "renderOrder" }, Ms = function(e) {
	As(t, e);
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
		return this.get(js.RENDER_ORDER);
	}, t.prototype.getStyle = function() {
		return this.style_;
	}, t.prototype.getStyleFunction = function() {
		return this.styleFunction_;
	}, t.prototype.getUpdateWhileAnimating = function() {
		return this.updateWhileAnimating_;
	}, t.prototype.getUpdateWhileInteracting = function() {
		return this.updateWhileInteracting_;
	}, t.prototype.renderDeclutter = function(e) {
		e.declutterTree ||= new ks.default(9), this.getRenderer().renderDeclutter(e);
	}, t.prototype.setRenderOrder = function(e) {
		this.set(js.RENDER_ORDER, e);
	}, t.prototype.setStyle = function(e) {
		this.style_ = e === void 0 ? Ds : e, this.styleFunction_ = e === null ? void 0 : Ts(this.style_), this.changed();
	}, t;
}(gr), X = {
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
}, Ns = [X.FILL], Ps = [X.STROKE], Fs = [X.BEGIN_PATH], Is = [X.CLOSE_PATH], Ls = function() {
	function e() {}
	return e.prototype.drawCustom = function(e, t, n, r) {}, e.prototype.drawGeometry = function(e) {}, e.prototype.setStyle = function(e) {}, e.prototype.drawCircle = function(e, t) {}, e.prototype.drawFeature = function(e, t) {}, e.prototype.drawGeometryCollection = function(e, t) {}, e.prototype.drawLineString = function(e, t) {}, e.prototype.drawMultiLineString = function(e, t) {}, e.prototype.drawMultiPoint = function(e, t) {}, e.prototype.drawMultiPolygon = function(e, t) {}, e.prototype.drawPoint = function(e, t) {}, e.prototype.drawPolygon = function(e, t) {}, e.prototype.drawText = function(e, t) {}, e.prototype.setFillStrokeStyle = function(e, t) {}, e.prototype.setImageStyle = function(e, t) {}, e.prototype.setTextStyle = function(e, t) {}, e;
}(), Rs = (function() {
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
})(), zs = function(e) {
	Rs(t, e);
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
		for (var n = this.getBufferedMaxExtent(), r = this.tmpCoordinate_, i = this.coordinates, a = i.length, o = 0, s = e.length; o < s; o += t) r[0] = e[o], r[1] = e[o + 1], Dt(n, r) && (i[a++] = r[0], i[a++] = r[1]);
		return a;
	}, t.prototype.appendFlatLineCoordinates = function(e, t, n, r, i, a) {
		var o = this.coordinates, s = o.length, c = this.getBufferedMaxExtent();
		a && (t += r);
		var l = e[t], u = e[t + 1], d = this.tmpCoordinate_, f = !0, p, m, h;
		for (p = t + r; p < n; p += r) d[0] = e[p], d[1] = e[p + 1], h = At(c, d), h === m ? h === bt.INTERSECTING ? (o[s++] = d[0], o[s++] = d[1], f = !1) : f = !0 : (f &&= (o[s++] = l, o[s++] = u, !1), o[s++] = d[0], o[s++] = d[1]), l = d[0], u = d[1], m = h;
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
			case U.MULTI_POLYGON:
				s = e.getOrientedFlatCoordinates(), u = [];
				var f = e.getEndss();
				d = 0;
				for (var p = 0, m = f.length; p < m; ++p) {
					var h = [];
					d = this.drawCustomCoordinates_(s, d, f[p], a, h), u.push(h);
				}
				this.instructions.push([
					X.CUSTOM,
					o,
					u,
					e,
					n,
					oa
				]), this.hitDetectionInstructions.push([
					X.CUSTOM,
					o,
					u,
					e,
					r || n,
					oa
				]);
				break;
			case U.POLYGON:
			case U.MULTI_LINE_STRING:
				l = [], s = i == U.POLYGON ? e.getOrientedFlatCoordinates() : e.getFlatCoordinates(), d = this.drawCustomCoordinates_(s, 0, e.getEnds(), a, l), this.instructions.push([
					X.CUSTOM,
					o,
					l,
					e,
					n,
					aa
				]), this.hitDetectionInstructions.push([
					X.CUSTOM,
					o,
					l,
					e,
					r || n,
					aa
				]);
				break;
			case U.LINE_STRING:
			case U.CIRCLE:
				s = e.getFlatCoordinates(), c = this.appendFlatLineCoordinates(s, 0, s.length, a, !1, !1), this.instructions.push([
					X.CUSTOM,
					o,
					c,
					e,
					n,
					ia
				]), this.hitDetectionInstructions.push([
					X.CUSTOM,
					o,
					c,
					e,
					r || n,
					ia
				]);
				break;
			case U.MULTI_POINT:
				s = e.getFlatCoordinates(), c = this.appendFlatPointCoordinates(s, a), c > o && (this.instructions.push([
					X.CUSTOM,
					o,
					c,
					e,
					n,
					ia
				]), this.hitDetectionInstructions.push([
					X.CUSTOM,
					o,
					c,
					e,
					r || n,
					ia
				]));
				break;
			case U.POINT:
				s = e.getFlatCoordinates(), this.coordinates.push(s[0], s[1]), c = this.coordinates.length, this.instructions.push([
					X.CUSTOM,
					o,
					c,
					e,
					n
				]), this.hitDetectionInstructions.push([
					X.CUSTOM,
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
			X.BEGIN_GEOMETRY,
			t,
			0,
			e
		], this.instructions.push(this.beginGeometryInstruction1_), this.beginGeometryInstruction2_ = [
			X.BEGIN_GEOMETRY,
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
		for (t = 0; t < n; ++t) r = e[t], i = r[0], i == X.END_GEOMETRY ? a = t : i == X.BEGIN_GEOMETRY && (r[2] = t, m(this.hitDetectionInstructions, a, t), a = -1);
	}, t.prototype.setFillStrokeStyle = function(e, t) {
		var n = this.state;
		if (e ? n.fillStyle = _s(e.getColor() || wr) : n.fillStyle = void 0, t) {
			n.strokeStyle = _s(t.getColor() || Or);
			var r = t.getLineCap();
			n.lineCap = r === void 0 ? Tr : r;
			var i = t.getLineDash();
			n.lineDash = i ? i.slice() : Er, n.lineDashOffset = t.getLineDashOffset() || 0;
			var a = t.getLineJoin();
			n.lineJoin = a === void 0 ? Dr : a;
			var o = t.getWidth();
			n.lineWidth = o === void 0 ? 1 : o;
			var s = t.getMiterLimit();
			n.miterLimit = s === void 0 ? 10 : s, n.lineWidth > this.maxLineWidth && (this.maxLineWidth = n.lineWidth, this.bufferedMaxExtent_ = null);
		} else n.strokeStyle = void 0, n.lineCap = void 0, n.lineDash = null, n.lineDashOffset = void 0, n.lineJoin = void 0, n.lineWidth = void 0, n.miterLimit = void 0;
	}, t.prototype.createFill = function(e) {
		var t = e.fillStyle, n = [X.SET_FILL_STYLE, t];
		return typeof t != "string" && n.push(!0), n;
	}, t.prototype.applyStroke = function(e) {
		this.instructions.push(this.createStroke(e));
	}, t.prototype.createStroke = function(e) {
		return [
			X.SET_STROKE_STYLE,
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
		var t = [X.END_GEOMETRY, e];
		this.instructions.push(t), this.hitDetectionInstructions.push(t);
	}, t.prototype.getBufferedMaxExtent = function() {
		if (!this.bufferedMaxExtent_ && (this.bufferedMaxExtent_ = Tt(this.maxExtent), this.maxLineWidth > 0)) {
			var e = this.resolution * (this.maxLineWidth + 1) / 2;
			wt(this.bufferedMaxExtent_, e, this.bufferedMaxExtent_);
		}
		return this.bufferedMaxExtent_;
	}, t;
}(Ls), Bs = (function() {
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
})(), Vs = function(e) {
	Bs(t, e);
	function t(t, n, r, i) {
		var a = e.call(this, t, n, r, i) || this;
		return a.hitDetectionImage_ = null, a.image_ = null, a.imagePixelRatio_ = void 0, a.anchorX_ = void 0, a.anchorY_ = void 0, a.height_ = void 0, a.opacity_ = void 0, a.originX_ = void 0, a.originY_ = void 0, a.rotateWithView_ = void 0, a.rotation_ = void 0, a.scale_ = void 0, a.width_ = void 0, a.declutterImageWithText_ = void 0, a;
	}
	return t.prototype.drawPoint = function(e, t) {
		if (this.image_) {
			this.beginGeometry(e, t);
			var n = e.getFlatCoordinates(), r = e.getStride(), i = this.coordinates.length, a = this.appendFlatPointCoordinates(n, r);
			this.instructions.push([
				X.DRAW_IMAGE,
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
				X.DRAW_IMAGE,
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
				X.DRAW_IMAGE,
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
				X.DRAW_IMAGE,
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
}(zs), Hs = (function() {
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
})(), Us = function(e) {
	Hs(t, e);
	function t(t, n, r, i) {
		return e.call(this, t, n, r, i) || this;
	}
	return t.prototype.drawFlatCoordinates_ = function(e, t, n, r) {
		var i = this.coordinates.length, a = this.appendFlatLineCoordinates(e, t, n, r, !1, !1), o = [
			X.MOVE_TO_LINE_TO,
			i,
			a
		];
		return this.instructions.push(o), this.hitDetectionInstructions.push(o), n;
	}, t.prototype.drawLineString = function(e, t) {
		var n = this.state, r = n.strokeStyle, i = n.lineWidth;
		if (!(r === void 0 || i === void 0)) {
			this.updateStrokeStyle(n, this.applyStroke), this.beginGeometry(e, t), this.hitDetectionInstructions.push([
				X.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				Er,
				0
			], Fs);
			var a = e.getFlatCoordinates(), o = e.getStride();
			this.drawFlatCoordinates_(a, 0, a.length, o), this.hitDetectionInstructions.push(Ps), this.endGeometry(t);
		}
	}, t.prototype.drawMultiLineString = function(e, t) {
		var n = this.state, r = n.strokeStyle, i = n.lineWidth;
		if (!(r === void 0 || i === void 0)) {
			this.updateStrokeStyle(n, this.applyStroke), this.beginGeometry(e, t), this.hitDetectionInstructions.push([
				X.SET_STROKE_STYLE,
				n.strokeStyle,
				n.lineWidth,
				n.lineCap,
				n.lineJoin,
				n.miterLimit,
				n.lineDash,
				n.lineDashOffset
			], Fs);
			for (var a = e.getEnds(), o = e.getFlatCoordinates(), s = e.getStride(), c = 0, l = 0, u = a.length; l < u; ++l) c = this.drawFlatCoordinates_(o, c, a[l], s);
			this.hitDetectionInstructions.push(Ps), this.endGeometry(t);
		}
	}, t.prototype.finish = function() {
		var t = this.state;
		return t.lastStroke != null && t.lastStroke != this.coordinates.length && this.instructions.push(Ps), this.reverseHitDetectionInstructions(), this.state = null, e.prototype.finish.call(this);
	}, t.prototype.applyStroke = function(t) {
		t.lastStroke != null && t.lastStroke != this.coordinates.length && (this.instructions.push(Ps), t.lastStroke = this.coordinates.length), t.lastStroke = 0, e.prototype.applyStroke.call(this, t), this.instructions.push(Fs);
	}, t;
}(zs), Ws = (function() {
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
})(), Gs = function(e) {
	Ws(t, e);
	function t(t, n, r, i) {
		return e.call(this, t, n, r, i) || this;
	}
	return t.prototype.drawFlatCoordinatess_ = function(e, t, n, r) {
		var i = this.state, a = i.fillStyle !== void 0, o = i.strokeStyle !== void 0, s = n.length;
		this.instructions.push(Fs), this.hitDetectionInstructions.push(Fs);
		for (var c = 0; c < s; ++c) {
			var l = n[c], u = this.coordinates.length, d = this.appendFlatLineCoordinates(e, t, l, r, !0, !o), f = [
				X.MOVE_TO_LINE_TO,
				u,
				d
			];
			this.instructions.push(f), this.hitDetectionInstructions.push(f), o && (this.instructions.push(Is), this.hitDetectionInstructions.push(Is)), t = l;
		}
		return a && (this.instructions.push(Ns), this.hitDetectionInstructions.push(Ns)), o && (this.instructions.push(Ps), this.hitDetectionInstructions.push(Ps)), t;
	}, t.prototype.drawCircle = function(e, t) {
		var n = this.state, r = n.fillStyle, i = n.strokeStyle;
		if (!(r === void 0 && i === void 0)) {
			this.setFillStrokeStyles_(), this.beginGeometry(e, t), n.fillStyle !== void 0 && this.hitDetectionInstructions.push([X.SET_FILL_STYLE, wr]), n.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
				X.SET_STROKE_STYLE,
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
			var c = [X.CIRCLE, s];
			this.instructions.push(Fs, c), this.hitDetectionInstructions.push(Fs, c), n.fillStyle !== void 0 && (this.instructions.push(Ns), this.hitDetectionInstructions.push(Ns)), n.strokeStyle !== void 0 && (this.instructions.push(Ps), this.hitDetectionInstructions.push(Ps)), this.endGeometry(t);
		}
	}, t.prototype.drawPolygon = function(e, t) {
		var n = this.state, r = n.fillStyle, i = n.strokeStyle;
		if (!(r === void 0 && i === void 0)) {
			this.setFillStrokeStyles_(), this.beginGeometry(e, t), n.fillStyle !== void 0 && this.hitDetectionInstructions.push([X.SET_FILL_STYLE, wr]), n.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
				X.SET_STROKE_STYLE,
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
			this.setFillStrokeStyles_(), this.beginGeometry(e, t), n.fillStyle !== void 0 && this.hitDetectionInstructions.push([X.SET_FILL_STYLE, wr]), n.strokeStyle !== void 0 && this.hitDetectionInstructions.push([
				X.SET_STROKE_STYLE,
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
		if (t !== 0) for (var n = this.coordinates, r = 0, i = n.length; r < i; ++r) n[r] = ta(n[r], t);
		return e.prototype.finish.call(this);
	}, t.prototype.setFillStrokeStyles_ = function() {
		var e = this.state;
		e.fillStyle !== void 0 && this.updateFillStyle(e, this.createFill), e.strokeStyle !== void 0 && this.updateStrokeStyle(e, this.applyStroke);
	}, t;
}(zs), Ks = {
	POINT: "point",
	LINE: "line"
};
//#endregion
//#region node_modules/ol/geom/flat/straightchunk.js
function qs(e, t, n, r, i) {
	var a = n, o = n, s = 0, c = 0, l = n, u, d, f, p, m, h, g, _, v, y;
	for (d = n; d < r; d += i) {
		var b = t[d], x = t[d + 1];
		m !== void 0 && (v = b - m, y = x - h, p = Math.sqrt(v * v + y * y), g !== void 0 && (c += f, u = Math.acos((g * v + _ * y) / (f * p)), u > e && (c > s && (s = c, a = l, o = d), c = 0, l = d - i)), f = p, g = v, _ = y), m = b, h = x;
	}
	return c += p, c > s ? [l, d] : [a, o];
}
//#endregion
//#region node_modules/ol/render/canvas/TextBuilder.js
var Js = (function() {
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
})(), Ys = {
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
}, Xs = {
	Circle: Gs,
	Default: zs,
	Image: Vs,
	LineString: Us,
	Polygon: Gs,
	Text: function(e) {
		Js(t, e);
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
				if (i.placement === Ks.LINE && (s == U.LINE_STRING || s == U.MULTI_LINE_STRING || s == U.POLYGON || s == U.MULTI_POLYGON)) {
					if (!Qt(this.getBufferedMaxExtent(), e.getExtent())) return;
					var u = void 0;
					if (c = e.getFlatCoordinates(), s == U.LINE_STRING) u = [c.length];
					else if (s == U.MULTI_LINE_STRING) u = e.getEnds();
					else if (s == U.POLYGON) u = e.getEnds().slice(0, 1);
					else if (s == U.MULTI_POLYGON) {
						var d = e.getEndss();
						u = [];
						for (var f = 0, p = d.length; f < p; ++f) u.push(d[f][0]);
					}
					this.beginGeometry(e, t);
					for (var m = i.textAlign, h = 0, g = void 0, _ = 0, v = u.length; _ < v; ++_) {
						if (m == null) {
							var y = qs(i.maxAngle, c, h, u[_], l);
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
						case U.POINT:
						case U.MULTI_POINT:
							c = e.getFlatCoordinates();
							break;
						case U.LINE_STRING:
							c = e.getFlatMidpoint();
							break;
						case U.CIRCLE:
							c = e.getCenter();
							break;
						case U.MULTI_LINE_STRING:
							c = e.getFlatMidpoints(), l = 2;
							break;
						case U.POLYGON:
							c = e.getFlatInteriorPoint(), i.overflow || x.push(c[2] / this.resolution), l = 3;
							break;
						case U.MULTI_POLYGON:
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
					if (w != jr && (i.scale[0] < 0 || i.scale[1] < 0)) {
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
						X.DRAW_IMAGE,
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
						w == jr ? jr : w.map(function(e) {
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
						X.DRAW_IMAGE,
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
			var s = this.pixelRatio, c = Ys[r.textBaseline], l = this.textOffsetY_ * s, u = this.text_, d = n ? n.lineWidth * Math.abs(r.scale[0]) / 2 : 0;
			this.instructions.push([
				X.DRAW_CHARS,
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
				X.DRAW_CHARS,
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
				a ? (r = this.textFillState_, r || (r = {}, this.textFillState_ = r), r.fillStyle = _s(a.getColor() || "#000")) : (r = null, this.textFillState_ = r);
				var o = e.getStroke();
				if (!o) i = null, this.textStrokeState_ = i;
				else {
					i = this.textStrokeState_, i || (i = {}, this.textStrokeState_ = i);
					var s = o.getLineDash(), c = o.getLineDashOffset(), l = o.getWidth(), u = o.getMiterLimit();
					i.lineCap = o.getLineCap() || "round", i.lineDash = s ? s.slice() : Er, i.lineDashOffset = c === void 0 ? 0 : c, i.lineJoin = o.getLineJoin() || "round", i.lineWidth = l === void 0 ? 1 : l, i.miterLimit = u === void 0 ? 10 : u, i.strokeStyle = _s(o.getColor() || "#000");
				}
				n = this.textState_;
				var d = e.getFont() || "10px sans-serif";
				Lr(d);
				var f = e.getScaleArray();
				n.overflow = e.getOverflow(), n.font = d, n.maxAngle = e.getMaxAngle(), n.placement = e.getPlacement(), n.textAlign = e.getTextAlign(), n.textBaseline = e.getTextBaseline() || "middle", n.backgroundFill = e.getBackgroundFill(), n.backgroundStroke = e.getBackgroundStroke(), n.padding = e.getPadding() || jr, n.scale = f === void 0 ? [1, 1] : f;
				var p = e.getOffsetX(), m = e.getOffsetY(), h = e.getRotateWithView(), g = e.getRotation();
				this.text_ = e.getText() || "", this.textOffsetX_ = p === void 0 ? 0 : p, this.textOffsetY_ = m === void 0 ? 0 : m, this.textRotateWithView_ = h !== void 0 && h, this.textRotation_ = g === void 0 ? 0 : g, this.strokeKey_ = i ? (typeof i.strokeStyle == "string" ? i.strokeStyle : I(i.strokeStyle)) + i.lineCap + i.lineDashOffset + "|" + i.lineWidth + i.lineJoin + i.miterLimit + "[" + i.lineDash.join() + "]" : "", this.textKey_ = n.font + n.scale + (n.textAlign || "?") + (n.textBaseline || "?"), this.fillKey_ = r ? typeof r.fillStyle == "string" ? r.fillStyle : "|" + I(r.fillStyle) : "";
			}
			this.declutterImageWithText_ = t;
		}, t;
	}(zs)
}, Zs = function() {
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
			var a = Xs[t];
			i = new a(this.tolerance_, this.maxExtent_, this.resolution_, this.pixelRatio_), r[t] = i;
		}
		return i;
	}, e;
}(), Z = {
	CIRCLE: "Circle",
	DEFAULT: "Default",
	IMAGE: "Image",
	LINE_STRING: "LineString",
	POLYGON: "Polygon",
	TEXT: "Text"
};
//#endregion
//#region node_modules/ol/geom/flat/textpath.js
function Qs(e, t, n, r, i, a, o, s, c, l, u, d) {
	var f = e[t], p = e[t + 1], m = 0, h = 0, g = 0, _ = 0;
	function v() {
		m = f, h = p, t += r, f = e[t], p = e[t + 1], _ += g, g = Math.sqrt((f - m) * (f - m) + (p - h) * (p - h));
	}
	do
		v();
	while (t < n - r && _ + g < a);
	for (var y = g === 0 ? 0 : (a - _) / g, b = Xe(m, f, y), x = Xe(h, p, y), S = t - r, C = _, w = a + s * c(l, i, u); t < n - r && _ + g < w;) v();
	y = g === 0 ? 0 : (w - _) / g;
	var T = Xe(m, f, y), E = Xe(h, p, y), D;
	if (d) {
		var O = [
			b,
			x,
			T,
			E
		];
		Fi(O, 0, 4, 2, d, O, O), D = O[0] > O[2];
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
	for (var F = 0, ee = i.length; F < ee;) {
		v();
		var I = Math.atan2(p - h, f - m);
		if (D && (I += I > 0 ? -k : k), M !== void 0) {
			var L = I - M;
			if (L += L > k ? -2 * k : L < -k ? 2 * k : 0, Math.abs(L) > o) return null;
		}
		M = I;
		for (var te = F, ne = 0; F < ee; ++F) {
			var R = s * c(l, i[D ? ee - F - 1 : F], u);
			if (t + r < n && _ + g < a + ne + R / 2) break;
			ne += R;
		}
		if (F !== te) {
			var re = D ? i.substring(ee - te, ee - F) : i.substring(te, F);
			y = g === 0 ? 0 : (a + ne / 2 - _) / g;
			var N = Xe(m, f, y), P = Xe(h, p, y);
			A.push([
				N,
				P,
				ne / 2,
				I,
				re
			]), a += ne;
		}
	}
	return A;
}
//#endregion
//#region node_modules/ol/render/canvas/Executor.js
var $s = jt(), ec = [], tc = [], nc = [], rc = [];
function ic(e) {
	return e[3].declutterBox;
}
var ac = /* @__PURE__ */ RegExp("[֑-ࣿיִ-﷿ﹰ-ﻼࠀ-࿿-]");
function oc(e, t) {
	return (t === "start" || t === "end") && !ac.test(e) && (t = t === "start" ? "left" : "right"), Ys[t];
}
var sc = function() {
	function e(e, t, n, r) {
		this.overlaps = n, this.pixelRatio = t, this.resolution = e, this.alignFill_, this.instructions = r.instructions, this.coordinates = r.coordinates, this.coordinateCache_ = {}, this.renderedTransform_ = zn(), this.hitDetectionInstructions = r.hitDetectionInstructions, this.pixelCoordinates_ = null, this.viewRotation_ = 0, this.fillStates = r.fillStates || {}, this.strokeStates = r.strokeStates || {}, this.textStates = r.textStates || {}, this.widths_ = {}, this.labels_ = {};
	}
	return e.prototype.createLabel = function(e, t, n, r) {
		var i = e + t + n + r;
		if (this.labels_[i]) return this.labels_[i];
		var a = r ? this.strokeStates[r] : null, o = n ? this.fillStates[n] : null, s = this.textStates[t], c = this.pixelRatio, l = [s.scale[0] * c, s.scale[1] * c], u = oc(e, s.textAlign || "center"), d = r && a.lineWidth ? a.lineWidth : 0, f = e.split("\n"), p = f.length, m = [], h = Hr(s.font, f, m), g = Rr(s.font), _ = g * p, v = h + d, y = [], b = (v + 2) * l[0], x = (_ + d) * l[1], S = {
			width: b < 0 ? Math.floor(b) : Math.ceil(b),
			height: x < 0 ? Math.floor(x) : Math.ceil(x),
			contextInstructions: y
		};
		(l[0] != 1 || l[1] != 1) && y.push("scale", l), y.push("font", s.font), r && (y.push("strokeStyle", a.strokeStyle), y.push("lineWidth", d), y.push("lineCap", a.lineCap), y.push("lineJoin", a.lineJoin), y.push("miterLimit", a.miterLimit), (le ? OffscreenCanvasRenderingContext2D : CanvasRenderingContext2D).prototype.setLineDash && (y.push("setLineDash", [a.lineDash]), y.push("lineDashOffset", a.lineDashOffset))), n && y.push("fillStyle", o.fillStyle), y.push("textBaseline", "middle"), y.push("textAlign", "center");
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
		(m || u !== 0) && (ec[0] = S, rc[0] = S, ec[1] = C, tc[1] = C, tc[0] = S + b, nc[0] = tc[0], nc[1] = C + x, rc[1] = nc[1]);
		var w;
		return u === 0 ? Mt(Math.min(S, S + b), Math.min(C, C + x), Math.max(S, S + b), Math.max(C, C + x), $s) : (w = Jn(zn(), n, r, 1, 1, u, -n, -r), W(w, ec), W(w, tc), W(w, nc), W(w, rc), Mt(Math.min(ec[0], tc[0], nc[0], rc[0]), Math.min(ec[1], tc[1], nc[1], rc[1]), Math.max(ec[0], tc[0], nc[0], rc[0]), Math.max(ec[1], tc[1], nc[1], rc[1]), $s)), f && (g = Math.round(g), _ = Math.round(_)), {
			drawImageX: g,
			drawImageY: _,
			drawImageW: v,
			drawImageH: y,
			originX: c,
			originY: l,
			declutterBox: {
				minX: $s[0],
				minY: $s[1],
				maxX: $s[2],
				maxY: $s[3],
				value: h
			},
			canvasTransform: w,
			scale: d
		};
	}, e.prototype.replayImageOrLabel_ = function(e, t, n, r, i, a, o) {
		var s = !!(a || o), c = r.declutterBox, l = e.canvas, u = o ? o[2] * r.scale[0] / 2 : 0;
		return c.minX - u <= l.width / t && c.maxX + u >= 0 && c.minY - u <= l.height / t && c.maxY + u >= 0 && (s && this.replayTextBackground_(e, ec, tc, nc, rc, a, o), Ur(e, r.canvasTransform, i, n, r.originX, r.originY, r.drawImageW, r.drawImageH, r.drawImageX, r.drawImageY, r.scale)), !0;
	}, e.prototype.fill_ = function(e) {
		if (this.alignFill_) {
			var t = W(this.renderedTransform_, [0, 0]), n = 512 * this.pixelRatio;
			e.save(), e.translate(t[0] % n, t[1] % n), e.rotate(this.viewRotation_);
		}
		e.fill(), this.alignFill_ && e.restore();
	}, e.prototype.setStrokeStyle_ = function(e, t) {
		e.strokeStyle = t[1], e.lineWidth = t[2], e.lineCap = t[3], e.lineJoin = t[4], e.miterLimit = t[5], e.setLineDash && (e.lineDashOffset = t[7], e.setLineDash(t[6]));
	}, e.prototype.drawLabelWithPointPlacement_ = function(e, t, n, r) {
		var i = this.textStates[t], a = this.createLabel(e, t, r, n), o = this.strokeStates[n], s = this.pixelRatio, c = oc(e, i.textAlign || "center"), l = Ys[i.textBaseline || "middle"], u = o && o.lineWidth ? o.lineWidth : 0;
		return {
			label: a,
			anchorX: c * (a.width / s - 2 * i.scale[0]) + 2 * (.5 - c) * u,
			anchorY: l * a.height / s + 2 * (.5 - l) * u
		};
	}, e.prototype.execute_ = function(e, t, n, r, i, a, o, s) {
		var c;
		this.pixelCoordinates_ && g(n, this.renderedTransform_) ? c = this.pixelCoordinates_ : (this.pixelCoordinates_ ||= [], c = Pi(this.coordinates, 0, this.coordinates.length, 2, n, this.pixelCoordinates_), Un(this.renderedTransform_, n));
		for (var l = 0, u = r.length, d = 0, f, p, m, h, _, v, y, b, x, S, C, w, T = 0, E = 0, D = null, O = null, k = this.coordinateCache_, A = this.viewRotation_, j = Math.round(Math.atan2(-n[1], n[0]) * 0xe8d4a51000) / 0xe8d4a51000, M = {
			context: e,
			pixelRatio: this.pixelRatio,
			resolution: this.resolution,
			rotation: A
		}, N = this.instructions != r || this.overlaps ? 0 : 200, P, F, ee, I; l < u;) {
			var L = r[l];
			switch (L[0]) {
				case X.BEGIN_GEOMETRY:
					P = L[1], I = L[3], P.getGeometry() ? o !== void 0 && !Qt(o, I.getExtent()) ? l = L[2] + 1 : ++l : l = L[2];
					break;
				case X.BEGIN_PATH:
					T > N && (this.fill_(e), T = 0), E > N && (e.stroke(), E = 0), !T && !E && (e.beginPath(), h = NaN, _ = NaN), ++l;
					break;
				case X.CIRCLE:
					d = L[1];
					var te = c[d], ne = c[d + 1], R = c[d + 2], re = c[d + 3], ie = R - te, ae = re - ne, oe = Math.sqrt(ie * ie + ae * ae);
					e.moveTo(te + oe, ne), e.arc(te, ne, oe, 0, 2 * Math.PI, !0), ++l;
					break;
				case X.CLOSE_PATH:
					e.closePath(), ++l;
					break;
				case X.CUSTOM:
					d = L[1], f = L[2];
					var se = L[3], ce = L[4], le = L.length == 6 ? L[5] : void 0;
					M.geometry = se, M.feature = P, l in k || (k[l] = []);
					var ue = k[l];
					le ? le(c, d, f, 2, ue) : (ue[0] = c[d], ue[1] = c[d + 1], ue.length = 2), ce(ue, M), ++l;
					break;
				case X.DRAW_IMAGE:
					d = L[1], f = L[2], b = L[3], p = L[4], m = L[5];
					var de = L[6], fe = L[7], pe = L[8], me = L[9], he = L[10], ge = L[11], _e = L[12], ve = L[13], ye = L[14];
					if (!b && L.length >= 19) {
						x = L[18], S = L[19], C = L[20], w = L[21];
						var be = this.drawLabelWithPointPlacement_(x, S, C, w);
						b = be.label, L[3] = b;
						var xe = L[22];
						p = (be.anchorX - xe) * this.pixelRatio, L[4] = p;
						var Se = L[23];
						m = (be.anchorY - Se) * this.pixelRatio, L[5] = m, de = b.height, L[6] = de, ve = b.width, L[13] = ve;
					}
					var Ce = void 0;
					L.length > 24 && (Ce = L[24]);
					var we = void 0, Te = void 0, Ee = void 0;
					L.length > 16 ? (we = L[15], Te = L[16], Ee = L[17]) : (we = jr, Te = !1, Ee = !1), he && j ? ge += A : !he && !j && (ge -= A);
					for (var De = 0; d < f; d += 2) if (!(Ce && Ce[De++] < ve / this.pixelRatio)) {
						var Oe = this.calculateImageOrLabelDimensions_(b.width, b.height, c[d], c[d + 1], ve, de, p, m, pe, me, ge, _e, i, we, Te || Ee, P), ke = [
							e,
							t,
							b,
							Oe,
							fe,
							Te ? D : null,
							Ee ? O : null
						], Ae = void 0, je = void 0;
						if (s && ye) {
							var Me = f - d;
							if (!ye[Me]) {
								ye[Me] = ke;
								continue;
							}
							if (Ae = ye[Me], delete ye[Me], je = ic(Ae), s.collides(je)) continue;
						}
						s && s.collides(Oe.declutterBox) || (Ae && (s && s.insert(je), this.replayImageOrLabel_.apply(this, Ae)), s && s.insert(Oe.declutterBox), this.replayImageOrLabel_.apply(this, ke));
					}
					++l;
					break;
				case X.DRAW_CHARS:
					var Ne = L[1], Pe = L[2], Fe = L[3], Ie = L[4];
					w = L[5];
					var Le = L[6], Re = L[7], ze = L[8];
					C = L[9];
					var Be = L[10];
					x = L[11], S = L[12];
					var z = [L[13], L[13]], Ve = this.textStates[S], He = Ve.font, B = [Ve.scale[0] * Re, Ve.scale[1] * Re], Ue = void 0;
					He in this.widths_ ? Ue = this.widths_[He] : (Ue = {}, this.widths_[He] = Ue);
					var We = ms(c, Ne, Pe, 2), Ge = Math.abs(B[0]) * Vr(He, x, Ue);
					if (Ie || Ge <= We) {
						var Ke = this.textStates[S].textAlign, qe = (We - Ge) * Ys[Ke], Je = Qs(c, Ne, Pe, 2, x, qe, Le, Math.abs(B[0]), Vr, He, Ue, j ? 0 : this.viewRotation_);
						drawChars: if (Je) {
							var Ye = [], Xe = void 0, Ze = void 0, Qe = void 0, $e = void 0, et = void 0;
							if (C) for (Xe = 0, Ze = Je.length; Xe < Ze; ++Xe) {
								et = Je[Xe], Qe = et[4], $e = this.createLabel(Qe, S, "", C), p = et[2] + (B[0] < 0 ? -Be : Be), m = Fe * $e.height + (.5 - Fe) * 2 * Be * B[1] / B[0] - ze;
								var Oe = this.calculateImageOrLabelDimensions_($e.width, $e.height, et[0], et[1], $e.width, $e.height, p, m, 0, 0, et[3], z, !1, jr, !1, P);
								if (s && s.collides(Oe.declutterBox)) break drawChars;
								Ye.push([
									e,
									t,
									$e,
									Oe,
									1,
									null,
									null
								]);
							}
							if (w) for (Xe = 0, Ze = Je.length; Xe < Ze; ++Xe) {
								et = Je[Xe], Qe = et[4], $e = this.createLabel(Qe, S, w, ""), p = et[2], m = Fe * $e.height - ze;
								var Oe = this.calculateImageOrLabelDimensions_($e.width, $e.height, et[0], et[1], $e.width, $e.height, p, m, 0, 0, et[3], z, !1, jr, !1, P);
								if (s && s.collides(Oe.declutterBox)) break drawChars;
								Ye.push([
									e,
									t,
									$e,
									Oe,
									1,
									null,
									null
								]);
							}
							s && s.load(Ye.map(ic));
							for (var tt = 0, nt = Ye.length; tt < nt; ++tt) this.replayImageOrLabel_.apply(this, Ye[tt]);
						}
					}
					++l;
					break;
				case X.END_GEOMETRY:
					if (a !== void 0) {
						P = L[1];
						var rt = a(P, I);
						if (rt) return rt;
					}
					++l;
					break;
				case X.FILL:
					N ? T++ : this.fill_(e), ++l;
					break;
				case X.MOVE_TO_LINE_TO:
					for (d = L[1], f = L[2], F = c[d], ee = c[d + 1], v = F + .5 | 0, y = ee + .5 | 0, (v !== h || y !== _) && (e.moveTo(F, ee), h = v, _ = y), d += 2; d < f; d += 2) F = c[d], ee = c[d + 1], v = F + .5 | 0, y = ee + .5 | 0, (d == f - 2 || v !== h || y !== _) && (e.lineTo(F, ee), h = v, _ = y);
					++l;
					break;
				case X.SET_FILL_STYLE:
					D = L, this.alignFill_ = L[2], T && (this.fill_(e), T = 0, E &&= (e.stroke(), 0)), e.fillStyle = L[1], ++l;
					break;
				case X.SET_STROKE_STYLE:
					O = L, E &&= (e.stroke(), 0), this.setStrokeStyle_(e, L), ++l;
					break;
				case X.STROKE:
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
}(), cc = [
	Z.POLYGON,
	Z.CIRCLE,
	Z.LINE_STRING,
	Z.IMAGE,
	Z.TEXT,
	Z.DEFAULT
], lc = function() {
	function e(e, t, n, r, i, a) {
		this.maxExtent_ = e, this.overlaps_ = r, this.pixelRatio_ = n, this.resolution_ = t, this.renderBuffer_ = a, this.executorsByZIndex_ = {}, this.hitDetectionContext_ = null, this.hitDetectionTransform_ = zn(), this.createExecutors_(i);
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
				n[i] = new sc(this.resolution_, this.pixelRatio_, this.overlaps_, a);
			}
		}
	}, e.prototype.hasExecutors = function(e) {
		for (var t in this.executorsByZIndex_) for (var n = this.executorsByZIndex_[t], r = 0, i = e.length; r < i; ++r) if (e[r] in n) return !0;
		return !1;
	}, e.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i, a) {
		r = Math.round(r);
		var o = r * 2 + 1, s = Jn(this.hitDetectionTransform_, r + .5, r + .5, 1 / t, -1 / t, -n, -e[0], -e[1]), c = !this.hitDetectionContext_;
		c && (this.hitDetectionContext_ = fe(o, o));
		var l = this.hitDetectionContext_;
		l.canvas.width !== o || l.canvas.height !== o ? (l.canvas.width = o, l.canvas.height = o) : c || l.clearRect(0, 0, o, o);
		var u;
		this.renderBuffer_ !== void 0 && (u = jt(), Rt(u, e), wt(u, t * (this.renderBuffer_ + r), u));
		var d = dc(r), p;
		function m(e, t) {
			for (var n = l.getImageData(0, 0, o, o).data, s = 0, c = d.length; s < c; s++) if (n[d[s]] > 0) {
				if (!a || p !== Z.IMAGE && p !== Z.TEXT || a.indexOf(e) !== -1) {
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
			for (v = this.executorsByZIndex_[x], _ = cc.length - 1; _ >= 0; --_) if (p = cc[_], y = v[p], y !== void 0 && (b = y.executeHitDetection(l, s, n, m, u), b)) return b;
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
		return Pi(o, 0, 8, 2, e, o), o;
	}, e.prototype.isEmpty = function() {
		return T(this.executorsByZIndex_);
	}, e.prototype.execute = function(e, t, n, r, i, a, o) {
		var s = Object.keys(this.executorsByZIndex_).map(Number);
		s.sort(f), this.maxExtent_ && (e.save(), this.clip(e, n));
		var c = a || cc, l, u, d, p, m, h;
		for (o && s.reverse(), l = 0, u = s.length; l < u; ++l) {
			var g = s[l].toString();
			for (m = this.executorsByZIndex_[g], d = 0, p = c.length; d < p; ++d) {
				var _ = c[d];
				h = m[_], h !== void 0 && h.execute(e, t, n, r, i, o);
			}
		}
		this.maxExtent_ && e.restore();
	}, e;
}(), uc = {};
function dc(e) {
	if (uc[e] !== void 0) return uc[e];
	for (var t = e * 2 + 1, n = e * e, r = Array(n + 1), i = 0; i <= e; ++i) for (var a = 0; a <= e; ++a) {
		var o = i * i + a * a;
		if (o > n) break;
		var s = r[o];
		s || (s = [], r[o] = s), s.push(((e + i) * t + (e + a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e + a)) * 4 + 3), a > 0 && (s.push(((e + i) * t + (e - a)) * 4 + 3), i > 0 && s.push(((e - i) * t + (e - a)) * 4 + 3));
	}
	for (var c = [], i = 0, l = r.length; i < l; ++i) r[i] && c.push.apply(c, r[i]);
	return uc[e] = c, c;
}
//#endregion
//#region node_modules/ol/render/canvas/Immediate.js
var fc = (function() {
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
})(), pc = function(e) {
	fc(t, e);
	function t(t, n, r, i, a, o, s) {
		var c = e.call(this) || this;
		return c.context_ = t, c.pixelRatio_ = n, c.extent_ = r, c.transform_ = i, c.viewRotation_ = a, c.squaredTolerance_ = o, c.userTransform_ = s, c.contextFillState_ = null, c.contextStrokeState_ = null, c.contextTextState_ = null, c.fillState_ = null, c.strokeState_ = null, c.image_ = null, c.imageAnchorX_ = 0, c.imageAnchorY_ = 0, c.imageHeight_ = 0, c.imageOpacity_ = 0, c.imageOriginX_ = 0, c.imageOriginY_ = 0, c.imageRotateWithView_ = !1, c.imageRotation_ = 0, c.imageScale_ = [0, 0], c.imageWidth_ = 0, c.text_ = "", c.textOffsetX_ = 0, c.textOffsetY_ = 0, c.textRotateWithView_ = !1, c.textRotation_ = 0, c.textScale_ = [0, 0], c.textFillState_ = null, c.textStrokeState_ = null, c.textState_ = null, c.pixelCoordinates_ = [], c.tmpLocalTransform_ = zn(), c;
	}
	return t.prototype.drawImages_ = function(e, t, n, r) {
		if (this.image_) {
			var i = Pi(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.tmpLocalTransform_, s = a.globalAlpha;
			this.imageOpacity_ != 1 && (a.globalAlpha = s * this.imageOpacity_);
			var c = this.imageRotation_;
			this.imageRotateWithView_ && (c += this.viewRotation_);
			for (var l = 0, u = i.length; l < u; l += 2) {
				var d = i[l] - this.imageAnchorX_, f = i[l + 1] - this.imageAnchorY_;
				if (c !== 0 || this.imageScale_[0] != 1 || this.imageScale_[1] != 1) {
					var p = d + this.imageAnchorX_, m = f + this.imageAnchorY_;
					Jn(o, p, m, 1, 1, c, -p, -m), a.setTransform.apply(a, o), a.translate(p, m), a.scale(this.imageScale_[0], this.imageScale_[1]), a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, -this.imageAnchorX_, -this.imageAnchorY_, this.imageWidth_, this.imageHeight_), a.setTransform(1, 0, 0, 1, 0, 0);
				} else a.drawImage(this.image_, this.imageOriginX_, this.imageOriginY_, this.imageWidth_, this.imageHeight_, d, f, this.imageWidth_, this.imageHeight_);
			}
			this.imageOpacity_ != 1 && (a.globalAlpha = s);
		}
	}, t.prototype.drawText_ = function(e, t, n, r) {
		if (!(!this.textState_ || this.text_ === "")) {
			this.textFillState_ && this.setContextFillState_(this.textFillState_), this.textStrokeState_ && this.setContextStrokeState_(this.textStrokeState_), this.setContextTextState_(this.textState_);
			var i = Pi(e, t, n, r, this.transform_, this.pixelCoordinates_), a = this.context_, o = this.textRotation_;
			for (this.textRotateWithView_ && (o += this.viewRotation_); t < n; t += r) {
				var s = i[t] + this.textOffsetX_, c = i[t + 1] + this.textOffsetY_;
				if (o !== 0 || this.textScale_[0] != 1 || this.textScale_[1] != 1) {
					var l = Jn(this.tmpLocalTransform_, s, c, 1, 1, o, -s, -c);
					a.setTransform.apply(a, l), a.translate(s, c), a.scale(this.textScale_[0], this.textScale_[1]), this.textStrokeState_ && a.strokeText(this.text_, 0, 0), this.textFillState_ && a.fillText(this.text_, 0, 0), a.setTransform(1, 0, 0, 1, 0, 0);
				} else this.textStrokeState_ && a.strokeText(this.text_, s, c), this.textFillState_ && a.fillText(this.text_, s, c);
			}
		}
	}, t.prototype.moveToLineTo_ = function(e, t, n, r, i) {
		var a = this.context_, o = Pi(e, t, n, r, this.transform_, this.pixelCoordinates_);
		a.moveTo(o[0], o[1]);
		var s = o.length;
		i && (s -= 2);
		for (var c = 2; c < s; c += 2) a.lineTo(o[c], o[c + 1]);
		return i && a.closePath(), n;
	}, t.prototype.drawRings_ = function(e, t, n, r) {
		for (var i = 0, a = n.length; i < a; ++i) t = this.moveToLineTo_(e, t, n[i], r, !0);
		return t;
	}, t.prototype.drawCircle = function(e) {
		if (Qt(this.extent_, e.getExtent())) {
			if (this.fillState_ || this.strokeState_) {
				this.fillState_ && this.setContextFillState_(this.fillState_), this.strokeState_ && this.setContextStrokeState_(this.strokeState_);
				var t = Gi(e, this.transform_, this.pixelCoordinates_), n = t[2] - t[0], r = t[3] - t[1], i = Math.sqrt(n * n + r * r), a = this.context_;
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
			case U.POINT:
				this.drawPoint(e);
				break;
			case U.LINE_STRING:
				this.drawLineString(e);
				break;
			case U.POLYGON:
				this.drawPolygon(e);
				break;
			case U.MULTI_POINT:
				this.drawMultiPoint(e);
				break;
			case U.MULTI_LINE_STRING:
				this.drawMultiLineString(e);
				break;
			case U.MULTI_POLYGON:
				this.drawMultiPolygon(e);
				break;
			case U.GEOMETRY_COLLECTION:
				this.drawGeometryCollection(e);
				break;
			case U.CIRCLE:
				this.drawCircle(e);
				break;
			default:
		}
	}, t.prototype.drawFeature = function(e, t) {
		var n = t.getGeometryFunction()(e);
		!n || !Qt(this.extent_, n.getExtent()) || (this.setStyle(t), this.drawGeometry(n));
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), Qt(this.extent_, e.getExtent())) {
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
		if (Qt(this.extent_, t)) {
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), Qt(this.extent_, e.getExtent())) {
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
		if (this.squaredTolerance_ && (e = e.simplifyTransformed(this.squaredTolerance_, this.userTransform_)), Qt(this.extent_, e.getExtent())) {
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
		var t = this.context_, n = this.contextTextState_, r = e.textAlign ? e.textAlign : kr;
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
			this.fillState_ = { fillStyle: _s(r || wr) };
		}
		if (!t) this.strokeState_ = null;
		else {
			var i = t.getColor(), a = t.getLineCap(), o = t.getLineDash(), s = t.getLineDashOffset(), c = t.getLineJoin(), l = t.getWidth(), u = t.getMiterLimit(), d = o || Er;
			this.strokeState_ = {
				lineCap: a === void 0 ? Tr : a,
				lineDash: this.pixelRatio_ === 1 ? d : d.map(function(e) {
					return e * n.pixelRatio_;
				}),
				lineDashOffset: (s || 0) * this.pixelRatio_,
				lineJoin: c === void 0 ? Dr : c,
				lineWidth: (l === void 0 ? 1 : l) * this.pixelRatio_,
				miterLimit: u === void 0 ? 10 : u,
				strokeStyle: _s(i || Or)
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
				this.textFillState_ = { fillStyle: _s(n || wr) };
			}
			var r = e.getStroke();
			if (!r) this.textStrokeState_ = null;
			else {
				var i = r.getColor(), a = r.getLineCap(), o = r.getLineDash(), s = r.getLineDashOffset(), c = r.getLineJoin(), l = r.getWidth(), u = r.getMiterLimit();
				this.textStrokeState_ = {
					lineCap: a === void 0 ? Tr : a,
					lineDash: o || Er,
					lineDashOffset: s || 0,
					lineJoin: c === void 0 ? Dr : c,
					lineWidth: l === void 0 ? 1 : l,
					miterLimit: u === void 0 ? 10 : u,
					strokeStyle: _s(i || Or)
				};
			}
			var d = e.getFont(), f = e.getOffsetX(), p = e.getOffsetY(), m = e.getRotateWithView(), h = e.getRotation(), g = e.getScaleArray(), _ = e.getText(), v = e.getTextAlign(), y = e.getTextBaseline();
			this.textState_ = {
				font: d === void 0 ? Cr : d,
				textAlign: v === void 0 ? kr : v,
				textBaseline: y === void 0 ? Ar : y
			}, this.text_ = _ === void 0 ? "" : _, this.textOffsetX_ = f === void 0 ? 0 : this.pixelRatio_ * f, this.textOffsetY_ = p === void 0 ? 0 : this.pixelRatio_ * p, this.textRotateWithView_ = m !== void 0 && m, this.textRotation_ = h === void 0 ? 0 : h, this.textScale_ = [this.pixelRatio_ * g[0], this.pixelRatio_ * g[1]];
		}
	}, t;
}(Ls), mc = {
	FRACTION: "fraction",
	PIXELS: "pixels"
}, hc = {
	BOTTOM_LEFT: "bottom-left",
	BOTTOM_RIGHT: "bottom-right",
	TOP_LEFT: "top-left",
	TOP_RIGHT: "top-right"
}, gc = (function() {
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
})(), _c = function(e) {
	gc(t, e);
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
}(D), vc = (function() {
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
	vc(t, e);
	function t(t, n, r, i, a, o) {
		var s = e.call(this, t, n, r, Y.IDLE) || this;
		return s.src_ = i, s.image_ = new Image(), a !== null && (s.image_.crossOrigin = a), s.unlisten_ = null, s.state = Y.IDLE, s.imageLoadFunction_ = o, s;
	}
	return t.prototype.getImage = function() {
		return this.image_;
	}, t.prototype.handleImageError_ = function() {
		this.state = Y.ERROR, this.unlistenImage_(), this.changed();
	}, t.prototype.handleImageLoad_ = function() {
		this.resolution === void 0 && (this.resolution = Jt(this.extent) / this.image_.height), this.state = Y.LOADED, this.unlistenImage_(), this.changed();
	}, t.prototype.load = function() {
		(this.state == Y.IDLE || this.state == Y.ERROR) && (this.state = Y.LOADING, this.changed(), this.imageLoadFunction_(this, this.src_), this.unlisten_ = yc(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
	}, t.prototype.setImage = function(e) {
		this.image_ = e, this.resolution = Jt(this.extent) / this.image_.height;
	}, t.prototype.unlistenImage_ = function() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}, t;
})(_c);
function yc(e, t, n) {
	var r = e;
	if (r.src && ue) {
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
var bc = (function() {
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
})(), xc = null, Sc = function(e) {
	bc(t, e);
	function t(t, n, r, i, a, o) {
		var s = e.call(this) || this;
		return s.hitDetectionImage_ = null, s.image_ = t || new Image(), i !== null && (s.image_.crossOrigin = i), s.canvas_ = {}, s.color_ = o, s.unlisten_ = null, s.imageState_ = a, s.size_ = r, s.src_ = n, s.tainted_, s;
	}
	return t.prototype.isTainted_ = function() {
		if (this.tainted_ === void 0 && this.imageState_ === Y.LOADED) {
			xc ||= fe(1, 1), xc.drawImage(this.image_, 0, 0);
			try {
				xc.getImageData(0, 0, 1, 1), this.tainted_ = !1;
			} catch {
				xc = null, this.tainted_ = !0;
			}
		}
		return this.tainted_ === !0;
	}, t.prototype.dispatchChangeEvent_ = function() {
		this.dispatchEvent(O.CHANGE);
	}, t.prototype.handleImageError_ = function() {
		this.imageState_ = Y.ERROR, this.unlistenImage_(), this.dispatchChangeEvent_();
	}, t.prototype.handleImageLoad_ = function() {
		this.imageState_ = Y.LOADED, this.size_ ? (this.image_.width = this.size_[0], this.image_.height = this.size_[1]) : this.size_ = [this.image_.width, this.image_.height], this.unlistenImage_(), this.dispatchChangeEvent_();
	}, t.prototype.getImage = function(e) {
		return this.replaceColor_(e), this.canvas_[e] ? this.canvas_[e] : this.image_;
	}, t.prototype.getPixelRatio = function(e) {
		return this.replaceColor_(e), this.canvas_[e] ? e : 1;
	}, t.prototype.getImageState = function() {
		return this.imageState_;
	}, t.prototype.getHitDetectionImage = function() {
		if (!this.hitDetectionImage_) if (this.isTainted_()) {
			var e = this.size_[0], t = this.size_[1], n = fe(e, t);
			n.fillRect(0, 0, e, t), this.hitDetectionImage_ = n.canvas;
		} else this.hitDetectionImage_ = this.image_;
		return this.hitDetectionImage_;
	}, t.prototype.getSize = function() {
		return this.size_;
	}, t.prototype.getSrc = function() {
		return this.src_;
	}, t.prototype.load = function() {
		if (this.imageState_ == Y.IDLE) {
			this.imageState_ = Y.LOADING;
			try {
				this.image_.src = this.src_;
			} catch {
				this.handleImageError_();
			}
			this.unlisten_ = yc(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this));
		}
	}, t.prototype.replaceColor_ = function(e) {
		if (!(!this.color_ || this.canvas_[e] || this.imageState_ !== Y.LOADED)) {
			var t = document.createElement("canvas");
			this.canvas_[e] = t, t.width = Math.ceil(this.image_.width * e), t.height = Math.ceil(this.image_.height * e);
			var n = t.getContext("2d");
			if (n.scale(e, e), n.drawImage(this.image_, 0, 0), n.globalCompositeOperation = "multiply", n.globalCompositeOperation === "multiply" || this.isTainted_()) n.fillStyle = tr(this.color_), n.fillRect(0, 0, t.width / e, t.height / e), n.globalCompositeOperation = "destination-in", n.drawImage(this.image_, 0, 0);
			else {
				for (var r = n.getImageData(0, 0, t.width, t.height), i = r.data, a = this.color_[0] / 255, o = this.color_[1] / 255, s = this.color_[2] / 255, c = this.color_[3], l = 0, u = i.length; l < u; l += 4) i[l] *= a, i[l + 1] *= o, i[l + 2] *= s, i[l + 3] *= c;
				n.putImageData(r, 0, 0);
			}
		}
	}, t.prototype.unlistenImage_ = function() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}, t;
}(D);
function Cc(e, t, n, r, i, a) {
	var o = ur.get(t, r, a);
	return o || (o = new Sc(e, t, n, r, i, a), ur.set(t, r, a, o)), o;
}
//#endregion
//#region node_modules/ol/style/Icon.js
var wc = (function() {
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
})(), Tc = function(e) {
	wc(t, e);
	function t(t) {
		var n = this, r = t || {}, i = r.opacity === void 0 ? 1 : r.opacity, a = r.rotation === void 0 ? 0 : r.rotation, o = r.scale === void 0 ? 1 : r.scale, s = r.rotateWithView !== void 0 && r.rotateWithView;
		n = e.call(this, {
			opacity: i,
			rotation: a,
			scale: o,
			displacement: r.displacement === void 0 ? [0, 0] : r.displacement,
			rotateWithView: s
		}) || this, n.anchor_ = r.anchor === void 0 ? [.5, .5] : r.anchor, n.normalizedAnchor_ = null, n.anchorOrigin_ = r.anchorOrigin === void 0 ? hc.TOP_LEFT : r.anchorOrigin, n.anchorXUnits_ = r.anchorXUnits === void 0 ? mc.FRACTION : r.anchorXUnits, n.anchorYUnits_ = r.anchorYUnits === void 0 ? mc.FRACTION : r.anchorYUnits, n.crossOrigin_ = r.crossOrigin === void 0 ? null : r.crossOrigin;
		var c = r.img === void 0 ? null : r.img, l = r.imgSize === void 0 ? null : r.imgSize, u = r.src;
		V(!(u !== void 0 && c), 4), V(!c || c && l, 5), (u === void 0 || u.length === 0) && c && (u = c.src || I(c)), V(u !== void 0 && u.length > 0, 6);
		var d = r.src === void 0 ? Y.LOADED : Y.IDLE;
		return n.color_ = r.color === void 0 ? null : ir(r.color), n.iconImage_ = Cc(c, u, l, n.crossOrigin_, d, n.color_), n.offset_ = r.offset === void 0 ? [0, 0] : r.offset, n.offsetOrigin_ = r.offsetOrigin === void 0 ? hc.TOP_LEFT : r.offsetOrigin, n.origin_ = null, n.size_ = r.size === void 0 ? null : r.size, n;
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
		if (this.anchorXUnits_ == mc.FRACTION || this.anchorYUnits_ == mc.FRACTION) {
			if (!t) return null;
			e = this.anchor_.slice(), this.anchorXUnits_ == mc.FRACTION && (e[0] *= t[0]), this.anchorYUnits_ == mc.FRACTION && (e[1] *= t[1]);
		}
		if (this.anchorOrigin_ != hc.TOP_LEFT) {
			if (!t) return null;
			e === this.anchor_ && (e = this.anchor_.slice()), (this.anchorOrigin_ == hc.TOP_RIGHT || this.anchorOrigin_ == hc.BOTTOM_RIGHT) && (e[0] = -e[0] + t[0]), (this.anchorOrigin_ == hc.BOTTOM_LEFT || this.anchorOrigin_ == hc.BOTTOM_RIGHT) && (e[1] = -e[1] + t[1]);
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
		if (this.offsetOrigin_ != hc.TOP_LEFT) {
			var t = this.getSize(), n = this.iconImage_.getSize();
			if (!t || !n) return null;
			e = e.slice(), (this.offsetOrigin_ == hc.TOP_RIGHT || this.offsetOrigin_ == hc.BOTTOM_RIGHT) && (e[0] = n[0] - t[0] - e[0]), (this.offsetOrigin_ == hc.BOTTOM_LEFT || this.offsetOrigin_ == hc.BOTTOM_RIGHT) && (e[1] = n[1] - t[1] - e[1]);
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
}(gs), Ec = "#333", Dc = function() {
	function e(e) {
		var t = e || {};
		this.font_ = t.font, this.rotation_ = t.rotation, this.rotateWithView_ = t.rotateWithView, this.scale_ = t.scale, this.scaleArray_ = za(t.scale === void 0 ? 1 : t.scale), this.text_ = t.text, this.textAlign_ = t.textAlign, this.textBaseline_ = t.textBaseline, this.fill_ = t.fill === void 0 ? new Ss({ color: Ec }) : t.fill, this.maxAngle_ = t.maxAngle === void 0 ? Math.PI / 4 : t.maxAngle, this.placement_ = t.placement === void 0 ? Ks.POINT : t.placement, this.overflow_ = !!t.overflow, this.stroke_ = t.stroke === void 0 ? null : t.stroke, this.offsetX_ = t.offsetX === void 0 ? 0 : t.offsetX, this.offsetY_ = t.offsetY === void 0 ? 0 : t.offsetY, this.backgroundFill_ = t.backgroundFill ? t.backgroundFill : null, this.backgroundStroke_ = t.backgroundStroke ? t.backgroundStroke : null, this.padding_ = t.padding === void 0 ? null : t.padding;
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
		this.scale_ = e, this.scaleArray_ = za(e === void 0 ? 1 : e);
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
}(), Oc = .5;
function kc(e, t, n, r, i, a, o) {
	var s = fe(e[0] * Oc, e[1] * Oc);
	s.imageSmoothingEnabled = !1;
	for (var c = s.canvas, l = new pc(s, Oc, i, null, o), u = n.length, d = Math.floor((256 * 256 * 256 - 1) / u), p = {}, m = 1; m <= u; ++m) {
		var h = n[m - 1], g = h.getStyleFunction() || r;
		if (r) {
			var _ = g(h, a);
			if (_) {
				Array.isArray(_) || (_ = [_]);
				for (var v = "#" + ("000000" + (m * d).toString(16)).slice(-6), y = 0, b = _.length; y < b; ++y) {
					var x = _[y], S = x.getGeometryFunction()(h);
					if (!(!S || !Qt(i, S.getExtent()))) {
						var C = x.clone(), w = C.getFill();
						w && w.setColor(v);
						var T = C.getStroke();
						T && (T.setColor(v), T.setLineDash(null)), C.setText(void 0);
						var E = x.getImage();
						if (E && E.getOpacity() !== 0) {
							var D = E.getImageSize();
							if (!D) continue;
							var O = fe(D[0], D[1], void 0, { alpha: !1 }), k = O.canvas;
							O.fillStyle = v, O.fillRect(0, 0, k.width, k.height), C.setImage(new Tc({
								img: k,
								imgSize: D,
								anchor: E.getAnchor(),
								anchorXUnits: mc.PIXELS,
								anchorYUnits: mc.PIXELS,
								offset: E.getOrigin(),
								opacity: 1,
								size: E.getSize(),
								scale: E.getScale(),
								rotation: E.getRotation(),
								rotateWithView: E.getRotateWithView()
							}));
						}
						var A = C.getZIndex() || 0, j = p[A];
						j || (j = {}, p[A] = j, j[U.POLYGON] = [], j[U.CIRCLE] = [], j[U.LINE_STRING] = [], j[U.POINT] = []), j[S.getType().replace("Multi", "")].push(S, C);
					}
				}
			}
		}
	}
	for (var M = Object.keys(p).map(Number).sort(f), m = 0, N = M.length; m < N; ++m) {
		var j = p[M[m]];
		for (var P in j) for (var F = j[P], y = 0, b = F.length; y < b; y += 2) {
			l.setStyle(F[y + 1]);
			for (var ee = 0, I = t.length; ee < I; ++ee) l.setTransform(t[ee]), l.drawGeometry(F[y]);
		}
	}
	return s.getImageData(0, 0, c.width, c.height);
}
function Ac(e, t, n) {
	var r = [];
	if (n) {
		var i = Math.floor(Math.round(e[0]) * Oc), a = Math.floor(Math.round(e[1]) * Oc), o = (B(i, 0, n.width - 1) + B(a, 0, n.height - 1) * n.width) * 4, s = n.data[o], c = n.data[o + 1], l = n.data[o + 2] + 256 * (c + 256 * s), u = Math.floor((256 * 256 * 256 - 1) / t.length);
		l && l % u === 0 && r.push(t[l / u - 1]);
	}
	return r;
}
//#endregion
//#region node_modules/ol/renderer/vector.js
var jc = .5, Mc = {
	Point: Wc,
	LineString: Vc,
	Polygon: Kc,
	MultiPoint: Gc,
	MultiLineString: Hc,
	MultiPolygon: Uc,
	GeometryCollection: Bc,
	Circle: Ic
};
function Nc(e, t) {
	return parseInt(I(e), 10) - parseInt(I(t), 10);
}
function Pc(e, t) {
	var n = Fc(e, t);
	return n * n;
}
function Fc(e, t) {
	return jc * e / t;
}
function Ic(e, t, n, r, i) {
	var a = n.getFill(), o = n.getStroke();
	if (a || o) {
		var s = e.getBuilder(n.getZIndex(), Z.CIRCLE);
		s.setFillStrokeStyle(a, o), s.drawCircle(t, r);
	}
	var c = n.getText();
	if (c && c.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Z.TEXT);
		l.setTextStyle(c), l.drawText(t, r);
	}
}
function Lc(e, t, n, r, i, a, o) {
	var s = !1, c = n.getImage();
	if (c) {
		var l = c.getImageState();
		l == Y.LOADED || l == Y.ERROR ? c.unlistenImageChange(i) : (l == Y.IDLE && c.load(), l = c.getImageState(), c.listenImageChange(i), s = !0);
	}
	return Rc(e, t, n, r, a, o), s;
}
function Rc(e, t, n, r, i, a) {
	var o = n.getGeometryFunction()(t);
	if (o) {
		var s = o.simplifyTransformed(r, i);
		if (n.getRenderer()) zc(e, s, n, t);
		else {
			var c = Mc[s.getType()];
			c(e, s, n, t, a);
		}
	}
}
function zc(e, t, n, r) {
	if (t.getType() == U.GEOMETRY_COLLECTION) {
		for (var i = t.getGeometries(), a = 0, o = i.length; a < o; ++a) zc(e, i[a], n, r);
		return;
	}
	e.getBuilder(n.getZIndex(), Z.DEFAULT).drawCustom(t, r, n.getRenderer(), n.getHitDetectionRenderer());
}
function Bc(e, t, n, r, i) {
	var a = t.getGeometriesArray(), o, s;
	for (o = 0, s = a.length; o < s; ++o) {
		var c = Mc[a[o].getType()];
		c(e, a[o], n, r, i);
	}
}
function Vc(e, t, n, r, i) {
	var a = n.getStroke();
	if (a) {
		var o = e.getBuilder(n.getZIndex(), Z.LINE_STRING);
		o.setFillStrokeStyle(null, a), o.drawLineString(t, r);
	}
	var s = n.getText();
	if (s && s.getText()) {
		var c = (i || e).getBuilder(n.getZIndex(), Z.TEXT);
		c.setTextStyle(s), c.drawText(t, r);
	}
}
function Hc(e, t, n, r, i) {
	var a = n.getStroke();
	if (a) {
		var o = e.getBuilder(n.getZIndex(), Z.LINE_STRING);
		o.setFillStrokeStyle(null, a), o.drawMultiLineString(t, r);
	}
	var s = n.getText();
	if (s && s.getText()) {
		var c = (i || e).getBuilder(n.getZIndex(), Z.TEXT);
		c.setTextStyle(s), c.drawText(t, r);
	}
}
function Uc(e, t, n, r, i) {
	var a = n.getFill(), o = n.getStroke();
	if (o || a) {
		var s = e.getBuilder(n.getZIndex(), Z.POLYGON);
		s.setFillStrokeStyle(a, o), s.drawMultiPolygon(t, r);
	}
	var c = n.getText();
	if (c && c.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Z.TEXT);
		l.setTextStyle(c), l.drawText(t, r);
	}
}
function Wc(e, t, n, r, i) {
	var a = n.getImage(), o = n.getText(), s;
	if (i && (e = i, s = a && o && o.getText() ? {} : void 0), a) {
		if (a.getImageState() != Y.LOADED) return;
		var c = e.getBuilder(n.getZIndex(), Z.IMAGE);
		c.setImageStyle(a, s), c.drawPoint(t, r);
	}
	if (o && o.getText()) {
		var l = e.getBuilder(n.getZIndex(), Z.TEXT);
		l.setTextStyle(o, s), l.drawText(t, r);
	}
}
function Gc(e, t, n, r, i) {
	var a = n.getImage(), o = n.getText(), s;
	if (i && (e = i, s = a && o && o.getText() ? {} : void 0), a) {
		if (a.getImageState() != Y.LOADED) return;
		var c = e.getBuilder(n.getZIndex(), Z.IMAGE);
		c.setImageStyle(a, s), c.drawMultiPoint(t, r);
	}
	if (o && o.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Z.TEXT);
		l.setTextStyle(o, s), l.drawText(t, r);
	}
}
function Kc(e, t, n, r, i) {
	var a = n.getFill(), o = n.getStroke();
	if (a || o) {
		var s = e.getBuilder(n.getZIndex(), Z.POLYGON);
		s.setFillStrokeStyle(a, o), s.drawPolygon(t, r);
	}
	var c = n.getText();
	if (c && c.getText()) {
		var l = (i || e).getBuilder(n.getZIndex(), Z.TEXT);
		l.setTextStyle(c), l.drawText(t, r);
	}
}
//#endregion
//#region node_modules/ol/renderer/canvas/VectorLayer.js
var qc = (function() {
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
})(), Jc = function(e) {
	qc(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.boundHandleStyleImageChange_ = n.handleStyleImageChange_.bind(n), n.animatingOrInteracting_, n.dirty_ = !1, n.hitDetectionImageData_ = null, n.renderedFeatures_ = null, n.renderedRevision_ = -1, n.renderedResolution_ = NaN, n.renderedExtent_ = jt(), n.wrappedRenderedExtent_ = jt(), n.renderedRotation_, n.renderedCenter_ = null, n.renderedProjection_ = null, n.renderedRenderOrder_ = null, n.replayGroup_ = null, n.replayGroupChanged = !0, n.declutterExecutorGroup = null, n.clipping = !0, n;
	}
	return t.prototype.useContainer = function(t, n, r) {
		r < 1 && (t = null), e.prototype.useContainer.call(this, t, n, r);
	}, t.prototype.renderWorlds = function(e, t, n) {
		var r = t.extent, i = t.viewState, a = i.center, o = i.resolution, s = i.projection, c = i.rotation, l = s.getExtent(), u = this.getLayer().getSource(), d = t.pixelRatio, f = t.viewHints, p = !(f[J.ANIMATING] || f[J.INTERACTING]), m = this.context, h = Math.round(t.size[0] * d), g = Math.round(t.size[1] * d), _ = u.getWrapX() && s.canWrapX(), v = _ ? H(l) : null, y = _ ? Math.ceil((r[2] - l[2]) / v) + 1 : 1, b = _ ? Math.floor((r[0] - l[0]) / v) : 0;
		do {
			var x = this.getRenderTransform(a, o, c, d, h, g, b * v);
			e.execute(m, 1, x, c, p, void 0, n);
		} while (++b < y);
	}, t.prototype.renderDeclutter = function(e) {
		this.declutterExecutorGroup && this.renderWorlds(this.declutterExecutorGroup, e, e.declutterTree);
	}, t.prototype.renderFrame = function(e, t) {
		var n = e.pixelRatio, r = e.layerStatesArray[e.layerIndex];
		Kn(this.pixelTransform, 1 / n, 1 / n), Yn(this.inversePixelTransform, this.pixelTransform);
		var i = Qn(this.pixelTransform);
		this.useContainer(t, i, r.opacity);
		var a = this.context, o = a.canvas, s = this.replayGroup_, c = this.declutterExecutorGroup;
		if ((!s || s.isEmpty()) && (!c || c.isEmpty())) return null;
		var l = Math.round(e.size[0] * n), u = Math.round(e.size[1] * n);
		o.width != l || o.height != u ? (o.width = l, o.height = u, o.style.transform !== i && (o.style.transform = i)) : this.containerReused || a.clearRect(0, 0, l, u), this.preRender(a, e);
		var d = e.viewState, f = d.projection, p = !1, m = !0;
		if (r.extent && this.clipping) {
			var h = jn(r.extent, f);
			m = Qt(h, e.extent), p = m && !Ot(h, e.extent), p && this.clipUnrotated(a, e, h);
		}
		m && this.renderWorlds(s, e), p && a.restore(), this.postRender(a, e);
		var g = Ae(r.opacity), _ = this.container;
		return g !== _.style.opacity && (_.style.opacity = g), this.renderedRotation_ !== d.rotation && (this.renderedRotation_ = d.rotation, this.hitDetectionImageData_ = null), this.container;
	}, t.prototype.getFeatures = function(e) {
		return new Promise(function(t) {
			if (!this.hitDetectionImageData_ && !this.animatingOrInteracting_) {
				var n = [this.context.canvas.width, this.context.canvas.height];
				W(this.pixelTransform, n);
				var r = this.renderedCenter_, i = this.renderedResolution_, a = this.renderedRotation_, o = this.renderedProjection_, s = this.wrappedRenderedExtent_, c = this.getLayer(), l = [], u = n[0] * Oc, d = n[1] * Oc;
				l.push(this.getRenderTransform(r, i, a, Oc, u, d, 0).slice());
				var f = c.getSource(), p = o.getExtent();
				if (f.getWrapX() && o.canWrapX() && !Ot(p, s)) {
					for (var m = s[0], h = H(p), g = 0, _ = void 0; m < p[0];) --g, _ = h * g, l.push(this.getRenderTransform(r, i, a, Oc, u, d, _).slice()), m += h;
					for (g = 0, m = s[2]; m > p[2];) ++g, _ = h * g, l.push(this.getRenderTransform(r, i, a, Oc, u, d, _).slice()), m -= h;
				}
				this.hitDetectionImageData_ = kc(n, l, this.renderedFeatures_, c.getStyleFunction(), s, i, a);
			}
			t(Ac(e, this.renderedFeatures_, this.hitDetectionImageData_));
		}.bind(this));
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {
		var a = this;
		if (this.replayGroup_) {
			var o = t.viewState.resolution, s = t.viewState.rotation, c = this.getLayer(), l = {}, u = function(e, t, n) {
				var a = I(e), o = l[a];
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
		var r = e.viewHints[J.ANIMATING], i = e.viewHints[J.INTERACTING], a = t.getUpdateWhileAnimating(), o = t.getUpdateWhileInteracting();
		if (!this.dirty_ && !a && r || !o && i) return this.animatingOrInteracting_ = !0, !0;
		this.animatingOrInteracting_ = !1;
		var s = e.extent, c = e.viewState, l = c.projection, u = c.resolution, d = e.pixelRatio, f = t.getRevision(), p = t.getRenderBuffer(), m = t.getRenderOrder();
		m === void 0 && (m = Nc);
		var h = c.center.slice(), _ = wt(s, p * u), v = _.slice(), y = [_.slice()], b = l.getExtent();
		if (n.getWrapX() && l.canWrapX() && !Ot(b, e.extent)) {
			var x = H(b), S = Math.max(H(_) / 2, x);
			_[0] = b[0] - S, _[2] = b[2] + S, dn(h, l);
			var C = rn(y[0], l);
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
		if (!this.dirty_ && this.renderedResolution_ == u && this.renderedRevision_ == f && this.renderedRenderOrder_ == m && Ot(this.wrappedRenderedExtent_, _)) return g(this.renderedExtent_, v) || (this.hitDetectionImageData_ = null, this.renderedExtent_ = v), this.renderedCenter_ = h, this.replayGroupChanged = !1, !0;
		this.replayGroup_ = null, this.dirty_ = !1;
		var w = new Zs(Fc(u, d), _, u, d), T;
		this.getLayer().getDeclutter() && (T = new Zs(Fc(u, d), _, u, d));
		var E = Dn(), D;
		if (E) {
			for (var O = 0, k = y.length; O < k; ++O) {
				var A = y[O], j = An(A, l);
				n.loadFeatures(j, Mn(u, l), E);
			}
			D = Cn(E, l);
		} else for (var O = 0, k = y.length; O < k; ++O) n.loadFeatures(y[O], u, l);
		var M = Pc(u, d), N = function(e) {
			var n, r = e.getStyleFunction() || t.getStyleFunction();
			if (r && (n = r(e, u)), n) {
				var i = this.renderFeature(e, M, n, w, D, T);
				this.dirty_ = this.dirty_ || i;
			}
		}.bind(this), P = An(_, l), F = n.getFeaturesInExtent(P);
		m && F.sort(m);
		for (var O = 0, k = F.length; O < k; ++O) N(F[O]);
		this.renderedFeatures_ = F;
		var ee = w.finish(), I = new lc(_, u, d, n.getOverlaps(), ee, t.getRenderBuffer());
		return T && (this.declutterExecutorGroup = new lc(_, u, d, n.getOverlaps(), T.finish(), t.getRenderBuffer())), this.renderedResolution_ = u, this.renderedRevision_ = f, this.renderedRenderOrder_ = m, this.renderedExtent_ = v, this.wrappedRenderedExtent_ = _, this.renderedCenter_ = h, this.renderedProjection_ = l, this.replayGroup_ = I, this.hitDetectionImageData_ = null, this.replayGroupChanged = !0, !0;
	}, t.prototype.renderFeature = function(e, t, n, r, i, a) {
		if (!n) return !1;
		var o = !1;
		if (Array.isArray(n)) for (var s = 0, c = n.length; s < c; ++s) o = Lc(r, e, n[s], t, this.boundHandleStyleImageChange_, i, a) || o;
		else o = Lc(r, e, n, t, this.boundHandleStyleImageChange_, i, a);
		return o;
	}, t;
}(co), Yc = (function() {
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
})(), Xc = function(e) {
	Yc(t, e);
	function t(t) {
		return e.call(this, t) || this;
	}
	return t.prototype.createRenderer = function() {
		return new Jc(this);
	}, t;
}(Ms), Zc = (function() {
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
})(), Qc = function(e) {
	Zc(t, e);
	function t(t) {
		var n = e.call(this) || this;
		n.projection = _n(t.projection), n.attributions_ = $c(t.attributions), n.attributionsCollapsible_ = t.attributionsCollapsible === void 0 || t.attributionsCollapsible, n.loading = !1, n.state_ = t.state === void 0 ? mr.READY : t.state, n.wrapX_ = t.wrapX !== void 0 && t.wrapX, n.viewResolver = null, n.viewRejector = null;
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
		this.attributions_ = $c(e), this.changed();
	}, t.prototype.setState = function(e) {
		this.state_ = e, this.changed();
	}, t;
}(R);
function $c(e) {
	return e ? Array.isArray(e) ? function(t) {
		return e;
	} : typeof e == "function" ? e : function(t) {
		return [e];
	} : null;
}
//#endregion
//#region node_modules/ol/source/VectorEventType.js
var el = {
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
function tl(e) {
	var t = e || {}, n = new ni(), r = new os(-.005, .05, 100);
	return (t.altShiftDragRotate === void 0 || t.altShiftDragRotate) && n.push(new Ko()), (t.doubleClickZoom === void 0 || t.doubleClickZoom) && n.push(new Do({
		delta: t.zoomDelta,
		duration: t.zoomDuration
	})), (t.dragPan === void 0 || t.dragPan) && n.push(new Wo({
		onFocusOnly: t.onFocusOnly,
		kinetic: r
	})), (t.pinchRotate === void 0 || t.pinchRotate) && n.push(new ds()), (t.pinchZoom === void 0 || t.pinchZoom) && n.push(new ps({ duration: t.zoomDuration })), (t.keyboard === void 0 || t.keyboard) && (n.push(new rs()), n.push(new as({
		delta: t.zoomDelta,
		duration: t.zoomDuration
	}))), (t.mouseWheelZoom === void 0 || t.mouseWheelZoom) && n.push(new ls({
		onFocusOnly: t.onFocusOnly,
		duration: t.zoomDuration
	})), (t.shiftDragZoom === void 0 || t.shiftDragZoom) && n.push(new es({ duration: t.zoomDuration })), n;
}
//#endregion
//#region node_modules/ol/Map.js
var nl = (function() {
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
})(), rl = function(e) {
	nl(t, e);
	function t(t) {
		var n = this;
		return t = S({}, t), t.controls ||= bo(), t.interactions ||= tl({ onFocusOnly: !0 }), n = e.call(this, t) || this, n;
	}
	return t.prototype.createRenderer = function() {
		return new Kr(this);
	}, t;
}(Va), il = (function() {
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
})(), al = function(e) {
	il(t, e);
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
			if (e.getState() == q.LOADED) return this.transition_ = 0, e;
			e = e.interimTile;
		} while (e);
		return this;
	}, t.prototype.refreshInterimChain = function() {
		if (this.interimTile) {
			var e = this.interimTile, t = this;
			do {
				if (e.getState() == q.LOADED) {
					e.interimTile = null;
					break;
				} else e.getState() == q.LOADING ? t = e : e.getState() == q.IDLE ? t.interimTile = e.interimTile : t = e;
				e = t.interimTile;
			} while (e);
		}
	}, t.prototype.getTileCoord = function() {
		return this.tileCoord;
	}, t.prototype.getState = function() {
		return this.state;
	}, t.prototype.setState = function(e) {
		if (this.state !== q.ERROR && this.state > e) throw Error("Tile load sequence violation");
		this.state = e, this.changed();
	}, t.prototype.load = function() {
		F();
	}, t.prototype.getAlpha = function(e, t) {
		if (!this.transition_) return 1;
		var n = this.transitionStarts_[e];
		if (!n) n = t, this.transitionStarts_[e] = n;
		else if (n === -1) return 1;
		var r = t - n + 1e3 / 60;
		return r >= this.transition_ ? 1 : ki(r / this.transition_);
	}, t.prototype.inTransition = function(e) {
		return this.transition_ ? this.transitionStarts_[e] !== -1 : !1;
	}, t.prototype.endTransition = function(e) {
		this.transition_ && (this.transitionStarts_[e] = -1);
	}, t;
}(D), sl = (function() {
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
})(), cl = function(e) {
	sl(t, e);
	function t(t, n, r, i, a, o) {
		var s = e.call(this, t, n, o) || this;
		return s.crossOrigin_ = i, s.src_ = r, s.key = r, s.image_ = new Image(), i !== null && (s.image_.crossOrigin = i), s.unlisten_ = null, s.tileLoadFunction_ = a, s;
	}
	return t.prototype.getImage = function() {
		return this.image_;
	}, t.prototype.setImage = function(e) {
		this.image_ = e, this.state = q.LOADED, this.unlistenImage_(), this.changed();
	}, t.prototype.handleImageError_ = function() {
		this.state = q.ERROR, this.unlistenImage_(), this.image_ = ll(), this.changed();
	}, t.prototype.handleImageLoad_ = function() {
		var e = this.image_;
		e.naturalWidth && e.naturalHeight ? this.state = q.LOADED : this.state = q.EMPTY, this.unlistenImage_(), this.changed();
	}, t.prototype.load = function() {
		this.state == q.ERROR && (this.state = q.IDLE, this.image_ = new Image(), this.crossOrigin_ !== null && (this.image_.crossOrigin = this.crossOrigin_)), this.state == q.IDLE && (this.state = q.LOADING, this.changed(), this.tileLoadFunction_(this, this.src_), this.unlisten_ = yc(this.image_, this.handleImageLoad_.bind(this), this.handleImageError_.bind(this)));
	}, t.prototype.unlistenImage_ = function() {
		this.unlisten_ &&= (this.unlisten_(), null);
	}, t;
}(al);
function ll() {
	var e = fe(1, 1);
	return e.fillStyle = "rgba(0,0,0,0)", e.fillRect(0, 0, 1, 1), e.canvas;
}
//#endregion
//#region node_modules/ol/tilecoord.js
function ul(e, t, n, r) {
	return r === void 0 ? [
		e,
		t,
		n
	] : (r[0] = e, r[1] = t, r[2] = n, r);
}
function dl(e, t, n) {
	return e + "/" + t + "/" + n;
}
function fl(e) {
	return dl(e[0], e[1], e[2]);
}
function pl(e) {
	return e.split("/").map(Number);
}
function ml(e) {
	return (e[1] << e[0]) + e[2];
}
function hl(e, t) {
	var n = e[0], r = e[1], i = e[2];
	if (t.getMinZoom() > n || n > t.getMaxZoom()) return !1;
	var a = t.getFullTileRange(n);
	return !a || a.containsXY(r, i);
}
//#endregion
//#region node_modules/ol/tilegrid/TileGrid.js
var gl = [
	0,
	0,
	0
], _l = function() {
	function e(e) {
		this.minZoom = e.minZoom === void 0 ? 0 : e.minZoom, this.resolutions_ = e.resolutions, V(_(this.resolutions_, function(e, t) {
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
		this.zoomFactor_ = t, this.maxZoom = this.resolutions_.length - 1, this.origin_ = e.origin === void 0 ? null : e.origin, this.origins_ = null, e.origins !== void 0 && (this.origins_ = e.origins, V(this.origins_.length == this.resolutions_.length, 20));
		var i = e.extent;
		i !== void 0 && !this.origin_ && !this.origins_ && (this.origin_ = Xt(i)), V(!this.origin_ && this.origins_ || this.origin_ && !this.origins_, 18), this.tileSizes_ = null, e.tileSizes !== void 0 && (this.tileSizes_ = e.tileSizes, V(this.tileSizes_.length == this.resolutions_.length, 19)), this.tileSize_ = e.tileSize === void 0 ? this.tileSizes_ ? null : 256 : e.tileSize, V(!this.tileSize_ && this.tileSizes_ || this.tileSize_ && !this.tileSizes_, 22), this.extent_ = i === void 0 ? null : i, this.fullTileRanges_ = null, this.tmpSize_ = [0, 0], this.tmpExtent_ = [
			0,
			0,
			0,
			0
		], e.sizes === void 0 ? i && this.calculateTileRanges_(i) : this.fullTileRanges_ = e.sizes.map(function(e, t) {
			var n = new lo(Math.min(0, e[0]), Math.max(e[0] - 1, -1), Math.min(0, e[1]), Math.max(e[1] - 1, -1));
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
			if (this.zoomFactor_ === 2 ? (a = Math.floor(a / 2), o = Math.floor(o / 2), i = uo(a, a, o, o, n)) : i = this.getTileRangeForExtentAndZ(s, c, n), t(c, i)) return !0;
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
				return uo(r, r + 1, i, i + 1, t);
			}
			var a = this.getTileCoordExtent(e, n || this.tmpExtent_);
			return this.getTileRangeForExtentAndZ(a, e[0] + 1, t);
		}
		return null;
	}, e.prototype.getTileRangeForTileCoordAndZ = function(e, t, n) {
		if (t > this.maxZoom || t < this.minZoom) return null;
		var r = e[0], i = e[1], a = e[2];
		if (t === r) return uo(i, a, i, a, n);
		if (this.zoomFactor_) {
			var o = this.zoomFactor_ ** +(t - r), s = Math.floor(i * o), c = Math.floor(a * o);
			return t < r ? uo(s, s, c, c, n) : uo(s, Math.floor(o * (i + 1)) - 1, c, Math.floor(o * (a + 1)) - 1, n);
		}
		var l = this.getTileCoordExtent(e, this.tmpExtent_);
		return this.getTileRangeForExtentAndZ(l, t, n);
	}, e.prototype.getTileRangeExtent = function(e, t, n) {
		var r = this.getOrigin(e), i = this.getResolution(e), a = za(this.getTileSize(e), this.tmpSize_), o = r[0] + t.minX * a[0] * i, s = r[0] + (t.maxX + 1) * a[0] * i;
		return Mt(o, r[1] + t.minY * a[1] * i, s, r[1] + (t.maxY + 1) * a[1] * i, n);
	}, e.prototype.getTileRangeForExtentAndZ = function(e, t, n) {
		var r = gl;
		this.getTileCoordForXYAndZ_(e[0], e[3], t, !1, r);
		var i = r[1], a = r[2];
		return this.getTileCoordForXYAndZ_(e[2], e[1], t, !0, r), uo(i, r[1], a, r[2], n);
	}, e.prototype.getTileCoordCenter = function(e) {
		var t = this.getOrigin(e[0]), n = this.getResolution(e[0]), r = za(this.getTileSize(e[0]), this.tmpSize_);
		return [t[0] + (e[1] + .5) * r[0] * n, t[1] - (e[2] + .5) * r[1] * n];
	}, e.prototype.getTileCoordExtent = function(e, t) {
		var n = this.getOrigin(e[0]), r = this.getResolution(e[0]), i = za(this.getTileSize(e[0]), this.tmpSize_), a = n[0] + e[1] * i[0] * r, o = n[1] - (e[2] + 1) * i[1] * r;
		return Mt(a, o, a + i[0] * r, o + i[1] * r, t);
	}, e.prototype.getTileCoordForCoordAndResolution = function(e, t, n) {
		return this.getTileCoordForXYAndResolution_(e[0], e[1], t, !1, n);
	}, e.prototype.getTileCoordForXYAndResolution_ = function(e, t, n, r, i) {
		var a = this.getZForResolution(n), o = n / this.getResolution(a), s = this.getOrigin(a), c = za(this.getTileSize(a), this.tmpSize_), l = r ? .5 : 0, u = r ? .5 : 0, d = Math.floor((e - s[0]) / n + l), f = Math.floor((s[1] - t) / n + u), p = o * d / c[0], m = o * f / c[1];
		return r ? (p = Math.ceil(p) - 1, m = Math.ceil(m) - 1) : (p = Math.floor(p), m = Math.floor(m)), ul(a, p, m, i);
	}, e.prototype.getTileCoordForXYAndZ_ = function(e, t, n, r, i) {
		var a = this.getOrigin(n), o = this.getResolution(n), s = za(this.getTileSize(n), this.tmpSize_), c = r ? .5 : 0, l = r ? .5 : 0, u = Math.floor((e - a[0]) / o + c), d = Math.floor((a[1] - t) / o + l), f = u / s[0], p = d / s[1];
		return r ? (f = Math.ceil(f) - 1, p = Math.ceil(p) - 1) : (f = Math.floor(f), p = Math.floor(p)), ul(n, f, p, i);
	}, e.prototype.getTileCoordForCoordAndZ = function(e, t, n) {
		return this.getTileCoordForXYAndZ_(e[0], e[1], t, !1, n);
	}, e.prototype.getTileCoordResolution = function(e) {
		return this.resolutions_[e[0]];
	}, e.prototype.getTileSize = function(e) {
		return this.tileSize_ ? this.tileSize_ : this.tileSizes_[e];
	}, e.prototype.getFullTileRange = function(e) {
		return this.fullTileRanges_ ? this.fullTileRanges_[e] : this.extent_ ? this.getTileRangeForExtentAndZ(this.extent_, e) : null;
	}, e.prototype.getZForResolution = function(e, t) {
		return B(p(this.resolutions_, e, t || 0), this.minZoom, this.maxZoom);
	}, e.prototype.calculateTileRanges_ = function(e) {
		for (var t = this.resolutions_.length, n = Array(t), r = this.minZoom; r < t; ++r) n[r] = this.getTileRangeForExtentAndZ(e, r);
		this.fullTileRanges_ = n;
	}, e;
}(), vl = .5, yl = 10, bl = .25, xl = function() {
	function e(e, t, n, r, i, a) {
		this.sourceProj_ = e, this.targetProj_ = t;
		var o = {}, s = wn(this.targetProj_, this.sourceProj_);
		this.transformInv_ = function(e) {
			var t = e[0] + "/" + e[1];
			return o[t] || (o[t] = s(e)), o[t];
		}, this.maxSourceExtent_ = r, this.errorThresholdSquared_ = i * i, this.triangles_ = [], this.wrapsXInSource_ = !1, this.canWrapXInSource_ = this.sourceProj_.canWrapX() && !!r && !!this.sourceProj_.getExtent() && H(r) == H(this.sourceProj_.getExtent()), this.sourceWorldWidth_ = this.sourceProj_.getExtent() ? H(this.sourceProj_.getExtent()) : null, this.targetWorldWidth_ = this.targetProj_.getExtent() ? H(this.targetProj_.getExtent()) : null;
		var c = Xt(n), l = Zt(n), u = Wt(n), d = Ut(n), f = this.transformInv_(c), p = this.transformInv_(l), m = this.transformInv_(u), h = this.transformInv_(d), g = yl + (a ? Math.max(0, Math.ceil(We(Ht(n) / (a * a * 256 * 256)))) : 0);
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
		var l = Ct([
			i,
			a,
			o,
			s
		]), u = this.sourceWorldWidth_ ? H(l) / this.sourceWorldWidth_ : null, d = this.sourceWorldWidth_, f = this.sourceProj_.canWrapX() && u > .5 && u < 1, p = !1;
		if (c > 0 && (this.targetProj_.isGlobal() && this.targetWorldWidth_ && (p = H(Ct([
			e,
			t,
			n,
			r
		])) / this.targetWorldWidth_ > bl || p), !f && this.sourceProj_.isGlobal() && u && (p = u > bl || p)), !(!p && this.maxSourceExtent_ && isFinite(l[0]) && isFinite(l[1]) && isFinite(l[2]) && isFinite(l[3]) && !Qt(l, this.maxSourceExtent_))) {
			var m = 0;
			if (!p && (!isFinite(i[0]) || !isFinite(i[1]) || !isFinite(a[0]) || !isFinite(a[1]) || !isFinite(o[0]) || !isFinite(o[1]) || !isFinite(s[0]) || !isFinite(s[1]))) {
				if (c > 0) p = !0;
				else if (m = (!isFinite(i[0]) || !isFinite(i[1]) ? 8 : 0) + (!isFinite(a[0]) || !isFinite(a[1]) ? 4 : 0) + (!isFinite(o[0]) || !isFinite(o[1]) ? 2 : 0) + +(!isFinite(s[0]) || !isFinite(s[1])), m != 1 && m != 2 && m != 4 && m != 8) return;
			}
			if (c > 0) {
				if (!p) {
					var h = [(e[0] + n[0]) / 2, (e[1] + n[1]) / 2], g = this.transformInv_(h), _ = void 0;
					_ = f ? (Ye(i[0], d) + Ye(o[0], d)) / 2 - Ye(g[0], d) : (i[0] + o[0]) / 2 - g[0];
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
		var e = jt();
		return this.triangles_.forEach(function(t, n, r) {
			var i = t.source;
			Rt(e, i[0]), Rt(e, i[1]), Rt(e, i[2]);
		}), e;
	}, e.prototype.getTriangles = function() {
		return this.triangles_;
	}, e;
}(), Sl = {
	imageSmoothingEnabled: !1,
	msImageSmoothingEnabled: !1
}, Cl;
function wl(e, t, n, r, i) {
	e.beginPath(), e.moveTo(0, 0), e.lineTo(t, n), e.lineTo(r, i), e.closePath(), e.save(), e.clip(), e.fillRect(0, 0, Math.max(t, r) + 1, Math.max(n, i)), e.restore();
}
function Tl(e, t) {
	return Math.abs(e[t * 4] - 210) > 2 || Math.abs(e[t * 4 + 3] - .75 * 255) > 2;
}
function El() {
	if (Cl === void 0) {
		var e = document.createElement("canvas").getContext("2d");
		e.globalCompositeOperation = "lighter", e.fillStyle = "rgba(210, 0, 0, 0.75)", wl(e, 4, 5, 4, 0), wl(e, 4, 5, 0, 5);
		var t = e.getImageData(0, 0, 3, 3).data;
		Cl = Tl(t, 0) || Tl(t, 4) || Tl(t, 8);
	}
	return Cl;
}
function Dl(e, t, n, r) {
	var i = Tn(n, t, e), a = vn(t, r, n), o = t.getMetersPerUnit();
	o !== void 0 && (a *= o);
	var s = e.getMetersPerUnit();
	s !== void 0 && (a /= s);
	var c = e.getExtent();
	if (!c || Dt(c, i)) {
		var l = vn(e, a, i) / a;
		isFinite(l) && l > 0 && (a /= l);
	}
	return a;
}
function Ol(e, t, n, r) {
	var i = Dl(e, t, Gt(n), r);
	return (!isFinite(i) || i <= 0) && Vt(n, function(n) {
		return i = Dl(e, t, n, r), isFinite(i) && i > 0;
	}), i;
}
function kl(e, t, n, r, i, a, o, s, c, l, u, d) {
	var f = fe(Math.round(n * e), Math.round(n * t));
	if (S(f, d), c.length === 0) return f.canvas;
	f.scale(n, n);
	function p(e) {
		return Math.round(e * n) / n;
	}
	f.globalCompositeOperation = "lighter";
	var m = jt();
	c.forEach(function(e, t, n) {
		Lt(m, e.extent);
	});
	var h = H(m), g = Jt(m), _ = fe(Math.round(n * h / r), Math.round(n * g / r));
	S(_, d);
	var v = n / r;
	c.forEach(function(e, t, n) {
		var r = e.extent[0] - m[0], i = -(e.extent[3] - m[3]), a = H(e.extent), o = Jt(e.extent);
		e.image.width > 0 && e.image.height > 0 && _.drawImage(e.image, l, l, e.image.width - 2 * l, e.image.height - 2 * l, r * v, i * v, a * v, o * v);
	});
	var y = Xt(o);
	return s.getTriangles().forEach(function(e, t, i) {
		var o = e.source, s = e.target, c = o[0][0], l = o[0][1], u = o[1][0], h = o[1][1], g = o[2][0], v = o[2][1], b = p((s[0][0] - y[0]) / a), x = p(-(s[0][1] - y[1]) / a), S = p((s[1][0] - y[0]) / a), C = p(-(s[1][1] - y[1]) / a), w = p((s[2][0] - y[0]) / a), T = p(-(s[2][1] - y[1]) / a), E = c, D = l;
		c = 0, l = 0, u -= E, h -= D, g -= E, v -= D;
		var O = qe([
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
			if (f.save(), f.beginPath(), El() || d === Sl) {
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
var Al = (function() {
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
})(), jl = function(e) {
	Al(t, e);
	function t(t, n, r, i, a, o, s, c, l, u, d, f) {
		var p = e.call(this, a, q.IDLE) || this;
		p.renderEdges_ = d !== void 0 && d, p.contextOptions_ = f, p.pixelRatio_ = s, p.gutter_ = c, p.canvas_ = null, p.sourceTileGrid_ = n, p.targetTileGrid_ = i, p.wrappedTileCoord_ = o || a, p.sourceTiles_ = [], p.sourcesListenerKeys_ = null, p.sourceZ_ = 0;
		var m = i.getTileCoordExtent(p.wrappedTileCoord_), h = p.targetTileGrid_.getExtent(), g = p.sourceTileGrid_.getExtent(), _ = h ? Yt(m, h) : m;
		if (Ht(_) === 0) return p.state = q.EMPTY, p;
		var v = t.getExtent();
		v && (g = g ? Yt(g, v) : v);
		var y = i.getResolution(p.wrappedTileCoord_[0]), b = Ol(t, r, _, y);
		if (!isFinite(b) || b <= 0 || (p.triangulation_ = new xl(t, r, _, g, b * (u === void 0 ? vl : u), y), p.triangulation_.getTriangles().length === 0)) return p.state = q.EMPTY, p;
		p.sourceZ_ = n.getZForResolution(b);
		var x = p.triangulation_.calculateSourceExtent();
		if (g && (t.canWrapX() ? (x[1] = B(x[1], g[1], g[3]), x[3] = B(x[3], g[1], g[3])) : x = Yt(x, g)), !Ht(x)) p.state = q.EMPTY;
		else {
			for (var S = n.getTileRangeForExtentAndZ(x, p.sourceZ_), C = S.minX; C <= S.maxX; C++) for (var w = S.minY; w <= S.maxY; w++) {
				var T = l(p.sourceZ_, C, w, s);
				T && p.sourceTiles_.push(T);
			}
			p.sourceTiles_.length === 0 && (p.state = q.EMPTY);
		}
		return p;
	}
	return t.prototype.getImage = function() {
		return this.canvas_;
	}, t.prototype.reproject_ = function() {
		var e = [];
		if (this.sourceTiles_.forEach(function(t, n, r) {
			t && t.getState() == q.LOADED && e.push({
				extent: this.sourceTileGrid_.getTileCoordExtent(t.tileCoord),
				image: t.getImage()
			});
		}.bind(this)), this.sourceTiles_.length = 0, e.length === 0) this.state = q.ERROR;
		else {
			var t = this.wrappedTileCoord_[0], n = this.targetTileGrid_.getTileSize(t), r = typeof n == "number" ? n : n[0], i = typeof n == "number" ? n : n[1], a = this.targetTileGrid_.getResolution(t), o = this.sourceTileGrid_.getResolution(this.sourceZ_), s = this.targetTileGrid_.getTileCoordExtent(this.wrappedTileCoord_);
			this.canvas_ = kl(r, i, this.pixelRatio_, o, this.sourceTileGrid_.getExtent(), a, s, this.triangulation_, e, this.gutter_, this.renderEdges_, this.contextOptions_), this.state = q.LOADED;
		}
		this.changed();
	}, t.prototype.load = function() {
		if (this.state == q.IDLE) {
			this.state = q.LOADING, this.changed();
			var e = 0;
			this.sourcesListenerKeys_ = [], this.sourceTiles_.forEach(function(t, n, r) {
				var i = t.getState();
				if (i == q.IDLE || i == q.LOADING) {
					e++;
					var a = k(t, O.CHANGE, function(n) {
						var r = t.getState();
						(r == q.LOADED || r == q.ERROR || r == q.EMPTY) && (j(a), e--, e === 0 && (this.unlistenSources_(), this.reproject_()));
					}, this);
					this.sourcesListenerKeys_.push(a);
				}
			}.bind(this)), e === 0 ? setTimeout(this.reproject_.bind(this), 0) : this.sourceTiles_.forEach(function(e, t, n) {
				e.getState() == q.IDLE && e.load();
			});
		}
	}, t.prototype.unlistenSources_ = function() {
		this.sourcesListenerKeys_.forEach(j), this.sourcesListenerKeys_ = null;
	}, t;
}(al), Ml = function() {
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
		return V(n !== void 0, 15), n === this.newest_ ? n.value_ : (n === this.oldest_ ? (this.oldest_ = this.oldest_.newer, this.oldest_.older = null) : (n.newer.older = n.older, n.older.newer = n.newer), n.newer = null, n.older = this.newest_, this.newest_.newer = n, this.newest_ = n, n.value_);
	}, e.prototype.remove = function(e) {
		var t = this.entries_[e];
		return V(t !== void 0, 15), t === this.newest_ ? (this.newest_ = t.older, this.newest_ && (this.newest_.newer = null)) : t === this.oldest_ ? (this.oldest_ = t.newer, this.oldest_ && (this.oldest_.older = null)) : (t.newer.older = t.older, t.older.newer = t.newer), delete this.entries_[e], --this.count_, t.value_;
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
		V(!(e in this.entries_), 16);
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
}(), Nl = (function() {
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
})(), Pl = function(e) {
	Nl(t, e);
	function t() {
		return e !== null && e.apply(this, arguments) || this;
	}
	return t.prototype.expireCache = function(e) {
		for (; this.canExpireCache() && !(this.peekLast().getKey() in e);) this.pop().release();
	}, t.prototype.pruneExceptNewestZ = function() {
		if (this.getCount() !== 0) {
			var e = pl(this.peekFirstKey())[0];
			this.forEach(function(t) {
				t.tileCoord[0] !== e && (this.remove(fl(t.tileCoord)), t.release());
			}.bind(this));
		}
	}, t;
}(Ml), Fl = {
	TILELOADSTART: "tileloadstart",
	TILELOADEND: "tileloadend",
	TILELOADERROR: "tileloaderror"
};
//#endregion
//#region node_modules/ol/tilegrid.js
function Il(e) {
	var t = e.getDefaultTileGrid();
	return t || (t = Bl(e), e.setDefaultTileGrid(t)), t;
}
function Ll(e, t, n) {
	var r = t[0], i = e.getTileCoordCenter(t), a = Vl(n);
	if (Dt(a, i)) return t;
	var o = H(a), s = Math.ceil((a[0] - i[0]) / o);
	return i[0] += o * s, e.getTileCoordForCoordAndZ(i, r);
}
function Rl(e, t, n, r) {
	var i = r === void 0 ? yt.TOP_LEFT : r, a = zl(e, t, n);
	return new _l({
		extent: e,
		origin: Kt(e, i),
		resolutions: a,
		tileSize: n
	});
}
function zl(e, t, n, r) {
	for (var i = t === void 0 ? 42 : t, a = Jt(e), o = H(e), s = za(n === void 0 ? 256 : n), c = r > 0 ? r : Math.max(o / s[0], a / s[1]), l = i + 1, u = Array(l), d = 0; d < l; ++d) u[d] = c / 2 ** d;
	return u;
}
function Bl(e, t, n, r) {
	return Rl(Vl(e), t, n, r);
}
function Vl(e) {
	e = _n(e);
	var t = e.getExtent();
	if (!t) {
		var n = 180 * Ve[z.DEGREES] / e.getMetersPerUnit();
		t = Mt(-n, -n, n, n);
	}
	return t;
}
//#endregion
//#region node_modules/ol/source/Tile.js
var Hl = (function() {
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
})(), Ul = function(e) {
	Hl(t, e);
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
		return i && za(i.getTileSize(i.getMinZoom()), r), n.tileCache = new Pl(t.cacheSize || 0), n.tmpSize = [0, 0], n.key_ = t.key || "", n.tileOptions = { transition: t.transition }, n.zDirection = t.zDirection ? t.zDirection : 0, n;
	}
	return t.prototype.canExpireCache = function() {
		return this.tileCache.canExpireCache();
	}, t.prototype.expireCache = function(e, t) {
		var n = this.getTileCacheForProjection(e);
		n && n.expireCache(t);
	}, t.prototype.forEachLoadedTile = function(e, t, n, r) {
		var i = this.getTileCacheForProjection(e);
		if (!i) return !1;
		for (var a = !0, o, s, c, l = n.minX; l <= n.maxX; ++l) for (var u = n.minY; u <= n.maxY; ++u) s = dl(t, l, u), c = !1, i.containsKey(s) && (o = i.get(s), c = o.getState() === q.LOADED, c &&= r(o) !== !1), c || (a = !1);
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
		return this.tileGrid ? this.tileGrid : Il(e);
	}, t.prototype.getTileCacheForProjection = function(e) {
		return V(Sn(this.getProjection(), e), 68), this.tileCache;
	}, t.prototype.getTilePixelRatio = function(e) {
		return this.tilePixelRatio_;
	}, t.prototype.getTilePixelSize = function(e, t, n) {
		var r = this.getTileGridForProjection(n), i = this.getTilePixelRatio(t), a = za(r.getTileSize(e), this.tmpSize);
		return i == 1 ? a : Ra(a, i, this.tmpSize);
	}, t.prototype.getTileCoordForTileUrlFunction = function(e, t) {
		var n = t === void 0 ? this.getProjection() : t, r = this.getTileGridForProjection(n);
		return this.getWrapX() && n.isGlobal() && (e = Ll(r, e, n)), hl(e, r) ? e : null;
	}, t.prototype.clear = function() {
		this.tileCache.clear();
	}, t.prototype.refresh = function() {
		this.clear(), e.prototype.refresh.call(this);
	}, t.prototype.updateCacheSize = function(e, t) {
		var n = this.getTileCacheForProjection(t);
		e > n.highWaterMark && (n.highWaterMark = e);
	}, t.prototype.useTile = function(e, t, n, r) {}, t;
}(Qc), Wl = function(e) {
	Hl(t, e);
	function t(t, n) {
		var r = e.call(this, t) || this;
		return r.tile = n, r;
	}
	return t;
}(l);
//#endregion
//#region node_modules/ol/tileurlfunction.js
function Gl(e, t) {
	var n = /\{z\}/g, r = /\{x\}/g, i = /\{y\}/g, a = /\{-y\}/g;
	return (function(o, s, c) {
		if (o) return e.replace(n, o[0].toString()).replace(r, o[1].toString()).replace(i, o[2].toString()).replace(a, function() {
			var e = o[0], n = t.getFullTileRange(e);
			return V(n, 55), (n.getHeight() - o[2] - 1).toString();
		});
	});
}
function Kl(e, t) {
	for (var n = e.length, r = Array(n), i = 0; i < n; ++i) r[i] = Gl(e[i], t);
	return ql(r);
}
function ql(e) {
	return e.length === 1 ? e[0] : (function(t, n, r) {
		if (t) return e[Ye(ml(t), e.length)](t, n, r);
	});
}
function Jl(e) {
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
var Yl = (function() {
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
		var t = e.target, n = I(t), r = t.getState(), i;
		r == q.LOADING ? (this.tileLoadingKeys_[n] = !0, i = Fl.TILELOADSTART) : n in this.tileLoadingKeys_ && (delete this.tileLoadingKeys_[n], i = r == q.ERROR ? Fl.TILELOADERROR : r == q.LOADED ? Fl.TILELOADEND : void 0), i != null && this.dispatchEvent(new Wl(i, t));
	}, t.prototype.setTileLoadFunction = function(e) {
		this.tileCache.clear(), this.tileLoadFunction = e, this.changed();
	}, t.prototype.setTileUrlFunction = function(e, t) {
		this.tileUrlFunction = e, this.tileCache.pruneExceptNewestZ(), t === void 0 ? this.changed() : this.setKey(t);
	}, t.prototype.setUrl = function(e) {
		var t = Jl(e);
		this.urls = t, this.setUrls(t);
	}, t.prototype.setUrls = function(e) {
		this.urls = e;
		var t = e.join("\n");
		this.generateTileUrlFunction_ ? this.setTileUrlFunction(Kl(e, this.tileGrid), t) : this.setKey(t);
	}, t.prototype.tileUrlFunction = function(e, t, n) {}, t.prototype.useTile = function(e, t, n) {
		var r = dl(e, t, n);
		this.tileCache.containsKey(r) && this.tileCache.get(r);
	}, t;
}(Ul), Zl = (function() {
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
})(), Ql = function(e) {
	Zl(t, e);
	function t(t) {
		var n = e.call(this, {
			attributions: t.attributions,
			cacheSize: t.cacheSize,
			opaque: t.opaque,
			projection: t.projection,
			state: t.state,
			tileGrid: t.tileGrid,
			tileLoadFunction: t.tileLoadFunction ? t.tileLoadFunction : $l,
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
		return n.crossOrigin = t.crossOrigin === void 0 ? null : t.crossOrigin, n.tileClass = t.tileClass === void 0 ? cl : t.tileClass, n.tileCacheForProjection = {}, n.tileGridForProjection = {}, n.reprojectionErrorThreshold_ = t.reprojectionErrorThreshold, n.contextOptions_ = t.imageSmoothing === !1 ? Sl : void 0, n.renderReprojectionEdges_ = !1, n;
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
		return this.getProjection() && e && !Sn(this.getProjection(), e) ? 0 : this.getGutter();
	}, t.prototype.getGutter = function() {
		return 0;
	}, t.prototype.getKey = function() {
		return e.prototype.getKey.call(this) + (this.contextOptions_ ? "\n" + JSON.stringify(this.contextOptions_) : "");
	}, t.prototype.getOpaque = function(t) {
		return this.getProjection() && t && !Sn(this.getProjection(), t) ? !1 : e.prototype.getOpaque.call(this, t);
	}, t.prototype.getTileGridForProjection = function(e) {
		var t = this.getProjection();
		if (this.tileGrid && (!t || Sn(t, e))) return this.tileGrid;
		var n = I(e);
		return n in this.tileGridForProjection || (this.tileGridForProjection[n] = Il(e)), this.tileGridForProjection[n];
	}, t.prototype.getTileCacheForProjection = function(e) {
		var t = this.getProjection();
		if (!t || Sn(t, e)) return this.tileCache;
		var n = I(e);
		return n in this.tileCacheForProjection || (this.tileCacheForProjection[n] = new Pl(this.tileCache.highWaterMark)), this.tileCacheForProjection[n];
	}, t.prototype.createTile_ = function(e, t, n, r, i, a) {
		var o = [
			e,
			t,
			n
		], s = this.getTileCoordForTileUrlFunction(o, i), c = s ? this.tileUrlFunction(s, r, i) : void 0, l = new this.tileClass(o, c === void 0 ? q.EMPTY : q.IDLE, c === void 0 ? "" : c, this.crossOrigin, this.tileLoadFunction, this.tileOptions);
		return l.key = a, l.addEventListener(O.CHANGE, this.handleTileChange.bind(this)), l;
	}, t.prototype.getTile = function(e, t, n, r, i) {
		var a = this.getProjection();
		if (!a || !i || Sn(a, i)) return this.getTileInternal(e, t, n, r, a || i);
		var o = this.getTileCacheForProjection(i), s = [
			e,
			t,
			n
		], c = void 0, l = fl(s);
		o.containsKey(l) && (c = o.get(l));
		var u = this.getKey();
		if (c && c.key == u) return c;
		var d = new jl(a, this.getTileGridForProjection(a), i, this.getTileGridForProjection(i), s, this.getTileCoordForTileUrlFunction(s, i), this.getTilePixelRatio(r), this.getGutter(), function(e, t, n, r) {
			return this.getTileInternal(e, t, n, r, a);
		}.bind(this), this.reprojectionErrorThreshold_, this.renderReprojectionEdges_, this.contextOptions_);
		return d.key = u, c ? (d.interimTile = c, d.refreshInterimChain(), o.replace(l, d)) : o.set(l, d), d;
	}, t.prototype.getTileInternal = function(e, t, n, r, i) {
		var a = null, o = dl(e, t, n), s = this.getKey();
		if (!this.tileCache.containsKey(o)) a = this.createTile_(e, t, n, r, i, s), this.tileCache.set(o, a);
		else if (a = this.tileCache.get(o), a.key != s) {
			var c = a;
			a = this.createTile_(e, t, n, r, i, s), c.getState() == q.IDLE ? a.interimTile = c.interimTile : a.interimTile = c, a.refreshInterimChain(), this.tileCache.replace(o, a);
		}
		return a;
	}, t.prototype.setRenderReprojectionEdges = function(e) {
		if (this.renderReprojectionEdges_ != e) {
			for (var t in this.renderReprojectionEdges_ = e, this.tileCacheForProjection) this.tileCacheForProjection[t].clear();
			this.changed();
		}
	}, t.prototype.setTileGridForProjection = function(e, t) {
		var n = _n(e);
		if (n) {
			var r = I(n);
			r in this.tileGridForProjection || (this.tileGridForProjection[r] = t);
		}
	}, t;
}(Xl);
function $l(e, t) {
	e.getImage().src = t;
}
//#endregion
//#region node_modules/ol/source/Zoomify.js
var eu = (function() {
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
})(), tu = {
	DEFAULT: "default",
	TRUNCATED: "truncated"
}, nu = function(e) {
	eu(t, e);
	function t(t, n, r, i, a, o, s) {
		var c = e.call(this, n, r, i, a, o, s) || this;
		return c.zoomifyImage_ = null, c.tileSize_ = t, c;
	}
	return t.prototype.getImage = function() {
		if (this.zoomifyImage_) return this.zoomifyImage_;
		var t = e.prototype.getImage.call(this);
		if (this.state == q.LOADED) {
			var n = this.tileSize_;
			if (t.width == n[0] && t.height == n[1]) return this.zoomifyImage_ = t, t;
			var r = fe(n[0], n[1]);
			return r.drawImage(t, 0, 0), this.zoomifyImage_ = r.canvas, r.canvas;
		} else return t;
	}, t;
}(cl), ru = function(e) {
	eu(t, e);
	function t(t) {
		var n = this, r = t, i = r.size, a = r.tierSizeCalculation === void 0 ? tu.DEFAULT : r.tierSizeCalculation, o = r.tilePixelRatio || 1, s = i[0], c = i[1], l = [], u = r.tileSize || 256, d = u * o;
		switch (a) {
			case tu.DEFAULT:
				for (; s > d || c > d;) l.push([Math.ceil(s / d), Math.ceil(c / d)]), d += d;
				break;
			case tu.TRUNCATED:
				for (var f = s, p = c; f > d || p > d;) l.push([Math.ceil(f / d), Math.ceil(p / d)]), f >>= 1, p >>= 1;
				break;
			default:
				V(!1, 53);
				break;
		}
		l.push([1, 1]), l.reverse();
		for (var m = [o], h = [0], g = 1, _ = l.length; g < _; g++) m.push(o << g), h.push(l[g - 1][0] * l[g - 1][1] + h[g - 1]);
		m.reverse();
		var v = new _l({
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
		var b = Jl(y), x = u * o;
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
		var C = ql(b.map(S)), w = nu.bind(null, za(u * o));
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
		var T = C(v.getTileCoordForCoordAndResolution(Gt(v.getExtent()), m[m.length - 1]), 1, null), E = new Image();
		return E.addEventListener("error", function() {
			x = u, this.changed();
		}.bind(n)), E.src = T, n;
	}
	return t;
}(Ql);
//#endregion
//#region node_modules/ol-ext/util/ext.js
window.ol && !ol.ext && (ol.ext = {});
var iu = function(e, t) {
	e.prototype = Object.create(t.prototype), e.prototype.constructor = e;
};
window.ol && (ol.inherits || (ol.inherits = iu)), window.NodeList && !NodeList.prototype.forEach && (NodeList.prototype.forEach = Array.prototype.forEach), window.Element && !Element.prototype.remove && (Element.prototype.remove = function() {
	this.parentNode && this.parentNode.removeChild(this);
});
//#endregion
//#region node_modules/ol-ext/util/getMapCanvas.js
var au = function(e) {
	if (!e) return null;
	var t = e.getViewport().getElementsByClassName("ol-fixedoverlay")[0];
	return t || (e.getViewport().querySelector(".ol-layers") ? (t = document.createElement("canvas"), t.className = "ol-fixedoverlay", e.getViewport().querySelector(".ol-layers").after(t), e.on("precompose", function(n) {
		t.width = e.getSize()[0] * n.frameState.pixelRatio, t.height = e.getSize()[1] * n.frameState.pixelRatio;
	})) : t = e.getViewport().querySelector("canvas")), t;
}, ou = function(e) {
	e ||= {}, this.setStyle(e.style), be.call(this, e);
};
iu(ou, be), ou.prototype.setMap = function(e) {
	this.getCanvas(e);
	var t = this.getMap();
	if (this._listener &&= (P(this._listener), null), be.prototype.setMap.call(this, e), t) try {
		t.renderSync();
	} catch {}
	e && (this._listener = e.on("postcompose", this._draw.bind(this)));
}, ou.prototype.getCanvas = function(e) {
	return au(e);
}, ou.prototype.getContext = function(e) {
	var t = e.context;
	if (!t && this.getMap()) {
		var n = this.getMap().getViewport().getElementsByClassName("ol-fixedoverlay")[0];
		t = n ? n.getContext("2d") : null;
	}
	return t;
}, ou.prototype.setStyle = function(e) {
	this._style = e || new ws({});
}, ou.prototype.getStyle = function() {
	return this._style;
}, ou.prototype.getStroke = function() {
	return this._style.getStroke() || this._style.setStroke(new Cs({
		color: "#000",
		width: 1.25
	})), this._style.getStroke();
}, ou.prototype.getFill = function() {
	return this._style.getFill() || this._style.setFill(new Ss({ color: "#fff" })), this._style.getFill();
}, ou.prototype.getTextStroke = function() {
	var e = this._style.getText();
	return e ||= new Dc({}), e.getStroke() || e.setStroke(new Cs({
		color: "#fff",
		width: 3
	})), e.getStroke();
}, ou.prototype.getTextFill = function() {
	var e = this._style.getText();
	return e ||= new Dc({}), e.getFill() || e.setFill(new Ss({ color: "#fff" })), e.getFill();
}, ou.prototype.getTextFont = function() {
	var e = this._style.getText();
	return e ||= new Dc({}), e.getFont() || e.setFont("12px sans-serif"), e.getFont();
}, ou.prototype._draw = function() {
	console.warn("[CanvasBase] draw function not implemented.");
};
//#endregion
//#region node_modules/ol-ext/control/Graticule.js
var su = function(e) {
	e ||= {};
	var t = document.createElement("div");
	t.className = "ol-graticule ol-unselectable ol-hidden", ou.call(this, { element: t }), this.set("projection", e.projection || "EPSG:4326");
	var n = new He({ code: this.get("projection") }).getMetersPerUnit();
	for (this.fac = 1; n / this.fac > 10;) this.fac *= 10;
	this.fac = 1e4 / this.fac, this.set("maxResolution", e.maxResolution || Infinity), this.set("step", e.step || .1), this.set("stepCoord", e.stepCoord || 1), this.set("spacing", e.spacing || 40), this.set("margin", e.margin || 0), this.set("borderWidth", e.borderWidth || 5), this.set("stroke", e.stroke !== !1), this.formatCoord = e.formatCoord || function(e) {
		return e;
	}, e.style instanceof ws ? this.setStyle(e.style) : this.setStyle(new ws({
		stroke: new Cs({
			color: "#000",
			width: 1
		}),
		fill: new Ss({ color: "#fff" }),
		text: new Dc({
			stroke: new Cs({
				color: "#fff",
				width: 2
			}),
			fill: new Ss({ color: "#000" })
		})
	}));
};
iu(su, ou), su.prototype.setStyle = function(e) {
	this._style = e;
}, su.prototype._draw = function(e) {
	if (!(this.get("maxResolution") < e.frameState.viewState.resolution)) {
		for (var t = this.getContext(e), n = t.canvas, r = e.frameState.pixelRatio, i = n.width / r, a = n.height / r, o = this.get("projection"), s = this.getMap(), c = [
			s.getCoordinateFromPixel([0, 0]),
			s.getCoordinateFromPixel([i, 0]),
			s.getCoordinateFromPixel([i, a]),
			s.getCoordinateFromPixel([0, a])
		], l = -Infinity, u = Infinity, d = -Infinity, f = Infinity, p = 0, m; m = c[p]; p++) c[p] = Tn(m, s.getView().getProjection(), o), l = Math.max(l, c[p][0]), u = Math.min(u, c[p][0]), d = Math.max(d, c[p][1]), f = Math.min(f, c[p][1]);
		var h = this.get("spacing"), g = this.get("step"), _ = this.get("stepCoord"), v = this.get("borderWidth"), y = this.get("margin");
		if ((l - u) / g * h > i) {
			var b = Math.round((l - u) / i * h / g);
			g *= b, g > this.fac && (g = Math.round(g / this.fac) * this.fac);
		}
		u = Math.floor(u / g) * g - g, f = Math.floor(f / g) * g - g, l = Math.floor(l / g) * g + 2 * g, d = Math.floor(d / g) * g + 2 * g;
		var x = _n(o).getExtent();
		x && (u < x[0] && (u = x[0]), f < x[1] && (f = x[1]), l > x[2] && (l = x[2] + g), d > x[3] && (d = x[3] + g));
		var S = this.getStyle().getStroke() && this.get("stroke"), C = this.getStyle().getText(), w = this.getStyle().getFill();
		t.save(), t.scale(r, r), t.beginPath(), t.rect(y, y, i - 2 * y, a - 2 * y), t.clip(), t.beginPath();
		var T = {
			top: [],
			left: [],
			bottom: [],
			right: []
		}, E, D, O, k, A;
		for (E = u; E < l; E += g) for (k = Tn([E, f], o, s.getView().getProjection()), k = s.getPixelFromCoordinate(k), S && t.moveTo(k[0], k[1]), O = k, D = f + g; D <= d; D += g) A = Tn([E, D], o, s.getView().getProjection()), A = s.getPixelFromCoordinate(A), S && t.lineTo(A[0], A[1]), O[1] > 0 && A[1] < 0 && T.top.push([E, O]), O[1] > a && A[1] < a && T.bottom.push([E, O]), O = A;
		for (D = f; D < d; D += g) for (k = Tn([u, D], o, s.getView().getProjection()), k = s.getPixelFromCoordinate(k), S && t.moveTo(k[0], k[1]), O = k, E = u + g; E <= l; E += g) A = Tn([E, D], o, s.getView().getProjection()), A = s.getPixelFromCoordinate(A), S && t.lineTo(A[0], A[1]), O[0] < 0 && A[0] > 0 && T.left.push([D, O]), O[0] < i && A[0] > i && T.right.push([D, O]), O = A;
		if (S && (t.strokeStyle = this.getStyle().getStroke().getColor(), t.lineWidth = this.getStyle().getStroke().getWidth(), t.stroke()), C) {
			t.fillStyle = this.getStyle().getText().getFill().getColor(), t.strokeStyle = this.getStyle().getText().getStroke().getColor(), t.lineWidth = this.getStyle().getText().getStroke().getWidth(), t.font = this.getStyle().getText().getFont(), t.textAlign = "center", t.textBaseline = "hanging";
			var j, M, N = (w ? v : 0) + y + 2;
			for (p = 0; j = T.top[p]; p++) Math.round(j[0] / this.get("step")) % _ || (M = this.formatCoord(j[0], "top"), t.strokeText(M, j[1][0], N), t.fillText(M, j[1][0], N));
			for (t.textBaseline = "alphabetic", p = 0; j = T.bottom[p]; p++) Math.round(j[0] / this.get("step")) % _ || (M = this.formatCoord(j[0], "bottom"), t.strokeText(M, j[1][0], a - N), t.fillText(M, j[1][0], a - N));
			for (t.textBaseline = "middle", t.textAlign = "left", p = 0; j = T.left[p]; p++) Math.round(j[0] / this.get("step")) % _ || (M = this.formatCoord(j[0], "left"), t.strokeText(M, N, j[1][1]), t.fillText(M, N, j[1][1]));
			for (t.textAlign = "right", p = 0; j = T.right[p]; p++) Math.round(j[0] / this.get("step")) % _ || (M = this.formatCoord(j[0], "right"), t.strokeText(M, i - N, j[1][1]), t.fillText(M, i - N, j[1][1]));
		}
		if (w) {
			var P = this.getStyle().getFill().getColor(), F, ee;
			for ((ee = this.getStyle().getStroke()) ? F = this.getStyle().getStroke().getColor() : (F = P, P = "#fff"), t.strokeStyle = F, t.lineWidth = ee ? ee.getWidth() : 1, p = 1; p < T.top.length; p++) t.beginPath(), t.rect(T.top[p - 1][1][0], y, T.top[p][1][0] - T.top[p - 1][1][0], v), t.fillStyle = Math.round(T.top[p][0] / g) % 2 ? F : P, t.fill(), t.stroke();
			for (p = 1; p < T.bottom.length; p++) t.beginPath(), t.rect(T.bottom[p - 1][1][0], a - v - y, T.bottom[p][1][0] - T.bottom[p - 1][1][0], v), t.fillStyle = Math.round(T.bottom[p][0] / g) % 2 ? F : P, t.fill(), t.stroke();
			for (p = 1; p < T.left.length; p++) t.beginPath(), t.rect(y, T.left[p - 1][1][1], v, T.left[p][1][1] - T.left[p - 1][1][1]), t.fillStyle = Math.round(T.left[p][0] / g) % 2 ? F : P, t.fill(), t.stroke();
			for (p = 1; p < T.right.length; p++) t.beginPath(), t.rect(i - v - y, T.right[p - 1][1][1], v, T.right[p][1][1] - T.right[p - 1][1][1]), t.fillStyle = Math.round(T.right[p][0] / g) % 2 ? F : P, t.fill(), t.stroke();
			t.beginPath(), t.fillStyle = F, t.rect(y, y, v, v), t.rect(y, a - v - y, v, v), t.rect(i - v - y, y, v, v), t.rect(i - v - y, a - v - y, v, v), t.fill();
		}
		t.restore();
	}
};
//#endregion
//#region node_modules/ol/layer/VectorTileRenderType.js
var cu = {
	IMAGE: "image",
	HYBRID: "hybrid",
	VECTOR: "vector"
}, lu = (function() {
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
})(), uu = {
	image: [
		Z.POLYGON,
		Z.CIRCLE,
		Z.LINE_STRING,
		Z.IMAGE,
		Z.TEXT
	],
	hybrid: [Z.POLYGON, Z.LINE_STRING],
	vector: []
}, du = {
	hybrid: [
		Z.IMAGE,
		Z.TEXT,
		Z.DEFAULT
	],
	vector: [
		Z.POLYGON,
		Z.CIRCLE,
		Z.LINE_STRING,
		Z.IMAGE,
		Z.TEXT,
		Z.DEFAULT
	]
}, fu = function(e) {
	lu(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.boundHandleStyleImageChange_ = n.handleStyleImageChange_.bind(n), n.dirty_ = !1, n.renderedLayerRevision_, n.renderedPixelToCoordinateTransform_ = null, n.renderedRotation_, n.tmpTransform_ = zn(), n;
	}
	return t.prototype.prepareTile = function(e, t, n) {
		var r, i = e.getState();
		return (i === q.LOADED || i === q.ERROR) && (this.updateExecutorGroup_(e, t, n), this.tileImageNeedsRender_(e) && (r = !0)), r;
	}, t.prototype.getTile = function(t, n, r, i) {
		var a = i.pixelRatio, o = i.viewState, s = o.resolution, c = o.projection, l = this.getLayer(), u = l.getSource().getTile(t, n, r, a, c), d = i.viewHints, f = !(d[J.ANIMATING] || d[J.INTERACTING]);
		return (f || !u.wantedResolution) && (u.wantedResolution = s), this.prepareTile(u, a, c) && (f || Date.now() - i.time < 8) && l.getRenderMode() !== cu.VECTOR && this.renderTileImage_(u, i), e.prototype.getTile.call(this, t, n, r, i);
	}, t.prototype.isDrawableTile = function(t) {
		var n = this.getLayer();
		return e.prototype.isDrawableTile.call(this, t) && (n.getRenderMode() === cu.VECTOR ? I(n) in t.executorGroups : t.hasContext(n));
	}, t.prototype.getTileImage = function(e) {
		return e.getImage(this.getLayer());
	}, t.prototype.prepareFrame = function(t) {
		var n = this.getLayer().getRevision();
		return this.renderedLayerRevision_ != n && (this.renderedTiles.length = 0), this.renderedLayerRevision_ = n, e.prototype.prepareFrame.call(this, t);
	}, t.prototype.updateExecutorGroup_ = function(e, t, n) {
		var r = this.getLayer(), i = r.getRevision(), a = r.getRenderOrder() || null, o = e.wantedResolution, s = e.getReplayState(r);
		if (!(!s.dirty && s.renderedResolution === o && s.renderedRevision == i && s.renderedRenderOrder == a)) {
			var c = r.getSource(), l = r.getDeclutter(), u = c.getTileGrid(), d = c.getTileGridForProjection(n).getTileCoordExtent(e.wrappedTileCoord), f = c.getSourceTiles(t, n, e), p = I(r);
			delete e.hitDetectionImageData[p], e.executorGroups[p] = [], l && (e.declutterExecutorGroups[p] = []);
			for (var m = function(n, i) {
				var m = f[n];
				if (m.getState() != q.LOADED) return "continue";
				var g = m.tileCoord, _ = u.getTileCoordExtent(g), v = Yt(d, _), y = wt(v, r.getRenderBuffer() * o, h.tmpExtent), b = It(_, v) ? null : y;
				s.dirty = !1;
				var x = new Zs(0, y, o, t), S = l ? new Zs(0, v, o, t) : void 0, C = Pc(o, t), w = function(e) {
					var t, n = e.getStyleFunction() || r.getStyleFunction();
					if (n && (t = n(e, o)), t) {
						var i = this.renderFeature(e, C, t, x, S);
						this.dirty_ = this.dirty_ || i, s.dirty = s.dirty || i;
					}
				}, T = m.getFeatures();
				a && a !== s.renderedRenderOrder && T.sort(a);
				for (var E = 0, D = T.length; E < D; ++E) {
					var O = T[E];
					(!b || Qt(b, O.getGeometry().getExtent())) && w.call(h, O);
				}
				var k = x.finish(), A = new lc(r.getRenderMode() !== cu.VECTOR && l && f.length === 1 ? null : v, o, t, c.getOverlaps(), k, r.getRenderBuffer());
				if (e.executorGroups[p].push(A), S) {
					var j = new lc(null, o, t, c.getOverlaps(), S.finish(), r.getRenderBuffer());
					e.declutterExecutorGroups[p].push(j);
				}
			}, h = this, g = 0, _ = f.length; g < _; ++g) m(g, _);
			s.renderedRevision = i, s.renderedRenderOrder = a, s.renderedResolution = o;
		}
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {
		var a = t.viewState.resolution, o = t.viewState.rotation;
		n ??= 0;
		var s = this.getLayer(), c = s.getSource().getTileGridForProjection(t.viewState.projection), l = Ct([e]);
		wt(l, a * n, l);
		for (var u = {}, d = function(e, t, n) {
			var a = e.getId();
			a === void 0 && (a = I(e));
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
			if (!Qt(c.getTileCoordExtent(u.wrappedTileCoord), l)) return "continue";
			var m = I(s), h = [u.executorGroups[m]], g = u.declutterExecutorGroups[m];
			g && h.push(g), h.some(function(r) {
				for (var i = r === g ? t.declutterTree.all().map(function(e) {
					return e.value;
				}) : null, s = 0, c = r.length; s < c; ++s) if (p = r[s].forEachFeatureAtCoordinate(e, a, o, n, d, i), p) return !0;
			});
		}, h = 0, g = f.length; !p && h < g; ++h) m(h, g);
		return p;
	}, t.prototype.getFeatures = function(e) {
		return new Promise(function(t, n) {
			for (var r = this.getLayer(), i = I(r), a = r.getSource(), o = this.renderedProjection, s = o.getExtent(), c = this.renderedResolution, l = a.getTileGridForProjection(o), u = W(this.renderedPixelToCoordinateTransform_, e.slice()), d = l.getTileCoordForCoordAndResolution(u, c), f, p = 0, m = this.renderedTiles.length; p < m; ++p) if (d.toString() === this.renderedTiles[p].tileCoord.toString()) {
				if (f = this.renderedTiles[p], f.getState() === q.LOADED) {
					var h = l.getTileCoordExtent(f.tileCoord);
					a.getWrapX() && o.canWrapX() && !Ot(s, h) && dn(u, o);
					break;
				}
				f = void 0;
			}
			if (!f || f.loadingSourceTiles > 0) {
				t([]);
				return;
			}
			var g = Xt(l.getTileCoordExtent(f.wrappedTileCoord)), _ = [(u[0] - g[0]) / c, (g[1] - u[1]) / c], v = f.getSourceTiles().reduce(function(e, t) {
				return e.concat(t.getFeatures());
			}, []), y = f.hitDetectionImageData[i];
			if (!y && !this.animatingOrInteracting_) {
				var b = za(l.getTileSize(l.getZForResolution(c))), x = this.renderedRotation_;
				y = kc(b, [this.getRenderTransform(l.getTileCoordCenter(f.wrappedTileCoord), c, 0, Oc, b[0] * Oc, b[1] * Oc, 0)], v, r.getStyleFunction(), l.getTileCoordExtent(f.wrappedTileCoord), f.getReplayState(r).renderedResolution, x), f.hitDetectionImageData[i] = y;
			}
			t(Ac(_, v, y));
		}.bind(this));
	}, t.prototype.handleFontsChanged = function() {
		var e = this.getLayer();
		e.getVisible() && this.renderedLayerRevision_ !== void 0 && e.changed();
	}, t.prototype.handleStyleImageChange_ = function(e) {
		this.renderIfReadyAndVisible();
	}, t.prototype.renderDeclutter = function(e) {
		var t = this.context, n = t.globalAlpha;
		t.globalAlpha = this.getLayer().getOpacity();
		for (var r = e.viewHints, i = !(r[J.ANIMATING] || r[J.INTERACTING]), a = this.renderedTiles, o = 0, s = a.length; o < s; ++o) {
			var c = a[o], l = c.declutterExecutorGroups[I(this.getLayer())];
			if (l) for (var u = l.length - 1; u >= 0; --u) l[u].execute(this.context, 1, this.getTileRenderTransform(c, e), e.viewState.rotation, i, void 0, e.declutterTree);
		}
		t.globalAlpha = n;
	}, t.prototype.getTileRenderTransform = function(e, t) {
		var n = t.pixelRatio, r = t.viewState, i = r.center, a = r.resolution, o = r.rotation, s = t.size, c = Math.round(s[0] * n), l = Math.round(s[1] * n), u = this.getLayer().getSource().getTileGridForProjection(t.viewState.projection), d = e.tileCoord, f = u.getTileCoordExtent(e.wrappedTileCoord), p = u.getTileCoordExtent(d, this.tmpExtent)[0] - f[0];
		return Vn(Gn(this.inversePixelTransform.slice(), 1 / n, 1 / n), this.getRenderTransform(i, a, o, n, c, l, p));
	}, t.prototype.renderFrame = function(t, n) {
		var r = t.viewHints, i = !(r[J.ANIMATING] || r[J.INTERACTING]);
		e.prototype.renderFrame.call(this, t, n), this.renderedPixelToCoordinateTransform_ = t.pixelToCoordinateTransform.slice(), this.renderedRotation_ = t.viewState.rotation;
		var a = this.getLayer(), o = a.getRenderMode(), s = this.context, c = s.globalAlpha;
		s.globalAlpha = a.getOpacity();
		for (var l = du[o], u = t.viewState.rotation, d = this.renderedTiles, f = [], p = [], m = d.length - 1; m >= 0; --m) for (var h = d[m], g = this.getTileRenderTransform(h, t), _ = h.executorGroups[I(a)], v = !1, y = 0, b = _.length; y < b; ++y) {
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
		if (Array.isArray(n)) for (var o = 0, s = n.length; o < s; ++o) a = Lc(r, e, n[o], t, this.boundHandleStyleImageChange_, void 0, i) || a;
		else a = Lc(r, e, n, t, this.boundHandleStyleImageChange_, void 0, i);
		return a;
	}, t.prototype.tileImageNeedsRender_ = function(e) {
		var t = this.getLayer();
		if (t.getRenderMode() === cu.VECTOR) return !1;
		var n = e.getReplayState(t), r = t.getRevision(), i = e.wantedResolution;
		return n.renderedTileResolution !== i || n.renderedTileRevision !== r;
	}, t.prototype.renderTileImage_ = function(e, t) {
		var n = this.getLayer(), r = e.getReplayState(n), i = n.getRevision(), a = e.executorGroups[I(n)];
		r.renderedTileRevision = i;
		var o = e.wrappedTileCoord, s = o[0], c = n.getSource(), l = t.pixelRatio, u = t.viewState.projection, d = c.getTileGridForProjection(u), f = d.getResolution(e.tileCoord[0]), p = t.pixelRatio / e.wantedResolution * f, m = d.getResolution(s), h = e.getContext(n);
		l = Math.round(Math.max(l, p / l));
		var g = c.getTilePixelSize(s, l, u);
		h.canvas.width = g[0], h.canvas.height = g[1];
		var _ = l / p;
		if (_ !== 1) {
			var v = Bn(this.tmpTransform_);
			Gn(v, _, _), h.setTransform.apply(h, v);
		}
		var y = d.getTileCoordExtent(o, this.tmpExtent), b = p / m, x = Bn(this.tmpTransform_);
		Gn(x, b, -b), qn(x, -y[0], -y[3]);
		for (var S = 0, C = a.length; S < C; ++S) a[S].execute(h, _, x, 0, !0, uu[n.getRenderMode()]);
		r.renderedTileResolution = e.wantedResolution;
	}, t;
}(po), pu = (function() {
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
})(), mu = function(e) {
	pu(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		delete i.preload, delete i.useInterimTilesOnError, n = e.call(this, i) || this, n.on, n.once, n.un, r.renderMode === cu.IMAGE && (console.warn("renderMode: \"image\" is deprecated. Option ignored."), r.renderMode = void 0);
		var a = r.renderMode || cu.HYBRID;
		return V(a == cu.HYBRID || a == cu.VECTOR, 28), n.renderMode_ = a, n.setPreload(r.preload ? r.preload : 0), n.setUseInterimTilesOnError(r.useInterimTilesOnError === void 0 || r.useInterimTilesOnError), n;
	}
	return t.prototype.createRenderer = function() {
		return new fu(this);
	}, t.prototype.getFeatures = function(t) {
		return e.prototype.getFeatures.call(this, t);
	}, t.prototype.getRenderMode = function() {
		return this.renderMode_;
	}, t.prototype.getPreload = function() {
		return this.get(no.PRELOAD);
	}, t.prototype.getUseInterimTilesOnError = function() {
		return this.get(no.USE_INTERIM_TILES_ON_ERROR);
	}, t.prototype.setPreload = function(e) {
		this.set(no.PRELOAD, e);
	}, t.prototype.setUseInterimTilesOnError = function(e) {
		this.set(no.USE_INTERIM_TILES_ON_ERROR, e);
	}, t;
}(Ms), hu = (function() {
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
})(), gu = function(e) {
	hu(t, e);
	function t(t) {
		var n = this, r = t || {};
		return n = e.call(this, r) || this, n;
	}
	return t;
}(gr), _u = (function() {
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
})(), vu = function(e) {
	_u(t, e);
	function t(t) {
		var n = e.call(this, t) || this;
		return n.image_ = null, n;
	}
	return t.prototype.getImage = function() {
		return this.image_ ? this.image_.getImage() : null;
	}, t.prototype.prepareFrame = function(e) {
		var t = e.layerStatesArray[e.layerIndex], n = e.pixelRatio, r = e.viewState, i = r.resolution, a = this.getLayer().getSource(), o = e.viewHints, s = e.extent;
		if (t.extent !== void 0 && (s = Yt(s, jn(t.extent, r.projection))), !o[J.ANIMATING] && !o[J.INTERACTING] && !$t(s)) if (a) {
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
		Jn(this.pixelTransform, e.size[0] / 2, e.size[1] / 2, 1 / s, 1 / s, h, -p / 2, -m / 2), Yn(this.inversePixelTransform, this.pixelTransform);
		var _ = Qn(this.pixelTransform);
		this.useContainer(t, _, o.opacity);
		var v = this.context, y = v.canvas;
		y.width != p || y.height != m ? (y.width = p, y.height = m) : this.containerReused || v.clearRect(0, 0, p, m);
		var b = !1, x = !0;
		if (o.extent) {
			var C = jn(o.extent, c.projection);
			x = Qt(C, e.extent), b = x && !Ot(C, e.extent), b && this.clipUnrotated(v, e, C);
		}
		var w = n.getImage(), T = Jn(this.tempTransform, p / 2, m / 2, f, f, 0, a * (r[0] - l[0]) / i, a * (l[1] - r[3]) / i);
		this.renderedResolution = i * s / a;
		var E = w.width * T[0], D = w.height * T[3];
		if (S(v, this.getLayer().getSource().getContextOptions()), this.preRender(v, e), x && E >= .5 && D >= .5) {
			var O = T[4], k = T[5], A = o.opacity, j = void 0;
			A !== 1 && (j = v.globalAlpha, v.globalAlpha = A), v.drawImage(w, 0, 0, +w.width, +w.height, Math.round(O), Math.round(k), Math.round(E), Math.round(D)), A !== 1 && (v.globalAlpha = j);
		}
		return this.postRender(v, e), b && v.restore(), _ !== y.style.transform && (y.style.transform = _), this.container;
	}, t;
}(co), yu = (function() {
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
})(), bu = function(e) {
	yu(t, e);
	function t(t) {
		return e.call(this, t) || this;
	}
	return t.prototype.createRenderer = function() {
		return new vu(this);
	}, t;
}(gu), xu = 34962, Su = 34963, Cu = 35040, wu = 35044, Tu = 35048, Eu = 5121, Du = 5123, Ou = 5125, ku = 5126, Au = [
	"experimental-webgl",
	"webgl",
	"webkit-3d",
	"moz-webgl"
];
function ju(e, t) {
	for (var n = Au.length, r = 0; r < n; ++r) try {
		var i = e.getContext(Au[r], t);
		if (i) return i;
	} catch {}
	return null;
}
//#endregion
//#region node_modules/ol/webgl/Buffer.js
var Mu = {
	STATIC_DRAW: wu,
	STREAM_DRAW: Cu,
	DYNAMIC_DRAW: Tu
}, Nu = function() {
	function e(e, t) {
		this.array = null, this.type = e, V(e === 34962 || e === 34963, 62), this.usage = t === void 0 ? Mu.STATIC_DRAW : t;
	}
	return e.prototype.ofSize = function(e) {
		this.array = new (Pu(this.type))(e);
	}, e.prototype.fromArray = function(e) {
		var t = Pu(this.type);
		this.array = t.from ? t.from(e) : new t(e);
	}, e.prototype.fromArrayBuffer = function(e) {
		this.array = new (Pu(this.type))(e);
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
function Pu(e) {
	switch (e) {
		case xu: return Float32Array;
		case Su: return Uint32Array;
		default: return Float32Array;
	}
}
//#endregion
//#region node_modules/ol/webgl/ContextEventType.js
var Fu = {
	LOST: "webglcontextlost",
	RESTORED: "webglcontextrestored"
}, Iu = "\n  precision mediump float;\n  \n  attribute vec2 a_position;\n  varying vec2 v_texCoord;\n  varying vec2 v_screenCoord;\n  \n  uniform vec2 u_screenSize;\n   \n  void main() {\n    v_texCoord = a_position * 0.5 + 0.5;\n    v_screenCoord = v_texCoord * u_screenSize;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n  }\n", Lu = "\n  precision mediump float;\n   \n  uniform sampler2D u_image;\n   \n  varying vec2 v_texCoord;\n   \n  void main() {\n    gl_FragColor = texture2D(u_image, v_texCoord);\n  }\n", Ru = function() {
	function e(e) {
		this.gl_ = e.webGlContext;
		var t = this.gl_;
		this.scaleRatio_ = e.scaleRatio || 1, this.renderTargetTexture_ = t.createTexture(), this.renderTargetTextureSize_ = null, this.frameBuffer_ = t.createFramebuffer();
		var n = t.createShader(t.VERTEX_SHADER);
		t.shaderSource(n, e.vertexShader || Iu), t.compileShader(n);
		var r = t.createShader(t.FRAGMENT_SHADER);
		t.shaderSource(r, e.fragmentShader || Lu), t.compileShader(r), this.renderTargetProgram_ = t.createProgram(), t.attachShader(this.renderTargetProgram_, n), t.attachShader(this.renderTargetProgram_, r), t.linkProgram(this.renderTargetProgram_), this.renderTargetVerticesBuffer_ = t.createBuffer(), t.bindBuffer(t.ARRAY_BUFFER, this.renderTargetVerticesBuffer_), t.bufferData(t.ARRAY_BUFFER, new Float32Array([
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
function zu() {
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
function Bu(e, t) {
	return e[0] = t[0], e[1] = t[1], e[4] = t[2], e[5] = t[3], e[12] = t[4], e[13] = t[5], e;
}
//#endregion
//#region node_modules/ol/webgl/Helper.js
var Vu = (function() {
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
})(), Hu = {
	PROJECTION_MATRIX: "u_projectionMatrix",
	OFFSET_SCALE_MATRIX: "u_offsetScaleMatrix",
	OFFSET_ROTATION_MATRIX: "u_offsetRotateMatrix",
	TIME: "u_time",
	ZOOM: "u_zoom",
	RESOLUTION: "u_resolution"
}, Uu = {
	UNSIGNED_BYTE: Eu,
	UNSIGNED_SHORT: Du,
	UNSIGNED_INT: Ou,
	FLOAT: ku
}, Wu = function(e) {
	Vu(t, e);
	function t(t) {
		var n = e.call(this) || this, r = t || {};
		n.boundHandleWebGLContextLost_ = n.handleWebGLContextLost.bind(n), n.boundHandleWebGLContextRestored_ = n.handleWebGLContextRestored.bind(n), n.canvas_ = document.createElement("canvas"), n.canvas_.style.position = "absolute", n.canvas_.style.left = "0", n.gl_ = ju(n.canvas_);
		var i = n.getGL();
		if (n.bufferCache_ = {}, n.extensionCache_ = {}, n.currentProgram_ = null, n.canvas_.addEventListener(Fu.LOST, n.boundHandleWebGLContextLost_), n.canvas_.addEventListener(Fu.RESTORED, n.boundHandleWebGLContextRestored_), n.offsetRotateMatrix_ = zn(), n.offsetScaleMatrix_ = zn(), n.tmpMat4_ = zu(), n.uniformLocations_ = {}, n.attribLocations_ = {}, n.uniforms_ = [], r.uniforms) for (var a in r.uniforms) n.uniforms_.push({
			name: a,
			value: r.uniforms[a]
		});
		return n.postProcessPasses_ = r.postProcesses ? r.postProcesses.map(function(e) {
			return new Ru({
				webGlContext: i,
				scaleRatio: e.scaleRatio,
				vertexShader: e.vertexShader,
				fragmentShader: e.fragmentShader,
				uniforms: e.uniforms
			});
		}) : [new Ru({ webGlContext: i })], n.shaderCompileErrors_ = null, n.startTime_ = Date.now(), n;
	}
	return t.prototype.getExtension = function(e) {
		if (e in this.extensionCache_) return this.extensionCache_[e];
		var t = this.gl_.getExtension(e);
		return this.extensionCache_[e] = t, t;
	}, t.prototype.bindBuffer = function(e) {
		var t = this.getGL(), n = I(e), r = this.bufferCache_[n];
		r || (r = {
			buffer: e,
			webGlBuffer: t.createBuffer()
		}, this.bufferCache_[n] = r), t.bindBuffer(e.getType(), r.webGlBuffer);
	}, t.prototype.flushBufferData = function(e) {
		var t = this.getGL();
		this.bindBuffer(e), t.bufferData(e.getType(), e.getArray(), e.getUsage());
	}, t.prototype.deleteBuffer = function(e) {
		var t = this.getGL(), n = I(e), r = this.bufferCache_[n];
		r && !t.isContextLost() && t.deleteBuffer(r.webGlBuffer), delete this.bufferCache_[n];
	}, t.prototype.disposeInternal = function() {
		this.canvas_.removeEventListener(Fu.LOST, this.boundHandleWebGLContextLost_), this.canvas_.removeEventListener(Fu.RESTORED, this.boundHandleWebGLContextRestored_);
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
		var t = e.size, n = e.viewState.rotation, r = Bn(this.offsetScaleMatrix_);
		Gn(r, 2 / t[0], 2 / t[1]);
		var i = Bn(this.offsetRotateMatrix_);
		n !== 0 && Wn(i, -n), this.setUniformMatrixValue(Hu.OFFSET_SCALE_MATRIX, Bu(this.tmpMat4_, r)), this.setUniformMatrixValue(Hu.OFFSET_ROTATION_MATRIX, Bu(this.tmpMat4_, i)), this.setUniformFloatValue(Hu.TIME, (Date.now() - this.startTime_) * .001), this.setUniformFloatValue(Hu.ZOOM, e.viewState.zoom), this.setUniformFloatValue(Hu.RESOLUTION, e.viewState.resolution);
	}, t.prototype.applyUniforms = function(e) {
		var t = this.getGL(), n, r = 0;
		this.uniforms_.forEach(function(i) {
			if (n = typeof i.value == "function" ? i.value(e) : i.value, n instanceof HTMLCanvasElement || n instanceof HTMLImageElement || n instanceof ImageData) i.texture ||= (i.prevValue = void 0, t.createTexture()), t.activeTexture(t["TEXTURE" + r]), t.bindTexture(t.TEXTURE_2D, i.texture), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), (!(n instanceof HTMLImageElement) || n.complete) && i.prevValue !== n && (i.prevValue = n, t.texImage2D(t.TEXTURE_2D, 0, t.RGBA, t.RGBA, t.UNSIGNED_BYTE, n)), t.uniform1i(this.getUniformLocation(i.name), r++);
			else if (Array.isArray(n) && n.length === 6) this.setUniformMatrixValue(i.name, Bu(this.tmpMat4_, n));
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
		return Bn(t), Jn(t, 0, 0, 2 / (i * n[0]), 2 / (i * n[1]), -r, -a[0], -a[1]), t;
	}, t.prototype.setUniformFloatValue = function(e, t) {
		this.getGL().uniform1f(this.getUniformLocation(e), t);
	}, t.prototype.setUniformMatrixValue = function(e, t) {
		this.getGL().uniformMatrix4fv(this.getUniformLocation(e), !1, t);
	}, t.prototype.enableAttributeArray_ = function(e, t, n, r, i) {
		var a = this.getAttributeLocation(e);
		a < 0 || (this.getGL().enableVertexAttribArray(a), this.getGL().vertexAttribPointer(a, t, n, !1, r, i));
	}, t.prototype.enableAttributes = function(e) {
		for (var t = Gu(e), n = 0, r = 0; r < e.length; r++) {
			var i = e[r];
			this.enableAttributeArray_(i.name, i.size, i.type || 5126, t, n), n += i.size * Ku(i.type);
		}
	}, t.prototype.handleWebGLContextLost = function() {
		C(this.bufferCache_), this.currentProgram_ = null;
	}, t.prototype.handleWebGLContextRestored = function() {}, t.prototype.createTexture = function(e, t, n) {
		var r = this.getGL(), i = n || r.createTexture(), a = 0, o = r.RGBA, s = 0, c = r.RGBA, l = r.UNSIGNED_BYTE;
		return r.bindTexture(r.TEXTURE_2D, i), t ? r.texImage2D(r.TEXTURE_2D, a, o, c, l, t) : r.texImage2D(r.TEXTURE_2D, a, o, e[0], e[1], s, c, l, null), r.texParameteri(r.TEXTURE_2D, r.TEXTURE_MIN_FILTER, r.LINEAR), r.texParameteri(r.TEXTURE_2D, r.TEXTURE_WRAP_S, r.CLAMP_TO_EDGE), r.texParameteri(r.TEXTURE_2D, r.TEXTURE_WRAP_T, r.CLAMP_TO_EDGE), i;
	}, t;
}(d);
function Gu(e) {
	for (var t = 0, n = 0; n < e.length; n++) {
		var r = e[n];
		t += r.size * Ku(r.type);
	}
	return t;
}
function Ku(e) {
	switch (e) {
		case Uu.UNSIGNED_BYTE: return Uint8Array.BYTES_PER_ELEMENT;
		case Uu.UNSIGNED_SHORT: return Uint16Array.BYTES_PER_ELEMENT;
		case Uu.UNSIGNED_INT: return Uint32Array.BYTES_PER_ELEMENT;
		case Uu.FLOAT:
		default: return Float32Array.BYTES_PER_ELEMENT;
	}
}
//#endregion
//#region node_modules/ol/renderer/webgl/Layer.js
var qu = (function() {
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
})(), Ju = { GENERATE_BUFFERS: "GENERATE_BUFFERS" }, Yu = function(e) {
	qu(t, e);
	function t(t, n) {
		var r = e.call(this, t) || this, i = n || {};
		return r.helper = new Wu({
			postProcesses: i.postProcesses,
			uniforms: i.uniforms
		}), i.className !== void 0 && (r.helper.getCanvas().className = i.className), r;
	}
	return t.prototype.disposeInternal = function() {
		this.helper.dispose(), delete this.helper, e.prototype.disposeInternal.call(this);
	}, t.prototype.dispatchRenderEvent_ = function(e, t) {
		var n = this.getLayer();
		if (n.hasListener(e)) {
			var r = new Sr(e, null, t, null);
			n.dispatchEvent(r);
		}
	}, t.prototype.preRender = function(e) {
		this.dispatchRenderEvent_(pr.PRERENDER, e);
	}, t.prototype.postRender = function(e) {
		this.dispatchRenderEvent_(pr.POSTRENDER, e);
	}, t;
}(oo);
function Xu(e, t) {
	var n = t || [], r = 256, i = r - 1;
	return n[0] = Math.floor(e / r / r / r) / i, n[1] = Math.floor(e / r / r) % r / i, n[2] = Math.floor(e / r) % r / i, n[3] = e % r / i, n;
}
function Zu(e) {
	var t = 0, n = 256, r = n - 1;
	return t += Math.round(e[0] * n * n * n * r), t += Math.round(e[1] * n * n * r), t += Math.round(e[2] * n * r), t += Math.round(e[3] * r), t;
}
//#endregion
//#region node_modules/ol/webgl/RenderTarget.js
var Qu = /* @__PURE__ */ new Uint8Array(4), $u = function() {
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
		if (e < 0 || t < 0 || e > this.size_[0] || t >= this.size_[1]) return Qu[0] = 0, Qu[1] = 0, Qu[2] = 0, Qu[3] = 0, Qu;
		this.readAll();
		var n = Math.floor(e) + (this.size_[1] - Math.floor(t) - 1) * this.size_[0];
		return Qu[0] = this.data_[n * 4], Qu[1] = this.data_[n * 4 + 1], Qu[2] = this.data_[n * 4 + 2], Qu[3] = this.data_[n * 4 + 3], Qu;
	}, e.prototype.getTexture = function() {
		return this.texture_;
	}, e.prototype.getFramebuffer = function() {
		return this.framebuffer_;
	}, e.prototype.updateSize_ = function() {
		var e = this.size_, t = this.helper_.getGL();
		this.texture_ = this.helper_.createTexture(e, null, this.texture_), t.bindFramebuffer(t.FRAMEBUFFER, this.framebuffer_), t.viewport(0, 0, e[0], e[1]), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, this.texture_, 0), this.data_ = new Uint8Array(e[0] * e[1] * 4);
	}, e;
}(), ed = new Blob(["var e=\"function\"==typeof Object.assign?Object.assign:function(e,n){if(null==e)throw new TypeError(\"Cannot convert undefined or null to object\");for(var t=Object(e),r=1,o=arguments.length;r<o;++r){var i=arguments[r];if(null!=i)for(var f in i)i.hasOwnProperty(f)&&(t[f]=i[f])}return t},n=\"GENERATE_BUFFERS\",t=[],r={vertexPosition:0,indexPosition:0};function o(e,n,t,r,o){e[n+0]=t,e[n+1]=r,e[n+2]=o}function i(e,n,i,f,s,u){var a=3+s,l=e[n+0],v=e[n+1],c=t;c.length=s;for(var g=0;g<c.length;g++)c[g]=e[n+2+g];var b=u?u.vertexPosition:0,h=u?u.indexPosition:0,d=b/a;return o(i,b,l,v,0),c.length&&i.set(c,b+3),o(i,b+=a,l,v,1),c.length&&i.set(c,b+3),o(i,b+=a,l,v,2),c.length&&i.set(c,b+3),o(i,b+=a,l,v,3),c.length&&i.set(c,b+3),b+=a,f[h++]=d,f[h++]=d+1,f[h++]=d+3,f[h++]=d+1,f[h++]=d+2,f[h++]=d+3,r.vertexPosition=b,r.indexPosition=h,r}var f=self;f.onmessage=function(t){var r=t.data;if(r.type===n){for(var o=r.customAttributesCount,s=2+o,u=new Float32Array(r.renderInstructions),a=u.length/s,l=4*a*(o+3),v=new Uint32Array(6*a),c=new Float32Array(l),g=null,b=0;b<u.length;b+=s)g=i(u,b,c,v,o,g);var h=e({vertexBuffer:c.buffer,indexBuffer:v.buffer,renderInstructions:u.buffer},r);f.postMessage(h,[c.buffer,v.buffer,u.buffer])}};"], { type: "application/javascript" }), td = URL.createObjectURL(ed);
function nd() {
	return new Worker(td);
}
//#endregion
//#region node_modules/ol/renderer/webgl/PointsLayer.js
var rd = (function() {
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
})(), id = function(e) {
	rd(t, e);
	function t(t, n) {
		var r = this, i = n.uniforms || {}, a = zn();
		i[Hu.PROJECTION_MATRIX] = a, r = e.call(this, t, {
			className: n.className,
			uniforms: i,
			postProcesses: n.postProcesses
		}) || this, r.sourceRevision_ = -1, r.verticesBuffer_ = new Nu(xu, Tu), r.hitVerticesBuffer_ = new Nu(xu, Tu), r.indicesBuffer_ = new Nu(Su, Tu), r.program_ = r.helper.getProgram(n.fragmentShader, n.vertexShader), r.hitDetectionEnabled_ = !!(n.hitFragmentShader && n.hitVertexShader), r.hitProgram_ = r.hitDetectionEnabled_ && r.helper.getProgram(n.hitFragmentShader, n.hitVertexShader);
		var o = n.attributes ? n.attributes.map(function(e) {
			return {
				name: "a_" + e.name,
				size: 1,
				type: Uu.FLOAT
			};
		}) : [];
		r.attributes = [{
			name: "a_position",
			size: 2,
			type: Uu.FLOAT
		}, {
			name: "a_index",
			size: 1,
			type: Uu.FLOAT
		}].concat(o), r.hitDetectionAttributes = [
			{
				name: "a_position",
				size: 2,
				type: Uu.FLOAT
			},
			{
				name: "a_index",
				size: 1,
				type: Uu.FLOAT
			},
			{
				name: "a_hitColor",
				size: 4,
				type: Uu.FLOAT
			},
			{
				name: "a_featureUid",
				size: 1,
				type: Uu.FLOAT
			}
		].concat(o), r.customAttributes = n.attributes ? n.attributes : [], r.previousExtent_ = jt(), r.currentTransform_ = a, r.renderTransform_ = zn(), r.invertRenderTransform_ = zn(), r.renderInstructions_ = /* @__PURE__ */ new Float32Array(), r.hitRenderInstructions_ = /* @__PURE__ */ new Float32Array(), r.hitRenderTarget_ = r.hitDetectionEnabled_ && new $u(r.helper), r.worker_ = nd(), r.worker_.addEventListener("message", function(e) {
			var t = e.data;
			if (t.type === Ju.GENERATE_BUFFERS) {
				var n = t.projectionTransform;
				t.hitDetection ? (this.hitVerticesBuffer_.fromArrayBuffer(t.vertexBuffer), this.helper.flushBufferData(this.hitVerticesBuffer_)) : (this.verticesBuffer_.fromArrayBuffer(t.vertexBuffer), this.helper.flushBufferData(this.verticesBuffer_)), this.indicesBuffer_.fromArrayBuffer(t.indexBuffer), this.helper.flushBufferData(this.indicesBuffer_), this.renderTransform_ = n, Yn(this.invertRenderTransform_, this.renderTransform_), t.hitDetection ? this.hitRenderInstructions_ = new Float32Array(e.data.renderInstructions) : this.renderInstructions_ = new Float32Array(e.data.renderInstructions), this.getLayer().changed();
			}
		}.bind(r)), r.featureCache_ = {}, r.featureCount_ = 0;
		var s = r.getLayer().getSource();
		return r.sourceListenKeys_ = [
			k(s, el.ADDFEATURE, r.handleSourceFeatureAdded_, r),
			k(s, el.CHANGEFEATURE, r.handleSourceFeatureChanged_, r),
			k(s, el.REMOVEFEATURE, r.handleSourceFeatureDelete_, r),
			k(s, el.CLEAR, r.handleSourceFeatureClear_, r)
		], s.forEachFeature(function(e) {
			this.featureCache_[I(e)] = {
				feature: e,
				properties: e.getProperties(),
				geometry: e.getGeometry()
			}, this.featureCount_++;
		}.bind(r)), r;
	}
	return t.prototype.handleSourceFeatureAdded_ = function(e) {
		var t = e.feature;
		this.featureCache_[I(t)] = {
			feature: t,
			properties: t.getProperties(),
			geometry: t.getGeometry()
		}, this.featureCount_++;
	}, t.prototype.handleSourceFeatureChanged_ = function(e) {
		var t = e.feature;
		this.featureCache_[I(t)] = {
			feature: t,
			properties: t.getProperties(),
			geometry: t.getGeometry()
		};
	}, t.prototype.handleSourceFeatureDelete_ = function(e) {
		var t = e.feature;
		delete this.featureCache_[I(t)], this.featureCount_--;
	}, t.prototype.handleSourceFeatureClear_ = function() {
		this.featureCache_ = {}, this.featureCount_ = 0;
	}, t.prototype.renderFrame = function(e) {
		this.preRender(e);
		var t = this.indicesBuffer_.getSize();
		this.helper.drawElements(0, t), this.helper.finalizeDraw(e);
		var n = this.helper.getCanvas(), r = e.layerStatesArray[e.layerIndex].opacity;
		return r !== parseFloat(n.style.opacity) && (n.style.opacity = String(r)), this.hitDetectionEnabled_ && (this.renderHitDetection(e), this.hitRenderTarget_.clearCachedData()), this.postRender(e), n;
	}, t.prototype.prepareFrame = function(e) {
		var t = this.getLayer(), n = t.getSource(), r = e.viewState, i = !e.viewHints[J.ANIMATING] && !e.viewHints[J.INTERACTING], a = !It(this.previousExtent_, e.extent), o = this.sourceRevision_ < n.getRevision();
		if (o && (this.sourceRevision_ = n.getRevision()), i && (a || o)) {
			var s = r.projection, c = r.resolution, l = t instanceof Ms ? t.getRenderBuffer() : 0, u = wt(e.extent, l * c);
			n.loadFeatures(u, c, s), this.rebuildBuffers_(e), this.previousExtent_ = e.extent.slice();
		}
		return this.helper.makeProjectionTransform(e, this.currentTransform_), Vn(this.currentTransform_, this.invertRenderTransform_), this.helper.useProgram(this.program_), this.helper.prepareDraw(e), this.helper.bindBuffer(this.verticesBuffer_), this.helper.bindBuffer(this.indicesBuffer_), this.helper.enableAttributes(this.attributes), !0;
	}, t.prototype.rebuildBuffers_ = function(e) {
		var t = zn();
		this.helper.makeProjectionTransform(e, t);
		var n = (2 + this.customAttributes.length) * this.featureCount_;
		if ((!this.renderInstructions_ || this.renderInstructions_.length !== n) && (this.renderInstructions_ = new Float32Array(n)), this.hitDetectionEnabled_) {
			var r = (7 + this.customAttributes.length) * this.featureCount_;
			(!this.hitRenderInstructions_ || this.hitRenderInstructions_.length !== r) && (this.hitRenderInstructions_ = new Float32Array(r));
		}
		var i, a, o = [], s = [], c = 0, l = 0, u;
		for (var d in this.featureCache_) if (i = this.featureCache_[d], a = i.geometry, !(!a || a.getType() !== U.POINT)) {
			o[0] = a.getFlatCoordinates()[0], o[1] = a.getFlatCoordinates()[1], W(t, o), u = Xu(l + 6, s), this.renderInstructions_[c++] = o[0], this.renderInstructions_[c++] = o[1], this.hitDetectionEnabled_ && (this.hitRenderInstructions_[l++] = o[0], this.hitRenderInstructions_[l++] = o[1], this.hitRenderInstructions_[l++] = u[0], this.hitRenderInstructions_[l++] = u[1], this.hitRenderInstructions_[l++] = u[2], this.hitRenderInstructions_[l++] = u[3], this.hitRenderInstructions_[l++] = Number(d));
			for (var f = void 0, p = 0; p < this.customAttributes.length; p++) f = this.customAttributes[p].callback(i.feature, i.properties), this.renderInstructions_[c++] = f, this.hitDetectionEnabled_ && (this.hitRenderInstructions_[l++] = f);
		}
		var m = {
			type: Ju.GENERATE_BUFFERS,
			renderInstructions: this.renderInstructions_.buffer,
			customAttributesCount: this.customAttributes.length
		};
		if (m.projectionTransform = t, this.worker_.postMessage(m, [this.renderInstructions_.buffer]), this.renderInstructions_ = null, this.hitDetectionEnabled_) {
			var h = {
				type: Ju.GENERATE_BUFFERS,
				renderInstructions: this.hitRenderInstructions_.buffer,
				customAttributesCount: 5 + this.customAttributes.length
			};
			h.projectionTransform = t, h.hitDetection = !0, this.worker_.postMessage(h, [this.hitRenderInstructions_.buffer]), this.hitRenderInstructions_ = null;
		}
	}, t.prototype.forEachFeatureAtCoordinate = function(e, t, n, r, i) {
		if (V(this.hitDetectionEnabled_, 66), this.hitRenderInstructions_) {
			var a = W(t.coordinateToPixelTransform, e.slice()), o = this.hitRenderTarget_.readPixel(a[0] / 2, a[1] / 2), s = Zu([
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
}(Yu), ad = (function() {
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
})(), od = {
	BLUR: "blur",
	GRADIENT: "gradient",
	RADIUS: "radius"
}, sd = [
	"#00f",
	"#0ff",
	"#0f0",
	"#ff0",
	"#f00"
], cd = function(e) {
	ad(t, e);
	function t(t) {
		var n = this, r = t || {}, i = S({}, r);
		delete i.gradient, delete i.radius, delete i.blur, delete i.weight, n = e.call(this, i) || this, n.gradient_ = null, n.addChangeListener(od.GRADIENT, n.handleGradientChanged_), n.setGradient(r.gradient ? r.gradient : sd), n.setBlur(r.blur === void 0 ? 15 : r.blur), n.setRadius(r.radius === void 0 ? 8 : r.radius);
		var a = r.weight ? r.weight : "weight";
		return typeof a == "string" ? n.weightFunction_ = function(e) {
			return e.get(a);
		} : n.weightFunction_ = a, n.setRenderOrder(null), n;
	}
	return t.prototype.getBlur = function() {
		return this.get(od.BLUR);
	}, t.prototype.getGradient = function() {
		return this.get(od.GRADIENT);
	}, t.prototype.getRadius = function() {
		return this.get(od.RADIUS);
	}, t.prototype.handleGradientChanged_ = function() {
		this.gradient_ = ld(this.getGradient());
	}, t.prototype.setBlur = function(e) {
		this.set(od.BLUR, e);
	}, t.prototype.setGradient = function(e) {
		this.set(od.GRADIENT, e);
	}, t.prototype.setRadius = function(e) {
		this.set(od.RADIUS, e);
	}, t.prototype.createRenderer = function() {
		return new id(this, {
			className: this.getClassName(),
			attributes: [{
				name: "weight",
				callback: function(e) {
					var t = this.weightFunction_(e);
					return t === void 0 ? 1 : B(t, 0, 1);
				}.bind(this)
			}],
			vertexShader: "\n        precision mediump float;\n        uniform mat4 u_projectionMatrix;\n        uniform mat4 u_offsetScaleMatrix;\n        uniform float u_size;\n        attribute vec2 a_position;\n        attribute float a_index;\n        attribute float a_weight;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n\n        void main(void) {\n          mat4 offsetMatrix = u_offsetScaleMatrix;\n          float offsetX = a_index == 0.0 || a_index == 3.0 ? -u_size / 2.0 : u_size / 2.0;\n          float offsetY = a_index == 0.0 || a_index == 1.0 ? -u_size / 2.0 : u_size / 2.0;\n          vec4 offsets = offsetMatrix * vec4(offsetX, offsetY, 0.0, 0.0);\n          gl_Position = u_projectionMatrix * vec4(a_position, 0.0, 1.0) + offsets;\n          float u = a_index == 0.0 || a_index == 3.0 ? 0.0 : 1.0;\n          float v = a_index == 0.0 || a_index == 1.0 ? 0.0 : 1.0;\n          v_texCoord = vec2(u, v);\n          v_weight = a_weight;\n        }",
			fragmentShader: "\n        precision mediump float;\n        uniform float u_blurSlope;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n\n        void main(void) {\n          vec2 texCoord = v_texCoord * 2.0 - vec2(1.0, 1.0);\n          float sqRadius = texCoord.x * texCoord.x + texCoord.y * texCoord.y;\n          float value = (1.0 - sqrt(sqRadius)) * u_blurSlope;\n          float alpha = smoothstep(0.0, 1.0, value) * v_weight;\n          gl_FragColor = vec4(alpha, alpha, alpha, alpha);\n        }",
			hitVertexShader: "\n        precision mediump float;\n        uniform mat4 u_projectionMatrix;\n        uniform mat4 u_offsetScaleMatrix;\n        uniform float u_size;\n        attribute vec2 a_position;\n        attribute float a_index;\n        attribute float a_weight;\n        attribute vec4 a_hitColor;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n        varying vec4 v_hitColor;\n\n        void main(void) {\n          mat4 offsetMatrix = u_offsetScaleMatrix;\n          float offsetX = a_index == 0.0 || a_index == 3.0 ? -u_size / 2.0 : u_size / 2.0;\n          float offsetY = a_index == 0.0 || a_index == 1.0 ? -u_size / 2.0 : u_size / 2.0;\n          vec4 offsets = offsetMatrix * vec4(offsetX, offsetY, 0.0, 0.0);\n          gl_Position = u_projectionMatrix * vec4(a_position, 0.0, 1.0) + offsets;\n          float u = a_index == 0.0 || a_index == 3.0 ? 0.0 : 1.0;\n          float v = a_index == 0.0 || a_index == 1.0 ? 0.0 : 1.0;\n          v_texCoord = vec2(u, v);\n          v_hitColor = a_hitColor;\n          v_weight = a_weight;\n        }",
			hitFragmentShader: "\n        precision mediump float;\n        uniform float u_blurSlope;\n\n        varying vec2 v_texCoord;\n        varying float v_weight;\n        varying vec4 v_hitColor;\n\n        void main(void) {\n          vec2 texCoord = v_texCoord * 2.0 - vec2(1.0, 1.0);\n          float sqRadius = texCoord.x * texCoord.x + texCoord.y * texCoord.y;\n          float value = (1.0 - sqrt(sqRadius)) * u_blurSlope;\n          float alpha = smoothstep(0.0, 1.0, value) * v_weight;\n          if (alpha < 0.05) {\n            discard;\n          }\n\n          gl_FragColor = v_hitColor;\n        }",
			uniforms: {
				u_size: function() {
					return (this.get(od.RADIUS) + this.get(od.BLUR)) * 2;
				}.bind(this),
				u_blurSlope: function() {
					return this.get(od.RADIUS) / Math.max(1, this.get(od.BLUR));
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
}(Xc);
function ld(e) {
	for (var t = 1, n = 256, r = fe(t, n), i = r.createLinearGradient(0, 0, t, n), a = 1 / (e.length - 1), o = 0, s = e.length; o < s; ++o) i.addColorStop(o * a, e[o]);
	return r.fillStyle = i, r.fillRect(0, 0, t, n), r.canvas;
}
//#endregion
//#region node_modules/ol-ext/util/input/Base.js
window.ol && (ol.ext.input = {});
var ud = function(e) {
	e ||= {}, R.call(this);
	var t = this.input = e.input;
	t || (t = this.input = document.createElement("INPUT"), e.type && t.setAttribute("type", e.type), e.min !== void 0 && t.setAttribute("min", e.min), e.max !== void 0 && t.setAttribute("max", e.max), e.step !== void 0 && t.setAttribute("step", e.step), e.parent && e.parent.appendChild(t)), e.disabled && (t.disabled = !0), e.checked !== void 0 && (t.checked = !!e.checked), e.val !== void 0 && (t.value = e.val), e.hidden && (t.style.display = "none");
};
iu(ud, R), ud.prototype._listenDrag = function(e, t) {
	var n = function(n) {
		this.moving = !0;
		var r = function(n) {
			n.type === "pointerup" && (document.removeEventListener("pointermove", r), document.removeEventListener("pointerup", r), document.removeEventListener("pointercancel", r), setTimeout(function() {
				this.moving = !1;
			}.bind(this))), n.target === e && t(n), n.stopPropagation(), n.preventDefault();
		}.bind(this);
		document.addEventListener("pointermove", r, !1), document.addEventListener("pointerup", r, !1), document.addEventListener("pointercancel", r, !1), n.stopPropagation(), n.preventDefault();
	}.bind(this);
	e.addEventListener("mousedown", n, !1), e.addEventListener("touchstart", n, !1);
}, ud.prototype.setValue = function(e) {
	e !== void 0 && (this.input.value = e), this.input.dispatchEvent(new Event("change"));
}, ud.prototype.getValue = function() {
	return this.input.value;
}, ud.prototype.getInputElement = function() {
	return this.input;
};
//#endregion
//#region node_modules/ol-ext/util/input/Checkbox.js
var dd = function(e) {
	e ||= {}, ud.call(this, e);
	var t = this.element = document.createElement("LABEL");
	e.html instanceof Element ? t.appendChild(e.html) : e.html !== void 0 && (t.innerHTML = e.html), t.className = ("ol-ext-check ol-ext-checkbox" + (e.className || "")).trim(), this.input.parentNode && this.input.parentNode.insertBefore(t, this.input), t.appendChild(this.input), t.appendChild(document.createElement("SPAN")), e.after && t.appendChild(document.createTextNode(e.after)), this.input.addEventListener("change", function() {
		this.dispatchEvent({
			type: "check",
			checked: this.input.checked,
			value: this.input.value
		});
	}.bind(this));
};
iu(dd, ud), dd.prototype.isChecked = function() {
	return this.input.checked;
};
//#endregion
//#region node_modules/ol-ext/util/input/Switch.js
var fd = function(e) {
	e ||= {}, dd.call(this, e), this.element.className = ("ol-ext-toggle-switch " + (e.className || "")).trim();
};
iu(fd, dd);
//#endregion
//#region node_modules/ol-ext/util/input/Radio.js
var pd = function(e, t) {
	t ||= {}, dd.call(this, e, t), this.element.className = ("ol-ext-check ol-ext-radio" + (t.className || "")).trim();
};
iu(pd, dd);
//#endregion
//#region node_modules/ol-ext/util/element.js
var Q = {};
Q.create = function(e, t) {
	t ||= {};
	var n;
	if (e === "TEXT") n = document.createTextNode(t.html || ""), t.parent && t.parent.appendChild(n);
	else for (var r in n = document.createElement(e), /button/i.test(e) && n.setAttribute("type", "button"), t) switch (r) {
		case "className":
			t.className && t.className.trim && n.setAttribute("class", t.className.trim());
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
			this.setStyle(n, t.style);
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
		default:
			n.setAttribute(r, t[r]);
			break;
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
	return new fd(Object.assign({ input: t }, e || {})), t;
}, Q.createCheck = function(e) {
	var t = Q.create("INPUT", {
		name: e.name,
		type: e.type === "radio" ? "radio" : "checkbox",
		on: e.on,
		parent: e.parent
	});
	console.log(t);
	var n = Object.assign({ input: t }, e || {});
	return e.type === "radio" ? new pd(n) : new dd(n), t;
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
		if (l = !0, n !== !1) {
			var f = (d ? -1 / u : 1) * (n - t[s]);
			e[c] += f, i = /* @__PURE__ */ new Date(), i - a && (r = (r + f / (i - a)) / 2), n = t[s], a = i, f && o(!0);
		}
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
};
//#endregion
//#region node_modules/ol-ext/control/LayerSwitcher.js
var $ = function(e) {
	e ||= {};
	var t = this;
	this.dcount = 0, this.show_progress = e.show_progress, this.oninfo = typeof e.oninfo == "function" ? e.oninfo : null, this.onextent = typeof e.onextent == "function" ? e.onextent : null, this.hasextent = e.extent || e.onextent, this.hastrash = e.trash, this.reordering = e.reordering !== !1, this._layers = [], this._layerGroup = e.layerGroup && e.layerGroup.getLayers ? e.layerGroup : null, typeof e.displayInLayerSwitcher == "function" && (this.displayInLayerSwitcher = e.displayInLayerSwitcher);
	var n;
	e.target ? n = Q.create("DIV", { className: e.switcherClass || "ol-layerswitcher" }) : (n = Q.create("DIV", { className: (e.switcherClass || "ol-layerswitcher") + " ol-unselectable ol-control" }), e.collapsed === !1 ? n.classList.add("ol-forceopen") : n.classList.add("ol-collapsed"), this.button = Q.create("BUTTON", {
		type: "button",
		parent: n
	}), this.button.addEventListener("touchstart", function(e) {
		n.classList.toggle("ol-forceopen"), n.classList.toggle("ol-collapsed"), t.dispatchEvent({
			type: "toggle",
			collapsed: n.classList.contains("ol-collapsed")
		}), e.preventDefault(), t.overflow();
	}), this.button.addEventListener("click", function() {
		n.classList.toggle("ol-forceopen"), n.classList.add("ol-collapsed"), t.dispatchEvent({
			type: "toggle",
			collapsed: !n.classList.contains("ol-forceopen")
		}), t.overflow();
	}), e.mouseover && (n.addEventListener("mouseleave", function() {
		n.classList.add("ol-collapsed"), t.dispatchEvent({
			type: "toggle",
			collapsed: !0
		});
	}), n.addEventListener("mouseover", function() {
		n.classList.remove("ol-collapsed"), t.dispatchEvent({
			type: "toggle",
			collapsed: !1
		});
	})), e.minibar && (e.noScroll = !0), e.noScroll || (this.topv = Q.create("DIV", {
		className: "ol-switchertopdiv",
		parent: n,
		click: function() {
			t.overflow("+50%");
		}
	}), this.botv = Q.create("DIV", {
		className: "ol-switcherbottomdiv",
		parent: n,
		click: function() {
			t.overflow("-50%");
		}
	})), this._noScroll = e.noScroll), this.panel_ = Q.create("UL", { className: "panel" }), this.panelContainer_ = Q.create("DIV", {
		className: "panel-container",
		html: this.panel_,
		parent: n
	}), !e.target && !e.noScroll && Q.addListener(this.panel_, "mousewheel DOMMouseScroll onmousewheel", function(e) {
		t.overflow(Math.max(-1, Math.min(1, e.wheelDelta || -e.detail))) && (e.stopPropagation(), e.preventDefault());
	}), this.header_ = Q.create("LI", {
		className: "ol-header",
		parent: this.panel_
	}), be.call(this, {
		element: n,
		target: e.target
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
};
iu($, be), $.prototype.tip = {
	up: "up/down",
	down: "down",
	info: "informations...",
	extent: "zoom to extent",
	trash: "remove layer",
	plus: "expand/shrink"
}, $.prototype.displayInLayerSwitcher = function(e) {
	return e.get("displayInLayerSwitcher") !== !1;
}, $.prototype.setMap = function(e) {
	if (be.prototype.setMap.call(this, e), this.drawPanel(), this._listener) for (var t in this._listener) P(this._listener[t]);
	this._listener = null, e && (this._listener = {
		moveend: e.on("moveend", this.viewChange.bind(this)),
		size: e.on("change:size", this.overflow.bind(this))
	}, this._layerGroup ? this._listener.change = this._layerGroup.getLayers().on("change:length", this.drawPanel.bind(this)) : this._listener.change = e.getLayerGroup().getLayers().on("change:length", this.drawPanel.bind(this)));
}, $.prototype.show = function() {
	this.element.classList.add("ol-forceopen"), this.overflow(), self.dispatchEvent({
		type: "toggle",
		collapsed: !1
	});
}, $.prototype.hide = function() {
	this.element.classList.remove("ol-forceopen"), this.overflow(), self.dispatchEvent({
		type: "toggle",
		collapsed: !0
	});
}, $.prototype.toggle = function() {
	this.element.classList.toggle("ol-forceopen"), this.overflow();
}, $.prototype.isOpen = function() {
	return this.element.classList.contains("ol-forceopen");
}, $.prototype.setHeader = function(e) {
	Q.setHTML(this.header_, e);
}, $.prototype.overflow = function(e) {
	if (this.button && !this._noScroll) {
		if (Q.hidden(this.panel_)) {
			Q.setStyle(this.element, { height: "auto" });
			return;
		}
		var t = Q.outerHeight(this.element), n = Q.outerHeight(this.panel_), r = this.button.offsetTop + Q.outerHeight(this.button), i = this.panel_.offsetTop - r;
		if (n > t - r) {
			Q.setStyle(this.element, { height: "100%" });
			var a = this.panel_.querySelectorAll("li.visible .li-content")[0], o = a ? 2 * Q.getStyle(a, "height") : 0;
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
			return i + n <= t - 3 * r / 2 ? (i = t - 3 * r / 2 - n, Q.hide(this.botv)) : Q.show(this.botv), i >= 0 ? (i = 0, Q.hide(this.topv)) : Q.show(this.topv), Q.setStyle(this.panel_, { top: i + "px" }), !0;
		} else return Q.setStyle(this.element, { height: "auto" }), Q.setStyle(this.panel_, { top: 0 }), Q.hide(this.botv), Q.hide(this.topv), !1;
	} else return !1;
}, $.prototype._setLayerForLI = function(e, t) {
	var n = [];
	t.getLayers && n.push(t.getLayers().on("change:length", this.drawPanel.bind(this))), e && (n.push(t.on("change:opacity", (function() {
		this.setLayerOpacity(t, e);
	}).bind(this))), n.push(t.on("change:visible", (function() {
		this.setLayerVisibility(t, e);
	}).bind(this)))), n.push(t.on("propertychange", (function(e) {
		(e.key === "displayInLayerSwitcher" || e.key === "openInLayerSwitcher") && this.drawPanel(e);
	}).bind(this))), this._layers.push({
		li: e,
		layer: t,
		listeners: n
	});
}, $.prototype.setLayerOpacity = function(e, t) {
	var n = t.querySelector(".layerswitcher-opacity-cursor");
	n && (n.style.left = e.getOpacity() * 100 + "%"), this.dispatchEvent({
		type: "layer:opacity",
		layer: e
	});
}, $.prototype.setLayerVisibility = function(e, t) {
	var n = t.querySelector(".ol-visibility");
	n && (n.checked = e.getVisible()), e.getVisible() ? t.classList.add("ol-visible") : t.classList.remove("ol-visible"), this.dispatchEvent({
		type: "layer:visible",
		layer: e
	});
}, $.prototype._clearLayerForLI = function() {
	this._layers.forEach(function(e) {
		e.listeners.forEach(function(e) {
			P(e);
		});
	}), this._layers = [];
}, $.prototype._getLayerForLI = function(e) {
	for (var t = 0, n; n = this._layers[t]; t++) if (n.li === e) return n.layer;
	return null;
}, $.prototype.viewChange = function() {
	this.panel_.querySelectorAll("li").forEach(function(e) {
		var t = this._getLayerForLI(e);
		t && (this.testLayerVisibility(t) ? e.classList.remove("ol-layer-hidden") : e.classList.add("ol-layer-hidden"));
	}.bind(this));
}, $.prototype.getPanel = function() {
	return this.panelContainer_;
}, $.prototype.drawPanel = function() {
	if (this.getMap()) {
		var e = this;
		this.dcount++, setTimeout(function() {
			e.drawPanel_();
		}, this.get("drawDelay") || 0);
	}
}, $.prototype.drawPanel_ = function() {
	if (!(--this.dcount || this.dragging_)) {
		var e = this.panelContainer_.scrollTop;
		this._clearLayerForLI(), this.panel_.querySelectorAll("li").forEach(function(e) {
			e.classList.contains("ol-header") || e.remove();
		}.bind(this)), this._layerGroup ? this.drawList(this.panel_, this._layerGroup.getLayers()) : this.getMap() && this.drawList(this.panel_, this.getMap().getLayers()), this.panelContainer_.scrollTop = e;
	}
}, $.prototype.switchLayerVisibility = function(e, t) {
	e.get("baseLayer") ? (e.getVisible() || e.setVisible(!0), t.forEach(function(t) {
		e !== t && t.get("baseLayer") && t.getVisible() && t.setVisible(!1);
	})) : e.setVisible(!e.getVisible());
}, $.prototype.testLayerVisibility = function(e) {
	if (!this.getMap()) return !0;
	var t = this.getMap().getView().getResolution(), n = this.getMap().getView().getZoom();
	if (e.getMaxResolution() <= t || e.getMinResolution() >= t || e.getMinZoom && (e.getMinZoom() >= n || e.getMaxZoom() < n)) return !1;
	var r = e.getExtent();
	return !r || Qt(this.getMap().getView().calculateExtent(this.getMap().getSize()), r);
}, $.prototype.dragOrdering_ = function(e) {
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
			var d = e.touches ? document.elementFromPoint(e.touches[0].clientX, e.touches[0].clientY) : e.target;
			for (d.classList.contains("ol-switcherbottomdiv") ? t.overflow(-1) : d.classList.contains("ol-switchertopdiv") && t.overflow(1); d && d.tagName !== "LI";) d = d.parentNode;
			(!d || !d.classList.contains("dropover")) && n.parentNode.querySelectorAll("li").forEach(function(e) {
				e.classList.remove("dropover"), e.classList.remove("dropover-after"), e.classList.remove("dropover-before");
			}), d && d.parentNode.classList.contains("drag") && d !== n ? (s = t._getLayerForLI(d), s && !s.get("allwaysOnTop") == !l.get("allwaysOnTop") ? (d.classList.add("dropover"), d.classList.add(n.offsetTop < d.offsetTop ? "dropover-after" : "dropover-before")) : s = !1, Q.show(c)) : (s = !1, d === n ? Q.hide(c) : Q.show(c)), s ? c.classList.remove("forbidden") : c.classList.add("forbidden");
		}
	}
	Q.addListener(document, "mousemove touchmove", f), Q.addListener(document, "mouseup touchend touchcancel", d);
}, $.prototype.dragOpacity_ = function(e) {
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
}, $.prototype.drawList = function(e, t) {
	var n = this, r = t.getArray(), i = function(e) {
		e.stopPropagation(), e.preventDefault();
		var r = n._getLayerForLI(this.parentNode.parentNode);
		n.switchLayerVisibility(r, t), n.get("selection") && r.getVisible() && n.selectLayer(r);
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
	function c(t) {
		if (!this.displayInLayerSwitcher(t)) {
			this._setLayerForLI(null, t);
			return;
		}
		var c = Q.create("LI", {
			className: (t.getVisible() ? "visible " : " ") + (t.get("baseLayer") ? "baselayer" : ""),
			parent: e
		});
		this._setLayerForLI(c, t), this._selectedLayer === t && c.classList.add("ol-layer-select");
		var u = Q.create("DIV", {
			className: "ol-layerswitcher-buttons",
			parent: c
		}), d = Q.create("DIV", {
			className: "li-content",
			parent: c
		});
		Q.create("INPUT", {
			type: t.get("baseLayer") ? "radio" : "checkbox",
			className: "ol-visibility",
			checked: t.getVisible(),
			click: i,
			parent: d
		});
		var f = Q.create("LABEL", {
			title: t.get("title") || t.get("name"),
			click: i,
			unselectable: "on",
			style: { userSelect: "none" },
			parent: d
		});
		if (f.addEventListener("selectstart", function() {
			return !1;
		}), Q.create("SPAN", {
			html: t.get("title") || t.get("name"),
			click: function(e) {
				this.get("selection") && (e.stopPropagation(), this.selectLayer(t));
			}.bind(this),
			parent: f
		}), this.reordering && (l < r.length - 1 && (t.get("allwaysOnTop") || !r[l + 1].get("allwaysOnTop")) || l > 0 && (!t.get("allwaysOnTop") || r[l - 1].get("allwaysOnTop"))) && Q.create("DIV", {
			className: "layerup ol-noscroll",
			title: this.tip.up,
			on: { "mousedown touchstart": function(e) {
				n.dragOrdering_(e);
			} },
			parent: u
		}), t.getLayers) {
			var p = 0;
			t.getLayers().forEach(function(e) {
				n.displayInLayerSwitcher(e) && p++;
			}), p && Q.create("DIV", {
				className: t.get("openInLayerSwitcher") ? "collapse-layers" : "expend-layers",
				title: this.tip.plus,
				click: function() {
					var e = n._getLayerForLI(this.parentNode.parentNode);
					e.set("openInLayerSwitcher", !e.get("openInLayerSwitcher"));
				},
				parent: u
			});
		}
		if (this.oninfo && Q.create("DIV", {
			className: "layerInfo",
			title: this.tip.info,
			click: a,
			parent: u
		}), this.hastrash && !t.get("noSwitcherDelete") && Q.create("DIV", {
			className: "layerTrash",
			title: this.tip.trash,
			click: s,
			parent: u
		}), this.hasextent && r[l].getExtent()) {
			var m = r[l].getExtent();
			m.length == 4 && m[0] < m[2] && m[1] < m[3] && Q.create("DIV", {
				className: "layerExtent",
				title: this.tip.extent,
				click: o,
				parent: u
			});
		}
		if (this.show_progress && t instanceof ho) {
			var h = Q.create("DIV", {
				className: "layerswitcher-progress",
				parent: d
			});
			this.setprogress_(t), t.layerswitcher_progress = Q.create("DIV", { parent: h });
		}
		var g = Q.create("DIV", {
			className: "layerswitcher-opacity",
			click: function(e) {
				if (e.target === this) {
					e.stopPropagation(), e.preventDefault();
					var t = Math.max(0, Math.min(1, e.offsetX / Q.getStyle(this, "width")));
					n._getLayerForLI(this.parentNode.parentNode).setOpacity(t);
				}
			},
			parent: d
		});
		if (Q.create("DIV", {
			className: "layerswitcher-opacity-cursor ol-noscroll",
			style: { left: t.getOpacity() * 100 + "%" },
			on: { "mousedown touchstart": function(e) {
				n.dragOpacity_(e);
			} },
			parent: g
		}), Q.create("DIV", {
			className: "layerswitcher-opacity-label",
			html: Math.round(t.getOpacity() * 100),
			parent: d
		}), t.getLayers && (c.classList.add("ol-layer-group"), t.get("openInLayerSwitcher") === !0)) {
			var _ = Q.create("UL", { parent: c });
			this.drawList(_, t.getLayers());
		}
		c.classList.add(this.getLayerClass(t)), this.dispatchEvent({
			type: "drawlist",
			layer: t,
			li: c
		});
	}
	for (var l = r.length - 1; l >= 0; l--) c.call(this, r[l]);
	this.viewChange(), e === this.panel_ && this.overflow();
}, $.prototype.getLayerClass = function(e) {
	return e ? e.getLayers ? "ol-layer-group" : e instanceof Xc ? "ol-layer-vector" : e instanceof mu ? "ol-layer-vectortile" : e instanceof ho ? "ol-layer-tile" : e instanceof bu ? "ol-layer-image" : e instanceof cd ? "ol-layer-heatmap" : e.getFeatures ? "ol-layer-vectorimage" : "unknown" : "none";
}, $.prototype.selectLayer = function(e, t) {
	if (!e) {
		if (!this.getMap()) return;
		e = this.getMap().getLayers().item(this.getMap().getLayers().getLength() - 1);
	}
	this._selectedLayer = e, this.drawPanel(), t || this.dispatchEvent({
		type: "select",
		layer: e
	});
}, $.prototype.getSelection = function() {
	return this._selectedLayer;
}, $.prototype.setprogress_ = function(e) {
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
};
//#endregion
//#region node_modules/ol-ext/control/Button.js
var md = function(e) {
	e ||= {};
	var t = document.createElement("div");
	t.className = (e.className || "") + " ol-button ol-unselectable ol-control";
	var n = this, r = this.button_ = document.createElement(/ol-text-button/.test(e.className) ? "div" : "button");
	r.type = "button", e.title && (r.title = e.title), e.name && (r.name = e.name), e.html instanceof Element ? r.appendChild(e.html) : r.innerHTML = e.html || "";
	var i = function(t) {
		t && t.preventDefault && (t.preventDefault(), t.stopPropagation()), e.handleClick && e.handleClick.call(n, t);
	};
	r.addEventListener("click", i), r.addEventListener("touchstart", i), t.appendChild(r), !e.title && r.firstElementChild && (r.title = r.firstElementChild.title), be.call(this, {
		element: t,
		target: e.target
	}), e.title && this.set("title", e.title), e.title && this.set("title", e.title), e.name && this.set("name", e.name);
};
iu(md, be), md.prototype.setVisible = function(e) {
	e ? Q.show(this.element) : Q.hide(this.element);
}, md.prototype.setTitle = function(e) {
	this.button_.setAttribute("title", e);
}, md.prototype.setHtml = function(e) {
	Q.setHTML(this.button_, e);
}, md.prototype.getButtonElement = function() {
	return this.button_;
};
//#endregion
//#region node_modules/ol-ext/control/Toggle.js
var hd = function(e) {
	e ||= {};
	var t = this;
	this.interaction_ = e.interaction, this.interaction_ && (this.interaction_.setActive(e.active), this.interaction_.on("change:active", function() {
		t.setActive(t.interaction_.getActive());
	})), e.toggleFn && (e.onToggle = e.toggleFn), e.handleClick = function() {
		t.toggle(), e.onToggle && e.onToggle.call(t, t.getActive());
	}, e.className = (e.className || "") + " ol-toggle", md.call(this, e), this.set("title", e.title), this.set("autoActivate", e.autoActivate), e.bar && this.setSubBar(e.bar), this.setActive(e.active), this.setDisable(e.disable);
};
iu(hd, md), hd.prototype.setMap = function(e) {
	!e && this.getMap() && (this.interaction_ && this.getMap().removeInteraction(this.interaction_), this.subbar_ && this.getMap().removeControl(this.subbar_)), md.prototype.setMap.call(this, e), e && (this.interaction_ && e.addInteraction(this.interaction_), this.subbar_ && e.addControl(this.subbar_));
}, hd.prototype.getSubBar = function() {
	return this.subbar_;
}, hd.prototype.setSubBar = function(e) {
	var t = this.getMap();
	t && this.subbar_ && t.removeControl(this.subbar_), this.subbar_ = e, e && (this.subbar_.setTarget(this.element), this.subbar_.element.classList.add("ol-option-bar"), t && t.addControl(this.subbar_));
}, hd.prototype.getDisable = function() {
	var e = this.element.querySelector("button");
	return e && e.disabled;
}, hd.prototype.setDisable = function(e) {
	this.getDisable() != e && (this.element.querySelector("button").disabled = e, e && this.getActive() && this.setActive(!1), this.dispatchEvent({
		type: "change:disable",
		key: "disable",
		oldValue: !e,
		disable: e
	}));
}, hd.prototype.getActive = function() {
	return this.element.classList.contains("ol-active");
}, hd.prototype.toggle = function() {
	this.getActive() ? this.setActive(!1) : this.setActive(!0);
}, hd.prototype.setActive = function(e) {
	this.interaction_ && this.interaction_.setActive(e), this.subbar_ && this.subbar_.setActive(e), this.getActive() !== e && (e ? this.element.classList.add("ol-active") : this.element.classList.remove("ol-active"), this.dispatchEvent({
		type: "change:active",
		key: "active",
		oldValue: !e,
		active: e
	}));
}, hd.prototype.setInteraction = function(e) {
	this.interaction_ = e;
}, hd.prototype.getInteraction = function() {
	return this.interaction_;
};
//#endregion
//#region src/main.js
async function gd() {
	let e = await fetch("/tileserver/session_id");
	if (!e.ok) throw Error("Failed to create TileServer session.");
	return (await e.json()).session_id;
}
async function _d(e) {
	let t = new FormData();
	if (t.append("slide_path", e), !(await fetch("/tileserver/slide", {
		method: "PUT",
		body: t
	})).ok) throw Error(`Failed to load slide: ${e}`);
	let n = await fetch("/tileserver/slide");
	if (!n.ok) throw Error("Failed to retrieve slide metadata.");
	return n.json();
}
function vd(e, t, n) {
	return new ru({
		url: `/tileserver/layer/slide/${e}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${n}`,
		size: t.slide_dimensions,
		crossOrigin: "anonymous",
		zDirection: -1
	});
}
var yd = document.getElementById("map");
if (yd === null) throw Error("The OpenLayers map element could not be found.");
var bd = JSON.parse(yd.dataset.layers ?? "[]"), xd = null, Sd = 0;
if (bd.length === 0) {
	let e = new URLSearchParams(window.location.search).get("slide");
	if (e === null) throw Error("No preloaded layers were supplied and no slide was selected.");
	xd = await gd();
	let t = await _d(e);
	bd = [{
		name: "slide",
		url: `/tileserver/layer/slide/${xd}/zoomify/{TileGroup}/{z}-{x}-{y}@1x.jpg`,
		size: t.slide_dimensions,
		mpp: t.mpp[0]
	}];
}
var Cd = bd.map((e) => {
	let t = new ru({
		url: e.url,
		size: e.size,
		crossOrigin: "anonymous",
		zDirection: -1
	});
	return new ho({
		title: e.name,
		source: t
	});
}), wd = Cd[0], Td = wd.getSource(), Ed = Td.getTileGrid(), Dd = Ed.getResolutions(), Od = Ed.getExtent(), kd = new He({
	code: "ZoomifyProjection",
	units: "pixels",
	extent: Od,
	metersPerUnit: bd[0].mpp * 1e-6,
	getPointResolution(e) {
		return e;
	}
});
hn(kd);
var Ad = new Aa({
	projection: kd,
	resolutions: Dd,
	constrainOnlyCenter: !0
}), jd = new rl({
	target: yd,
	layers: Cd,
	view: Ad
}), Md = new to({
	units: "metric",
	bar: !0,
	steps: 10,
	minWidth: 256
});
jd.addControl(Md);
var Nd = new qa({
	className: "ol-overviewmap ol-custom-overviewmap",
	layers: [new ho({ source: Td })]
});
jd.addControl(Nd);
var Pd = new Ln({
	coordinateFormat: (e) => sn([e[0], -e[1]], "{x}, {y}", 0),
	projection: kd,
	className: "ol-mouse-position",
	placeholder: "\xA0"
});
jd.addControl(Pd);
var Fd = new Ya({
	autoHide: !1,
	className: "ol-rotate"
});
jd.addControl(Fd);
var Id = new Pe();
jd.addControl(Id);
var Ld = new $();
jd.addControl(Ld);
var Rd = 64, zd = 64, Bd = new ws({
	stroke: new Cs({
		color: "rgba(0, 0, 0, 0.5)",
		width: 1
	}),
	text: new Dc({
		font: "12px Calibri,sans-serif",
		fill: new Ss({ color: "rgba(0, 0, 0, 1)" }),
		stroke: new Cs({
			color: "rgba(255, 255, 255, 1)",
			width: 3
		})
	})
}), Vd = new su({
	projection: kd.getCode(),
	margin: zd,
	style: Bd,
	spacing: Rd,
	formatCoord(e, t) {
		let n;
		return n = t === "left" || t === "right" ? -Math.floor(e) : Math.floor(e), n >= 1e6 && (n = n.toExponential(3), n = n.replace("+", "")), n;
	}
}), Hd = Rd, Ud = zd, Wd = new su({
	projection: kd.getCode(),
	spacing: Hd,
	margin: Ud,
	style: Bd,
	formatCoord(e, t) {
		let n = jd.getView().calculateExtent(jd.getSize()), r = jd.getView().getResolution(), i = n[0] + r * Ud, a = n[3] - r * Ud, o;
		if (o = t === "left" || t === "right" ? -(e - a) : e - i, o = Math.floor(o / r / Hd), t === "left" || t === "right") {
			let e = "";
			do
				e += String.fromCharCode(65 + o % 26), o = Math.floor(o / 26);
			while (o > 0);
			return e.split("").reverse().join("");
		}
		return o;
	}
}), Gd = new hd({
	html: "<i class=\"fas fa-ruler-combined\"></i>",
	className: "ol-graticule",
	title: "Toggle Graticule",
	onToggle(e) {
		e ? (Kd.setActive(!1), Wd.setMap(null), Vd.setMap(jd)) : Vd.setMap(null);
	}
}), Kd;
jd.addControl(Gd), Kd = new hd({
	html: "<i class=\"fas fa-border-all\"></i>",
	className: "ol-screen-space-graticule",
	title: "Toggle Screen Space Graticule",
	onToggle(e) {
		e ? (Gd.setActive(!1), Vd.setMap(null), Wd.setMap(jd)) : Wd.setMap(null);
	}
}), jd.addControl(Kd), jd.getView().fit(Od);
async function qd(e) {
	if (xd === null) throw Error("Dynamic slide switching requires a TileServer session.");
	let t = await _d(e);
	Sd += 1;
	let n = vd(xd, t, Sd), r = n.getTileGrid(), i = r.getExtent(), a = r.getResolutions(), o = new He({
		code: "zoomify",
		units: "pixels",
		extent: i,
		metersPerUnit: t.mpp[0] * 1e-6
	});
	hn(o), wd.setSource(n), jd.setView(new Aa({
		projection: o,
		resolutions: a,
		extent: i,
		constrainOnlyCenter: !0
	})), jd.getView().fit(i);
}
Object.assign(window, {
	extent: Od,
	fullscreen: Id,
	graticule: Vd,
	graticuleToggle: Gd,
	layerSwitcher: Ld,
	layers: Cd,
	layersData: bd,
	map: jd,
	mousePositionControl: Pd,
	overviewMapControl: Nd,
	projection: kd,
	resolutions: Dd,
	rotate: Fd,
	scaleLineControl: Md,
	screenSpaceGraticule: Wd,
	screenSpaceGraticuleToggle: Kd,
	switchSlide: qd,
	view: Ad
});
//#endregion
