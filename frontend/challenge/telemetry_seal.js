function _b64FromBytes(bytes) {
    let bin = "";
    const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
    return btoa(bin);
}

function _bytesFromB64(text) {
    const pad = "=".repeat((4 - (text.length % 4)) % 4);
    const normalized = String(text).replace(/-/g, "+").replace(/_/g, "/") + pad;
    const bin = atob(normalized);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

const _SHA_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

function _rotr(n, x) {
    return (x >>> n) | (x << (32 - n));
}

function sha256Bytes(data) {
    const msg = data instanceof Uint8Array ? data : new Uint8Array(data);
    const bitLen = msg.length * 8;
    const withPad = ((msg.length + 9 + 63) >> 6) << 6;
    const buf = new Uint8Array(withPad);
    buf.set(msg);
    buf[msg.length] = 0x80;
    const view = new DataView(buf.buffer);
    view.setUint32(withPad - 4, bitLen >>> 0);
    view.setUint32(withPad - 8, Math.floor(bitLen / 0x100000000));

    let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    const w = new Uint32Array(64);
    for (let off = 0; off < withPad; off += 64) {
        for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4);
        for (let i = 16; i < 64; i++) {
            const s0 = _rotr(7, w[i - 15]) ^ _rotr(18, w[i - 15]) ^ (w[i - 15] >>> 3);
            const s1 = _rotr(17, w[i - 2]) ^ _rotr(19, w[i - 2]) ^ (w[i - 2] >>> 10);
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
        }
        let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
        for (let i = 0; i < 64; i++) {
            const S1 = _rotr(6, e) ^ _rotr(11, e) ^ _rotr(25, e);
            const ch = (e & f) ^ (~e & g);
            const t1 = (h + S1 + ch + _SHA_K[i] + w[i]) >>> 0;
            const S0 = _rotr(2, a) ^ _rotr(13, a) ^ _rotr(22, a);
            const maj = (a & b) ^ (a & c) ^ (b & c);
            const t2 = (S0 + maj) >>> 0;
            h = g; g = f; f = e; e = (d + t1) >>> 0;
            d = c; c = b; b = a; a = (t1 + t2) >>> 0;
        }
        h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0;
        h4 = (h4 + e) >>> 0; h5 = (h5 + f) >>> 0; h6 = (h6 + g) >>> 0; h7 = (h7 + h) >>> 0;
    }
    const out = new Uint8Array(32);
    const ov = new DataView(out.buffer);
    ov.setUint32(0, h0); ov.setUint32(4, h1); ov.setUint32(8, h2); ov.setUint32(12, h3);
    ov.setUint32(16, h4); ov.setUint32(20, h5); ov.setUint32(24, h6); ov.setUint32(28, h7);
    return out;
}

function hmacSha256(keyBytes, dataBytes) {
    const block = 64;
    let key = keyBytes.length > block ? sha256Bytes(keyBytes) : keyBytes;
    const ipad = new Uint8Array(block);
    const opad = new Uint8Array(block);
    ipad.set(key);
    opad.set(key);
    for (let i = 0; i < block; i++) {
        ipad[i] ^= 0x36;
        opad[i] ^= 0x5c;
    }
    const inner = new Uint8Array(block + dataBytes.length);
    inner.set(ipad);
    inner.set(dataBytes, block);
    const innerHash = sha256Bytes(inner);
    const outer = new Uint8Array(block + 32);
    outer.set(opad);
    outer.set(innerHash, block);
    return sha256Bytes(outer);
}

const _SBOX = new Uint8Array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]);

function _xtime(a) {
    return ((a << 1) ^ (((a >> 7) & 1) * 0x1b)) & 0xff;
}

function _mixWord(a, b, c, d) {
    return _xtime(a) ^ _xtime(b) ^ b ^ c ^ d;
}

function aes256ExpandKey(key) {
    const w = new Uint8Array(240);
    w.set(key);
    let rcon = 1;
    for (let i = 8; i < 60; i++) {
        let t0 = w[(i - 1) * 4], t1 = w[(i - 1) * 4 + 1], t2 = w[(i - 1) * 4 + 2], t3 = w[(i - 1) * 4 + 3];
        if (i % 8 === 0) {
            const x = t0;
            t0 = _SBOX[t1] ^ rcon;
            t1 = _SBOX[t2];
            t2 = _SBOX[t3];
            t3 = _SBOX[x];
            rcon = _xtime(rcon);
        } else if (i % 8 === 4) {
            t0 = _SBOX[t0];
            t1 = _SBOX[t1];
            t2 = _SBOX[t2];
            t3 = _SBOX[t3];
        }
        const p = (i - 8) * 4;
        w[i * 4] = w[p] ^ t0;
        w[i * 4 + 1] = w[p + 1] ^ t1;
        w[i * 4 + 2] = w[p + 2] ^ t2;
        w[i * 4 + 3] = w[p + 3] ^ t3;
    }
    return w;
}

function aes256EncryptBlock(roundKeys, input, output, outOff) {
    const s = new Uint8Array(16);
    s.set(input);
    for (let i = 0; i < 16; i++) s[i] ^= roundKeys[i];
    for (let round = 1; round < 14; round++) {
        for (let i = 0; i < 16; i++) s[i] = _SBOX[s[i]];
        const t1 = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t1;
        const t2 = s[2]; s[2] = s[10]; s[10] = t2;
        const t6 = s[6]; s[6] = s[14]; s[14] = t6;
        const t3 = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t3;
        for (let c = 0; c < 4; c++) {
            const i = 4 * c;
            const a = s[i], b = s[i + 1], cc = s[i + 2], d = s[i + 3];
            s[i] = _mixWord(a, b, cc, d);
            s[i + 1] = _mixWord(b, cc, d, a);
            s[i + 2] = _mixWord(cc, d, a, b);
            s[i + 3] = _mixWord(d, a, b, cc);
        }
        const rk = round * 16;
        for (let i = 0; i < 16; i++) s[i] ^= roundKeys[rk + i];
    }
    for (let i = 0; i < 16; i++) s[i] = _SBOX[s[i]];
    const u1 = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = u1;
    const u2 = s[2]; s[2] = s[10]; s[10] = u2;
    const u6 = s[6]; s[6] = s[14]; s[14] = u6;
    const u3 = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = u3;
    for (let i = 0; i < 16; i++) output[outOff + i] = s[i] ^ roundKeys[224 + i];
}

function aes256CtrXor(key, iv, plaintext) {
    const roundKeys = aes256ExpandKey(key);
    const out = new Uint8Array(plaintext.length);
    const counter = new Uint8Array(16);
    counter.set(iv);
    const stream = new Uint8Array(16);
    for (let off = 0; off < plaintext.length; off += 16) {
        aes256EncryptBlock(roundKeys, counter, stream, 0);
        const n = Math.min(16, plaintext.length - off);
        for (let i = 0; i < n; i++) out[off + i] = plaintext[off + i] ^ stream[i];
        for (let i = 15; i >= 0; i--) {
            counter[i] = (counter[i] + 1) & 0xff;
            if (counter[i] !== 0) break;
        }
    }
    return out;
}

function sealAesCtrHmac(keyBytes, plaintext) {
    const iv = crypto.getRandomValues(new Uint8Array(16));
    const ciphertext = aes256CtrXor(keyBytes, iv, plaintext);
    const macInput = new Uint8Array(iv.length + ciphertext.length);
    macInput.set(iv);
    macInput.set(ciphertext, iv.length);
    const mac = hmacSha256(keyBytes, macInput);
    return {
        alg: "AES-256-CTR-HMAC",
        iv: _b64FromBytes(iv),
        ciphertext: _b64FromBytes(ciphertext),
        mac: _b64FromBytes(mac),
    };
}

async function sealAesGcmSubtle(keyBytes, plaintext) {
    const key = await crypto.subtle.importKey("raw", keyBytes, "AES-GCM", false, ["encrypt"]);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, plaintext);
    return {
        alg: "AES-GCM",
        iv: _b64FromBytes(iv),
        ciphertext: _b64FromBytes(new Uint8Array(ciphertext)),
    };
}

async function persistSealedLocally(sessionId, sealed) {
    if (!sessionId || !window.indexedDB) return;
    try {
        const req = indexedDB.open("bcaptcha-local-capture", 1);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains("captures")) {
                db.createObjectStore("captures", { keyPath: "sessionId" });
            }
        };
        req.onsuccess = () => {
            const db = req.result;
            const tx = db.transaction("captures", "readwrite");
            tx.objectStore("captures").put({ sessionId, sealed, savedAt: Date.now() });
            tx.oncomplete = () => db.close();
            tx.onerror = () => db.close();
        };
    } catch (err) {
        /* local backup is optional */
    }
}

async function sealTelemetryEvents(keyB64, events) {
    const keyBytes = _bytesFromB64(keyB64);
    const plaintext = new TextEncoder().encode(JSON.stringify({ events }));
    if (window.isSecureContext && crypto.subtle) {
        try {
            return await sealAesGcmSubtle(keyBytes, plaintext);
        } catch (err) {
            /* LAN HTTP has no WebCrypto; fall through */
        }
    }
    return sealAesCtrHmac(keyBytes, plaintext);
}
