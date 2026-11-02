//-----------------------------------------------------------
// Runs the JavaScript modules gen_c_export writes (ann_export_js) with
// ann_rt.js and checks they give libann's float results (ann_predict) for the
// same inputs, to float32 rounding.
//
// Usage: node test_js_export.mjs <directory written by gen_c_export>
//-----------------------------------------------------------

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const dir = resolve(process.argv[2] ?? ".");
const expected = JSON.parse(readFileSync(join(dir, "expected.json"), "utf8"));
let failures = 0, checks = 0;

function check(name, ok, detail = "") {
	checks++;
	if (!ok) {
		failures++;
		console.log(`FAIL ${name}${detail ? ": " + detail : ""}`);
	}
}

for (const [name, { x, y }] of Object.entries(expected)) {
	const mod = await import(pathToFileURL(join(dir, `${name}.js`)).href);
	check(`${name}: input and output sizes`,
	      mod.model.input_size === x[0].length && mod.model.output_size === y[0].length);
	let worst = 0;
	for (let r = 0; r < x.length; r++) {
		// plain arrays, Float32Arrays and (for token ids) Int32Arrays all work
		const input = r % 3 === 0 ? x[r] : r % 3 === 1 ? Float32Array.from(x[r]) :
			name === "txt" ? Int32Array.from(x[r]) : Float64Array.from(x[r]);
		const out = mod.predict(input);
		for (let i = 0; i < out.length; i++)
			worst = Math.max(worst, Math.abs(out[i] - y[r][i]) / (1e-4 + Math.abs(y[r][i])));
		if (r === 0)
			check(`${name}: argmax`, mod.argmax(out) === mod.argmax(Float32Array.from(y[r])));
	}
	check(`${name}: matches ann_predict`, worst < 1e-3, `largest relative difference ${worst.toExponential(2)}`);
	console.log(`${name}: largest relative difference ${worst.toExponential(2)}`);
}

let threw = false;
try {
	(await import(pathToFileURL(join(dir, "dense.js")).href)).predict([1, 2, 3]);
} catch (e) {
	threw = e instanceof RangeError;
}
check("a wrong input size throws a RangeError", threw);

console.log(`${checks - failures} of ${checks} checks passed`);
process.exit(failures ? 1 : 0);
