"use strict";
import $ from "jquery";
import * as tf from "@tensorflow/tfjs";
import {
	MnistData
} from "./data";
//import * as ui from "./ui";
$(document).ready(function () {
	//console.log('hello world');
	var mousePressed = false;
	var lastX, lastY;
	var ctx;
	var canvas = document.getElementById("myCanvas");
	var ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	ctx.strokeStyle = "#fff";
	ctx.lineWidth = 5;
	//we need 28*28
	var scaledCanvas = document.getElementById("scaledCanvas");
	var scaledContext = scaledCanvas.getContext("2d");
	scaledCanvas.width = 28;
	scaledCanvas.height = 28;
	// Set up mouse events for drawing
	var drawing = false;
	var mousePos = {
		x: 0,
		y: 0
	};
	var lastPos = mousePos;
	canvas.addEventListener(
		"mousedown",
		function (e) {
			drawing = true;
			lastPos = getMousePos(canvas, e);
		},
		false
	);
	canvas.addEventListener(
		"mouseup",
		function (e) {
			drawing = false;
		},
		false
	);
	canvas.addEventListener(
		"mousemove",
		function (e) {
			mousePos = getMousePos(canvas, e);
		},
		false
	);

	// Get the position of the mouse relative to the canvas
	function getMousePos(canvasDom, mouseEvent) {
		var rect = canvasDom.getBoundingClientRect();
		return {
			x: mouseEvent.clientX - rect.left,
			y: mouseEvent.clientY - rect.top
		};
	}

	// Get a regular interval for drawing to the screen
	window.requestAnimFrame = (function (callback) {
		return (
			window.requestAnimationFrame ||
			window.webkitRequestAnimationFrame ||
			window.mozRequestAnimationFrame ||
			window.oRequestAnimationFrame ||
			window.msRequestAnimaitonFrame ||
			function (callback) {
				window.setTimeout(callback, 1000 / 60);
			}
		);
	})();

	function renderCanvas() {
		if (drawing) {
			ctx.moveTo(lastPos.x, lastPos.y);
			ctx.lineTo(mousePos.x, mousePos.y);
			ctx.stroke();
			lastPos = mousePos;
		}
	}

	// Allow for animation
	(function drawLoop() {
		requestAnimFrame(drawLoop);
		renderCanvas();
	})();
	// Set up touch events for mobile, etc
	canvas.addEventListener(
		"touchstart",
		function (e) {
			mousePos = getTouchPos(canvas, e);
			var touch = e.touches[0];
			var mouseEvent = new MouseEvent("mousedown", {
				clientX: touch.clientX,
				clientY: touch.clientY
			});
			canvas.dispatchEvent(mouseEvent);
		},
		false
	);
	canvas.addEventListener(
		"touchend",
		function (e) {
			var mouseEvent = new MouseEvent("mouseup", {});
			canvas.dispatchEvent(mouseEvent);
		},
		false
	);
	canvas.addEventListener(
		"touchmove",
		function (e) {
			var touch = e.touches[0];
			var mouseEvent = new MouseEvent("mousemove", {
				clientX: touch.clientX,
				clientY: touch.clientY
			});
			canvas.dispatchEvent(mouseEvent);
		},
		false
	);

	// Get the position of a touch relative to the canvas
	function getTouchPos(canvasDom, touchEvent) {
		var rect = canvasDom.getBoundingClientRect();
		return {
			x: touchEvent.touches[0].clientX - rect.left,
			y: touchEvent.touches[0].clientY - rect.top
		};
	}
	// Prevent scrolling when touching the canvas
	document.body.addEventListener(
		"touchstart",
		function (e) {
			if (e.target == canvas) {
				e.preventDefault();
			}
		},
		false
	);
	document.body.addEventListener(
		"touchend",
		function (e) {
			if (e.target == canvas) {
				e.preventDefault();
			}
		},
		false
	);
	document.body.addEventListener(
		"touchmove",
		function (e) {
			if (e.target == canvas) {
				e.preventDefault();
			}
		},
		false
	);

	function clearArea() {
		canvas.width = canvas.width;
		scaledCanvas.width = scaledCanvas.width;
		ctx.fillStyle = "black";
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.strokeStyle = "#fff";
		ctx.lineWidth = 5;
		$("#guessNumber").html("");
	}

	function saveAsImage() {
		scaledContext.scale(0.1, 0.1);
		scaledContext.drawImage(canvas, 0, 0);
		tf.tidy(() => {
			var imageData = scaledContext.getImageData(
				0,
				0,
				scaledCanvas.width,
				scaledCanvas.height
			);
			var data = imageData.data;

			for (var i = 0; i < data.length; i += 4) {
				var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
				data[i] = avg; // red
				data[i + 1] = avg; // green
				data[i + 2] = avg; // blue
			}
			//console.log(imageData.data)
			scaledContext.putImageData(imageData, 0, 0);
			//image.print();
			showPredictions(imageData);
		});
	}
	$(".clearButton").click(function (event) {
		event.preventDefault();
		clearArea();
	});
	$(".guessButton").click(function (event) {
		event.preventDefault();
		saveAsImage();
	});
});
const model = tf.sequential();

model.add(
	tf.layers.conv2d({
		inputShape: [28, 28, 1],
		kernelSize: 5,
		filters: 8,
		strides: 1,
		activation: "relu",
		kernelInitializer: "varianceScaling"
	})
);
model.add(
	tf.layers.maxPooling2d({
		poolSize: [2, 2],
		strides: [2, 2]
	})
);
model.add(
	tf.layers.conv2d({
		kernelSize: 5,
		filters: 16,
		strides: 1,
		activation: "relu",
		kernelInitializer: "varianceScaling"
	})
);
model.add(
	tf.layers.maxPooling2d({
		poolSize: [2, 2],
		strides: [2, 2]
	})
);

model.add(tf.layers.flatten());
model.add(
	tf.layers.dense({
		units: 10,
		kernelInitializer: "varianceScaling",
		activation: "softmax"
	})
);

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
	optimizer: optimizer,
	loss: "categoricalCrossentropy",
	metrics: ["accuracy"]
});

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 200;
// Every few batches, test accuracy over many examples. Ideally, we'd compute
// accuracy over the whole test set, but for performance we'll use a subset.
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

async function train() {
	//ui.isTraining();

	const lossValues = [];
	const accuracyValues = [];

	for (let i = 0; i < TRAIN_BATCHES; i++) {
		const batch = data.nextTrainBatch(BATCH_SIZE);

		let testBatch;
		let validationData;
		// Every few batches test the accuracy of the mode.
		if (i % TEST_ITERATION_FREQUENCY === 0) {
			testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
			validationData = [
				testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]),
				testBatch.labels
			];
		}

		// The entire dataset doesn't fit into memory so we call fit repeatedly
		// with batches.
		const history = await model.fit(
			batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
			batch.labels, {
				batchSize: BATCH_SIZE,
				validationData,
				epochs: 2
			}
		);

		const loss = history.history.loss[0];
		const accuracy = history.history.acc[0];

		// Plot loss / accuracy.
		lossValues.push({
			batch: i,
			loss: loss,
			set: "train"
		});
		//ui.plotLosses(lossValues);

		if (testBatch != null) {
			accuracyValues.push({
				batch: i,
				accuracy: accuracy,
				set: "train"
			});
			//ui.plotAccuracies(accuracyValues);
			//console.log(accuracy.dataSync());
		}

		batch.xs.dispose();
		batch.labels.dispose();
		if (testBatch != null) {
			testBatch.xs.dispose();
			testBatch.labels.dispose();
		}

		await tf.nextFrame();
	}
}

async function showPredictions(imageData) {
	tf.tidy(() => {
		// Convert the canvas pixels to
		let img = tf.fromPixels(imageData, 1);
		img = img.reshape([-1, 28, 28, 1]);
		img = tf.cast(img, "float32");
		const output = model1.predict(img);
		const axis = 1;
		const predictions = Array.from(output.argMax(axis).dataSync());
		$("#guessNumber").html(predictions[0]);
	});
}
// async function showPredictions1() {
// 	const testExamples = 100;
// 	const batch = data.nextTestBatch(testExamples);

// 	// Code wrapped in a tf.tidy() function callback will have their tensors freed
// 	// from GPU memory after execution without having to call dispose().
// 	// The tf.tidy callback runs synchronously.
// 	tf.tidy(() => {
// 	  const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

// 	  // tf.argMax() returns the indices of the maximum values in the tensor along
// 	  // a specific axis. Categorical classification tasks like this one often
// 	  // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
// 	  // one element for each output class. All values in the vector are 0
// 	  // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
// 	  // output from model.predict() will be a probability distribution, so we use
// 	  // argMax to get the index of the vector element that has the highest
// 	  // probability. This is our prediction.
// 	  // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
// 	  // dataSync() synchronously downloads the tf.tensor values from the GPU so
// 	  // that we can use them in our normal CPU JavaScript code
// 	  // (for a non-blocking version of this function, use data()).
// 	  const axis = 1;
// 	  const labels = Array.from(batch.labels.argMax(axis).dataSync());
// 	  const predictions = Array.from(output.argMax(axis).dataSync());

// 	  ui.showTestResults(batch, predictions, labels);
// 	});
//   }

let data;
async function load() {
	data = new MnistData();
	await data.load();
}
var model1;
async function mnist() {
	//await load();
	//await train();
	//await model.save("downloads://my-model-1");
	model1 = await tf.loadModel('https://raw.githubusercontent.com/shivangidas/digit-recognizer/master/model/my-model-1.json');
	//showPredictions1();
	//const jsonUpload = document.getElementById("modelFile");
	//const weightsUpload = document.getElementById("weightsFile");
	//model1 = await tf.loadModel(tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
}
//$("#addModelButton").click(function () {
	mnist();
//});