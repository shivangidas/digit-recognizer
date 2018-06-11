"use strict";
import $ from "jquery";
import * as tf from "@tensorflow/tfjs";
import {
  MnistData
} from './data';

$(document).ready(function () {
  //console.log('hello world');
  var mousePressed = false;
  var lastX, lastY;
  var ctx;
  var canvas = document.getElementById("myCanvas");
  var ctx = canvas.getContext("2d");
  ctx.strokeStyle = "#222222";
  ctx.lineWidth = 5;
  //we need 28*28
  var scaledCanvas = document.createElement('canvas');
  var scaledContext = scaledCanvas.getContext('2d');
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
  }

  function saveAsImage() {
    scaledContext.scale(0.1, 0.1);
    scaledContext.drawImage(canvas, 0, 0);
    tf.tidy(() => {
      var imgd = scaledContext.getImageData(0, 0, 28, 28);
      //var pix = imgd.data;
      //console.log(pix);
      var image = tf.fromPixels(imgd);
      image = tf.tensor(image.dataSync(), image.shape,'float32');
      //image.print();
      showPredictions(image);
    })

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

model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
  units: 10,
  kernelInitializer: 'varianceScaling',
  activation: 'softmax'
}));

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
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
        testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
      ];
    }

    // The entire dataset doesn't fit into memory so we call fit repeatedly
    // with batches.
    const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels, {
        batchSize: BATCH_SIZE,
        validationData,
        epochs: 1
      });

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    // Plot loss / accuracy.
    lossValues.push({
      'batch': i,
      'loss': loss,
      'set': 'train'
    });
   // ui.plotLosses(lossValues);

    if (testBatch != null) {
      accuracyValues.push({
        'batch': i,
        'accuracy': accuracy,
        'set': 'train'
      });
     // ui.plotAccuracies(accuracyValues);
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

async function showPredictions(image) {
  const imageTensor = image;

  tf.tidy(() => {
    const output = model.predict(image.reshape([-1, 28, 28, 1]));
    console.log(output.dataSync());
    const axis = 1;
    //const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());
    console.log(predictions);

    //ui.showTestResults(batch, predictions, labels);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  await load();
  await train();
  //showPredictions();
}
mnist();