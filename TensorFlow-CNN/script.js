/**
 * @description 使用 CNN 识别手写数字
 * @author YHW 2022-01-10
 */

import { MnistData } from "./data.js";

// 加载数据
async function showExamples(data) {
    const surface = tfvis.visor().surface({ name: "Input Data Examples", tab: "Input Data" });

    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = "margin: 4px;";
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

// 定义模型架构
function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 10;

    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: 'softmax'
    }));

    const optimizer = tf.train.adam();
    model.compile({
        optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);
}

document.addEventListener("DOMContentLoaded", run);