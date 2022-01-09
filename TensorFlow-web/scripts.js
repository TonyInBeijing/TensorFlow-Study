/**
 * @description TensorFlow.js 根据2D数据进行预测
 * @author YHW 2022-01-09
 */

async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.jconson();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

// 定义模型
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    return model;
}

// 准备数据
function convertToTensor(data) {
    return tf.tidy(() => {
        // 1.打乱数据顺序
        tf.util.shuffle(data);
        // 2.将 data 转换成 Tensor
        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // 3.将 Tensor 按照标准化到0-1范围
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }




    });
}

async function run() {
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg
    }));

    console.log(values);

    tfvis.render.scatterplot(
        { name: "Horsepower V MPG" },
        { values },
        {
            xLabel: "Horsepower",
            yLabel: "MPG",
            height: 300
        }
    );

    const model = createModel();
    tfvis.show.modelSummary({ name: "modelSummary" }, model);
}

document.addEventListener("DOMContentLoaded", run);