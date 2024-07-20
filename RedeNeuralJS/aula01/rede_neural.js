function sigmoid(x){
    return 1 / (1+ Math.exp(-x));
}

function dsigmoid(x){
    return x * (1-x);
}

class Rede_neural{
    constructor(i_nodes, h_nodes, o_nodes){
        this.i_nodes = i_nodes;
        this.h_nodes = h_nodes;
        this.o_nodes = o_nodes;

        this.bias_ih = new Matriz(this.h_nodes, 1);
        this.bias_ih.randomize();

        this.bias_ho = new Matriz(this.o_nodes, 1);
        this.bias_ho.randomize();
        // this.bias_ih.print();
        // this.bias_ho.print();
        this.weigths_ih = new Matriz(this.h_nodes, this.i_nodes);
        this.weigths_ih.randomize();

        this.weigths_ho = new Matriz(this.o_nodes, this.h_nodes);
        this.weigths_ho.randomize();
        //this.weigths_ho.print();
        //this.weigths_ih.print();

        this.learning_rate = 0.1;
    }

    train(arr, target){
        //INPUT -> HIDDEN
        let input = Matriz.arrayToMatriz(arr);
        let hidden = Matriz.multiply(this.weigths_ih, input);
        hidden = Matriz.add(hidden, this.bias_ih);

        hidden.map(sigmoid);

        // HIDDEN -> OUTPUT
        let output = Matriz.multiply(this.weigths_ho, hidden);
        output = Matriz.add(output, this.bias_ho);
        output.map(sigmoid);

        // BACKPROPAGATION

        // OUTPUT -> HIDDEN
        let expected = Matriz.arrayToMatriz(target);
        let output_error = Matriz.subtract(expected, output);
        let d_output = Matriz.map(output, dsigmoid);
        let hidden_T = Matriz.transpose(hidden);

        let gradient = Matriz.hadamard(d_output, output_error);
        gradient = Matriz.escalar_multiply(gradient, this.learning_rate);

        // ajustar os bias O -> H
        this.bias_ho = Matriz.add(this.bias_ho, gradient);
        // ajustar os pessos O -> H
        let weigths_ho_deltas = Matriz.multiply(gradient, hidden_T);
        this.weigths_ho = Matriz.add(this.weigths_ho, weigths_ho_deltas);

        // HIDDEN -> INPUT
        let weigths_ho_T = Matriz.transpose(this.weigths_ho);
        let hidden_error = Matriz.multiply(weigths_ho_T, output_error);
        let d_hidden = Matriz.map(hidden, dsigmoid);
        let input_T = Matriz.transpose(input);

        let gradient_H = Matriz.hadamard(d_hidden, hidden_error);
        gradient_H = Matriz.escalar_multiply(gradient_H, this.learning_rate);

        // ajustando os bias H -> I
        this.bias_ih = Matriz.add(this.bias_ih, gradient_H);
        // ajustando pesos H -> I
        let weigths_ih_deltas = Matriz.multiply(gradient_H, input_T);
        this.weigths_ih = Matriz.add(this.weigths_ih, weigths_ih_deltas)
    }

    predict(arr){
        // INPUT -> HIDDEN
        let input = Matriz.arrayToMatriz(arr);

        let hidden = Matriz.multiply(this.weigths_ih, input);
        hidden = Matriz.add(hidden, this.bias_ih);

        hidden.map(sigmoid);

        // HIDDEN -> OUTPUT
        let output = Matriz.multiply(this.weigths_ho, hidden);
        output = Matriz.add(output, this.bias_ho);
        output.map(sigmoid);
        output = Matriz.MatrizToArray(output);

        return output;
    }

    feedforward(arr){
        // INPUT -> HIDDEN

        let input = Matriz.arrayToMatriz(arr);
        let hidden = Matriz.multiply(this.weigths_ih, input);
        hidden = Matriz.add(hidden, this.bias_ih);

        hidden.map(sigmoid);

        //HIDDEN -> OUTPUT

        let output = Matriz.multiply(this.weigths_ho, hidden);
        output = Matriz.add(output, this.bias_ho);
        output.map(sigmoid);
        output.print();
    }
}