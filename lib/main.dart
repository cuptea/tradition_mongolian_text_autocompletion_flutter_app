import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/enums/dtype.dart';

import 'package:mongol/mongol.dart';

void main() => runApp(MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  Model _customModel;
  TextEditingController _controller;

  // variable used to share the auto completed word between two functions:
  // 1). _sample and 2). runCustomModel
  List<dynamic> _tcac;

  // placeholder for auto generated words
  Set<String> _prediction;

  // mappings from token to characters and vise versa.
  Map<String, dynamic> _char_to_token;
  Map<int, dynamic> _token_to_char;

  // configuration variables
  int _blockSize; // maximum context to look back when running model inference.
  int _numberOfSampleWords; // max number of attempts for generating words.
  int _maxLengthOfWord; // max length of a word

  // random number generator. It is used for sampling next character given a
  // probability distribution of characters.
  var rng = new Random();

  @override
  Future<void> initState() {
    super.initState();
    // setup configuration variables
    _blockSize = 20;
    _numberOfSampleWords = 10;
    _maxLengthOfWord = 20;

    _controller = new TextEditingController(text: 'ᢌᡪᡱᡱᡭᢙᡪᡫ ᡥᡪᢑᡪᡪᡪᢙᡪᡪᡪᢔᡪᡧ ᢜᡪᢐᡨ ᢈᡭᢜᡭᡪᡪᢔᡪᡧ ᡳ ᢋᡭᡬᢔᡪᢊᢔᡪᢊᡪᢝ ᢌᡬᡱᡳ ᡸᡪᡬᡬᢝᡨ');
    // ignore: deprecated_member_use
    _prediction = new Set<String>();

    print('loading model');
    loadModel();

    print('loading mappings');
    loadMappings();
  }

  //load your model
  Future loadModel() async {
    String pathCustomModel = "assets/models/model.pt";
    try {
      _customModel = await PyTorchMobile.loadModel(pathCustomModel);
    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  Future loadMappings() async {
    String pathMappings = "assets/mappings/char_to_token.json";
    try {
      String jsonString = await rootBundle.loadString(pathMappings);
      _char_to_token = jsonDecode(jsonString);
      _token_to_char = _char_to_token.map((k, v) => MapEntry(v, k));
      for (var i = 0; i < 87; i++) {
        var char = _token_to_char[i] as String;
        print(char);
      }
    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  // softmax function. It turns a list of number into a probability
  // distribution which sum to one.
  // Definition:
  //    softmax([x1,x2,...,xn]) = [
  //      exp(x1)/(exp(x1)+exp(x2)+...+exp(xn)),
  //      exp(x2)/(exp(x1)+exp(x2)+...+exp(xn)),
  //      ...
  //      exp(xn)/(exp(x1)+exp(x2)+...+exp(xn))]
  List _softmax(List X) {
    var XExp = X.map((xi) => exp(xi)).toList();
    var sum = XExp.reduce((double a, double b) => a + b);
    List probs = XExp.map((xexpi) => (xexpi / sum)).toList();
    return probs;
  }

  // Generate multiple words given tokenized context.
  // The tokenContent is a list of number, each of which maps to certain
  // Mongolian basic character, e.g. 'ᠠ','ᠯ'. The mapping is defined in the
  // assets/mappings/char_to_token.json file.
  Future _sample(List tokenContext, int wordMaxLength, int sampleNumber) async {
    // deep copy context to avoid accumulation
    List x = new List.from(tokenContext);

    // Make sure the context only look back maximum _blockSize number of
    // characters.
    if (x.length >_blockSize) {
      x = x.sublist(x.length - _blockSize,x.length);
    }

    // cast from int to double. This step could be removed for a better code
    // quality
    x = x.map((i) => (i.toDouble() as double)).toList();

    // Create sampleNumber of repeating words as the root for sampling
    var xMultiple = [];
    // Variable to track the status of word completeness
    var isComplete = [];

    // Initialization
    for (var i = 0; i < sampleNumber; i++) {
      xMultiple.add(new List.from(x));
      isComplete.add(false);
    }

    // Generate one character at a time until 'space' or 'new line' character
    // are generated for all words or wordMaxLength iteration is reached.
    for (var l = 0; l < wordMaxLength; l++) {

      // Check whether there is a completed word
      // The xMultiple and isComplete has to be updated accordingly when there
      // is any completed words
      // The xMultiple only contains words haven't been completed all the time.
      var xMultipleNew = [];
      for (var m = 0; m < xMultiple.length; m++) {
        //if yes, map word from token to string and add it to the result _prediction
        if (isComplete[m]) {
          var word = new List.from(xMultiple[m]);
          //map token to char and add it to the result _prediction
          _prediction.add(word
              .map((token) => (_token_to_char[token] as String))
              .toList()
              .join());
        }
        // otherwise, grow the word by one char again
        else {
          xMultipleNew.add(new List.from( xMultiple[m].sublist(xMultiple[m].length - _blockSize,xMultiple[m].length)));
        }
      }

      //prepare the new words to complete
      xMultiple = xMultipleNew;
      isComplete = [];
      for (var m = 0; m < xMultiple.length; m++) {
        isComplete.add(false);
      }

      //if all the words are complete, end the sampling procedure
      if (xMultiple.length == 0) {
        break;
      }
      //other wise sample one new char for each words that is not completed yet
      else {

        var currentWordLength = xMultiple[0].length;
        //prepare the input for model inference
        var xCondMultiple_1d = xMultiple.expand((i) => i).toList();
        xCondMultiple_1d =
            xCondMultiple_1d.map((i) => (i.toDouble() as double)).toList();

        // interact with the model and get the logits. Logits has shape
        // [xMultiple.length, xCond.length, _char_to_token.length] and it includes
        // the raw number that can be turned into probability for generating next
        // character.
        var logits = await _customModel.getPrediction(xCondMultiple_1d,
            [xMultiple.length, currentWordLength], DType.int64);

        // we still sample one char from the 87 potential chars for each words
        // Todo:further speed up can be reached by vectorizing the softmax, random
        // number generation and sampling parts.
        for (var k = 1; k <= xMultiple.length; k++) {
          // only consider the last slide of the logits. Others are logits based on
          // shorter previous context.
          var logitsK = logits.sublist(
              (k * currentWordLength - 1) * _char_to_token.length,
              k * currentWordLength * _char_to_token.length);

          // turn the logits into Probability Mass Function (PMF)
          List probs = _softmax(logitsK);

          // get the Cumulative Probability Mass Function (CMF). It is needed since I
          // don't know how to sample from a distribution in dart. The idea here is
          // to build a sampling tool based on a known CMF.
          probs[0] = probs[0] as double;
          for (var j = 1; j < probs.length; j++) {
            probs[j] = (probs[j - 1] + probs[j]) as double;
          }

          double randNum = rng.nextDouble();
          num ix = probs.indexWhere((p) => (p > randNum) as bool);

          // append the sampled new character to the context
          xMultiple[k - 1].add(ix.toDouble());
          // if a space or new line token is obtained, it means one word is
          // completed and we can record this in the isComplete list.
          if (ix == 0 || ix == 1) {
            isComplete[k - 1] = true;
          }
        }
      }
    }


  }

  // Generate auto completed words given the context.
  Future runCustomModel() async {
    // refresh the old auto completions
    _prediction = new Set<String>();

    // get the context from the textField
    String context = _controller.text;

    // tokenize the characters into integer numbers, the mapping are defined in
    // assets/mappings/char_to_token.json file.
    List tokenizedContext =
        context.split('').map((ch) => (_char_to_token[ch] as int)).toList();
    tokenizedContext =
        tokenizedContext.map((i) => (i.toDouble() as double)).toList();

    // generate _numberOfSample number of auto completed words
    await _sample(tokenizedContext, _maxLengthOfWord, _numberOfSampleWords);

    // construct the word by mapping back the tokens into the characters and
    // join the characters into a string.

    print('all auto completed words: $_prediction');
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(fontFamily: 'z52chimegtig'),
      home: Scaffold(
        appBar: AppBar(
          title: const Text(' ZMongol:\n Input Auto Completion Demo'),
        ),
        body: Column(
          //mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Container(
              child: Center(
                child: SizedBox(
                  width: 200,
                  height: 200,
                  child: MongolTextField(
                    controller: _controller,
                    decoration: null,
                    maxLines: null,
                  ),
                ),
              ),
            ),
            Container(
              child: TextButton(
                onPressed: runCustomModel,
                child: Text(
                  "Run custom model",
                  style: TextStyle(
                    color: Colors.black,
                  ),
                ),
              ),
            ),
            Container(
              child: Visibility(
                visible: _prediction != null,
                child: MongolText(
                  _prediction.join('\n'),
                  style: TextStyle(fontFamily: 'z52chimegtig', fontSize: 24),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
