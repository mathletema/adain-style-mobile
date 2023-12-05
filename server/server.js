const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const { spawn } = require("child_process");

const storage = multer.diskStorage({
    destination: './uploads/',
    filename: function (req, file, cb) {
        cb(null, file.originalname);
    },
});

const upload = multer({ storage });

const app = express();
const port = 8000;

const allowedOrigins = ['https://mathletema.github.io/',];
app.use(cors({
  origin: function(origin, callback){
    if (!origin) {
      return callback(null, true);
    }

    if (allowedOrigins.includes(origin)) {
      const msg = 'The CORS policy for this site does not allow access from the specified Origin.';
      return callback(new Error(msg), false);
    }
    return callback(null, true);
  }

}));

const python_dir = "C:\\Users\\Ishank\\python-3.10\\"
const fastnet_py = path.join(__dirname, "../fastnst-python/fastnst.py")

var global_res;

console.log("attempt to spawn fastnst")
let fastnst = spawn(`python.exe`, [`${fastnet_py}`], {cwd: python_dir})
console.log("fastnst spawned!")

fastnst.stdout.on("data", (data) => {
    const output = data.toString();
    console.log(`fastnst out: ${output}`);
    if (output.includes("done")) {
        console.log("fastnst done!\n");
        global_res.status(200).send("done");
    }
})

fastnst.stderr.on("data", (data) => {
    const output = data.toString();
    console.log(`fastnst err: ${output}`);
})

app.use('/css', express.static("../site/css"))
app.use('/public', express.static("./outs"))
app.get('/', (req, res) => {res.sendFile(path.join(__dirname, "../site/index.html"));})

app.post('/submit', upload.any(), async (req, res, next) => {
    console.log("Got a submission!");
    global_res = res;

    if (req.files || req.file) {
        content = req.files[0];
        style = req.files[1];
        console.log("content:")
        console.log(content);
        console.log("style:");
        console.log(style);

        fastnst.stdin.write(`${path.join(__dirname, content.path)},${path.join(__dirname, style.path)},${path.join(__dirname, 'outs\\out.jpg')}\n`);

        // fastnst will send the res.status(200)
    } else {
        return res.status(400).json({ msg: 'An error occurred!' });
    }
});

app.listen(port);
console.log(`Server started at http://localhost:${port}`);