const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require("child_process");

const storage = multer.diskStorage({
    destination: './uploads/',
    filename: function (req, file, cb) {
        cb(null, file.originalname);
    },
});

const upload = multer({ storage });

const app = express();
const port = 8000;

const python = "C:\\Users\\Ishank\\python-3.10\\python.exe"
const fastnet = "..\\fastnst-python\\fastnst.py"

app.use('/css', express.static("../site/css"))
app.use('/public', express.static("./outs"))

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, "../site/index.html"));
})

app.post('/submit', upload.any(), (req, res, next) => {
    // console.log(req.files || req.file);
    if (req.files || req.file) {
        content = req.files[0];
        style = req.files[1];
        console.log("content:")
        console.log(content);
        console.log("style:");
        console.log(style);
        

        exec(`${python} ${fastnet} "${content.path}" "${style.path}" outs\\out.png`, 
        (err, stdout, stderr) => {
            console.log(`stderr: ${stderr}`)
            console.log(`stdout: ${stdout}`)
            console.log("done!");
            return res.status(200).json(req.files || req.file); 
        })
        
    } else {
        return res.status(400).json({ msg: 'An error occurred!' });
    }
});

app.listen(port);
console.log(`Server started at http://localhost:${port}`);