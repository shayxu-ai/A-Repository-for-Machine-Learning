SvgCaptcha = require("svg-captcha")
const fs = require('fs')

let n = SvgCaptcha.create({
        size: 4,
        ignoreChars: "0o1itILl",
        noise: 1,
        color: true
    })

// console.log(n['data'])
fs.writeFile('data/' + n['text'] + '.svg', n['data'], err => {
    if (err) {
      console.error(err)
      return
    }
    //文件写入成功。
  })