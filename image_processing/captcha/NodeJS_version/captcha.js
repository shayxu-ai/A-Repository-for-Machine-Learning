// cd C:\Users\yxu94\PycharmProjects\A-Repository-for-Machine-Learning\image_processing\captcha\NodeJS_version
// node C:\Users\yxu94\PycharmProjects\A-Repository-for-Machine-Learning\image_processing\captcha\NodeJS_version\captcha.js

SvgCaptcha = require("svg-captcha")
const fs = require('fs')
let i = 0

while(i<1) {
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
      i -= 1
    }
  })
i += 1
}