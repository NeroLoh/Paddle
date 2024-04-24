# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

<!DOCTYPE html><html lang=""><head><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width,maximum-scale=1,user-scalable=0,initial-scale=1"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="Cache-Control" content="no-cache, must-revalidate"><meta http-equiv="expires" content="0"><link rel="icon" href="/uuapstatic/img/ph-logo.5e92fca2.svg"><title>统一认证中心</title><script>function ieBrowser() {
            var userAgent = navigator.userAgent; //取得浏览器的userAgent字符串
            var isIE = userAgent.indexOf("compatible") > -1 && userAgent.indexOf("MSIE") > -1; //判断是否IE<11浏览器
            var isIE11 = userAgent.indexOf('Trident') > -1 && userAgent.indexOf("rv:11.0") > -1;
            return (isIE || isIE11) ? true : false;
        }
        window.onload = function() {
            var isIE = ieBrowser();
            if (isIE) {
                var nav = window.navigator;
                var isZh = nav.userLanguage.indexOf('zh') !== -1 && nav.systemLanguage.indexOf('zh') !== -1 && nav.browserLanguage.indexOf('zh') !== -1;
                document.getElementById(isZh ? 'enLang' : 'chLang').style.display = 'none';
                document.getElementById('too-low').style.display = 'block';
            }
        };</script><noscript><meta http-equiv="refresh" content="0; url=./noscript.html"></noscript><link href="/uuapstatic/css/app.0196c6c7.css" rel="preload" as="style"><link href="/uuapstatic/css/chunk-vendors.ec2db238.css" rel="preload" as="style"><link href="/uuapstatic/js/app.a97acfba.js" rel="preload" as="script"><link href="/uuapstatic/js/chunk-vendors.39371eef.js" rel="preload" as="script"><link href="/uuapstatic/css/chunk-vendors.ec2db238.css" rel="stylesheet"><link href="/uuapstatic/css/app.0196c6c7.css" rel="stylesheet"></head><body><div id="too-low" class="too-low" style="display: none;"><div class="header"><img class="cursor-pointer" src="/uuapstatic/logo.svg" alt="logo" height="38"></div><div class="too-low-con"><div><img src="/uuapstatic/error-auth.png" alt="error-auth.png"><br><br><div id="enLang">Your browser version is too old. In order to have a better access experience, please change your browser to firebox,<br>Safari, chrome (including chrome kernel browser) and edge. Chrome is recommended.<br><br>Browser compatible version:</div><div id="chLang">您的浏览器版本过旧，为了更好的访问体验，请更换浏览器为firefox，Safari，<br>Chrome（包含Chromium内核浏览器）及edge，推荐使用Chrome。<br><br>浏览器兼容版本：</div>firefox(58+), safari(10+), chrome(76+), edge(16+)</div></div></div><div id="uuap"></div><script src="/uuapstatic/js/chunk-vendors.39371eef.js"></script><script src="/uuapstatic/js/app.a97acfba.js"></script></body><script src="https://wappass.baidu.com/static/machine/js/api/mkd.js"></script><script>function initQRCode({appToken, language}) {
        return new Promise((resolve, reject) => {
            new Beep_QRCode({
                el: 'qrCode',
                // appToken 请选择颁发的线上appToken
                appToken,
                language,
                success: function (res) {
                    resolve(res);
                },
                error: function (res) {
                    reject(res);
                }
            });
        });
    }

    function dongting(token) {
        return new DongTing('#dongting-wrapper', {
            page: location.href,
            token
        });
    }

    function initDongting() {
        let dongtingJs = '';
        let appId = '';
        if (window.location.host === 'uuap.baidu.com') {
            dongtingJs = 'https://dongting.cdn.bcebos.com/dongting-front/prod/dongting_launch.min.js';
            appId = '0618a830025d7f3ae8777b4e27eb5c91';
        }
        if (window.location.host === 'uuap.dev.weiyun.baidu.com' || window.location.host.includes('localhost')) {
            dongtingJs = 'https://dongting.cdn.bcebos.com/dongting-front/dev/dongting_launch.min.js';
            appId = '8e3c139d68ee6c59ffe01e6b34b9cb8b';
        }
        if (window.location.host === 'itebeta.baidu.com') {
            dongtingJs = 'https://dongting.cdn.bcebos.com/dongting-front/dev/dongting_launch.min.js';
            appId = 'bb92be22add6df6ff2477a82e454c125';
        }
        if (!dongtingJs) {
            return;
        }
        const dongtingScript = document.createElement('script');
        dongtingScript.src = dongtingJs;
        document.body.appendChild(dongtingScript);

        dongtingScript.onload = function() {
            window.dongting(appId).trigger('dongting: page:', window.location.href);
        };
    };</script></html>
