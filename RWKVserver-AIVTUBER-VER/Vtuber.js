const OpenCC = require('opencc-js');
const { WebSocket } = require('ws');

class Vtuber {
    constructor(config) {
        this.port = config.port || 3033;
        this.baseURL = config.baseURL || ("ws://localhost:" + this.port)

        this._tw2s = OpenCC.Converter({ from: 'tw', to: 'cn' });
        this._s2tw = OpenCC.Converter({ from: 'cn', to: 'tw' });

        /**
         * @type {Array} 對話歷史紀錄
         */
        this.history = [];

        /**
         * @type {boolean} 是否正在等待
         */
        this.isWaittingReply = false;

        /**
         * @type {Array} 等帶回覆的句子
         */
        this.waitList = [];

        /**
         * @type {boolean} ws連線是否初始化
         */
        this.isInitWs = false;

        /**
         * @type {boolean} 資料處理程序是否初始化
         */
        this.isInitProcess = false;

        /**
         * @type {Array} 存放收到的字元陣列
         */
        this.toSpeak = [];

        this.setting = {
            USER: "粉丝",
            ASSISTANT: "樱氏",
            INTERFACE: ":",

            MAX_GENERATION_LENGTH: 250,

            TEMPERATURE: 1.5,  // 0.8

            TOP_P: 0.6,  // 0.5
        }
    }

    getURL = (r) => {
        let url = new URL(r, this.baseURL);
        return url;
    }

    /**
    * 繁轉簡
    */
    tw2s = (input) => {
        return this._tw2s(input);
    }

    /**
     * 簡轉繁
     */
    s2tw = (input) => {
        return this._s2tw(input);
    }

    getRespondPkg = (authorId, content) => {
        return {
            authorId: authorId,
            content: content,
            temperature: this.setting.TEMPERATURE,
            top_p: this.setting.TOP_P
        }
    }

    generateMessage = (role, content) => {
        return `${role}${this.setting.INTERFACE} ${content}`
    }

    pushHistory = (authorId, role, content) => {
        this.history.push({
            "role": role,
            "authorId": authorId,
            "content": content
        })
    }

    copyString = (str) => {
        return (' ' + str).slice(1);
    }

    copyArray = (ary) => {
        return JSON.parse(JSON.stringify(ary));
    }

    onWsMsg = (data) => {
        try {
            console.log("onWsMsg");
            const buffer = Buffer.from(data);
            data = JSON.parse(buffer.toString());
            if(data?.respond?.content) {
                let content = data.respond.content;
                this.onVtuberMsg(content);
            }
        } catch(err) {
            console.error(err)
        }
    }

    onVtuberMsg = (content) => {
        // callTTS
        console.log("onVtuberMsg", content);
        this.isWaittingReply = false;
        console.log("isWaittingReply", this.isWaittingReply);
    }

    onUserMsg = (authorId, content) => {
        console.log("onUserMsg", authorId, content);
        this.waitList.push({ authorId, content });
    }

    reply = (dataPkg) => {
        console.log("reply", dataPkg)
        this.client.send(JSON.stringify(dataPkg));
    }

    initWs = () => {
        this.client = new WebSocket(this.baseURL);

        this.client.on('open', (error) => {
            console.log("connect to tts server.");
            this.isInitWs = true;
            this.onUserMsg("主人", "早安");
            this.onUserMsg("主人", "請你做個自我介紹");
            this.onUserMsg("主人", "早上好～");
            this.onUserMsg("主人", "櫻氏早安阿");
        });

        this.client.on('message', this.onWsMsg);
    }

    initProcess = async () => {
        this.setting.USER = this.tw2s(this.setting.USER)
        this.setting.ASSISTANT = this.tw2s(this.setting.ASSISTANT)

        const checkProcess = setInterval(async () => {
            //console.log(this.isInitWs, this.isInitProcess, this.isWaittingReply,  this.waitList.length > 0)
            if (this.isInitWs && this.isInitProcess && !this.isWaittingReply && this.waitList.length > 0) {
                this.isWaittingReply = true;
                let { authorId, content } = this.waitList.shift();
                let content_s = this.tw2s(content);

                let text = "";

                text += this.generateMessage(this.setting.USER, content_s) + "\n\n";
                text += this.generateMessage(this.setting.ASSISTANT, "");

                let dataPkg = this.getRespondPkg(authorId, text);

                this.pushHistory(authorId, this.setting.USER, content);

                this.reply(dataPkg);
            }
        }, 500);

        this.isInitProcess = true;
    };
}

module.exports = Vtuber;

const vtuber = new Vtuber({
    port: 3033
});

vtuber.initWs();

vtuber.initProcess();