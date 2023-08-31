# How to setup Copilot for lecturing
The recommended workflow contains the following three building blocks
1. Editor: Visual Studio Code (VS Code), a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux.
2. Markdown: A file format for writing plain text documents that can be converted to HTML. Or equivalently, you could use latex to write your lecture notes.
3. Github Copilot: An AI pair programmer that helps you write code faster and with less work. 

## Step 1: Get Visual Studio Code
1. Go to [VS Code website](https://code.visualstudio.com/) and download the latest version.
2. Install & open VS Code.
3. Add the following plugins
    - [Github Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
    To add `Github Copilot`, please click the extension icon on the left side of the VS Code window and search for `Github Copilot`. Click `Install` to install the extension. Other extensions can be installed in the same way.
    - [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced)
    - [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
4. Open a new file and save it as `test.md`. Then, click the `Open Preview to the Side` button in the top right corner of the VS Code window. The left side is the editor and the right side is the preview. You can edit the file on the left side and the preview will be updated automatically.
5. Now, you can start writing your own markdown file. The markdown grammar is specified at [Markdown](https://www.markdownguide.org/basic-syntax/). The GitHub website also renders math equations in Markdown. For example, the following code
    ~~~
    ```math
    \frac{1}{2}
    ```
    ~~~
    will be rendered as
    ```math
    \frac{1}{2}
    ```
## Step 2: Link to Github Copilot
To use Copilot, you need to have a Github account first.
Github is a code hosting platform for version control and collaboration.
1. Go to [Github website](https://github.com/) and sign up an account.
