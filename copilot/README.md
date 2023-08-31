# How to setup Copilot for writing
## Sign up Github
Github is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.
1. Go to [Github website](https://github.com/) and sign up an account.

## Get Visual Studio Code and Copilot
Visual Studio Code (VS Code) is a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux. Github Copilot is a VS Code extension that uses AI to help you write code faster and with less work. In the following, I will show how to setup a workflow based on
```
VS Code + Markdown + Github Copilot
```
1. Go to [VS Code website](https://code.visualstudio.com/) and download the latest version.
2. Install & open VS Code.
3. Add the following plugins
    - [Github Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
    To add `Github Copilot`, please click the extension icon on the left side of the VS Code window and search for `Github Copilot`. Click `Install` to install the extension. Other extensions can be installed in the same way.
    - [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced)
    - [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
4. Open a new file and save it as `test.md`. Then, click the `Open Preview to the Side` button on the top right corner of the VS Code window. You will see the following window.
    ![Markdown Preview](./images/markdown_preview.png)
    The left side is the editor and the right side is the preview. You can edit the file on the left side and the preview will be updated automatically. The preview is rendered by the `Markdown Preview Enhanced` plugin. You can click the `Open Preview to the Side` button again to close the preview.
5. Now, you can start writing your own markdown file. For example, you can write the following code in the editor
    ~~~
    # Test
    This is a test file.
    ~~~
    and the preview will be updated automatically.
    ![Markdown Preview](./images/markdown_preview_test.png)
6. You can also use `Github Copilot` to help you write the markdown file. For example, you can write the following code in the editor.

## Resources
1. Learn [Markdown](https://www.markdownguide.org/basic-syntax/). The Github website also renders math equations in Markdown. For example, the following code
    ~~~
    ```math
    \frac{1}{2}
    ```
    ~~~
    will be rendered as
    ```math
    \frac{1}{2}
    ```