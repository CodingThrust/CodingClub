# How to setup Copilot for lecturing
The recommended workflow contains the following three building blocks
1. Markdown: A file format for writing plain text documents that can be converted to HTML for preview. Or equivalently, you could use latex to write your lecture notes.
2. Editor: Visual Studio Code (VS Code), a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux. We use it to write and preview markdown files.
3. Github Copilot: An extension of VSCode that helps you write code. Here we use it to generate lecture notes.


## Step 1: Link to Github Copilot
To use Github Copilot, you need to have a Github account first. Github is a code hosting platform for version control and collaboration, which can host your markdown files and provide preview. You should go to [Github website](https://github.com/) and sign up an account.
## Step 2: Get Visual Studio Code
1. Go to [VS Code website](https://code.visualstudio.com/) and download the latest version.
2. Install & open VS Code.
3. Add the following VS Code extensions. To add an VS Code extension, please click the `Extensions` button in the left side bar, search for the extension name and click the `Install` button.
    - [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced)
    - [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
5. Open a new file and save it as `test.md`. Then, click the `Open Preview to the Side` button in the top right corner of the VS Code window. The left side is the editor and the right side is the preview. You can edit the file on the left side and the preview will be updated automatically.
6. Now, you can start writing your own markdown file. You can learn markdown from [here](https://www.markdownguide.org/basic-syntax/). The GitHub website also renders math equations in Markdown. For example, the following code
    ~~~
    ```math
    \frac{1}{2}
    ```
    ~~~
    will be rendered as
    ```math
    \frac{1}{2}
    ```
    I think is better than latex because the preview is dynamically rendered.

7. Install the following VS Code extension
   - [Github Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot). It requires you to connect with your Github account, and you can use it for free for 30 days. After 30 days, you can still use it for free but you need to join the waitlist. You can also use it for free if you are a student. You can apply for a student account [here](https://education.github.com/pack). Please check this [YouTube video](https://youtu.be/HDG4PQK7DK8?si=sOR7PqNcGAnrV4Tm) for more details
8. You might need to activate the Github Copilot extension by clicking the `Activate` button in the bottom right corner of the VS Code window to make it work. Then you can type some text in the editor and press `Tab` to generate lecture notes.
## Step 3: Using Github to sync your files (optional)
If you want to sync your files across different devices, you can use Github. Github is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere. You can use Github to host your markdown files and provide preview.

This tutorial does not cover how to use Github. You can learn it from the [YouTube video](https://www.youtube.com/watch?v=RGOj5yH7evk) or the [official guide](https://guides.github.com/activities/hello-world/).