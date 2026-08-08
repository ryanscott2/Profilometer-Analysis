pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    width: 1360
    height: 900
    minimumWidth: 1120
    minimumHeight: 760
    visible: true
    title: "PFLM  ·  Profilometer Analysis"
    color: theme.appBg

    QtObject {
        id: theme
        readonly property color appBg:        "#202020"
        readonly property color cardBg:       "#2b2b2b"
        readonly property color cardStroke:   "#363636"
        readonly property color surfaceBg:    "#191919"
        readonly property color textPrimary:  "#ffffff"
        readonly property color textSecond:   "#c5c5c5"
        readonly property color textTertiary: "#8a8a8a"
        readonly property color accent:       "#4cc2ff"
        readonly property color danger:       "#ff99a4"
        readonly property int   radius:       8
        readonly property int   gap:          12
        readonly property int   pad:          12
        readonly property string face:        "Segoe UI Variable Text"
        readonly property string mono:        "Cascadia Mono"
    }

    property string selectedFigure: ""
    property var calibrationPicks: []
    property var cellSpecs: ({})

    function setCellSpec(sample, spec) {
        var next = {}
        for (var key in root.cellSpecs)
            next[key] = root.cellSpecs[key]
        if (spec === "")
            delete next[sample]
        else
            next[sample] = spec
        root.cellSpecs = next
    }

    // Which of the three things the Run button does, following the visible tab.
    readonly property int mode: tabs.currentIndex
    readonly property bool isRowMode: mode === 1
    readonly property string runLabel: mode === 0 ? "Run analysis"
                                     : mode === 1 ? "Run wafer row" : "Calibrate depth"

    function refreshRows() {
        var info = bridge.rowInfo(vk4Field.text, mapField.text)
        mapField.placeholderText = info.mapPath !== "" ? info.mapPath : "No wafer_map.csv found"
        rowProblem.text = info.problem
        rowCombo.model = info.rows
        if (info.rows.length > 0 && rowCombo.currentIndex < 0)
            rowCombo.currentIndex = 0
    }

    function currentRow() {
        return rowCombo.currentIndex < 0 ? 0 : parseInt(rowCombo.currentText)
    }

    // Figures, refresh and export all follow whichever result the active tab means.
    function activeResultName() {
        return root.isRowMode ? bridge.rowName(vk4Field.text, root.currentRow())
                              : (sampleCombo.currentIndex < 0 ? "" : sampleCombo.currentText)
    }

    function doRun() {
        if (bridge.busy) { bridge.stop(); return }
        if (root.mode === 0)
            bridge.run(sampleCombo.currentText, root.values())
        else if (root.mode === 1)
            bridge.runRow(vk4Field.text, dxfField.text, root.currentRow(), mapField.text)
        else
            bridge.runCalibration(targetsField.text, root.calibrationPicks,
                                  bandsArea.text, legacyQcBox.checked, root.cellSpecs)
    }

    Component.onCompleted: bandsArea.text = bridge.defaultBands()
    // Set from the command line so a sample can be opened directly.
    property string initialSample: ""
    property int initialTab: -1
    onInitialTabChanged: if (initialTab >= 0) tabs.currentIndex = initialTab
    onInitialSampleChanged: {
        var index = bridge.sampleNames.indexOf(initialSample)
        if (index >= 0) {
            sampleCombo.currentIndex = index
            root.applySample(initialSample)
        }
    }

    function values() {
        return {
            "dxf": dxfField.text,
            "vk4_dir": vk4Field.text,
            "csv_text": csvArea.text,
            "radial_text": radialArea.text
        }
    }

    function applySample(name) {
        var info = bridge.loadSample(name)
        if (!info.ok)
            return
        dxfField.text = info.dxf
        vk4Field.text = info.vk4_dir
        csvArea.text = info.csv_text
        radialArea.text = info.radial_text
        root.selectedFigure = ""
        root.refreshRows()
    }

    FileDialog {
        id: dxfDialog
        title: "Select DXF"
        nameFilters: ["DXF (*.dxf)", "All files (*)"]
        onAccepted: dxfField.text = selectedFile.toString().replace("file:///", "")
    }
    FolderDialog {
        id: vk4Dialog
        title: "Select VK4 folder"
        onAccepted: {
            vk4Field.text = selectedFolder.toString().replace("file:///", "")
            root.refreshRows()
        }
    }
    FileDialog {
        id: mapDialog
        title: "Select wafer map CSV"
        nameFilters: ["CSV (*.csv)", "All files (*)"]
        onAccepted: {
            mapField.text = selectedFile.toString().replace("file:///", "")
            root.refreshRows()
        }
    }
    FileDialog {
        id: zipDialog
        title: "Save figures zip"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "zip"
        nameFilters: ["Zip archive (*.zip)"]
        onAccepted: {
            var problem = bridge.exportZip(root.activeResultName(),
                                           selectedFile.toString(), root.isRowMode)
            exportProblem.text = problem
        }
    }
    Dialog {
        id: saveDialog
        title: "Save sample"
        anchors.centerIn: parent
        modal: true
        standardButtons: Dialog.Save | Dialog.Cancel
        onAccepted: {
            bridge.saveSample(nameField.text, root.values())
            sampleCombo.currentIndex = bridge.sampleNames.indexOf(nameField.text)
        }
        ColumnLayout {
            spacing: 8
            Label { text: "Sample name"; color: theme.textSecond; font.family: theme.face }
            TextField {
                id: nameField
                implicitWidth: 340
                placeholderText: "e.g. 080826 D300 3x3"
                font.family: theme.face
            }
        }
    }

    // ------------------------------------------------------------- header

    header: Rectangle {
        height: 62
        color: theme.appBg
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1
                    color: theme.cardStroke }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: theme.pad
            anchors.rightMargin: theme.pad
            spacing: theme.gap

            ColumnLayout {
                spacing: 0
                Label {
                    text: "PFLM profilometer analysis"
                    color: theme.textPrimary
                    font.family: theme.face
                    font.pixelSize: 17
                    font.weight: Font.DemiBold
                }
                Label {
                    text: "tiled VK4 analysis against a DXF cell map"
                    color: theme.textTertiary
                    font.family: theme.face
                    font.pixelSize: 12
                }
            }

            Item { Layout.fillWidth: true }

            Label {
                text: "Sample"
                color: theme.textTertiary
                font.family: theme.face
                font.pixelSize: 12
            }
            ComboBox {
                id: sampleCombo
                implicitWidth: 260
                model: bridge.sampleNames
                // A model resets currentIndex to 0, which would show a sample name
                // while none of its fields were actually loaded.
                currentIndex: -1
                displayText: currentIndex < 0 ? "none" : currentText
                onActivated: root.applySample(currentText)
                onModelChanged: currentIndex = -1
            }
            Button {
                text: "Save"
                onClicked: {
                    nameField.text = sampleCombo.currentIndex >= 0 ? sampleCombo.currentText : ""
                    saveDialog.open()
                }
            }
            Button {
                text: "Delete"
                enabled: sampleCombo.currentIndex >= 0
                onClicked: {
                    bridge.deleteSample(sampleCombo.currentText)
                    sampleCombo.currentIndex = -1
                }
            }

            Rectangle { width: 1; height: 28; color: theme.cardStroke }

            BusyIndicator {
                running: bridge.busy
                visible: bridge.busy
                implicitWidth: 22
                implicitHeight: 22
            }
            Button {
                text: bridge.busy ? "Stop" : root.runLabel
                highlighted: !bridge.busy
                enabled: bridge.busy
                         || (root.mode === 0 && sampleCombo.currentIndex >= 0)
                         || (root.mode === 1 && rowCombo.currentIndex >= 0)
                         || root.mode === 2
                onClicked: root.doRun()
            }
        }
    }

    // --------------------------------------------------------------- body

    RowLayout {
        anchors.fill: parent
        anchors.margins: theme.pad
        spacing: theme.gap

        // ---- left: three modes
        ColumnLayout {
            Layout.preferredWidth: 400
            Layout.minimumWidth: 364
            Layout.maximumWidth: 424
            Layout.fillHeight: true
            spacing: theme.gap

            TabBar {
                id: tabs
                Layout.fillWidth: true
                TabButton { text: "Sample" }
                TabButton { text: "Wafer row" }
                TabButton { text: "Depth cal" }
            }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: tabs.currentIndex

            ColumnLayout {
                spacing: theme.gap

            Rectangle {
                Layout.fillWidth: true
                color: theme.cardBg
                radius: theme.radius
                border.color: theme.cardStroke
                border.width: 1
                implicitHeight: inputsCol.implicitHeight + theme.pad * 2
                ColumnLayout {
                    id: inputsCol
                    anchors.fill: parent
                    anchors.margins: theme.pad
                    spacing: 8
                    Label {
                        text: "INPUTS"; color: theme.textTertiary
                        font.family: theme.face; font.pixelSize: 11
                        font.weight: Font.DemiBold; font.letterSpacing: 0.6
                    }
                    Label {
                        text: "DXF cell map"; color: theme.textSecond
                        font.family: theme.face; font.pixelSize: 13
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        TextField {
                            id: dxfField
                            Layout.fillWidth: true
                            placeholderText: "No DXF selected"
                            font.family: theme.face
                            font.pixelSize: 12
                        }
                        Button { text: "Browse"; onClicked: dxfDialog.open() }
                    }
                    Label {
                        text: "VK4 tile folder"; color: theme.textSecond
                        font.family: theme.face; font.pixelSize: 13
                        Layout.topMargin: 4
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        TextField {
                            id: vk4Field
                            Layout.fillWidth: true
                            placeholderText: "No folder selected"
                            font.family: theme.face
                            font.pixelSize: 12
                        }
                        Button { text: "Browse"; onClicked: vk4Dialog.open() }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Label {
                            text: "classifies as"; color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                        }
                        Rectangle {
                            radius: 4
                            color: "#333333"
                            implicitWidth: modeChip.implicitWidth + 14
                            implicitHeight: 20
                            Label {
                                id: modeChip
                                anchors.centerIn: parent
                                text: bridge.classify(vk4Field.text)
                                color: theme.accent
                                font.family: theme.face
                                font.pixelSize: 11
                                font.weight: Font.DemiBold
                            }
                        }
                        Item { Layout.fillWidth: true }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: theme.cardBg
                radius: theme.radius
                border.color: theme.cardStroke
                border.width: 1
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.pad
                    spacing: 6
                    Label {
                        text: "CELL PARAMS"; color: theme.textTertiary
                        font.family: theme.face; font.pixelSize: 11
                        font.weight: Font.DemiBold; font.letterSpacing: 0.6
                    }
                    Label {
                        text: "one row per cell row, comma separated"
                        color: theme.textTertiary
                        font.family: theme.face; font.pixelSize: 11
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 6
                        color: theme.surfaceBg
                        border.color: theme.cardStroke
                        border.width: 1
                        clip: true
                        ScrollView {
                            anchors.fill: parent
                            anchors.margins: 6
                            TextArea {
                                id: csvArea
                                wrapMode: TextArea.NoWrap
                                color: theme.textPrimary
                                background: null
                                font.family: theme.mono
                                font.pixelSize: 12
                            }
                        }
                    }
                    Label {
                        text: "RADIAL SETS"; color: theme.textTertiary
                        font.family: theme.face; font.pixelSize: 11
                        font.weight: Font.DemiBold; font.letterSpacing: 0.6
                        Layout.topMargin: 4
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 92
                        radius: 6
                        color: theme.surfaceBg
                        border.color: theme.cardStroke
                        border.width: 1
                        clip: true
                        ScrollView {
                            anchors.fill: parent
                            anchors.margins: 6
                            TextArea {
                                id: radialArea
                                wrapMode: TextArea.NoWrap
                                color: theme.textPrimary
                                background: null
                                font.family: theme.mono
                                font.pixelSize: 12
                            }
                        }
                    }
                }
            }
            }

            // ---------------- wafer row ----------------
            ColumnLayout {
                spacing: theme.gap
                Rectangle {
                    Layout.fillWidth: true
                    color: theme.cardBg
                    radius: theme.radius
                    border.color: theme.cardStroke
                    border.width: 1
                    implicitHeight: rowCol.implicitHeight + theme.pad * 2
                    ColumnLayout {
                        id: rowCol
                        anchors.fill: parent
                        anchors.margins: theme.pad
                        spacing: 8
                        Label {
                            text: "WAFER ROW BATCH"; color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            font.weight: Font.DemiBold; font.letterSpacing: 0.6
                        }
                        Label {
                            Layout.fillWidth: true
                            text: "One run covering every sample in a wafer row, using the VK4 "
                                  + "folder and DXF from the Sample tab."
                            color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                        Label {
                            text: "Wafer map"; color: theme.textSecond
                            font.family: theme.face; font.pixelSize: 13
                            Layout.topMargin: 4
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            TextField {
                                id: mapField
                                Layout.fillWidth: true
                                placeholderText: "Searched automatically"
                                font.family: theme.face
                                font.pixelSize: 12
                                onEditingFinished: root.refreshRows()
                            }
                            Button { text: "Browse"; onClicked: mapDialog.open() }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Label {
                                text: "Row"; color: theme.textSecond
                                font.family: theme.face; font.pixelSize: 13
                            }
                            ComboBox {
                                id: rowCombo
                                Layout.preferredWidth: 104
                                currentIndex: -1
                                displayText: currentIndex < 0 ? "none" : currentText
                            }
                            Button { text: "Rescan"; onClicked: root.refreshRows() }
                            Item { Layout.fillWidth: true }
                        }
                        Label {
                            Layout.fillWidth: true
                            visible: rowCombo.currentIndex >= 0
                            text: "writes results/" + bridge.rowName(vk4Field.text,
                                                                    root.currentRow())
                            color: theme.textTertiary
                            font.family: theme.mono; font.pixelSize: 11
                            elide: Text.ElideMiddle
                        }
                        Label {
                            id: rowProblem
                            Layout.fillWidth: true
                            visible: text !== ""
                            color: theme.danger
                            font.family: theme.face; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                    }
                }
                Item { Layout.fillHeight: true }
            }

            // ---------------- depth calibration ----------------
            ColumnLayout {
                spacing: theme.gap
                Rectangle {
                    Layout.fillWidth: true
                    color: theme.cardBg
                    radius: theme.radius
                    border.color: theme.cardStroke
                    border.width: 1
                    implicitHeight: calCol.implicitHeight + theme.pad * 2
                    ColumnLayout {
                        id: calCol
                        anchors.fill: parent
                        anchors.margins: theme.pad
                        spacing: 8
                        Label {
                            text: "DEPTH CALIBRATION"; color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            font.weight: Font.DemiBold; font.letterSpacing: 0.6
                        }
                        Label {
                            Layout.fillWidth: true
                            text: "Pools the per-sample legacy measurements across results and "
                                  + "solves for the dose that reaches each target depth."
                            color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Label {
                                text: "Target"; color: theme.textSecond
                                font.family: theme.face; font.pixelSize: 13
                            }
                            TextField {
                                id: targetsField
                                Layout.fillWidth: true
                                text: "5"
                                placeholderText: "e.g. 5 or 3,5,8"
                                font.family: theme.face
                                font.pixelSize: 12
                            }
                            Label {
                                text: "um"; color: theme.textTertiary
                                font.family: theme.face; font.pixelSize: 12
                            }
                        }
                        CheckBox { id: legacyQcBox; text: "Allow legacy QC rows" }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: theme.cardBg
                    radius: theme.radius
                    border.color: theme.cardStroke
                    border.width: 1
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: theme.pad
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            Label {
                                text: "POOL"; color: theme.textTertiary
                                font.family: theme.face; font.pixelSize: 11
                                font.weight: Font.DemiBold; font.letterSpacing: 0.6
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                text: root.calibrationPicks.length === 0
                                      ? "all results" : root.calibrationPicks.length + " picked"
                                color: theme.textTertiary
                                font.family: theme.face; font.pixelSize: 11
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 6
                            color: theme.surfaceBg
                            border.color: theme.cardStroke
                            border.width: 1
                            clip: true
                            ListView {
                                id: poolList
                                anchors.fill: parent
                                anchors.margins: 4
                                clip: true
                                spacing: 2
                                model: bridge.calibrationCandidates()
                                delegate: RowLayout {
                                    id: poolRow
                                    required property string modelData
                                    width: poolList.width - 8
                                    spacing: 4
                                    CheckBox {
                                        text: poolRow.modelData
                                        font.family: theme.face
                                        font.pixelSize: 12
                                        Layout.fillWidth: true
                                        onToggled: {
                                            var picks = root.calibrationPicks.slice()
                                            var at = picks.indexOf(poolRow.modelData)
                                            if (checked && at < 0) picks.push(poolRow.modelData)
                                            else if (!checked && at >= 0) picks.splice(at, 1)
                                            root.calibrationPicks = picks
                                        }
                                    }
                                    TextField {
                                        id: specField
                                        Layout.preferredWidth: 104
                                        placeholderText: "all cells"
                                        font.family: theme.mono
                                        font.pixelSize: 11
                                        // Red while invalid, so a bad spec is visible before the run.
                                        color: bridge.validateCellSpec(text) === ""
                                               ? theme.textPrimary : theme.danger
                                        onEditingFinished: {
                                            var problem = bridge.validateCellSpec(text)
                                            cellProblem.text = problem === "" ? ""
                                                : poolRow.modelData + ": " + problem
                                            if (problem === "")
                                                root.setCellSpec(poolRow.modelData, text.trim())
                                        }
                                    }
                                }
                            }
                        }
                        Label {
                            Layout.fillWidth: true
                            text: "Per-sample cell filter: cell_ids to keep, e.g. 1-5, 8, 12-16. "
                                  + "A leading ! excludes instead. Blank uses every cell."
                            color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                        Label {
                            id: cellProblem
                            Layout.fillWidth: true
                            visible: text !== ""
                            color: theme.danger
                            font.family: theme.face; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                        Label {
                            text: "BANDS"; color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            font.weight: Font.DemiBold; font.letterSpacing: 0.6
                            Layout.topMargin: 4
                        }
                        Label {
                            Layout.fillWidth: true
                            text: "min diameter, max diameter, pitch in um. Blank uses the "
                                  + "measurements' own band column."
                            color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 72
                            radius: 6
                            color: theme.surfaceBg
                            border.color: theme.cardStroke
                            border.width: 1
                            clip: true
                            ScrollView {
                                anchors.fill: parent
                                anchors.margins: 6
                                TextArea {
                                    id: bandsArea
                                    wrapMode: TextArea.NoWrap
                                    color: theme.textPrimary
                                    background: null
                                    font.family: theme.mono
                                    font.pixelSize: 12
                                }
                            }
                        }
                    }
                }
            }
            }
        }

        // ---- centre: figures then console
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: theme.gap

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: theme.cardBg
                radius: theme.radius
                border.color: theme.cardStroke
                border.width: 1
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.pad
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Label {
                            text: "FIGURES"; color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            font.weight: Font.DemiBold; font.letterSpacing: 0.6
                        }
                        Label {
                            text: bridge.figures.length + " found"
                            color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            id: exportProblem
                            color: theme.danger
                            font.family: theme.face; font.pixelSize: 11
                            elide: Text.ElideRight
                            Layout.maximumWidth: 300
                        }
                        Button {
                            text: "Export .zip"
                            enabled: root.activeResultName() !== ""
                            onClicked: {
                                exportProblem.text = ""
                                zipDialog.currentFile = "file:///"
                                    + root.activeResultName().replace(/ /g, "_")
                                    + "_figures.zip"
                                zipDialog.open()
                            }
                        }
                        Button {
                            text: "Refresh"
                            enabled: root.activeResultName() !== ""
                            onClicked: bridge.refreshFigures(root.activeResultName())
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 6
                        color: theme.surfaceBg
                        border.color: theme.cardStroke
                        border.width: 1
                        clip: true

                        Label {
                            anchors.centerIn: parent
                            visible: bridge.figures.length === 0
                            text: root.activeResultName() === ""
                                  ? "Pick a sample or wafer row to browse its results"
                                  : "No figures yet for " + root.activeResultName()
                            color: theme.textTertiary
                            font.family: theme.face
                            font.pixelSize: 13
                        }

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            visible: root.selectedFigure !== ""
                            source: root.selectedFigure
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            asynchronous: true
                        }

                        GridView {
                            id: grid
                            anchors.fill: parent
                            anchors.margins: 8
                            visible: root.selectedFigure === "" && bridge.figures.length > 0
                            cellWidth: 172
                            cellHeight: 148
                            clip: true
                            model: bridge.figures
                            delegate: Item {
                                id: cell
                                required property string modelData
                                width: grid.cellWidth - 8
                                height: grid.cellHeight - 8
                                Rectangle {
                                    anchors.fill: parent
                                    radius: 6
                                    color: hover.hovered ? "#242424" : "#1d1d1d"
                                    border.color: hover.hovered ? theme.accent : theme.cardStroke
                                    border.width: 1
                                    Image {
                                        anchors.fill: parent
                                        anchors.margins: 8
                                        anchors.bottomMargin: 22
                                        source: cell.modelData
                                        fillMode: Image.PreserveAspectFit
                                        asynchronous: true
                                        smooth: true
                                    }
                                    Label {
                                        anchors.bottom: parent.bottom
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        anchors.bottomMargin: 5
                                        width: parent.width - 12
                                        horizontalAlignment: Text.AlignHCenter
                                        elide: Text.ElideMiddle
                                        text: cell.modelData.split("/").pop()
                                        color: theme.textTertiary
                                        font.family: theme.face
                                        font.pixelSize: 10
                                    }
                                    HoverHandler { id: hover }
                                    TapHandler {
                                        onTapped: root.selectedFigure = cell.modelData
                                    }
                                }
                            }
                        }

                        Button {
                            anchors.top: parent.top
                            anchors.right: parent.right
                            anchors.margins: 8
                            visible: root.selectedFigure !== ""
                            text: "Back to all"
                            onClicked: root.selectedFigure = ""
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 200
                color: theme.cardBg
                radius: theme.radius
                border.color: theme.cardStroke
                border.width: 1
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.pad
                    spacing: 8
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "CONSOLE"; color: theme.textTertiary
                            font.family: theme.face; font.pixelSize: 11
                            font.weight: Font.DemiBold; font.letterSpacing: 0.6
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: bridge.status
                            color: theme.textSecond
                            font.family: theme.face
                            font.pixelSize: 12
                            elide: Text.ElideRight
                        }
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 6
                        color: theme.surfaceBg
                        border.color: theme.cardStroke
                        border.width: 1
                        clip: true
                        ScrollView {
                            anchors.fill: parent
                            anchors.margins: 8
                            TextArea {
                                id: consoleArea
                                readOnly: true
                                wrapMode: TextArea.NoWrap
                                color: "#a8d8a8"
                                background: null
                                font.family: theme.mono
                                font.pixelSize: 12
                            }
                        }
                    }
                }
            }
        }
    }

    Connections {
        target: bridge
        function onLogAppended(chunk) {
            consoleArea.text += chunk
            consoleArea.cursorPosition = consoleArea.length
        }
        function onLogCleared() { consoleArea.text = "" }
    }
}
